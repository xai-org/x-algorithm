use crate::inputs::{CandidateScoringInputs, QueryScoringContext};
use crate::weights::{MIN_NOT_DWELLED_POWER_OFFSET, NEGATIVE_SCORES_OFFSET, ValueModelWeights};

const MIN_SKIP_BASE: f64 = 1e-6;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValueScores {
    pub weighted: Vec<f64>,
    pub scores: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostFusionMultiplier {
    pub author_diversity: f64,
    pub oon: f64,
}

impl PostFusionMultiplier {
    pub fn combined(&self) -> f64 {
        self.author_diversity * self.oon
    }

    pub fn apply(&self, score: f64) -> f64 {
        score * self.author_diversity * self.oon
    }
}

// These weights reflect a combination of how much an action is
// valued in ranking and typical propensities of these actions
// across the X network (e.g. negative feedback is overall rare).

// Each weight multiplies the *predicted* probability of that
// action (P(favorite), P(repost), …) or a continuous value e.g.
// watch time -- the weights do not multiply raw engagement counts.
// One common misinterpretation is that you can read these weight
// ratios as count equivalences, e.g. the incorrect statement that
// "one report cancels 468 likes" -- this is incorrect because the
// weights apply to the predicted probabilities rather than raw counts.

// And the baseline probability of a Report is more than 1000x lower
// than a Like, so it’s weighted more to allow the prediction to affect
// the final ranking at all.

// Related to the above is a misunderstanding that bad actors engaging
// in mass blocking/reporting will significantly suppress reach. There
// are multiple things inhibiting this:
// 1. It’s predicting your likelihood of the action, not summing up
// raw weights on counts. Also, recommendations are personalized, so
// reports from bad actors will primarily affect recommendations for
// users who are similar to the bad actors, rather than having the same
// effect on the post's ranking to everyone.
// 2. For an account to count in the algorithms recommendation system,
// it must take place on a post served in Home Timeline. Directly
// navigating to a post (i.e., coordinating via groupchat) has no
// ranking impact. And users cannot manufacture a post to show up in
// their Timeline in any consistently reproducible way.
fn apply(score: Option<f64>, weight: f64) -> f64 {
    score.unwrap_or(0.0) * weight
}

fn product(probability: Option<f64>, conditional_value: Option<f64>) -> Option<f64> {
    Some(probability? * conditional_value?)
}

pub fn fuse_heads(weights: &ValueModelWeights, candidate: &CandidateScoringInputs) -> f64 {
    if weights.multiplicative {
        multiplicative_score(weights, candidate)
    } else {
        offset_score(compute_weighted_score(weights, candidate), weights)
    }
}

fn skip_score(weights: &ValueModelWeights, candidate: &CandidateScoringInputs) -> f64 {
    let Some(p) = candidate.phoenix_scores.not_dwelled_score else {
        return 1.0;
    };
    if !p.is_finite() || weights.not_dwelled_power_exponent == 0.0 {
        return 1.0;
    }
    (p + weights
        .not_dwelled_power_offset
        .max(MIN_NOT_DWELLED_POWER_OFFSET))
    .max(MIN_SKIP_BASE)
    .powf(weights.not_dwelled_power_exponent)
}

fn engage_sum(weights: &ValueModelWeights, candidate: &CandidateScoringInputs) -> f64 {
    weighted_terms(weights, candidate)
        .iter()
        .filter(|(name, _)| *name != "dwell_time" && *name != "not_dwelled")
        .map(|(_, t)| *t)
        .sum()
}

fn time_factor(weights: &ValueModelWeights, candidate: &CandidateScoringInputs) -> f64 {
    1.0 + apply(candidate.phoenix_scores.dwell_time, weights.cont_dwell_time).max(0.0)
}

fn multiplicative_score(weights: &ValueModelWeights, candidate: &CandidateScoringInputs) -> f64 {
    let engage = engage_sum(weights, candidate) + weights.engage_bias;
    if engage < 0.0 {
        return offset_score(engage, weights);
    }
    offset_score(
        skip_score(weights, candidate) * engage * time_factor(weights, candidate),
        weights,
    )
}

pub fn compute_weighted_score(
    weights: &ValueModelWeights,
    candidate: &CandidateScoringInputs,
) -> f64 {
    weighted_terms(weights, candidate)
        .iter()
        .map(|(_, t)| t)
        .sum()
}

pub fn weighted_terms(
    weights: &ValueModelWeights,
    candidate: &CandidateScoringInputs,
) -> [(&'static str, f64); 30] {
    let scores = &candidate.phoenix_scores;

    let vqv_weight = if candidate.vqv_eligible {
        weights.vqv
    } else {
        0.0
    };
    let quoted_vqv_weight = if candidate.quoted_vqv_eligible {
        weights.quoted_vqv
    } else {
        0.0
    };
    let post_unexplored_weight =
        if weights.post_unexplored_include_out_of_network || candidate.in_network == Some(true) {
            weights.post_unexplored
        } else {
            0.0
        };

    [
        ("favorite", apply(scores.favorite_score, weights.favorite)),
        (
            "reply",
            apply(scores.reply_score, weights.reply_weight_for(candidate)),
        ),
        ("retweet", apply(scores.retweet_score, weights.retweet)),
        (
            "photo_expand",
            apply(scores.photo_expand_score, weights.photo_expand),
        ),
        (
            "video_open",
            apply(scores.video_open_score, weights.video_open),
        ),
        ("click", apply(scores.click_score, weights.click)),
        (
            "open_link",
            apply(scores.open_link_score, weights.open_link),
        ),
        (
            "profile_click",
            apply(scores.profile_click_score, weights.profile_click),
        ),
        ("vqv", apply(scores.vqv_score, vqv_weight)),
        ("share", apply(scores.share_score, weights.share)),
        (
            "share_via_dm",
            apply(scores.share_via_dm_score, weights.share_via_dm),
        ),
        (
            "share_via_copy_link",
            apply(
                scores.share_via_copy_link_score,
                weights.share_via_copy_link,
            ),
        ),
        (
            "dwell",
            apply(scores.dwell_score, weights.dwell_weight_for(candidate)),
        ),
        ("quote", apply(scores.quote_score, weights.quote)),
        (
            "quoted_click",
            apply(scores.quoted_click_score, weights.quoted_click),
        ),
        (
            "quoted_vqv",
            apply(scores.quoted_vqv_score, quoted_vqv_weight),
        ),
        (
            "dwell_time",
            apply(scores.dwell_time, weights.cont_dwell_time),
        ),
        (
            "click_dwell_time",
            apply(scores.click_dwell_time, weights.cont_click_dwell_time),
        ),
        (
            "video_continuation",
            apply(
                product(scores.video_open_score, scores.home_video_continuation_secs),
                weights.video_continuation,
            ),
        ),
        (
            "user_video_continuation",
            apply(
                product(
                    scores.video_open_score,
                    candidate.user_video_continuation_secs,
                ),
                weights.user_video_continuation,
            ),
        ),
        (
            "profile_visit_secs",
            apply(
                product(scores.profile_click_score, scores.home_profile_visit_secs),
                weights.profile_visit_secs,
            ),
        ),
        (
            "follow_author",
            apply(scores.follow_author_score, weights.follow_author),
        ),
        (
            "not_interested",
            apply(scores.not_interested_score, weights.not_interested),
        ),
        (
            "block_author",
            apply(scores.block_author_score, weights.block_author),
        ),
        (
            "mute_author",
            apply(scores.mute_author_score, weights.mute_author),
        ),
        ("report", apply(scores.report_score, weights.report)),
        (
            "not_dwelled",
            apply(scores.not_dwelled_score, weights.not_dwelled),
        ),
        (
            "post_unexplored",
            apply(scores.post_unexplored_score, post_unexplored_weight),
        ),
        (
            "post_video",
            apply(scores.post_video_score, weights.post_video),
        ),
        (
            "grok_analyze",
            apply(scores.grok_analyze_score, weights.grok_analyze),
        ),
    ]
}

pub fn set_user_video_continuation(candidates: &mut [CandidateScoringInputs]) {
    let (open_weighted_secs, opens) = candidates
        .iter()
        .filter_map(|c| {
            let scores = &c.phoenix_scores;
            Some((
                scores.video_open_score?,
                scores.home_video_continuation_secs?,
            ))
        })
        .fold((0.0, 0.0), |(secs, opens), (open, continuation)| {
            (secs + open * continuation, opens + open)
        });
    let user_secs = (opens > 0.0).then(|| open_weighted_secs / opens);
    for c in candidates {
        c.user_video_continuation_secs = user_secs;
    }
}

pub fn offset_score(combined_score: f64, w: &ValueModelWeights) -> f64 {
    let total_sum = w.offset_total_sum();
    if total_sum == 0.0 {
        combined_score.max(0.0)
    } else if combined_score < 0.0 {
        (combined_score + w.offset_negative_sum()) / total_sum * NEGATIVE_SCORES_OFFSET
    } else {
        combined_score + NEGATIVE_SCORES_OFFSET
    }
}

pub fn unoffset_score(weighted_score: f64, w: &ValueModelWeights) -> f64 {
    let total_sum = w.offset_total_sum();
    if total_sum == 0.0 {
        weighted_score
    } else if weighted_score < NEGATIVE_SCORES_OFFSET {
        weighted_score / NEGATIVE_SCORES_OFFSET * total_sum - w.offset_negative_sum()
    } else {
        weighted_score - NEGATIVE_SCORES_OFFSET
    }
}

pub fn diversity_multiplier(decay_factor: f64, floor: f64, exponent: f64) -> f64 {
    (1.0 - floor) * decay_factor.powf(exponent) + floor
}

pub fn author_pool_counts(author_ids: &[u64], ordering_scores: &[f64]) -> Vec<u32> {
    let mut indexed: Vec<(usize, f64)> = ordering_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    let mut counts = vec![0u32; author_ids.len()];
    let mut author_counts: FxHashMap<u64, u32> = FxHashMap::default();
    for (idx, _) in indexed {
        let author_id = author_ids[idx];
        let k = author_counts.get(&author_id).copied().unwrap_or(0);
        counts[idx] = k;
        author_counts.insert(author_id, k + 1);
    }
    counts
}

fn oon_applies(weights: &ValueModelWeights, candidate: &CandidateScoringInputs) -> bool {
    match candidate.in_network {
        Some(false) => true,
        Some(true) => {
            weights.oon_rescore_in_network_replies_retweets
                && (candidate.is_reply || candidate.is_retweet)
        }
        None => false,
    }
}

fn class_factor_score(
    weights: &ValueModelWeights,
    ctx: &QueryScoringContext,
    candidate: &CandidateScoringInputs,
    score: f64,
) -> Option<f64> {
    if weights.multiplicative {
        return None;
    }
    let (factors, level) = if candidate.in_network? {
        let level = if oon_applies(weights, candidate) {
            ctx.effective_oon_weight
        } else {
            1.0
        };
        (ctx.in_network_factors?, level)
    } else {
        (ctx.oon_factors?, 1.0)
    };
    let dwell_time = apply(candidate.phoenix_scores.dwell_time, weights.cont_dwell_time);
    let dwell_time = if score >= NEGATIVE_SCORES_OFFSET && dwell_time.is_finite() {
        dwell_time.clamp(0.0, score)
    } else {
        0.0
    };
    Some(level * (factors.engagement * (score - dwell_time) + factors.dwell_time * dwell_time))
}

pub fn post_fusion_multipliers(
    weights: &ValueModelWeights,
    ctx: &QueryScoringContext,
    candidates: &[CandidateScoringInputs],
    ordering_scores: &[f64],
) -> Vec<PostFusionMultiplier> {
    let author_diversity: Vec<f64> = if weights.enable_author_diversity {
        let author_ids: Vec<u64> = candidates.iter().map(|c| c.author_id).collect();
        author_pool_counts(&author_ids, ordering_scores)
            .into_iter()
            .map(|k| {
                diversity_multiplier(
                    weights.author_diversity_decay,
                    weights.author_diversity_floor,
                    f64::from(k),
                )
            })
            .collect()
    } else {
        vec![1.0; candidates.len()]
    };

    candidates
        .iter()
        .zip(author_diversity)
        .map(|(c, author_diversity)| PostFusionMultiplier {
            author_diversity,
            oon: if oon_applies(weights, c) {
                ctx.effective_oon_weight
            } else {
                1.0
            },
        })
        .collect()
}

pub fn compute_value_scores(
    weights: &ValueModelWeights,
    ctx: &QueryScoringContext,
    candidates: &[CandidateScoringInputs],
) -> ValueScores {
    compute_value_scores_with_adjustment(weights, ctx, candidates, |scores| {
        apply_cold_start_decisions(scores, candidates)
    })
}

pub fn apply_cold_start_decisions(
    scores: &[f64],
    candidates: &[CandidateScoringInputs],
) -> Vec<f64> {
    let mut effective = scores.to_vec();
    let lift = candidates
        .iter()
        .enumerate()
        .find_map(|(i, c)| c.cold_start_lift_to_rank.map(|rank| (i, rank as usize)));
    if let Some((index, rank)) = lift
        && rank < scores.len()
    {
        let mut ranked = scores.to_vec();
        ranked.sort_by(|a, b| b.total_cmp(a));
        effective[index] = effective[index].max(ranked[rank]);
    }
    effective
}

pub fn compute_value_scores_with_adjustment<F>(
    weights: &ValueModelWeights,
    ctx: &QueryScoringContext,
    candidates: &[CandidateScoringInputs],
    adjust_base_scores: F,
) -> ValueScores
where
    F: FnOnce(&[f64]) -> Vec<f64>,
{
    let weighted: Vec<f64> = candidates
        .iter()
        .map(|c| match c.weighted_score {
            Some(weighted) if !weights.multiplicative => weighted,
            _ => fuse_heads(weights, c),
        })
        .collect();

    if weights.multiplier_pre_offset {
        let multipliers = post_fusion_multipliers(weights, ctx, candidates, &weighted);
        let pre_offset_scaled: Vec<f64> = weighted
            .iter()
            .zip(candidates)
            .zip(&multipliers)
            .map(|((&weighted, c), m)| {
                let net = unoffset_score(weighted, weights) + c.author_exploration_bonus;
                let scaled = if net >= 0.0 { m.combined() * net } else { net };
                offset_score(scaled, weights)
            })
            .collect();
        let scores = adjust_base_scores(&pre_offset_scaled);
        return ValueScores { weighted, scores };
    }

    let base: Vec<f64> = weighted
        .iter()
        .zip(candidates)
        .map(|(&w, c)| add_exploration_bonus(w, c.author_exploration_bonus, weights))
        .collect();
    let adjusted = adjust_base_scores(&base);
    let multipliers = post_fusion_multipliers(weights, ctx, candidates, &adjusted);
    let scores = adjusted
        .iter()
        .zip(&multipliers)
        .zip(candidates)
        .map(
            |((&s, m), c)| match class_factor_score(weights, ctx, c, s) {
                Some(class_score) => class_score * m.author_diversity,
                None => m.apply(s),
            },
        )
        .collect();
    ValueScores { weighted, scores }
}

fn add_exploration_bonus(weighted: f64, bonus: f64, weights: &ValueModelWeights) -> f64 {
    if bonus == 0.0 {
        return weighted;
    }
    offset_score(unoffset_score(weighted, weights) + bonus, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::ClassFactors;
    use crate::phoenix_scores::PhoenixScores;

    #[test]
    fn hand_computed_parity() {
        let weights = ValueModelWeights {
            favorite: 2.0,
            reply: 4.0,
            retweet: 1.0,
            dwell: 0.5,
            vqv: 3.0,
            cont_dwell_time: 0.1,
            cont_click_dwell_time: 0.2,
            post_unexplored: 1.5,
            post_video: 2.5,
            report: -100.0,
            not_interested: -10.0,
            bidirectional_follow_reply_weight_boost: 1.0,
            bidirectional_follow_dwell_weight_boost: 0.5,
            enable_author_diversity: true,
            author_diversity_decay: 0.5,
            author_diversity_floor: 0.25,
            oon_rescore_in_network_replies_retweets: true,
            ..Default::default()
        };
        let ctx = QueryScoringContext {
            effective_oon_weight: 0.75,
            ..Default::default()
        };

        let mutual_original = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.5),
                reply_score: Some(0.1),
                dwell_score: Some(0.4),
                dwell_time: Some(10.0),
                post_unexplored_score: Some(0.2),
                post_video_score: Some(0.04),
                ..Default::default()
            },
            author_id: 1,
            in_network: Some(true),
            is_mutual_follow_author: true,
            ..Default::default()
        };
        let ineligible_video_oon = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.2),
                vqv_score: Some(0.9),
                click_dwell_time: Some(5.0),
                post_unexplored_score: Some(0.2),
                ..Default::default()
            },
            author_id: 2,
            in_network: Some(false),
            vqv_eligible: false,
            ..Default::default()
        };
        let eligible_video_same_author = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.05),
                vqv_score: Some(0.5),
                click_dwell_time: Some(5.0),
                ..Default::default()
            },
            author_id: 1,
            in_network: Some(true),
            vqv_eligible: true,
            ..Default::default()
        };
        let reported_in_network_reply = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.01),
                report_score: Some(0.05),
                ..Default::default()
            },
            author_id: 3,
            in_network: Some(true),
            is_reply: true,
            is_mutual_follow_author: true,
            ..Default::default()
        };
        let candidates = [
            mutual_original,
            ineligible_video_oon,
            eligible_video_same_author,
            reported_in_network_reply,
        ];

        let offset = 0.001;
        let positive_weight_sum = 2.0 + 4.0 + 1.0 + 0.5 + 3.0 + 1.5 + 2.5;
        let negative_weight_magnitude = 10.0 + 100.0;
        let total_weight_sum = positive_weight_sum + negative_weight_magnitude;

        let boosted_reply_weight = 4.0 + 1.0;
        let boosted_dwell_weight = 0.5 + 0.5;
        let weighted_mutual_original = 2.0 * 0.5
            + boosted_reply_weight * 0.1
            + boosted_dwell_weight * 0.4
            + 0.1 * 10.0
            + 1.5 * 0.2
            + 2.5 * 0.04
            + offset;

        let weighted_ineligible_video_oon = 2.0 * 0.2 + 0.2 * 5.0 + offset;

        let weighted_eligible_video_same_author = 2.0 * 0.05 + 3.0 * 0.5 + 0.2 * 5.0 + offset;

        let net_reported_reply = 2.0 * 0.01 - 100.0 * 0.05;
        let weighted_reported_reply =
            (net_reported_reply + negative_weight_magnitude) / total_weight_sum * offset;

        let diversity_second_from_author = (1.0 - 0.25) * 0.5 + 0.25;

        let expected_scores = [
            weighted_mutual_original,
            weighted_ineligible_video_oon * 0.75,
            weighted_eligible_video_same_author * diversity_second_from_author,
            weighted_reported_reply * 0.75,
        ];
        let expected_weighted = [
            weighted_mutual_original,
            weighted_ineligible_video_oon,
            weighted_eligible_video_same_author,
            weighted_reported_reply,
        ];

        let result = compute_value_scores(&weights, &ctx, &candidates);
        for (i, (got, want)) in result.scores.iter().zip(expected_scores).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "candidate {i}: got {got} want {want}"
            );
        }
        for (i, (got, want)) in result.weighted.iter().zip(expected_weighted).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "weighted {i}: got {got} want {want}"
            );
        }
        assert!((fuse_heads(&weights, &candidates[0]) - weighted_mutual_original).abs() < 1e-12);
    }

    fn class_factor_weights() -> ValueModelWeights {
        ValueModelWeights {
            favorite: 2.0,
            cont_dwell_time: 0.1,
            report: -100.0,
            oon_rescore_in_network_replies_retweets: true,
            ..Default::default()
        }
    }

    fn class_factor_candidate(
        in_network: Option<bool>,
        favorite: f64,
        dwell_secs: Option<f64>,
    ) -> CandidateScoringInputs {
        CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(favorite),
                dwell_time: dwell_secs,
                ..Default::default()
            },
            in_network,
            ..Default::default()
        }
    }

    fn assert_scores(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (got, want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "candidate {i}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn class_factors_scale_dwell_time_and_engagement_separately() {
        let weights = class_factor_weights();
        let ctx = QueryScoringContext {
            effective_oon_weight: 0.75,
            oon_factors: ClassFactors::new(0.6, 0.9),
            in_network_factors: ClassFactors::new(1.2, 0.7),
        };
        let reply = CandidateScoringInputs {
            is_reply: true,
            ..class_factor_candidate(Some(true), 0.2, Some(4.0))
        };
        let reported = |in_network, report| CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                report_score: Some(report),
                dwell_time: Some(4.0),
                ..Default::default()
            },
            in_network: Some(in_network),
            ..Default::default()
        };
        let candidates = [
            class_factor_candidate(Some(false), 0.2, Some(4.0)),
            class_factor_candidate(Some(true), 0.2, Some(4.0)),
            reply,
            class_factor_candidate(Some(false), 0.2, None),
            class_factor_candidate(None, 0.2, Some(4.0)),
            reported(false, 0.003),
            reported(true, 0.003),
            reported(false, 0.05),
        ];

        let offset = 0.001;
        let engagement = 2.0 * 0.2 + offset;
        let dwell_time = 0.1 * 4.0;
        let capped = 0.4 - 100.0 * 0.003 + offset;
        let negative = (0.4 - 100.0 * 0.05 + 100.0) / 102.0 * offset;
        let want = [
            0.6 * engagement + 0.9 * dwell_time,
            1.2 * engagement + 0.7 * dwell_time,
            0.75 * (1.2 * engagement + 0.7 * dwell_time),
            0.6 * engagement,
            engagement + dwell_time,
            0.9 * capped,
            0.7 * capped,
            0.6 * negative,
        ];
        let got = compute_value_scores(&weights, &ctx, &candidates);
        assert_scores(&got.scores, &want);
        assert!((got.weighted[0] - (engagement + dwell_time)).abs() < 1e-12);
    }

    #[test]
    fn neutral_class_factors_match_the_single_oon_factor() {
        let weights = ValueModelWeights {
            enable_author_diversity: true,
            author_diversity_decay: 0.5,
            author_diversity_floor: 0.25,
            ..class_factor_weights()
        };
        let single = QueryScoringContext {
            effective_oon_weight: 0.75,
            ..Default::default()
        };
        let neutral = QueryScoringContext {
            oon_factors: ClassFactors::new(0.75, 0.75),
            in_network_factors: ClassFactors::new(1.0, 1.0),
            ..single.clone()
        };
        let retweet = CandidateScoringInputs {
            is_retweet: true,
            ..class_factor_candidate(Some(true), 0.3, Some(8.0))
        };
        let negative = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                report_score: Some(0.05),
                dwell_time: Some(4.0),
                ..Default::default()
            },
            in_network: Some(false),
            ..Default::default()
        };
        let candidates = [
            class_factor_candidate(Some(false), 0.2, Some(4.0)),
            class_factor_candidate(Some(false), 0.1, Some(9.0)),
            class_factor_candidate(Some(true), 0.2, Some(4.0)),
            retweet,
            class_factor_candidate(Some(false), 0.2, None),
            class_factor_candidate(None, 0.2, Some(4.0)),
            negative,
        ];
        let want = compute_value_scores(&weights, &single, &candidates);
        let got = compute_value_scores(&weights, &neutral, &candidates);
        assert_scores(&got.scores, &want.scores);
        assert!(want.scores[1] < 0.75 * want.weighted[1]);
    }

    #[test]
    fn class_factors_use_a_cached_weighted_score() {
        let weights = class_factor_weights();
        let ctx = QueryScoringContext {
            effective_oon_weight: 0.75,
            oon_factors: ClassFactors::new(0.6, 0.9),
            in_network_factors: None,
        };
        let cached = CandidateScoringInputs {
            weighted_score: Some(1.0),
            ..class_factor_candidate(Some(false), 0.2, Some(4.0))
        };
        let in_network = CandidateScoringInputs {
            weighted_score: Some(1.0),
            ..class_factor_candidate(Some(true), 0.2, Some(4.0))
        };
        let got = compute_value_scores(&weights, &ctx, &[cached, in_network]);
        assert_scores(&got.scores, &[0.6 * (1.0 - 0.4) + 0.9 * 0.4, 1.0]);
        assert_scores(&got.weighted, &[1.0, 1.0]);
    }

    #[test]
    fn class_factors_are_ignored_in_multiplicative_and_pre_offset_modes() {
        let ctx = QueryScoringContext {
            effective_oon_weight: 0.75,
            oon_factors: ClassFactors::new(0.6, 0.9),
            in_network_factors: ClassFactors::new(1.2, 0.7),
        };
        let single = QueryScoringContext {
            effective_oon_weight: 0.75,
            ..Default::default()
        };
        let candidates = [
            class_factor_candidate(Some(false), 0.2, Some(4.0)),
            class_factor_candidate(Some(true), 0.2, Some(4.0)),
        ];
        for weights in [
            ValueModelWeights {
                multiplicative: true,
                ..class_factor_weights()
            },
            ValueModelWeights {
                multiplier_pre_offset: true,
                ..class_factor_weights()
            },
        ] {
            let want = compute_value_scores(&weights, &single, &candidates);
            let got = compute_value_scores(&weights, &ctx, &candidates);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn class_factors_reject_invalid_values() {
        assert_eq!(
            ClassFactors::new(0.6, 0.9),
            Some(ClassFactors {
                engagement: 0.6,
                dwell_time: 0.9
            })
        );
        assert!(ClassFactors::new(0.0, 0.0).is_some());
        assert_eq!(ClassFactors::new(-1.0, 0.9), None);
        assert_eq!(ClassFactors::new(0.6, -1.0), None);
        assert_eq!(ClassFactors::new(f64::NAN, 0.9), None);
        assert_eq!(ClassFactors::new(0.6, f64::INFINITY), None);
    }

    #[test]
    fn multiplicative_score_is_skip_times_engage_times_time() {
        let weights = ValueModelWeights {
            favorite: 0.5,
            video_continuation: 0.003,
            report: -100.0,
            cont_dwell_time: 0.1,
            multiplicative: true,
            not_dwelled_power_exponent: -0.5,
            not_dwelled_power_offset: 0.11,
            engage_bias: 0.1,
            ..Default::default()
        };
        let positive = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.4),
                video_open_score: Some(0.5),
                home_video_continuation_secs: Some(10.0),
                dwell_time: Some(20.0),
                not_dwelled_score: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        };
        let want = (0.36_f64).powf(-0.5) * 0.315 * 3.0 + 0.001;
        let got = fuse_heads(&weights, &positive);
        assert!((got - want).abs() < 1e-12, "got {got} want {want}");

        let toxic = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                report_score: Some(0.05),
                dwell_time: Some(20.0),
                not_dwelled_score: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        };
        let toxic_score = fuse_heads(&weights, &toxic);
        assert!(toxic_score < 0.002);
        assert!(toxic_score < got);

        let stale = CandidateScoringInputs {
            weighted_score: Some(99.0),
            phoenix_scores: positive.phoenix_scores.clone(),
            ..Default::default()
        };
        let ctx = QueryScoringContext::default();
        let recomputed =
            compute_value_scores(&weights, &ctx, std::slice::from_ref(&stale)).weighted[0];
        assert!((recomputed - want).abs() < 1e-12);
        let additive = ValueModelWeights {
            multiplicative: false,
            ..weights
        };
        let cached =
            compute_value_scores(&additive, &ctx, std::slice::from_ref(&stale)).weighted[0];
        assert!((cached - 99.0).abs() < 1e-12);
    }
}
