use crate::inputs::{CandidateScoringInputs, QueryScoringContext};
use crate::score_order::compare_scores_desc;
use crate::weights::{NEGATIVE_SCORES_OFFSET, ValueModelWeights};
use rustc_hash::FxHashMap;

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
    offset_score(compute_weighted_score(weights, candidate), weights)
}

pub fn compute_weighted_score(
    weights: &ValueModelWeights,
    candidate: &CandidateScoringInputs,
) -> f64 {
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
        apply(scores.favorite_score, weights.favorite),
        apply(scores.reply_score, weights.reply_weight_for(candidate)),
        apply(scores.retweet_score, weights.retweet),
        apply(scores.photo_expand_score, weights.photo_expand),
        apply(scores.video_open_score, weights.video_open),
        apply(scores.click_score, weights.click),
        apply(scores.open_link_score, weights.open_link),
        apply(scores.profile_click_score, weights.profile_click),
        apply(scores.vqv_score, vqv_weight),
        apply(scores.share_score, weights.share),
        apply(scores.share_via_dm_score, weights.share_via_dm),
        apply(
            scores.share_via_copy_link_score,
            weights.share_via_copy_link,
        ),
        apply(scores.dwell_score, weights.dwell_weight_for(candidate)),
        apply(scores.quote_score, weights.quote),
        apply(scores.quoted_click_score, weights.quoted_click),
        apply(scores.quoted_vqv_score, quoted_vqv_weight),
        apply(scores.dwell_time, weights.cont_dwell_time),
        apply(scores.click_dwell_time, weights.cont_click_dwell_time),
        apply(
            product(scores.video_open_score, scores.home_video_continuation_secs),
            weights.video_continuation,
        ),
        apply(
            product(
                scores.video_open_score,
                candidate.user_video_continuation_secs,
            ),
            weights.user_video_continuation,
        ),
        apply(
            product(scores.profile_click_score, scores.home_profile_visit_secs),
            weights.profile_visit_secs,
        ),
        apply(scores.follow_author_score, weights.follow_author),
        apply(scores.not_interested_score, weights.not_interested),
        apply(scores.block_author_score, weights.block_author),
        apply(scores.mute_author_score, weights.mute_author),
        apply(scores.report_score, weights.report),
        apply(scores.not_dwelled_score, weights.not_dwelled),
        apply(scores.post_unexplored_score, post_unexplored_weight),
        apply(scores.post_video_score, weights.post_video),
    ]
    .iter()
    .sum()
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
    let total_sum = w.total_sum();
    if total_sum == 0.0 {
        combined_score.max(0.0)
    } else if combined_score < 0.0 {
        (combined_score + w.negative_sum()) / total_sum * NEGATIVE_SCORES_OFFSET
    } else {
        combined_score + NEGATIVE_SCORES_OFFSET
    }
}

pub fn unoffset_score(weighted_score: f64, w: &ValueModelWeights) -> f64 {
    let total_sum = w.total_sum();
    if total_sum == 0.0 {
        weighted_score
    } else if weighted_score < NEGATIVE_SCORES_OFFSET {
        weighted_score / NEGATIVE_SCORES_OFFSET * total_sum - w.negative_sum()
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
    indexed.sort_by(|(_, a), (_, b)| compare_scores_desc(*a, *b));

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
            Some(weighted) => weighted,
            None => offset_score(compute_weighted_score(weights, c), weights),
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
        .map(|(&s, m)| m.apply(s))
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
    use crate::phoenix_scores::PhoenixScores;

    #[test]
    fn author_counts_nan_witness() {
        let scores = [
            19.0,
            10.0,
            7.0,
            18.0,
            8.0,
            5.0,
            2.0,
            17.0,
            16.0,
            3.0,
            15.0,
            20.0,
            21.0,
            11.0,
            6.0,
            4.0,
            1.0,
            f64::NAN,
            13.0,
            12.0,
            14.0,
            9.0,
        ];
        let distinct_authors: Vec<u64> = (0..scores.len() as u64).collect();
        assert_eq!(
            author_pool_counts(&distinct_authors, &scores),
            vec![0; scores.len()]
        );
        // Reusing an author makes the rank of each score observable in the counts.
        let expected: Vec<u32> = scores
            .iter()
            .map(|&score| {
                if score.is_nan() {
                    21
                } else {
                    21 - score as u32
                }
            })
            .collect();
        assert_eq!(author_pool_counts(&[7; 22], &scores), expected);
    }

    #[test]
    fn author_counts_keep_invalid_scores_after_finite_scores() {
        let scores = [f64::NAN, 5.0, f64::INFINITY, -2.0, f64::NEG_INFINITY, 10.0];
        assert_eq!(author_pool_counts(&[7; 6], &scores), vec![3, 1, 4, 2, 5, 0]);
    }

    #[test]
    fn author_counts_preserve_finite_ties_and_author_groups() {
        assert_eq!(
            author_pool_counts(&[7, 7, 8, 7, 7], &[0.0, -0.0, 10.0, 5.0, 5.0]),
            vec![2, 3, 0, 0, 1],
        );
        assert!(author_pool_counts(&[], &[]).is_empty());
    }

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
}
