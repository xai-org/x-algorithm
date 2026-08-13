use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    AuthorIsControl, AuthorIsTreatment, ColdStartFollowerCap, ColdStartFollowerDecayWidth,
    ColdStartImpressionDecayWidth, ColdStartImpressionThreshold, ColdStartMaxPostAgeSecs,
    ColdStartSlotMax, ColdStartSlotMin, EnableViewerColdStart, LowImpressionsMaxPositionRatio,
    PhoenixMoeCodivertViewerIsControl, PhoenixMoeCodivertViewerIsTreatment,
};
use crate::util::author_rules::AuthorRulesEvaluator;
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_home_mixer_proto as pb;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewerArm {
    Holdout,
    Control,
    Treatment,
}

impl ViewerArm {
    fn as_str(self) -> &'static str {
        match self {
            Self::Holdout => "holdout",
            Self::Control => "control",
            Self::Treatment => "treatment",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthorCorpus {
    NotBucketed,
    Control,
    Treatment,
}

fn viewer_arm(query: &ScoredPostsQuery) -> ViewerArm {
    if query.params.get(PhoenixMoeCodivertViewerIsTreatment) {
        return ViewerArm::Treatment;
    }
    if query.params.get(PhoenixMoeCodivertViewerIsControl) {
        return ViewerArm::Control;
    }
    ViewerArm::Holdout
}

fn positions_among_nonzero(scores: &[f64]) -> (Vec<usize>, usize) {
    let mut order: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| **s != 0.0)
        .map(|(i, _)| i)
        .collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let nonzero = order.len();
    let mut positions = vec![usize::MAX; scores.len()];
    for (position, &i) in order.iter().enumerate() {
        positions[i] = position;
    }
    (positions, nonzero)
}

fn record_cold_started_posts(is_moe: bool, viewer_arm: &str, count: u64) {
    if count == 0 {
        return;
    }
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.incr(
            "home_mixer.cold_started_posts_total",
            &[
                ("is_moe", if is_moe { "true" } else { "false" }),
                ("viewer_arm", viewer_arm),
            ],
            count,
        );
    }
}

fn is_phoenix_moe(c: &PostCandidate) -> bool {
    c.served_type == Some(pb::ServedType::ForYouPhoenixRetrievalMoe)
}

/// Returns a smooth, bounded confidence that is full through `threshold` and
/// decays to zero over `decay_width`. A zero-width rollout preserves the prior
/// binary comparison.
fn threshold_confidence(
    value: u64,
    threshold: u64,
    decay_width: u64,
    legacy_threshold_inclusive: bool,
) -> f64 {
    if decay_width == 0 {
        return if value < threshold || (legacy_threshold_inclusive && value == threshold) {
            1.0
        } else {
            0.0
        };
    }
    if value <= threshold {
        return 1.0;
    }
    let end = threshold.saturating_add(decay_width);
    if value >= end {
        return 0.0;
    }
    let progress = (value - threshold) as f64 / decay_width as f64;
    let smoothstep = progress * progress * (3.0 - 2.0 * progress);
    (1.0 - smoothstep).clamp(0.0, 1.0)
}

fn follower_confidence(followers: i32, follower_cap: i64, decay_width: u32) -> f64 {
    if decay_width == 0 {
        return if (followers as i64) <= follower_cap {
            1.0
        } else {
            0.0
        };
    }
    if follower_cap < 0 {
        return 0.0;
    }
    threshold_confidence(
        followers.max(0) as u64,
        follower_cap as u64,
        decay_width as u64,
        true,
    )
}

pub(crate) fn cold_start_base_eligible(
    c: &PostCandidate,
    follower_cap: i64,
    follower_decay_width: u32,
) -> bool {
    c.in_reply_to_tweet_id.is_none()
        && c.retweeted_tweet_id.is_none()
        && c.author_followers_count.is_some_and(|followers| {
            follower_confidence(followers, follower_cap, follower_decay_width) > 0.0
        })
}

fn author_corpus(
    author_rules: &AuthorRulesEvaluator,
    candidates: &[PostCandidate],
) -> Vec<AuthorCorpus> {
    candidates
        .iter()
        .map(|c| {
            if author_rules.get(c.author_id, AuthorIsTreatment) {
                AuthorCorpus::Treatment
            } else if author_rules.get(c.author_id, AuthorIsControl) {
                AuthorCorpus::Control
            } else {
                AuthorCorpus::NotBucketed
            }
        })
        .collect()
}

fn apply_moe_ranking_policy(
    arm: ViewerArm,
    candidates: &[PostCandidate],
    corpus: &[AuthorCorpus],
    scores: &[f64],
) -> Vec<f64> {
    let mut out = scores.to_vec();
    for (i, c) in candidates.iter().enumerate() {
        if !is_phoenix_moe(c) {
            continue;
        }
        let keep = matches!(arm, ViewerArm::Treatment) && corpus[i] == AuthorCorpus::Treatment;
        if !keep {
            out[i] = 0.0;
        }
    }
    out
}

fn cold_start_target(query: &ScoredPostsQuery, scores: &[f64]) -> Option<f64> {
    let mut ranked = scores.to_vec();
    ranked.sort_by(|a, b| b.total_cmp(a));
    let hi = (query.params.get(ColdStartSlotMax) as usize).min(ranked.len());
    let lo = (query.params.get(ColdStartSlotMin) as usize).min(hi);
    if lo >= hi {
        return None;
    }
    Some(ranked[rand::rng().random_range(lo..hi)])
}

fn cold_start_corpus_eligible(arm: ViewerArm, c: &PostCandidate, corpus: AuthorCorpus) -> bool {
    match arm {
        ViewerArm::Holdout => !is_phoenix_moe(c),
        ViewerArm::Control => corpus == AuthorCorpus::Control && !is_phoenix_moe(c),
        ViewerArm::Treatment => corpus == AuthorCorpus::Treatment,
    }
}

fn cold_start_freshness_eligible(arm: ViewerArm, c: &PostCandidate, max_age: Duration) -> bool {
    if arm != ViewerArm::Treatment {
        return true;
    }
    duration_since_creation_opt(c.tweet_id).is_some_and(|age| age <= max_age)
}

fn apply_cold_start(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
    scores: &[f64],
    corpus: &[AuthorCorpus],
    arm: ViewerArm,
    target: f64,
) -> Vec<f64> {
    let follower_cap = query.params.get(ColdStartFollowerCap);
    let follower_decay_width = query.params.get(ColdStartFollowerDecayWidth);
    let threshold = query.params.get(ColdStartImpressionThreshold) as u64;
    let impression_decay_width = query.params.get(ColdStartImpressionDecayWidth) as u64;
    let max_post_age = Duration::from_secs(query.params.get(ColdStartMaxPostAgeSecs));
    let (positions, nonzero) = positions_among_nonzero(scores);
    let max_cold_start_slot =
        (query.params.get(LowImpressionsMaxPositionRatio) * nonzero as f64) as usize;

    let best = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if !cold_start_base_eligible(c, follower_cap, follower_decay_width)
                || !cold_start_corpus_eligible(arm, c, corpus[i])
                || !cold_start_freshness_eligible(arm, c, max_post_age)
                || positions[i] >= max_cold_start_slot
            {
                return None;
            }
            let impression_confidence = c.view_count.map_or(0.0, |impressions| {
                threshold_confidence(impressions, threshold, impression_decay_width, false)
            });
            let author_confidence = c.author_followers_count.map_or(0.0, |followers| {
                follower_confidence(followers, follower_cap, follower_decay_width)
            });
            let confidence = (impression_confidence * author_confidence).clamp(0.0, 1.0);
            if confidence > 0.0 {
                Some((i, confidence))
            } else {
                None
            }
        })
        .max_by(|(i, _), (j, _)| scores[*i].total_cmp(&scores[*j]));

    let Some((best_idx, confidence)) = best else {
        return scores.to_vec();
    };

    let mut effective = scores.to_vec();
    let boost = (target - effective[best_idx]).max(0.0) * confidence;
    effective[best_idx] += boost;
    record_cold_started_posts(is_phoenix_moe(&candidates[best_idx]), arm.as_str(), 1);
    effective
}

#[derive(Clone)]
pub struct AuthorColdStart {
    pub author_rules: Arc<AuthorRulesEvaluator>,
}

impl AuthorColdStart {
    pub(crate) fn apply(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        scores: &[f64],
    ) -> Vec<f64> {
        if !query.params.get(EnableViewerColdStart) {
            return scores.to_vec();
        }

        let arm = viewer_arm(query);
        let corpus = match arm {
            ViewerArm::Holdout => vec![AuthorCorpus::NotBucketed; candidates.len()],
            ViewerArm::Control | ViewerArm::Treatment => {
                author_corpus(&self.author_rules, candidates)
            }
        };

        let mut effective = apply_moe_ranking_policy(arm, candidates, &corpus, scores);
        if let Some(target) = cold_start_target(query, scores) {
            effective = apply_cold_start(query, candidates, &effective, &corpus, arm, target);
        }
        effective
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_candidate_pipeline::component_library::utils::current_time_to_id;
    use xai_feature_switches::{
        BucketMembership, ExperimentBucket, ExperimentBucketsChooser, FeatureSwitches,
        NullBucketImpressor, Recipient,
    };

    #[derive(Debug)]
    struct ArmChooser {
        treatment: Vec<u64>,
        control: Vec<u64>,
    }

    impl ExperimentBucketsChooser for ArmChooser {
        fn choose_buckets(&self, _recipient: &dyn Recipient) -> BucketMembership {
            BucketMembership::new()
        }

        fn choose_bucket_without_overrides(
            &self,
            experiment_key: &str,
            recipient: &dyn Recipient,
        ) -> Option<ExperimentBucket> {
            let uid = recipient.user_id()?;
            if experiment_key != "moe_exp" {
                return None;
            }
            let arm = if self.treatment.contains(&uid) {
                "treatment"
            } else if self.control.contains(&uid) {
                "control"
            } else {
                return None;
            };
            Some(ExperimentBucket::new(experiment_key, arm).with_version(1))
        }
    }

    fn cold_start_with_arms(treatment: Vec<u64>, control: Vec<u64>) -> AuthorColdStart {
        let yaml = r#"
rust_home_mixer:
  description: "x"
  owner: "t@example.com"
  parameters:
    rust_home_mixer_author_is_control:
      type: boolean
      default: false
    rust_home_mixer_author_is_treatment:
      type: boolean
      default: false
  rules:
    - description: "moe treatment corpus"
      query: >
        [moe_exp author_bucket_membership treatment]
      values:
        rust_home_mixer_author_is_treatment: true
    - description: "moe control corpus"
      query: >
        [moe_exp author_bucket_membership control]
      values:
        rust_home_mixer_author_is_control: true
"#;
        let features = xai_feature_switches::load_yaml_string(yaml).unwrap();
        let fs = Arc::new(
            FeatureSwitches::with_options(
                features,
                Arc::new(ArmChooser { treatment, control }),
                Arc::new(NullBucketImpressor::new()),
                None,
                false,
                None,
            )
            .unwrap(),
        );
        AuthorColdStart {
            author_rules: Arc::new(AuthorRulesEvaluator::new(fs)),
        }
    }

    fn minutes(n: u64) -> Duration {
        Duration::from_secs(n * 60)
    }

    fn tweet_id_with_age(age: Duration) -> u64 {
        current_time_to_id() as u64 - ((age.as_millis() as u64) << 22)
    }

    fn cold_start_candidate(author_id: u64, age: Duration, view_count: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            tweet_id: tweet_id_with_age(age),
            author_followers_count: Some(100),
            view_count: Some(view_count),
            ..Default::default()
        }
    }

    fn moe_candidate(author_id: u64, age: Duration, view_count: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            tweet_id: tweet_id_with_age(age),
            author_followers_count: Some(100),
            view_count: Some(view_count),
            served_type: Some(pb::ServedType::ForYouPhoenixRetrievalMoe),
            ..Default::default()
        }
    }

    fn base_query() -> ScoredPostsQuery {
        let mut query = ScoredPostsQuery::default();
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_viewer_cold_start_boost".to_string(),
            "true",
        );
        results.override_fs(
            "rust_home_mixer_low_impressions_max_boost_position_ratio".to_string(),
            "1.0",
        );
        results.override_fs(
            "rust_home_mixer_cold_start_impression_threshold".to_string(),
            "100",
        );
        results.override_fs(
            "rust_home_mixer_cold_start_impression_decay_width".to_string(),
            "100",
        );
        results.override_fs("rust_home_mixer_cold_start_slot_min".to_string(), "0");
        results.override_fs("rust_home_mixer_cold_start_slot_max".to_string(), "1");
        results.override_fs(
            "rust_home_mixer_cold_start_follower_cap".to_string(),
            "1000",
        );
        results.override_fs(
            "rust_home_mixer_cold_start_follower_decay_width".to_string(),
            "100",
        );
        results.override_fs(
            "rust_home_mixer_cold_start_max_post_age_secs".to_string(),
            "7200",
        );
        query.params = results.into();
        query
    }

    fn codivert_query(viewer_control: bool, viewer_treatment: bool) -> ScoredPostsQuery {
        let mut query = base_query();
        let mut results = query.params.0.expect("params set");
        results.override_fs(
            "rust_home_mixer_phoenix_moe_codivert_viewer_is_control".to_string(),
            if viewer_control { "true" } else { "false" },
        );
        results.override_fs(
            "rust_home_mixer_phoenix_moe_codivert_viewer_is_treatment".to_string(),
            if viewer_treatment { "true" } else { "false" },
        );
        query.params = results.into();
        query
    }

    fn query_with_max_post_age(viewer_treatment: bool, max_post_age_secs: u64) -> ScoredPostsQuery {
        let mut query = codivert_query(!viewer_treatment, viewer_treatment);
        let mut results = query.params.0.expect("params set");
        results.override_fs(
            "rust_home_mixer_cold_start_max_post_age_secs".to_string(),
            &max_post_age_secs.to_string(),
        );
        query.params = results.into();
        query
    }

    #[test]
    fn holdout_zeros_moe_and_cold_starts_non_moe() {
        let author_cold_start = cold_start_with_arms(vec![], vec![]);
        let candidates = vec![
            cold_start_candidate(1, minutes(10), 3),
            moe_candidate(2, minutes(20), 3),
        ];
        let result =
            author_cold_start.apply(&codivert_query(false, false), &candidates, &[5.0, 40.0]);
        assert_eq!(result, vec![40.0, 0.0]);
    }

    #[test]
    fn control_viewer_zeros_moe_and_cold_starts_control_corpus_only() {
        let author_cold_start = cold_start_with_arms(vec![2], vec![1]);
        let candidates = vec![
            cold_start_candidate(1, minutes(10), 3),
            cold_start_candidate(2, minutes(20), 3),
            cold_start_candidate(3, minutes(30), 3),
            moe_candidate(1, minutes(40), 1000),
        ];
        let result = author_cold_start.apply(
            &codivert_query(true, false),
            &candidates,
            &[40.0, 100.0, 90.0, 80.0],
        );
        assert_eq!(result[0], 100.0);
        assert_eq!(result[1], 100.0);
        assert_eq!(result[2], 90.0);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn treatment_viewer_keeps_treatment_moe_and_cold_starts_treatment_corpus() {
        let author_cold_start = cold_start_with_arms(vec![1, 2], vec![3]);
        let candidates = vec![
            moe_candidate(1, minutes(10), 3),
            cold_start_candidate(2, minutes(20), 3),
            cold_start_candidate(3, minutes(30), 3),
        ];
        let result = author_cold_start.apply(
            &codivert_query(false, true),
            &candidates,
            &[80.0, 50.0, 90.0],
        );
        assert_eq!(result[0], 90.0);
        assert_eq!(result[1], 50.0);
        assert_eq!(result[2], 90.0);
    }

    #[test]
    fn treatment_skips_post_older_than_max_post_age() {
        let author_cold_start = cold_start_with_arms(vec![1], vec![]);
        let candidates = vec![
            cold_start_candidate(1, minutes(180), 3),
            cold_start_candidate(2, minutes(30), 1000),
        ];
        let result = author_cold_start.apply(
            &query_with_max_post_age(true, 7200),
            &candidates,
            &[10.0, 90.0],
        );
        assert_eq!(result, vec![10.0, 90.0]);
    }

    #[test]
    fn control_ignores_max_post_age() {
        let author_cold_start = cold_start_with_arms(vec![], vec![1]);
        let candidates = vec![
            cold_start_candidate(1, minutes(180), 3),
            cold_start_candidate(2, minutes(30), 1000),
        ];
        let result = author_cold_start.apply(
            &query_with_max_post_age(false, 7200),
            &candidates,
            &[10.0, 90.0],
        );
        assert_eq!(result, vec![90.0, 90.0]);
    }

    #[test]
    fn cold_start_respects_max_position_ratio() {
        let author_cold_start = cold_start_with_arms(vec![], vec![1, 2, 3, 4]);
        let candidates = vec![
            cold_start_candidate(1, minutes(10), 3),
            cold_start_candidate(2, minutes(20), 1000),
            cold_start_candidate(3, minutes(30), 1000),
            cold_start_candidate(4, minutes(40), 1000),
        ];

        let mut query = codivert_query(true, false);
        let mut results = query.params.0.expect("params set");
        results.override_fs(
            "rust_home_mixer_low_impressions_max_boost_position_ratio".to_string(),
            "0.5",
        );
        query.params = results.into();

        let result = author_cold_start.apply(&query, &candidates, &[10.0, 40.0, 30.0, 20.0]);
        assert_eq!(result, vec![10.0, 40.0, 30.0, 20.0]);
    }

    #[test]
    fn threshold_confidence_is_exact_at_boundaries_and_monotonic() {
        assert_eq!(threshold_confidence(999, 1000, 200, false), 1.0);
        assert_eq!(threshold_confidence(1000, 1000, 200, false), 1.0);
        assert_eq!(threshold_confidence(1100, 1000, 200, false), 0.5);
        assert_eq!(threshold_confidence(1200, 1000, 200, false), 0.0);
        assert_eq!(threshold_confidence(1201, 1000, 200, false), 0.0);

        let confidences: Vec<f64> = (900..=1250)
            .map(|value| threshold_confidence(value, 1000, 200, false))
            .collect();
        assert!(confidences.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(confidences.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn zero_decay_width_preserves_legacy_threshold_comparisons() {
        assert_eq!(threshold_confidence(99, 100, 0, false), 1.0);
        assert_eq!(threshold_confidence(100, 100, 0, false), 0.0);
        assert_eq!(threshold_confidence(1000, 1000, 0, true), 1.0);
        assert_eq!(threshold_confidence(1001, 1000, 0, true), 0.0);
        assert_eq!(follower_confidence(-1, -1, 0), 1.0);
        assert_eq!(follower_confidence(0, -1, 0), 0.0);
    }

    #[test]
    fn impression_decay_applies_a_partial_bounded_boost() {
        let author_cold_start = cold_start_with_arms(vec![], vec![1, 2]);
        let candidates = vec![
            cold_start_candidate(1, minutes(10), 150),
            cold_start_candidate(2, minutes(10), 1000),
        ];

        let result =
            author_cold_start.apply(&codivert_query(true, false), &candidates, &[20.0, 100.0]);

        assert_eq!(result, vec![60.0, 100.0]);
    }

    #[test]
    fn follower_decay_applies_a_partial_bounded_boost() {
        let author_cold_start = cold_start_with_arms(vec![], vec![1, 2]);
        let mut partial = cold_start_candidate(1, minutes(10), 3);
        partial.author_followers_count = Some(1050);
        let candidates = vec![partial, cold_start_candidate(2, minutes(10), 1000)];

        let result =
            author_cold_start.apply(&codivert_query(true, false), &candidates, &[20.0, 100.0]);

        assert_eq!(result, vec![60.0, 100.0]);
    }

    #[test]
    fn decay_reaches_zero_without_crossing_target_or_lowering_score() {
        let author_cold_start = cold_start_with_arms(vec![], vec![1, 2]);
        let candidates = vec![
            cold_start_candidate(1, minutes(10), 200),
            cold_start_candidate(2, minutes(10), 1000),
        ];

        let result =
            author_cold_start.apply(&codivert_query(true, false), &candidates, &[20.0, 100.0]);

        assert_eq!(result, vec![20.0, 100.0]);
    }
}
