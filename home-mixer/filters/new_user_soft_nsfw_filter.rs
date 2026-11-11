use crate::models::candidate::PostCandidate;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{
    EnableNewUserSoftNsfwFilter, NewUserSoftNsfwFilterMaxAccountAgeSecs,
    NewUserSoftNsfwFilterMaxScoringSeqLength,
};
use std::time::Duration;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct NewUserSoftNsfwFilter;

pub(crate) fn new_user_soft_nsfw_filter_enabled(query: &ScoredPostsQuery) -> bool {
    let for_you_without_topics =
        query.request_type == RequestType::ForYou && !query.is_topic_request();
    if !for_you_without_topics || !query.params.get(EnableNewUserSoftNsfwFilter) {
        return false;
    }
    let max_age = Duration::from_secs(query.params.get(NewUserSoftNsfwFilterMaxAccountAgeSecs));
    duration_since_creation_opt(query.user_id).is_some_and(|age| age < max_age)
        && below_max_scoring_seq_length(query)
}

fn below_max_scoring_seq_length(query: &ScoredPostsQuery) -> bool {
    let max_length = query.params.get(NewUserSoftNsfwFilterMaxScoringSeqLength);
    max_length == 0 || scoring_sequence_length(query).is_none_or(|length| length < max_length)
}

fn scoring_sequence_length(query: &ScoredPostsQuery) -> Option<u64> {
    query
        .scoring_sequence
        .as_ref()?
        .metadata
        .as_ref()
        .map(|metadata| metadata.length)
}

fn is_labeled_soft_nsfw_out_of_network(candidate: &PostCandidate) -> bool {
    candidate.in_network == Some(false) && candidate.soft_nsfw == Some(true)
}

impl Filter<ScoredPostsQuery, PostCandidate> for NewUserSoftNsfwFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        new_user_soft_nsfw_filter_enabled(query)
    }

    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        FilterResult::partition(candidates, |c| !is_labeled_soft_nsfw_out_of_network(c))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use xai_candidate_pipeline::component_library::utils::current_time_to_id;

    const SNOWFLAKE_EPOCH_MS: u64 = 1_288_834_974_657;
    const ENABLE: &str = "rust_home_mixer_enable_new_user_soft_nsfw_filter";
    const MAX_AGE: &str = "rust_home_mixer_new_user_soft_nsfw_filter_max_account_age_secs";

    pub(crate) fn new_user_id() -> u64 {
        current_time_to_id() as u64
    }

    pub(crate) fn old_user_id() -> u64 {
        (1_600_000_000_000 - SNOWFLAKE_EPOCH_MS) << 22
    }

    pub(crate) fn query(user_id: u64, enabled: bool, max_age_secs: u64) -> ScoredPostsQuery {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(ENABLE.to_string(), if enabled { "true" } else { "false" });
        results.override_fs(MAX_AGE.to_string(), &max_age_secs.to_string());
        ScoredPostsQuery {
            user_id,
            request_type: RequestType::ForYou,
            params: results.into(),
            ..Default::default()
        }
    }

    pub(crate) fn enabled_query() -> ScoredPostsQuery {
        query(new_user_id(), true, 86_400)
    }

    pub(crate) fn candidate(
        tweet_id: u64,
        in_network: Option<bool>,
        soft_nsfw: Option<bool>,
    ) -> PostCandidate {
        PostCandidate {
            tweet_id,
            in_network,
            soft_nsfw,
            ..Default::default()
        }
    }

    fn removed_and_kept(candidates: Vec<PostCandidate>) -> (Vec<u64>, Vec<u64>) {
        let result = NewUserSoftNsfwFilter.filter(&enabled_query(), candidates);
        (
            result.removed.iter().map(|c| c.tweet_id).collect(),
            result.kept.iter().map(|c| c.tweet_id).collect(),
        )
    }

    #[test]
    fn removes_out_of_network_soft_nsfw_post() {
        let (removed, kept) = removed_and_kept(vec![
            candidate(1, Some(false), Some(true)),
            candidate(2, Some(false), Some(false)),
        ]);
        assert_eq!(removed, vec![1]);
        assert_eq!(kept, vec![2]);
    }

    #[test]
    fn keeps_in_network_soft_nsfw_post() {
        let (removed, kept) = removed_and_kept(vec![candidate(1, Some(true), Some(true))]);
        assert!(removed.is_empty());
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn keeps_post_whose_label_is_unknown() {
        let (removed, kept) = removed_and_kept(vec![candidate(1, Some(false), None)]);
        assert!(removed.is_empty());
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn switch_gate() {
        assert!(NewUserSoftNsfwFilter.enable(&query(new_user_id(), true, 86_400)));
        assert!(!NewUserSoftNsfwFilter.enable(&query(new_user_id(), false, 86_400)));
    }

    #[test]
    fn account_age_gate() {
        assert!(!NewUserSoftNsfwFilter.enable(&query(old_user_id(), true, 86_400)));
        assert!(!NewUserSoftNsfwFilter.enable(&query(new_user_id(), true, 0)));
        assert!(NewUserSoftNsfwFilter.enable(&query(old_user_id(), true, u64::MAX / 2)));
    }

    #[test]
    fn for_you_gate_excludes_topic_and_other_request_types() {
        let topic = ScoredPostsQuery {
            topic_ids: vec![7],
            ..enabled_query()
        };
        assert!(!NewUserSoftNsfwFilter.enable(&topic));
        for request_type in [
            RequestType::ScoredPosts,
            RequestType::RankedFollowing,
            RequestType::Following,
            RequestType::PhoenixScores,
        ] {
            let other = ScoredPostsQuery {
                request_type,
                ..enabled_query()
            };
            assert!(!NewUserSoftNsfwFilter.enable(&other), "{request_type:?}");
        }
    }
}
