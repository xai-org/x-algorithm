use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableServedFilterAllRequests;
use crate::util::candidates_util::related_post_ids_iter;
use std::collections::HashSet;
use xai_candidate_pipeline::component_library::utils::client_utils::RequestContext::{
    self, ForegroundTruncate,
};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslyServedPostsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslyServedPostsFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        let req_context = RequestContext::parse(&query.request_context);
        let enable_all = query.params.get(EnableServedFilterAllRequests);

        enable_all || (query.is_bottom_request && req_context != ForegroundTruncate)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let served_ids: HashSet<u64> = query.served_ids.iter().copied().collect();

        FilterResult::partition(candidates, |c| {
            !related_post_ids_iter(c).any(|id| served_ids.contains(&id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};

    #[test]
    fn served_filter_gate_and_related_post_partitions_are_viewer_cache_independent() {
        for cached in [false, true] {
            for enabled in [false, true] {
                let mut fs = FeatureSwitches::new(vec![])
                    .unwrap()
                    .match_recipient(&RecipientBuilder::new().build());
                fs.override_fs(
                    "rust_home_mixer_enable_served_filter_all_requests".to_string(),
                    if enabled { "true" } else { "false" },
                );
                let query = ScoredPostsQuery {
                    has_cached_posts: cached,
                    served_ids: vec![1, 10, 20, 30],
                    params: fs.into(),
                    ..Default::default()
                };
                assert_eq!(PreviouslyServedPostsFilter.enable(&query), enabled);
                if enabled {
                    let mut input: Vec<_> = (1..=5)
                        .map(|id| PostCandidate {
                            tweet_id: id,
                            tweet_text: format!("post-{id}"),
                            ..Default::default()
                        })
                        .collect();
                    input[1].retweeted_tweet_id = Some(10);
                    input[2].in_reply_to_tweet_id = Some(20);
                    input[3].ancestors = vec![30];
                    input[4].quoted_tweet_id = Some(10);
                    let originals = input.clone();
                    let result =
                        PreviouslyServedPostsFilter.run(&query, input, PipelineStage::Filter);
                    assert_eq!(
                        result.kept.iter().map(|c| c.tweet_id).collect::<Vec<_>>(),
                        originals[3..]
                            .iter()
                            .map(|c| c.tweet_id)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        result
                            .removed
                            .iter()
                            .map(|c| c.tweet_id)
                            .collect::<Vec<_>>(),
                        originals[..3]
                            .iter()
                            .map(|c| c.tweet_id)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        serde_json::to_value(&result.kept).unwrap(),
                        serde_json::to_value(&originals[3..]).unwrap()
                    );
                    assert_eq!(
                        serde_json::to_value(&result.removed).unwrap(),
                        serde_json::to_value(&originals[..3]).unwrap()
                    );
                }
            }
        }
    }
}
