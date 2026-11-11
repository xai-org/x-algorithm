use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableSimclustersSource;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_home_mixer_proto::ServedType;

pub struct OONNsfwSimclustersFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for OONNsfwSimclustersFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableSimclustersSource)
    }

    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        FilterResult::partition(candidates, |c| {
            !(c.served_type == Some(ServedType::ForYouSimclusters)
                && c.in_network == Some(false)
                && c.nsfw_author == Some(true))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_only_out_of_network_nsfw_simclusters_posts() {
        let post = |tweet_id, served_type, in_network, nsfw_author| PostCandidate {
            tweet_id,
            served_type: Some(served_type),
            in_network: Some(in_network),
            nsfw_author: Some(nsfw_author),
            ..Default::default()
        };
        let candidates = vec![
            post(1, ServedType::ForYouSimclusters, false, true),
            post(2, ServedType::ForYouSimclusters, true, true),
            post(3, ServedType::ForYouSimclusters, false, false),
            post(4, ServedType::ForYouPhoenixRetrieval, false, true),
            post(5, ServedType::ForYouSimclusters, false, true),
        ];
        let result = OONNsfwSimclustersFilter.filter(&ScoredPostsQuery::default(), candidates);
        let ids = |posts: &[PostCandidate]| posts.iter().map(|c| c.tweet_id).collect::<Vec<_>>();
        assert_eq!(ids(&result.kept), [2, 3, 4]);
        assert_eq!(ids(&result.removed), [1, 5]);
    }
}
