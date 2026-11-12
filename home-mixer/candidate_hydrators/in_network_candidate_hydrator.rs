use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

pub struct InNetworkCandidateHydrator;

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for InNetworkCandidateHydrator {
    type Fetched = Vec<bool>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<bool>, String> {
        let followed_ids: HashSet<u64> = query
            .user_features
            .followed_user_ids
            .iter()
            .map(|&id| id as u64)
            .collect();
        Ok(candidates
            .iter()
            .map(|candidate| {
                candidate.author_id == query.user_id || followed_ids.contains(&candidate.author_id)
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        in_network: Vec<bool>,
    ) {
        for (candidate, in_network) in candidates.iter_mut().zip(in_network) {
            candidate.in_network = Some(in_network);
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.in_network = hydrated.in_network;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::user_features::UserFeatures;

    #[tokio::test]
    async fn in_network_comes_from_the_author_ids_fetch_saw() {
        let query = ScoredPostsQuery {
            user_id: 1,
            user_features: UserFeatures {
                followed_user_ids: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut candidates = vec![PostCandidate {
            tweet_id: 10,
            author_id: 0,
            ..Default::default()
        }];
        let hydrator = InNetworkCandidateHydrator;

        let fetched = hydrator.fetch(&query, &candidates).await.unwrap();
        candidates[0].author_id = 2;
        hydrator.apply(&query, &mut candidates, fetched);

        assert_eq!(candidates[0].in_network, Some(false));
    }
}
