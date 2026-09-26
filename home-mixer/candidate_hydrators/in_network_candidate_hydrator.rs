use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;

pub struct InNetworkCandidateHydrator;

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for InNetworkCandidateHydrator {
    fn enable(&self, _query: &ScoredPostsQuery) -> bool {
        // FollowedUserIdsQueryHydrator still runs on cache hits. Skipping here
        // would keep the cached in_network bit after follow/unfollow.
        true
    }

    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let viewer_id = query.user_id;
        let followed_ids: HashSet<u64> = query
            .user_features
            .followed_user_ids
            .iter()
            .copied()
            .map(|id| id as u64)
            .collect();

        candidates
            .iter()
            .map(|candidate| {
                let is_self = candidate.author_id == viewer_id;
                let is_in_network = is_self || followed_ids.contains(&candidate.author_id);
                Ok(PostCandidate {
                    in_network: Some(is_in_network),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.in_network = hydrated.in_network;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::user_features::UserFeatures;

    fn query(user_id: u64, followed: Vec<i64>, has_cached_posts: bool) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_id,
            has_cached_posts,
            user_features: UserFeatures {
                followed_user_ids: followed,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn candidate(tweet_id: u64, author_id: u64, in_network: Option<bool>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            in_network,
            ..Default::default()
        }
    }

    #[test]
    fn enable_on_cache_hit_and_miss() {
        let hydrator = InNetworkCandidateHydrator;
        assert!(hydrator.enable(&query(1, vec![10], true)));
        assert!(hydrator.enable(&query(1, vec![10], false)));
    }

    #[tokio::test]
    async fn cache_hit_recomputes_from_current_follow_list() {
        let hydrator = InNetworkCandidateHydrator;
        let q = query(1, vec![10], true);
        let mut candidates = vec![
            candidate(1, 10, Some(false)),
            candidate(2, 20, Some(true)),
            candidate(3, 1, Some(false)),
        ];

        let hydrated = hydrator.hydrate(&q, &candidates).await;
        for (c, h) in candidates.iter_mut().zip(hydrated) {
            hydrator.update(c, h.expect("hydrate ok"));
        }

        assert_eq!(candidates[0].in_network, Some(true));
        assert_eq!(candidates[1].in_network, Some(false));
        assert_eq!(candidates[2].in_network, Some(true));
    }
}
