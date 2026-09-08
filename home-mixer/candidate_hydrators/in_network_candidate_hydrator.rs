use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;

pub struct InNetworkCandidateHydrator;

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for InNetworkCandidateHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
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

    fn query_following(viewer_id: u64, followed: Vec<i64>) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_id: viewer_id,
            user_features: UserFeatures {
                followed_user_ids: followed,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn stamp(query: &ScoredPostsQuery, author_id: u64) -> Option<bool> {
        let hydrator = InNetworkCandidateHydrator;
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            author_id,
            ..Default::default()
        }];
        let hydrated = hydrator.hydrate(query, &candidates).await;
        let mut candidate = candidates.into_iter().next().unwrap();
        hydrator.update(&mut candidate, hydrated.into_iter().next().unwrap().unwrap());
        candidate.in_network
    }

    #[tokio::test]
    async fn followed_author_is_in_network() {
        let query = query_following(1, vec![42]);
        assert_eq!(stamp(&query, 42).await, Some(true));
    }

    #[tokio::test]
    async fn unfollowed_author_is_oon() {
        let query = query_following(1, vec![42]);
        assert_eq!(stamp(&query, 99).await, Some(false));
    }

    #[tokio::test]
    async fn missing_author_id_is_stamped_oon() {
        let query = query_following(1, vec![42]);
        assert_eq!(
            stamp(&query, 0).await,
            Some(false),
            "author_id 0 is not in the follow set, so this stamps OON"
        );
    }

    #[tokio::test]
    async fn tes_author_id_must_be_present_before_stamp() {
        let query = query_following(1, vec![42]);
        let hydrator = InNetworkCandidateHydrator;

        // Wrong order: stamp while author_id is still 0, then TES fills the followed author.
        let mut too_early = PostCandidate {
            tweet_id: 1,
            author_id: 0,
            ..Default::default()
        };
        let early = hydrator.hydrate(&query, &[too_early.clone()]).await;
        hydrator.update(&mut too_early, early.into_iter().next().unwrap().unwrap());
        too_early.author_id = 42;
        assert_eq!(
            too_early.in_network,
            Some(false),
            "stale OON stamp after TES fills a followed author"
        );

        // Right order: TES author_id first.
        assert_eq!(stamp(&query, 42).await, Some(true));
    }
}
