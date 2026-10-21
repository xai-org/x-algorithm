use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::hydrator::Hydrator;

pub struct FollowingBlockedByHydrator {
    socialgraph_client: Arc<dyn SocialGraphClientOps>,
}

impl FollowingBlockedByHydrator {
    pub async fn new(socialgraph_client: Arc<dyn SocialGraphClientOps>) -> Self {
        Self { socialgraph_client }
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for FollowingBlockedByHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let user_ids: Vec<u64> = candidates
            .iter()
            .flat_map(|c| {
                std::iter::once(c.author_id)
                    .chain(c.quoted_user_id)
                    .chain(c.retweeted_user_id)
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let blocked_by_user_ids = match self
            .socialgraph_client
            .check_blocked_by(query.user_id, &user_ids)
            .await
        {
            Ok(ids) => ids,
            Err(e) => {
                let err_msg = e.to_string();
                return candidates.iter().map(|_| Err(err_msg.clone())).collect();
            }
        };
        candidates
            .iter()
            .map(|candidate| {
                let author_blocks_viewer = blocked_by_user_ids.contains(&candidate.author_id)
                    || candidate
                        .retweeted_user_id
                        .is_some_and(|uid| blocked_by_user_ids.contains(&uid));
                let quoted_author_blocks_viewer = candidate
                    .quoted_user_id
                    .map(|uid| blocked_by_user_ids.contains(&uid));
                Ok(PostCandidate {
                    author_blocks_viewer: Some(author_blocks_viewer),
                    quoted_author_blocks_viewer,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.author_blocks_viewer = hydrated.author_blocks_viewer;
        if hydrated.quoted_author_blocks_viewer.is_some() {
            candidate.quoted_author_blocks_viewer = hydrated.quoted_author_blocks_viewer;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Status;

    struct MockSocialGraph {
        blocked_by: HashSet<u64>,
    }

    #[async_trait]
    impl SocialGraphClientOps for MockSocialGraph {
        async fn get_following_list(&self, _user_id: u64) -> Result<Vec<u64>, Status> {
            Ok(vec![])
        }
        async fn check_blocked_by(
            &self,
            _viewer_id: u64,
            author_ids: &[u64],
        ) -> Result<HashSet<u64>, Status> {
            Ok(author_ids
                .iter()
                .copied()
                .filter(|id| self.blocked_by.contains(id))
                .collect())
        }
        async fn check_followed_by(
            &self,
            _viewer_id: u64,
            _user_ids: &[u64],
        ) -> Result<HashSet<u64>, Status> {
            Ok(HashSet::new())
        }
        async fn get_blocked_user_ids(&self, _viewer_id: u64) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
        async fn get_muted_user_ids(&self, _viewer_id: u64) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
        async fn get_followed_user_ids(&self, _viewer_id: u64) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
        async fn get_follower_ids(&self, _user_id: u64) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
        async fn get_subscribed_user_ids(&self, _viewer_id: u64) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
        async fn get_device_following_user_ids(&self, _viewer_id: u64) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
        async fn get_hide_recommendations_user_ids(
            &self,
            _viewer_id: u64,
        ) -> Result<Vec<i64>, Status> {
            Ok(vec![])
        }
    }

    fn hydrator(blocked_by: HashSet<u64>) -> FollowingBlockedByHydrator {
        FollowingBlockedByHydrator {
            socialgraph_client: Arc::new(MockSocialGraph { blocked_by }),
        }
    }

    #[tokio::test]
    async fn native_author_who_blocked_viewer_is_marked() {
        let hydrator = hydrator(HashSet::from([10]));
        let mut native = PostCandidate {
            tweet_id: 1,
            author_id: 10,
            ..Default::default()
        };
        let query = ScoredPostsQuery {
            user_id: 1,
            ..Default::default()
        };
        let hydrated = hydrator.hydrate(&query, &[native.clone()]).await;
        hydrator.update(&mut native, hydrated[0].clone().unwrap());
        assert_eq!(native.author_blocks_viewer, Some(true));
    }

    #[tokio::test]
    async fn retweet_of_author_who_blocked_viewer_is_still_marked() {
        let hydrator = hydrator(HashSet::from([99]));
        let rt = PostCandidate {
            tweet_id: 1,
            author_id: 10,
            retweeted_user_id: Some(99),
            ..Default::default()
        };
        let native = PostCandidate {
            tweet_id: 2,
            author_id: 10,
            ..Default::default()
        };
        let query = ScoredPostsQuery {
            user_id: 1,
            ..Default::default()
        };
        let hydrated = hydrator.hydrate(&query, &[rt.clone(), native.clone()]).await;
        let mut rt = rt;
        let mut native = native;
        hydrator.update(&mut rt, hydrated[0].clone().unwrap());
        hydrator.update(&mut native, hydrated[1].clone().unwrap());
        assert_eq!(rt.author_blocks_viewer, Some(true));
        assert_eq!(native.author_blocks_viewer, Some(false));
    }

    #[tokio::test]
    async fn retweeter_who_blocked_viewer_is_marked() {
        let hydrator = hydrator(HashSet::from([10]));
        let mut rt = PostCandidate {
            tweet_id: 1,
            author_id: 10,
            retweeted_user_id: Some(99),
            ..Default::default()
        };
        let query = ScoredPostsQuery {
            user_id: 1,
            ..Default::default()
        };
        let hydrated = hydrator.hydrate(&query, &[rt.clone()]).await;
        hydrator.update(&mut rt, hydrated[0].clone().unwrap());
        assert_eq!(rt.author_blocks_viewer, Some(true));
    }

    #[tokio::test]
    async fn quoted_author_who_blocked_viewer_is_still_marked() {
        let hydrator = hydrator(HashSet::from([77]));
        let mut quote = PostCandidate {
            tweet_id: 1,
            author_id: 10,
            quoted_user_id: Some(77),
            ..Default::default()
        };
        let query = ScoredPostsQuery {
            user_id: 1,
            ..Default::default()
        };
        let hydrated = hydrator.hydrate(&query, &[quote.clone()]).await;
        hydrator.update(&mut quote, hydrated[0].clone().unwrap());
        assert_eq!(quote.quoted_author_blocks_viewer, Some(true));
        assert_eq!(quote.author_blocks_viewer, Some(false));
    }
}
