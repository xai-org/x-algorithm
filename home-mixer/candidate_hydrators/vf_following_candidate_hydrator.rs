use crate::candidate_hydrators::vf_candidate_hydrator::should_drop_ancillary;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableXaiVfClient;
use anyhow::Result;
use futures::future::join;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_twittercontext_proto::{GetTwitterContextViewer, TwitterContextViewer};
use xai_visibility_filtering::models::FilteredReason;
use xai_visibility_filtering::vf_client::SafetyLevel;
use xai_visibility_filtering::vf_client::SafetyLevel::{TimelineHome, TimelineHomeRecommendations};
use xai_visibility_filtering::vf_client::{TweetVisibility, VfClient};

pub struct VFFollowingCandidateHydrator {
    pub strato_vf_client: Arc<dyn VfClient + Send + Sync>,
    pub xai_vf_client: Arc<dyn VfClient + Send + Sync>,
}

impl VFFollowingCandidateHydrator {
    pub fn new(
        strato_vf_client: Arc<dyn VfClient + Send + Sync>,
        xai_vf_client: Arc<dyn VfClient + Send + Sync>,
    ) -> Self {
        Self {
            strato_vf_client,
            xai_vf_client,
        }
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for VFFollowingCandidateHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let context = query.get_viewer();
        let client = if query.params.get(EnableXaiVfClient) {
            &self.xai_vf_client
        } else {
            &self.strato_vf_client
        };

        let mut home_ids: Vec<u64> = Vec::new();
        let mut recs_ids: Vec<u64> = Vec::new();
        for candidate in candidates {
            home_ids.push(candidate.tweet_id);
            recs_ids.extend(candidate.ancestors.iter().copied());
            if let Some(quoted_post_id) = candidate.quoted_tweet_id {
                recs_ids.push(quoted_post_id);
            }
            if let Some(reposted_post_id) = candidate.retweeted_tweet_id {
                home_ids.push(reposted_post_id);
            }
        }
        home_ids.sort_unstable();
        home_ids.dedup();
        recs_ids.sort_unstable();
        recs_ids.dedup();

        let home_future = fetch_vf(client, home_ids, TimelineHome, query.user_id, context.clone());
        let recs_future = fetch_vf(
            client,
            recs_ids,
            TimelineHomeRecommendations,
            query.user_id,
            context,
        );
        let (home_results, recs_results) = join(home_future, recs_future).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let primary_result = home_results.get(&candidate.tweet_id);
            let visibility_reason = match primary_result {
                Some(Ok(Some(reason))) => Some(reason.clone()),
                _ => None,
            };

            let drop_ancillary = should_drop_following_ancillary(
                candidate,
                &home_results,
                &recs_results,
            );

            let hydrated = match primary_result {
                Some(Err(err)) => Err(err.to_string()),
                _ => Ok(PostCandidate {
                    visibility_reason,
                    drop_ancillary_posts: Some(drop_ancillary),
                    ..Default::default()
                }),
            };
            hydrated_candidates.push(hydrated);
        }
        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.visibility_reason = hydrated.visibility_reason;
        candidate.drop_ancillary_posts = hydrated.drop_ancillary_posts;
    }
}

async fn fetch_vf(
    client: &Arc<dyn VfClient + Send + Sync>,
    tweet_ids: Vec<u64>,
    safety_level: SafetyLevel,
    for_user_id: u64,
    context: Option<TwitterContextViewer>,
) -> HashMap<u64, Result<Option<FilteredReason>>> {
    if tweet_ids.is_empty() {
        return HashMap::new();
    }
    client
        .get_result(tweet_ids, safety_level, for_user_id, context)
        .await
        .into_iter()
        .map(|(id, r)| (id, r.map(|t| t.reason)))
        .collect()
}

fn should_drop_following_ancillary(
    candidate: &PostCandidate,
    home_results: &HashMap<u64, Result<Option<FilteredReason>>>,
    recs_results: &HashMap<u64, Result<Option<FilteredReason>>>,
) -> bool {
    let mut quote_and_ancestors = candidate.clone();
    quote_and_ancestors.retweeted_tweet_id = None;
    if should_drop_ancillary(&quote_and_ancestors, recs_results) {
        return true;
    }
    let mut retweet_only = candidate.clone();
    retweet_only.ancestors.clear();
    retweet_only.quoted_tweet_id = None;
    should_drop_ancillary(&retweet_only, home_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_safety_label_store::types::SafetyLabelMap;
    use xai_visibility_filtering::models::{
        Action, DropReason, SafetyResult, SafetyResultReason,
    };
    use xai_visibility_filtering::vf_client::SafetyLevel;

    struct LevelClient {
        home: HashMap<u64, FilteredReason>,
        recs: HashMap<u64, FilteredReason>,
        seen_home: std::sync::Mutex<Vec<u64>>,
        seen_recs: std::sync::Mutex<Vec<u64>>,
    }

    fn vis(reason: FilteredReason) -> TweetVisibility {
        TweetVisibility {
            reason: Some(reason),
            safety_labels: Ok(SafetyLabelMap::default()),
        }
    }

    fn interstitial() -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult {
            reason: Some(SafetyResultReason::NsfwHighPrecision),
            action: Action::Interstitial,
        })
    }

    fn recs_drop() -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult {
            reason: Some(SafetyResultReason::NsfwHighPrecision),
            action: Action::Drop(DropReason {}),
        })
    }

    #[async_trait]
    impl VfClient for LevelClient {
        async fn get_result(
            &self,
            post_ids: Vec<u64>,
            safety_level: SafetyLevel,
            _for_user_id: u64,
            _context: Option<TwitterContextViewer>,
        ) -> HashMap<u64, Result<TweetVisibility>> {
            let mut out = HashMap::new();
            match safety_level {
                SafetyLevel::TimelineHome => {
                    self.seen_home.lock().unwrap().extend(post_ids.iter().copied());
                    for id in post_ids {
                        if let Some(reason) = self.home.get(&id) {
                            out.insert(id, Ok(vis(reason.clone())));
                        }
                    }
                }
                SafetyLevel::TimelineHomeRecommendations => {
                    self.seen_recs.lock().unwrap().extend(post_ids.iter().copied());
                    for id in post_ids {
                        if let Some(reason) = self.recs.get(&id) {
                            out.insert(id, Ok(vis(reason.clone())));
                        }
                    }
                }
                _ => {}
            }
            out
        }
    }

    fn hydrator(client: Arc<LevelClient>) -> VFFollowingCandidateHydrator {
        VFFollowingCandidateHydrator::new(client.clone(), client)
    }

    #[tokio::test]
    async fn quote_of_recs_drop_is_ancillary_drop() {
        let client = Arc::new(LevelClient {
            home: HashMap::from([(1, interstitial())]),
            recs: HashMap::from([(99, recs_drop())]),
            seen_home: std::sync::Mutex::new(Vec::new()),
            seen_recs: std::sync::Mutex::new(Vec::new()),
        });
        let results = hydrator(client.clone())
            .hydrate(
                &ScoredPostsQuery::default(),
                &[PostCandidate {
                    tweet_id: 1,
                    quoted_tweet_id: Some(99),
                    ..Default::default()
                }],
            )
            .await;
        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(hydrated.drop_ancillary_posts, Some(true));
        assert!(matches!(
            hydrated.visibility_reason,
            Some(FilteredReason::SafetyResult(ref s)) if s.action == Action::Interstitial
        ));
        assert!(client.seen_home.lock().unwrap().contains(&1));
        assert!(!client.seen_home.lock().unwrap().contains(&99));
        assert!(client.seen_recs.lock().unwrap().contains(&99));
        assert!(!client.seen_recs.lock().unwrap().contains(&1));
    }

    #[tokio::test]
    async fn ancestor_recs_drop_is_ancillary_drop() {
        let client = Arc::new(LevelClient {
            home: HashMap::from([(2, interstitial())]),
            recs: HashMap::from([(50, recs_drop())]),
            seen_home: std::sync::Mutex::new(Vec::new()),
            seen_recs: std::sync::Mutex::new(Vec::new()),
        });
        let results = hydrator(client)
            .hydrate(
                &ScoredPostsQuery::default(),
                &[PostCandidate {
                    tweet_id: 2,
                    ancestors: vec![50],
                    ..Default::default()
                }],
            )
            .await;
        assert_eq!(results[0].as_ref().unwrap().drop_ancillary_posts, Some(true));
    }

    #[tokio::test]
    async fn followee_nsfw_primary_stays_interstitial() {
        let client = Arc::new(LevelClient {
            home: HashMap::from([(3, interstitial())]),
            recs: HashMap::from([(3, recs_drop())]),
            seen_home: std::sync::Mutex::new(Vec::new()),
            seen_recs: std::sync::Mutex::new(Vec::new()),
        });
        let results = hydrator(client)
            .hydrate(
                &ScoredPostsQuery::default(),
                &[PostCandidate {
                    tweet_id: 3,
                    ..Default::default()
                }],
            )
            .await;
        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(hydrated.drop_ancillary_posts, Some(false));
        assert!(matches!(
            hydrated.visibility_reason,
            Some(FilteredReason::SafetyResult(ref s)) if s.action == Action::Interstitial
        ));
    }

    #[tokio::test]
    async fn retweet_original_stays_on_home() {
        let client = Arc::new(LevelClient {
            home: HashMap::from([(4, interstitial()), (200, interstitial())]),
            recs: HashMap::from([(200, recs_drop())]),
            seen_home: std::sync::Mutex::new(Vec::new()),
            seen_recs: std::sync::Mutex::new(Vec::new()),
        });
        let results = hydrator(client.clone())
            .hydrate(
                &ScoredPostsQuery::default(),
                &[PostCandidate {
                    tweet_id: 4,
                    retweeted_tweet_id: Some(200),
                    ..Default::default()
                }],
            )
            .await;
        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(hydrated.drop_ancillary_posts, Some(false));
        assert!(client.seen_home.lock().unwrap().contains(&200));
        assert!(!client.seen_recs.lock().unwrap().contains(&200));
    }
}
