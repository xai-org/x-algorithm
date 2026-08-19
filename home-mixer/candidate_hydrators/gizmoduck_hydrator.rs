use crate::clients::gizmoduck_client::GizmoduckClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};
use xai_x_thrift::user_labels::LabelValue;

pub struct GizmoduckCandidateHydrator {
    pub gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
    pub cache: QuickCache<GizmoduckCacheKey, GizmoduckCacheValue>,
}

impl GizmoduckCandidateHydrator {
    pub async fn new(gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>) -> Self {
        let cache = default_quick_cache();
        Self {
            gizmoduck_client,
            cache,
        }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for GizmoduckCandidateHydrator {
    type CacheKey = GizmoduckCacheKey;
    type CacheValue = GizmoduckCacheValue;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }
    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        GizmoduckCacheKey {
            author_id: candidate.author_id,
            retweeted_user_id: candidate.retweeted_user_id,
        }
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        GizmoduckCacheValue {
            author_followers_count: hydrated.author_followers_count,
            author_screen_name: hydrated.author_screen_name.clone(),
            retweeted_screen_name: hydrated.retweeted_screen_name.clone(),
            nsfw_author: hydrated.nsfw_author,
            nsfw_author_ads: hydrated.nsfw_author_ads,
            nsfw_author_phoenix: hydrated.nsfw_author_phoenix,
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            author_followers_count: value.author_followers_count,
            author_screen_name: value.author_screen_name,
            retweeted_screen_name: value.retweeted_screen_name,
            nsfw_author: value.nsfw_author,
            nsfw_author_ads: value.nsfw_author_ads,
            nsfw_author_phoenix: value.nsfw_author_phoenix,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let client = &self.gizmoduck_client;

        let user_ids_to_fetch: Vec<i64> = candidates
            .iter()
            .map(|c| c.author_id as i64)
            .chain(
                candidates
                    .iter()
                    .filter_map(|c| c.retweeted_user_id)
                    .map(|id| id as i64),
            )
            .collect::<HashSet<i64>>()
            .into_iter()
            .collect();

        let users = client.get_users(user_ids_to_fetch).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let user = users.get(&(candidate.author_id as i64));
            let user = match user {
                Some(Ok(Some(user))) => Ok(Some(user)),
                Some(Ok(None)) | None => Ok(None),
                Some(Err(err)) => Err(err.to_string()),
            };

            let retweet_user = candidate
                .retweeted_user_id
                .and_then(|retweeted_user_id| users.get(&(retweeted_user_id as i64)));
            let retweet_user = match retweet_user {
                Some(Ok(Some(user))) => Ok(Some(user)),
                Some(Ok(None)) | None => Ok(None),
                Some(Err(err)) => Err(err.to_string()),
            };

            let hydrated = match (user, retweet_user) {
                (Ok(user), Ok(retweet_user)) => {
                    let user_counts = user.and_then(|user| user.user.as_ref().map(|u| &u.counts));
                    let user_profile = user.and_then(|user| user.user.as_ref().map(|u| &u.profile));

                    let author_followers_count: Option<i32> =
                        user_counts.map(|x| x.followers_count).map(|x| x as i32);
                    let author_screen_name: Option<String> =
                        user_profile.map(|x| x.screen_name.clone());

                    let retweet_profile =
                        retweet_user.and_then(|user| user.user.as_ref().map(|u| &u.profile));
                    let retweeted_screen_name: Option<String> =
                        retweet_profile.map(|x| x.screen_name.clone());

                    let author = user.and_then(|u| u.user.as_ref());
                    let nsfw_author: Option<bool> = author.map(|u| {
                        u.safety.nsfw_admin
                            || u.safety.nsfw_user
                            || u.labels
                                .labels
                                .iter()
                                .any(|label| label.label_value == LabelValue::NSFW_HIGH_PRECISION.0)
                    });
                    let nsfw_author_ads: Option<bool> = author.map(|u| {
                        nsfw_author.unwrap_or(false)
                            || u.labels.labels.iter().any(|label| {
                                label.label_value == LabelValue::POSSIBLY_NSFW_ACCOUNT.0
                            })
                    });
                    let nsfw_author_phoenix: Option<bool> = author.map(|u| {
                        u.safety.nsfw_user
                            || u.safety.nsfw_admin
                            || u.labels.labels.iter().any(|l| {
                                matches!(
                                    LabelValue(l.label_value),
                                    LabelValue::NSFW_HIGH_PRECISION
                                        | LabelValue::POSSIBLY_NSFW_ACCOUNT
                                )
                            })
                    });

                    Ok(PostCandidate {
                        author_followers_count,
                        author_screen_name,
                        retweeted_screen_name,
                        nsfw_author,
                        nsfw_author_ads,
                        nsfw_author_phoenix,
                        ..Default::default()
                    })
                }
                (Err(err), _) | (_, Err(err)) => Err(err),
            };
            hydrated_candidates.push(hydrated);
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.author_followers_count = hydrated.author_followers_count;
        candidate.author_screen_name = hydrated.author_screen_name;
        candidate.retweeted_screen_name = hydrated.retweeted_screen_name;
        candidate.nsfw_author = hydrated.nsfw_author;
        candidate.nsfw_author_ads = hydrated.nsfw_author_ads;
        candidate.nsfw_author_phoenix = hydrated.nsfw_author_phoenix;
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GizmoduckCacheKey {
    pub author_id: u64,
    pub retweeted_user_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct GizmoduckCacheValue {
    pub author_followers_count: Option<i32>,
    pub author_screen_name: Option<String>,
    pub retweeted_screen_name: Option<String>,
    pub nsfw_author: Option<bool>,
    pub nsfw_author_ads: Option<bool>,
    pub nsfw_author_phoenix: Option<bool>,
}
