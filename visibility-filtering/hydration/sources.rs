use crate::clients::about_this_account_client::AboutThisAccountClient;
use crate::clients::article_client::ArticleClient;
use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::clients::socialgraph_client::{EdgeQuery, SocialgraphClient};
use crate::clients::trusted_friends_client::TrustedFriendsClient;
use crate::clients::user_devices_client::UserDevicesClient;
use crate::clients::user_location_client::UserLocationClient;
use crate::clients::wingman_client::WingmanClient;
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::community_source::{CommunityPost, CommunitySource};
use crate::hydration::decode::article::decode_lifecycle;
pub(crate) use crate::hydration::decode::author::{
    actor_batch, author_batch, DecodedActor, DecodedAuthor,
};
use crate::hydration::decode::tweet::pure_core;
use crate::hydration::decode::viewer::decode_viewer;
pub(crate) use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::fetcher::Owner;
use crate::hydration::tweet_source::{decode_tweet, decode_tweet_state, TweetSource, TweetState};
use crate::models::{ArticleLifecycle, CommunityModeration, PureCore, TweetFeatures};
use crate::post_card::{PostCard, PostCards};
use futures::future::join;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::iter;
use std::sync::Arc;
use tonic::metadata::{MetadataMap, MetadataValue};
use tracing::warn;
use wingman_client::Exists;
use xai_core_entities::entities::{ConversationControl, GizmoduckUser, QuotedTweet};
use xai_core_entities::gizmoduck_client::QueryFields;
use xai_core_entities::tweet_entity_service_client::TESClient;
use xai_safety_label_source::SafetyLabelSource;
use xai_strato::Bytes;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Expired;

pub(crate) type Edges = Result<Vec<RawHydrationBatch<bool>>, Expired>;

#[tonic::async_trait]
pub(crate) trait Sources: Send + Sync {
    async fn pure_cores(&self, owner: Owner, tweet_ids: &[u64]) -> RawHydrationBatch<PureCore>;

    async fn tweets(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetFeatures>;

    async fn conversation_controls(
        &self,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<ConversationControl>;

    async fn quoted_tweets(&self, outer_ids: &[u64]) -> RawHydrationBatch<QuotedTweet>;

    async fn tweet_states(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetState>;

    async fn safety_labels(
        &self,
        owner: Owner,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>;

    async fn viewer(
        &self,
        viewer_id: u64,
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedViewer>;

    async fn users(
        &self,
        owner: Owner,
        user_ids: &[u64],
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedAuthor>;

    async fn actors(
        &self,
        user_ids: &[u64],
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedActor>;

    async fn confirmed_emails(&self, user_ids: &[u64]) -> RawHydrationBatch<bool>;

    async fn verified_phones(&self, user_ids: &[u64]) -> RawHydrationBatch<bool>;

    async fn select_edges(
        &self,
        owner: Owner,
        viewer_id: Option<u64>,
        queries: &[EdgeQuery],
    ) -> Edges;

    async fn viewer_country(&self, viewer_id: u64) -> RawHydrationBatch<Arc<str>>;

    async fn second_degree(
        &self,
        viewer_id: u64,
        root_author_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn community_moderations(
        &self,
        posts: &[CommunityPost],
    ) -> RawHydrationBatch<CommunityModeration>;

    async fn community_moderators(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn community_viewer_removals(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn article_lifecycles(&self, article_ids: &[u64]) -> RawHydrationBatch<ArticleLifecycle>;

    async fn article_media_entities(
        &self,
        viewer_id: Option<u64>,
        article_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn article_cover_media(
        &self,
        viewer_id: Option<u64>,
        article_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn trusted_friends(&self, viewer_id: u64, list_ids: &[u64]) -> RawHydrationBatch<bool>;

    async fn outside_places(&self, viewer_id: u64, place_ids: &[u64]) -> RawHydrationBatch<bool>;
}

fn expired(status: &tonic::Status) -> bool {
    let timer_expired = iter::successors(std::error::Error::source(status), |err| err.source())
        .any(|err| err.is::<tonic::TimeoutExpired>());
    timer_expired || status.code() == tonic::Code::DeadlineExceeded
}

fn select_result(
    sets: Result<&[Option<HashSet<u64>>], &tonic::Status>,
    edges: Vec<RawHydrationBatch<bool>>,
) -> Edges {
    match sets {
        Err(status) if expired(status) => Err(Expired),
        Ok(_) | Err(_) => Ok(edges),
    }
}

fn edge_batches(
    queries: &[EdgeQuery],
    sets: Result<&[Option<HashSet<u64>>], &tonic::Status>,
) -> Vec<RawHydrationBatch<bool>> {
    queries
        .iter()
        .enumerate()
        .map(|(position, query)| {
            let set = sets
                .as_ref()
                .map(|sets| sets.get(position).and_then(Option::as_ref));
            let answers = query.destination_ids.iter().map(|&destination| {
                let answer = match set {
                    Err(_) => Hydrated::Failed,
                    Ok(None) => Hydrated::Partial(false),
                    Ok(Some(set)) => Hydrated::Found(set.contains(&destination)),
                };
                (destination, answer)
            });
            HydrationBatch::from_hydrated(answers.collect())
        })
        .collect()
}

fn tweet_batch<V>(
    tweet_ids: &[u64],
    values: HashMap<u64, anyhow::Result<Bytes>>,
    decode: fn(&[u8]) -> anyhow::Result<Option<V>>,
) -> RawHydrationBatch<V> {
    let tweets: FxHashMap<_, _> = values
        .into_iter()
        .map(|(id, value)| (id, value.and_then(|bytes| decode(&bytes))))
        .collect();
    HydrationBatch::from_results(tweet_ids.iter().copied(), tweets)
}

fn viewer_batch<E>(
    viewer_id: u64,
    viewer: Result<Option<GizmoduckUser>, E>,
    fields: &[QueryFields],
) -> RawHydrationBatch<DecodedViewer> {
    let viewer = viewer.map(|user| Some(decode_viewer(user.as_ref(), fields)));
    one_viewer_batch(viewer_id, viewer)
}

fn one_viewer_batch<V, E>(viewer_id: u64, answer: Result<Option<V>, E>) -> RawHydrationBatch<V> {
    HydrationBatch::from_results([viewer_id], HashMap::from([(viewer_id, answer)]))
}

fn lifecycle_batch(rows: RawHydrationBatch<i32>) -> RawHydrationBatch<ArticleLifecycle> {
    let decoded = rows.into_hydrated().into_iter().map(|(id, row)| {
        let lifecycle = match row {
            Hydrated::Found(value) => {
                decode_lifecycle(value).map_or(Hydrated::Failed, Hydrated::Found)
            }
            Hydrated::Partial(value) => {
                decode_lifecycle(value).map_or(Hydrated::Failed, Hydrated::Partial)
            }
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed => Hydrated::Failed,
            Hydrated::Unanswered => Hydrated::Unanswered,
        };
        (id, lifecycle)
    });
    HydrationBatch::from_hydrated(decoded.collect())
}

pub(crate) struct Backends {
    pub(crate) tes: Arc<dyn TESClient + Send + Sync>,
    pub(crate) tweets: TweetSource,
    pub(crate) gizmoduck: GizmoduckLookup,
    pub(crate) socialgraph: Arc<dyn SocialgraphClient + Send + Sync>,
    pub(crate) about_this_account: AboutThisAccountClient,
    pub(crate) wingman: Arc<dyn WingmanClient>,
    pub(crate) articles: ArticleClient,
    pub(crate) trusted_friends: TrustedFriendsClient,
    pub(crate) user_location: UserLocationClient,
    pub(crate) safety_labels: Arc<SafetyLabelSource>,
    pub(crate) communities: CommunitySource,
    pub(crate) user_devices: UserDevicesClient,
}

pub(crate) struct ProdSources {
    backends: Backends,
}

impl ProdSources {
    pub(crate) fn new(backends: Backends) -> Self {
        Self { backends }
    }
}

const TEST_USERS_DTAB: &str = "/s/gizmoduck/test-users-permanent=>/s/gizmoduck/gizmoduck";

pub fn test_users_metadata() -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert("dtab-local", MetadataValue::from_static(TEST_USERS_DTAB));
    metadata
}

#[tonic::async_trait]
impl Sources for ProdSources {
    async fn pure_cores(&self, _: Owner, tweet_ids: &[u64]) -> RawHydrationBatch<PureCore> {
        let cores = self
            .backends
            .tes
            .get_tweet_core_datas(tweet_ids.to_vec())
            .await;
        HydrationBatch::from_results(tweet_ids.iter().copied(), cores).map(|core| pure_core(&core))
    }

    async fn tweets(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetFeatures> {
        let values = self.backends.tweets.get_tweet_values(tweet_ids).await;
        tweet_batch(tweet_ids, values, decode_tweet)
    }

    async fn conversation_controls(
        &self,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<ConversationControl> {
        let controls = self
            .backends
            .tes
            .get_conversation_controls(tweet_ids.to_vec())
            .await;
        HydrationBatch::from_results(tweet_ids.iter().copied(), controls)
    }

    async fn quoted_tweets(&self, outer_ids: &[u64]) -> RawHydrationBatch<QuotedTweet> {
        let quoted = self
            .backends
            .tes
            .get_quoted_tweets(outer_ids.to_vec())
            .await;
        HydrationBatch::from_results(outer_ids.iter().copied(), quoted)
    }

    async fn tweet_states(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetState> {
        let values = self.backends.tweets.get_tweet_states(tweet_ids).await;
        tweet_batch(tweet_ids, values, decode_tweet_state)
    }

    async fn safety_labels(
        &self,
        _: Owner,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>> {
        let labels: FxHashMap<_, _> = self
            .backends
            .safety_labels
            .get(tweet_ids)
            .await
            .into_iter()
            .map(|(id, labels)| (id, labels.map(Some)))
            .collect();
        HydrationBatch::from_results(tweet_ids.iter().copied(), labels)
    }

    async fn viewer(
        &self,
        viewer_id: u64,
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedViewer> {
        let viewer = self
            .backends
            .gizmoduck
            .get_viewer(viewer_id, fields)
            .await
            .inspect_err(|error| warn!(%error, "Gizmoduck viewer lookup failed; failing open"));
        viewer_batch(viewer_id, viewer, fields)
    }

    async fn users(
        &self,
        _: Owner,
        user_ids: &[u64],
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedAuthor> {
        let users = self
            .backends
            .gizmoduck
            .get_users(user_ids.to_vec(), fields)
            .await;
        author_batch(HydrationBatch::from_results(
            user_ids.iter().copied(),
            users,
        ))
    }

    async fn actors(
        &self,
        user_ids: &[u64],
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedActor> {
        let users = self
            .backends
            .gizmoduck
            .get_users(user_ids.to_vec(), fields)
            .await;
        actor_batch(HydrationBatch::from_results(
            user_ids.iter().copied(),
            users,
        ))
    }

    async fn confirmed_emails(&self, user_ids: &[u64]) -> RawHydrationBatch<bool> {
        let confirmed = self
            .backends
            .user_devices
            .confirmed_emails(user_ids)
            .await
            .inspect_err(|error| warn!(%error, "confirmed email lookup failed"));
        HydrationBatch::from_results(
            user_ids.iter().copied(),
            held_by(user_ids, confirmed.as_ref().ok()),
        )
    }

    async fn verified_phones(&self, user_ids: &[u64]) -> RawHydrationBatch<bool> {
        let verified = self
            .backends
            .user_devices
            .verified_phones(user_ids)
            .await
            .inspect_err(|error| warn!(%error, "verified phone lookup failed"));
        HydrationBatch::from_results(
            user_ids.iter().copied(),
            held_by(user_ids, verified.as_ref().ok()),
        )
    }

    async fn select_edges(
        &self,
        _owner: Owner,
        viewer_id: Option<u64>,
        queries: &[EdgeQuery],
    ) -> Edges {
        let sets = self
            .backends
            .socialgraph
            .select_edges(viewer_id, queries)
            .await;
        let edges = edge_batches(queries, sets.as_deref());
        select_result(sets.as_deref(), edges)
    }

    async fn viewer_country(&self, viewer_id: u64) -> RawHydrationBatch<Arc<str>> {
        let country = self
            .backends
            .about_this_account
            .tfe_top_country(viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "tfe_top_country lookup failed"));
        one_viewer_batch(viewer_id, country.map(|country| country.map(Arc::from)))
    }

    async fn second_degree(
        &self,
        viewer_id: u64,
        root_author_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let answers = self
            .backends
            .wingman
            .batch_exists_intersect(viewer_id, root_author_ids)
            .await;
        let answers: HashMap<_, _> = root_author_ids
            .iter()
            .copied()
            .zip(answers.into_iter().flatten())
            .map(|(root, answer)| {
                let answer = match answer {
                    Exists::Found => Ok(Some(true)),
                    Exists::NotFound => Ok(Some(false)),
                    Exists::Incomplete | Exists::ItemError => Err(answer),
                };
                (root, answer)
            })
            .collect();
        HydrationBatch::from_results(root_author_ids.iter().copied(), answers)
    }

    async fn community_moderations(
        &self,
        posts: &[CommunityPost],
    ) -> RawHydrationBatch<CommunityModeration> {
        let moderations = self.backends.communities.moderations(posts).await;
        HydrationBatch::from_results(posts.iter().map(|post| post.tweet_id), moderations)
    }

    async fn community_moderators(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let moderators = self
            .backends
            .communities
            .moderators(viewer_id, community_ids)
            .await;
        HydrationBatch::from_results(community_ids.iter().copied(), moderators)
    }

    async fn community_viewer_removals(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let removals = self
            .backends
            .communities
            .viewer_removals(viewer_id, community_ids)
            .await;
        HydrationBatch::from_results(community_ids.iter().copied(), removals)
    }

    async fn article_media_entities(
        &self,
        viewer_id: Option<u64>,
        article_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let rows = self
            .backends
            .articles
            .media_entities(viewer_id, article_ids)
            .await;
        HydrationBatch::from_results(article_ids.iter().copied(), rows)
    }

    async fn article_cover_media(
        &self,
        viewer_id: Option<u64>,
        article_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let rows = self
            .backends
            .articles
            .cover_media(viewer_id, article_ids)
            .await;
        HydrationBatch::from_results(article_ids.iter().copied(), rows)
    }

    async fn article_lifecycles(&self, article_ids: &[u64]) -> RawHydrationBatch<ArticleLifecycle> {
        let rows = self.backends.articles.lifecycles(article_ids).await;
        lifecycle_batch(HydrationBatch::from_results(
            article_ids.iter().copied(),
            rows,
        ))
    }

    async fn trusted_friends(&self, viewer_id: u64, list_ids: &[u64]) -> RawHydrationBatch<bool> {
        let answers = self
            .backends
            .trusted_friends
            .batch_is_member_or_owner(viewer_id, list_ids)
            .await;
        let answers: HashMap<_, _> = list_ids
            .iter()
            .copied()
            .zip(answers)
            .map(|(list_id, answer)| (list_id, answer.map(Some)))
            .collect();
        HydrationBatch::from_results(list_ids.iter().copied(), answers)
    }

    async fn outside_places(&self, viewer_id: u64, place_ids: &[u64]) -> RawHydrationBatch<bool> {
        let places = self
            .backends
            .user_location
            .places(viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "Geoduck userLocation lookup failed"));
        let answers: HashMap<_, _> = place_ids
            .iter()
            .map(|&place| {
                let outside = places.as_ref().map(|places| Some(!places.contains(&place)));
                (place, outside)
            })
            .collect();
        HydrationBatch::from_results(place_ids.iter().copied(), answers)
    }
}

#[tonic::async_trait]
impl PostCards for ProdSources {
    async fn post_card(&self, tweet_id: u64) -> Option<PostCard> {
        let (core, media) = join(
            self.backends.tes.get_tweet_core_datas(vec![tweet_id]),
            self.backends.tes.get_tweet_media_entities(vec![tweet_id]),
        )
        .await;
        let core = only(core)?;
        let author = only(
            self.backends
                .gizmoduck
                .get_users(vec![core.author_id], &[QueryFields::PROFILE])
                .await,
        )
        .and_then(|result| result.user);
        Some(PostCard::new(
            tweet_id,
            core,
            author.as_ref().map(|user| &user.profile),
            &only(media).unwrap_or_default(),
        ))
    }
}

fn held_by(
    user_ids: &[u64],
    holders: Option<&HashSet<u64>>,
) -> HashMap<u64, Result<Option<bool>, ()>> {
    user_ids
        .iter()
        .map(|&user_id| {
            let held = holders
                .map(|holders| Some(holders.contains(&user_id)))
                .ok_or(());
            (user_id, held)
        })
        .collect()
}

fn only<V, E>(answers: HashMap<u64, Result<Option<V>, E>>) -> Option<V> {
    answers.into_values().next()?.ok().flatten()
}

#[cfg(test)]
pub(crate) use in_memory::{control, found_user, suspended, Fault, InMemorySources};

#[cfg(test)]
mod fake_backends;

#[cfg(test)]
mod in_memory;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::Graph;
    use tonic::Status;

    #[test]
    fn a_select_past_its_forwarded_deadline_expires_and_another_error_fails_every_key() {
        let queries = [EdgeQuery::forward(Graph::Follows, vec![1, 2])];
        let client_timer = Status::from_error(Box::new(tonic::TimeoutExpired(())));
        for (status, expired) in [
            (client_timer, true),
            (Status::deadline_exceeded(""), true),
            (Status::unavailable(""), false),
        ] {
            let batches = edge_batches(&queries, Err(&status));
            let failed = Some(Hydrated::Failed);
            assert_eq!(
                [1, 2].map(|key| batches[0].hydrated(&key).cloned()),
                [failed.clone(), failed],
                "{status:?}"
            );
            let result = select_result(Err(&status), batches);
            assert_eq!(result.is_err(), expired, "{status:?}");
        }
    }
}
