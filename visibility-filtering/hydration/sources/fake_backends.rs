use super::*;
use crate::clients::socialgraph_client::{EdgeDirection, EdgeFrom, EdgeQuery, Graph};
use crate::clients::strato::answer::{found, no_row, Thrift};
use crate::clients::{trusted_friends_client, user_devices_client, user_location_client};
use crate::hydration::batch::Hydrated;
use crate::hydration::community_source::CommunityPost;
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::fetcher::Owner;
use crate::hydration::sources::Source;
use crate::hydration::tweet_source::TweetState;
use crate::models::NsfwFeature;
use crate::models::{ArticleLifecycle, CommunityModeration, TweetFeatures};
use std::fmt;
use std::num::NonZeroU64;
use thrift::protocol::TType;
use tonic::{Request, Response, Status};
use tower::util::BoxCloneSyncService;
use tower::ServiceExt;
use wingman_client::Exists;
use xai_core_entities::entities::{
    ApiCounts, CashtagAttachments, ConversationControl, ConversationControlArm, EditControl,
    EscherbirdEntityAnnotation, ExclusiveTweetControl, GizmoduckAccount, GizmoduckUser,
    GizmoduckUserResult, MediaEntities, NoteTweet, PCFLabel, PureCoreData, QuotedTweet,
    ReactionContext, Safety, TakedownReason, TrustedFriendsControl, UrlEntities, UserResponseState,
};
use xai_core_entities::gizmoduck_client::{
    GizmoduckClient, LookupContext, MockGizmoduckClient, UserFields, ViewerData,
};
use xai_core_entities::tweet_entity_service_client::{MockTESClient, TESClient};
use xai_safety_label_source::testing::FakeLabels;
use xai_strato::strato_proto::rpc_server::{Rpc, RpcServer};
use xai_strato::strato_proto::{result, Call3, Issue3, ResultList};
use xai_strato::StratoGrpc;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::entities::ApiMediaEntity;
use xai_x_thrift::tweet_safety_label::{PerspectivalUser, SafetyLabel, SafetyLabelType};
use xai_x_thrift::tweets::ApiPerspective;

const VIEWER: u64 = 50;
const AUTHOR: u64 = 10;
const SUSPENDED_AUTHOR: u64 = 11;
const SOURCE_AUTHOR: u64 = 12;
const REPLIED_AUTHOR: u64 = 13;
const INVITED: u64 = 14;
const LABELED: u64 = 15;
const EXCLUSIVE_AUTHOR: u64 = 16;
const TWEET: u64 = 1;
const SOURCE_TWEET: u64 = 2;
const REPLIED_TWEET: u64 = 3;
const CONVERSATION: u64 = 4;
const INITIAL_TWEET: u64 = 5;
const EDIT_TWEET: u64 = 6;
const CIRCLE: u64 = 7;
const ROOT: u64 = 20;
const FOLLOWER: u64 = 30;
const COMMUNITY: u64 = 40;
const ARTICLE: u64 = 60;
const PLACE: u64 = 70;
const MISSING: u64 = 8;
const FAILED: u64 = 9;
const ABSENT: u64 = 17;
const EDGES: [(Graph, EdgeDirection, bool); 3] = [
    (Graph::Follows, EdgeDirection::Forward, false),
    (Graph::Follows, EdgeDirection::Reverse, true),
    (Graph::Blocks, EdgeDirection::Forward, true),
];
const FIELDS: [&[QueryFields]; 2] = [&[QueryFields::SAFETY], &[QueryFields::ACCOUNT]];
const POST: CommunityPost = CommunityPost {
    tweet_id: TWEET,
    author_id: AUTHOR,
    community_id: COMMUNITY,
};

type Users = HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>>;

struct Gizmoduck(MockGizmoduckClient);

#[tonic::async_trait]
impl GizmoduckClient for Gizmoduck {
    async fn get_users_with_context(
        &self,
        ids: Vec<i64>,
        context: Option<LookupContext>,
        fields: &[QueryFields],
    ) -> Users {
        let mut users = self.0.get_users_with_context(ids, context, fields).await;
        users.remove(&ABSENT.cast_signed());
        if let Some(user) = users.get_mut(&FAILED.cast_signed()) {
            *user = Err(anyhow::anyhow!("{FAILED} fails"));
        }
        users
    }

    async fn get_users(&self, _: Vec<i64>) -> Users {
        unreachable!()
    }

    async fn get_users_with_perspective(&self, _: i64, _: Vec<i64>) -> Users {
        unreachable!()
    }

    async fn get_viewer_roles(&self, _: u64) -> anyhow::Result<Vec<String>> {
        unreachable!()
    }

    async fn get_viewer_data(&self, _: u64) -> anyhow::Result<ViewerData> {
        unreachable!()
    }

    async fn get_viewer_data_with_fields(
        &self,
        _: u64,
        _: &[QueryFields],
    ) -> anyhow::Result<ViewerData> {
        unreachable!()
    }

    async fn get_pcf_labels(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<PCFLabel>> {
        unreachable!()
    }

    async fn get_profile_description_languages(
        &self,
        _: Vec<i64>,
    ) -> HashMap<i64, anyhow::Result<Option<String>>> {
        unreachable!()
    }

    async fn get_user_fields(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<UserFields>> {
        unreachable!()
    }

    async fn get_by_screen_name(&self, _: &str) -> anyhow::Result<Option<GizmoduckUserResult>> {
        unreachable!()
    }
}

type Answers<V> = HashMap<u64, anyhow::Result<Option<V>>>;

struct Tes(MockTESClient);

impl Tes {
    fn failing<V>(mut answers: Answers<V>) -> Answers<V> {
        answers.remove(&ABSENT);
        if let Some(answer) = answers.get_mut(&FAILED) {
            *answer = Err(anyhow::anyhow!("{FAILED} fails"));
        }
        answers
    }
}

#[tonic::async_trait]
impl TESClient for Tes {
    async fn get_tweet_core_datas(&self, ids: Vec<u64>) -> Answers<PureCoreData> {
        Self::failing(self.0.get_tweet_core_datas(ids).await)
    }

    async fn get_conversation_controls(&self, ids: Vec<u64>) -> Answers<ConversationControl> {
        Self::failing(self.0.get_conversation_controls(ids).await)
    }

    async fn get_quoted_tweets(&self, ids: Vec<u64>) -> Answers<QuotedTweet> {
        Self::failing(self.0.get_quoted_tweets(ids).await)
    }

    async fn get_tweet_media_entities(&self, _: Vec<u64>) -> Answers<MediaEntities> {
        unreachable!()
    }

    async fn get_subscription_author_ids(&self, _: Vec<u64>) -> Answers<u64> {
        unreachable!()
    }

    async fn get_reaction_context(&self, _: Vec<u64>) -> Answers<ReactionContext> {
        unreachable!()
    }

    async fn get_min_video_durations(&self, _: Vec<u64>) -> Answers<i64> {
        unreachable!()
    }

    async fn get_media_count(&self, _: Vec<u64>) -> Answers<i64> {
        unreachable!()
    }

    async fn get_nullcast(&self, _: Vec<u64>) -> Answers<bool> {
        unreachable!()
    }

    async fn get_community(&self, _: Vec<u64>) -> Answers<i64> {
        unreachable!()
    }

    async fn get_nsfw_user(&self, _: Vec<u64>) -> Answers<bool> {
        unreachable!()
    }

    async fn get_nsfw_admin(&self, _: Vec<u64>) -> Answers<bool> {
        unreachable!()
    }

    async fn get_has_takedown(&self, _: Vec<u64>) -> Answers<bool> {
        unreachable!()
    }

    async fn get_takedown_country_codes(&self, _: Vec<u64>) -> Answers<Vec<String>> {
        unreachable!()
    }

    async fn get_takedown_reasons(&self, _: Vec<u64>) -> Answers<Vec<TakedownReason>> {
        unreachable!()
    }

    async fn get_language_code(&self, _: Vec<u64>) -> Answers<String> {
        unreachable!()
    }

    async fn get_api_counts(&self, _: Vec<u64>) -> Answers<ApiCounts> {
        unreachable!()
    }

    async fn get_core_data_and_api_counts(
        &self,
        _: u64,
    ) -> (
        anyhow::Result<Option<PureCoreData>>,
        anyhow::Result<Option<ApiCounts>>,
    ) {
        unreachable!()
    }

    async fn get_is_article(&self, _: Vec<u64>) -> Answers<bool> {
        unreachable!()
    }

    async fn get_is_premium(&self, _: Vec<u64>) -> Answers<bool> {
        unreachable!()
    }

    async fn get_urls(&self, _: Vec<u64>) -> Answers<UrlEntities> {
        unreachable!()
    }

    async fn get_note_tweets(&self, _: Vec<u64>) -> Answers<NoteTweet> {
        unreachable!()
    }

    async fn get_cashtag_attachments(&self, _: Vec<u64>) -> Answers<CashtagAttachments> {
        unreachable!()
    }

    async fn get_exclusive_controls(&self, _: Vec<u64>) -> Answers<ExclusiveTweetControl> {
        unreachable!()
    }

    async fn get_trusted_friends_controls(&self, _: Vec<u64>) -> Answers<TrustedFriendsControl> {
        unreachable!()
    }

    async fn get_grok_post_ids(&self, _: Vec<u64>) -> Answers<String> {
        unreachable!()
    }

    async fn get_edit_control(&self, _: Vec<u64>) -> Answers<EditControl> {
        unreachable!()
    }

    async fn get_status_perspectives(
        &self,
        _: Vec<u64>,
        _: Option<&MetadataMap>,
    ) -> Answers<ApiPerspective> {
        unreachable!()
    }

    async fn get_api_media_entities(
        &self,
        _: Vec<u64>,
        _: Option<&MetadataMap>,
    ) -> Answers<Vec<ApiMediaEntity>> {
        unreachable!()
    }

    async fn get_escherbird_entity_annotations(
        &self,
        _: Vec<u64>,
    ) -> Answers<Vec<EscherbirdEntityAnnotation>> {
        unreachable!()
    }
}

struct Fake;

#[tonic::async_trait]
impl WingmanClient for Fake {
    async fn batch_exists_intersect(&self, _: u64, ids: &[u64]) -> Option<Vec<Exists>> {
        let exists = ids.iter().map(|&id| match id {
            MISSING => Exists::NotFound,
            FAILED => Exists::ItemError,
            ABSENT => Exists::Incomplete,
            _ => Exists::Found,
        });
        Some(exists.collect())
    }
}

#[tonic::async_trait]
impl SocialgraphClient for Fake {
    async fn select_edges(
        &self,
        _: Option<u64>,
        queries: &[EdgeQuery],
    ) -> Result<Vec<Option<HashSet<u64>>>, Status> {
        if queries
            .iter()
            .any(|query| query.destination_ids.contains(&FAILED))
        {
            return Err(Status::unavailable("select failed"));
        }
        let sets = queries.iter().map(|query| {
            let holds = EDGES.contains(&(query.graph, query.direction, true));
            let set = query.destination_ids.iter().copied().filter(|_| holds);
            (query.graph != Graph::Mutes).then(|| set.collect())
        });
        Ok(sets.collect())
    }
}

fn labels() -> vf_pb::SafetyLabelMap {
    let label = vf_pb::SafetyLabel {
        applicable_users: vec![LABELED.cast_signed()],
        ..Default::default()
    };
    vf_pb::SafetyLabelMap {
        labels: [(i32::from(SafetyLabelType::SPAM), label)].into(),
    }
}

fn id(id: u64) -> Thrift {
    Thrift::I64(id.cast_signed())
}

fn tweet() -> Thrift {
    use Thrift::{Bool, List, Str, Struct};
    let edit_control_initial = Struct(vec![(1, List(TType::I64, vec![id(EDIT_TWEET)]))]);
    let edit = Struct(vec![(1, id(INITIAL_TWEET)), (2, edit_control_initial)]);
    Struct(vec![
        (1, id(TWEET)),
        (2, Struct(vec![(1, id(AUTHOR)), (9, Bool(true))])),
        (
            125,
            Struct(vec![(1, List(TType::I64, vec![id(COMMUNITY)]))]),
        ),
        (155, Struct(vec![(1, id(EXCLUSIVE_AUTHOR))])),
        (156, Struct(vec![(1, id(CIRCLE))])),
        (157, Struct(vec![(2, edit)])),
        (170, Struct(vec![(1, id(ARTICLE))])),
        (172, Struct(vec![(1, Str(format!("{PLACE:x}")))])),
    ])
}

fn tweet_features() -> TweetFeatures {
    TweetFeatures {
        nsfw: NsfwFeature {
            user: true,
            admin: false,
        },
        community_id: NonZeroU64::new(COMMUNITY),
        exclusive_conversation_author_id: Some(EXCLUSIVE_AUTHOR),
        trusted_friends_list_id: Some(CIRCLE),
        latest_edit_tweet_id: Some(EDIT_TWEET),
        article_id: NonZeroU64::new(ARTICLE),
        narrowcast_place_id: Some(PLACE),
        ..Default::default()
    }
}

fn located() -> Thrift {
    use Thrift::{Map, Set, Struct, I32};
    const CITY: i32 = 3;
    let places = vec![(I32(CITY), Set(TType::I64, vec![id(PLACE)]))];
    let viewer = Struct(vec![
        (1, id(VIEWER)),
        (5, Map(TType::I32, TType::Set, places)),
    ]);
    let found = Map(TType::I64, TType::Struct, vec![(id(VIEWER), viewer)]);
    Struct(vec![(1, found)])
}

fn held_by_author(entry: Thrift) -> Thrift {
    Thrift::Struct(vec![(1, Thrift::List(TType::Struct, vec![entry]))])
}

struct Strato;

#[tonic::async_trait]
impl Rpc for Strato {
    async fn issue3(&self, request: Request<Issue3>) -> Result<Response<ResultList>, Status> {
        use Thrift::{Bool, Set, Str, Struct, I32, I64};
        const SOFT_DELETED: i32 = 2;
        const BOUNCE_DELETED: i32 = 15;
        fn asks(call: &Call3, id: u64) -> bool {
            let key = match call.path.as_str() {
                "trusted-friends/visibility-filtering/isMember.TrustedFriendsList"
                | "trusted-friends/visibility-filtering/isOwner.TrustedFriendsList" => {
                    trusted_friends_client::key(id, VIEWER)
                }
                "geo/service/userLocation" => user_location_client::key(id),
                user_devices_client::CONFIRMED_EMAILS_COLUMN => {
                    user_devices_client::emails_key(&[id])
                }
                user_devices_client::VERIFIED_PHONES_COLUMN => {
                    user_devices_client::phones_key(&[id])
                }
                _ => xai_strato::encode(&(id, ())),
            };
            call.args.first().is_some_and(|asked| *asked == key)
        }
        let results = request
            .get_ref()
            .calls
            .iter()
            .filter(|call| !asks(call, ABSENT))
            .map(|call| {
                if asks(call, FAILED) {
                    return Ok(xai_strato::strato_proto::Result {
                        result_type: Some(result::ResultType::Err(
                            xai_strato::strato_proto::Error {
                                code: 5,
                                message: "fails".to_owned(),
                            },
                        )),
                    });
                }
                let value = match call.path.as_str() {
                    _ if asks(call, MISSING) => None,
                    "tweetypie/federated/tweetForVisibility.Tweet" => Some(tweet()),
                    "tweetypie/federated/tweetState.Tweet" => Some(I32(BOUNCE_DELETED)),
                    "communities/moderationState.TweetCommunityRelationship"
                    | "communities/moderationState.UserCommunityRelationship" => {
                        Some(Struct(vec![(2, Struct(vec![]))]))
                    }
                    "communities/visibility/visibilityFeatures.Community" => {
                        Some(Struct(vec![(1, Struct(vec![(4, Bool(true))]))]))
                    }
                    "communities/isRemoved.Community" => Some(Bool(true)),
                    "about_this_account/tfe_top_country.User" => {
                        Some(Struct(vec![(2, Struct(vec![(1, Str("us".to_owned()))]))]))
                    }
                    "article/fields/mediaEntities.ArticleEntity" => Some(Set(
                        TType::Struct,
                        vec![Struct(vec![(1, Struct(vec![(1, I64(7)), (2, I64(0))]))])],
                    )),
                    "article/fields/coverMedia.ArticleEntity" => {
                        Some(Struct(vec![(1, Struct(vec![(1, I64(7)), (2, I64(0))]))]))
                    }
                    "article/fields/lifecycleState.ArticleEntity" => {
                        Some(Struct(vec![(1, I32(SOFT_DELETED))]))
                    }
                    "trusted-friends/visibility-filtering/isMember.TrustedFriendsList"
                    | "trusted-friends/visibility-filtering/isOwner.TrustedFriendsList" => {
                        Some(Bool(true))
                    }
                    "geo/service/userLocation" => Some(located()),
                    user_devices_client::CONFIRMED_EMAILS_COLUMN => {
                        Some(held_by_author(Struct(vec![
                            (1, I64(AUTHOR.cast_signed())),
                            (2, Struct(vec![(1, Str("obfuscated".to_owned()))])),
                        ])))
                    }
                    user_devices_client::VERIFIED_PHONES_COLUMN => {
                        Some(held_by_author(Struct(vec![
                            (1, Str("obfuscated".to_owned())),
                            (2, I64(AUTHOR.cast_signed())),
                        ])))
                    }
                    path => return Err(Status::unimplemented(path)),
                };
                let answer = value.map_or_else(no_row, found);
                Ok(xai_strato::strato_proto::Result {
                    result_type: Some(result::ResultType::Ok(answer.into())),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Response::new(ResultList { results }))
    }
}

fn strato() -> StratoGrpc {
    StratoGrpc::from_boxed_service(BoxCloneSyncService::new(ServiceExt::<
        tonic::codegen::http::Request<tonic::body::Body>,
    >::map_err(
        RpcServer::new(Strato),
        |never| match never {},
    )))
}

fn core() -> PureCoreData {
    PureCoreData {
        author_id: AUTHOR,
        source_tweet_id: Some(SOURCE_TWEET),
        source_user_id: Some(SOURCE_AUTHOR),
        in_reply_to_tweet_id: Some(REPLIED_TWEET),
        in_reply_to_user_id: Some(REPLIED_AUTHOR),
        conversation_id: Some(CONVERSATION),
        ..Default::default()
    }
}

fn conversation_control() -> ConversationControl {
    ConversationControl {
        invited_user_ids: vec![INVITED],
        ..control(ConversationControlArm::ByInvitation, AUTHOR, &[])
    }
}

fn quoted_tweet() -> QuotedTweet {
    QuotedTweet {
        tweet_id: SOURCE_TWEET,
        user_id: SOURCE_AUTHOR,
    }
}

fn viewer() -> GizmoduckUser {
    GizmoduckUser {
        user_id: VIEWER,
        account: GizmoduckAccount {
            nsfw_view: true,
            country_code: Some("US".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn suspended_author() -> GizmoduckUserResult {
    GizmoduckUserResult {
        user: Some(GizmoduckUser {
            user_id: SUSPENDED_AUTHOR,
            safety: Safety {
                suspended: true,
                ..Default::default()
            },
            ..Default::default()
        }),
        response_state: Some(UserResponseState::Found),
    }
}

fn prod_sources() -> ProdSources {
    let tes = MockTESClient {
        core_data: HashMap::from([(TWEET, Some(core()))]),
        conversation_controls: HashMap::from([(TWEET, Some(conversation_control()))]),
        quoted_tweets: HashMap::from([(TWEET, Some(quoted_tweet()))]),
        ..Default::default()
    };
    let found = |user| GizmoduckUserResult {
        user: Some(user),
        response_state: Some(UserResponseState::Found),
    };
    let gizmoduck = MockGizmoduckClient {
        users: HashMap::from([
            (VIEWER.cast_signed(), Some(found(viewer()))),
            (AUTHOR.cast_signed(), Some(found_user(AUTHOR))),
            (SUSPENDED_AUTHOR.cast_signed(), Some(suspended_author())),
        ]),
        ..Default::default()
    };
    let fake = Arc::new(Fake);
    let label = SafetyLabel {
        applicable_users: Some(vec![PerspectivalUser {
            user_id: LABELED.cast_signed(),
        }]),
        ..Default::default()
    };
    ProdSources::new(Backends {
        tes: Arc::new(Tes(tes)),
        tweets: TweetSource {
            grpc_client: Arc::new(strato()),
        },
        gizmoduck: GizmoduckLookup::new(Arc::new(Gizmoduck(gizmoduck))),
        socialgraph: Arc::clone(&fake) as _,
        about_this_account: AboutThisAccountClient::new(strato()),
        wingman: fake,
        articles: ArticleClient::new(strato()),
        trusted_friends: TrustedFriendsClient::new(strato()),
        user_location: UserLocationClient::new(strato()),
        safety_labels: Arc::new(
            FakeLabels::default()
                .label(TWEET, SafetyLabelType::SPAM, label)
                .not_found(MISSING)
                .failing(FAILED)
                .into_source(),
        ),
        communities: CommunitySource {
            grpc_client: strato(),
        },
        user_devices: UserDevicesClient::new(strato()),
    })
}

fn failing_keys(mut sources: InMemorySources) -> InMemorySources {
    for fetcher in FetcherId::all() {
        sources = sources.fail_key(fetcher, FAILED).fail_key(fetcher, ABSENT);
    }
    sources
        .fail_edge(Graph::Follows, FAILED)
        .fail_edge(Graph::Blocks, FAILED)
        .miss_graph(Graph::Mutes)
}

fn held() -> InMemorySources {
    failing_keys(
        InMemorySources::default()
            .pure_core(TWEET, core())
            .tweet_features(TWEET, tweet_features())
            .control(TWEET, conversation_control())
            .quote(TWEET, SOURCE_TWEET, SOURCE_AUTHOR)
            .tweet_state(TWEET, TweetState::BounceDeleted)
            .labels(TWEET, labels())
            .labels(MISSING, vf_pb::SafetyLabelMap::default())
            .viewer(VIEWER, viewer())
            .user(AUTHOR, found_user(AUTHOR))
            .user(SUSPENDED_AUTHOR, suspended_author())
            .confirmed_email(AUTHOR)
            .verified_phone(AUTHOR)
            .edge(Graph::Follows, FOLLOWER, VIEWER)
            .edge(Graph::Blocks, VIEWER, FOLLOWER)
            .edge(Graph::Blocks, AUTHOR, FOLLOWER)
            .country(VIEWER, "us")
            .second_degree_path(ROOT, VIEWER)
            .community_moderation(
                TWEET,
                CommunityModeration {
                    is_hidden: true,
                    is_author_removed: true,
                },
            )
            .community_moderation(
                MISSING,
                CommunityModeration {
                    is_hidden: false,
                    is_author_removed: true,
                },
            )
            .moderator_of(COMMUNITY)
            .removed_from(COMMUNITY)
            .article_media_entities(ARTICLE, true)
            .article_cover_media(ARTICLE, true)
            .lifecycle(ARTICLE, ArticleLifecycle::SoftDeleted)
            .trusted_friend(CIRCLE, VIEWER)
            .located_in(PLACE),
    )
}

fn defaults() -> InMemorySources {
    let user = || GizmoduckUserResult {
        user: Some(GizmoduckUser::default()),
        response_state: Some(UserResponseState::Found),
    };
    failing_keys(
        InMemorySources::default()
            .pure_core(TWEET, PureCoreData::default())
            .labels(TWEET, vf_pb::SafetyLabelMap::default())
            .viewer(VIEWER, GizmoduckUser::default())
            .user(AUTHOR, user())
            .user(SUSPENDED_AUTHOR, user())
            .country(VIEWER, ""),
    )
}

fn shown<V: fmt::Debug>(
    unanswered: &mut Vec<u64>,
    batch: &RawHydrationBatch<V>,
    keys: &[u64],
) -> String {
    let shown: Vec<String> = keys
        .iter()
        .map(|&key| {
            let answer = batch.hydrated(&key);
            if matches!(answer, None | Some(Hydrated::Unanswered)) {
                unanswered.push(key);
            }
            match answer {
                Some(Hydrated::Unanswered) => format!("{key}: {:?}", Some(Hydrated::<V>::Failed)),
                answer => format!("{key}: {answer:?}"),
            }
        })
        .collect();
    shown.join(", ")
}

fn input(fetcher: FetcherId, keys: &[u64]) -> Input<'_> {
    Input {
        keys,
        viewer_id: Some(VIEWER),
        fields: &[],
        queries: &[],
        community_posts: &[],
        fetcher,
    }
}

async fn fetched_by<F: SourceFetch>(
    sources: &(impl Fetches<F> + ?Sized),
    input: Input<'_>,
) -> F::Reply {
    sources.fetch(&input).await.unwrap()
}

async fn fetch_shown(
    sources: &dyn Sources,
    fetcher: FetcherId,
    unanswered: &mut Vec<u64>,
) -> String {
    match fetcher.source {
        Source::TesPureCoreData => {
            let keys = [TWEET, MISSING, FAILED, ABSENT];
            let cores = fetched_by::<source::TesPureCoreData>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &cores, &keys)
        }
        Source::TesTweetForVisibility => {
            let keys = [TWEET, MISSING, FAILED, ABSENT];
            let tweets =
                fetched_by::<source::TesTweetForVisibility>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &tweets, &keys)
        }
        Source::TesConversationControl => {
            let keys = [TWEET, MISSING, FAILED, ABSENT];
            let controls =
                fetched_by::<source::TesConversationControl>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &controls, &keys)
        }
        Source::TesQuotedTweet => {
            let keys = [TWEET, MISSING, FAILED, ABSENT];
            let quoted = fetched_by::<source::TesQuotedTweet>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &quoted, &keys)
        }
        Source::TesTweetState => {
            let keys = [TWEET, MISSING, FAILED, ABSENT];
            let states = fetched_by::<source::TesTweetState>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &states, &keys)
        }
        Source::SafetyLabels => {
            let keys = [TWEET, MISSING, FAILED];
            let labels = fetched_by::<source::SafetyLabels>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &labels, &keys)
        }
        Source::GizmoduckViewer => {
            let inputs = FIELDS
                .map(|fields| (VIEWER, fields))
                .into_iter()
                .chain([MISSING, FAILED].map(|viewer| (viewer, FIELDS[0])));
            let mut answers = vec![];
            for (viewer_id, fields) in inputs {
                let viewer_ids = [viewer_id];
                let call_input = Input {
                    viewer_id: Some(viewer_id),
                    fields,
                    ..input(fetcher, &viewer_ids)
                };
                let viewer = fetched_by::<source::GizmoduckViewer>(sources, call_input).await;
                let viewer = viewer.map(|viewer| {
                    let DecodedViewer {
                        profile,
                        has_age_verified_18_label,
                        state,
                    } = viewer;
                    (profile, has_age_verified_18_label, state)
                });
                answers.push(shown(unanswered, &viewer, &viewer_ids));
            }
            answers.join(", ")
        }
        Source::GizmoduckAuthor | Source::GizmoduckActor => {
            let users: [&[u64]; 2] = [&[AUTHOR, MISSING, FAILED, ABSENT], &[SUSPENDED_AUTHOR]];
            let mut answers = vec![];
            for (users, fields) in users.into_iter().zip(FIELDS) {
                let call_input = || Input {
                    fields,
                    ..input(fetcher, users)
                };
                answers.push(if fetcher.source == Source::GizmoduckActor {
                    let actors = fetched_by::<source::GizmoduckActor>(sources, call_input()).await;
                    shown(unanswered, &actors, users)
                } else {
                    let authors =
                        fetched_by::<source::GizmoduckAuthor>(sources, call_input()).await;
                    shown(unanswered, &authors, users)
                });
            }
            answers.join(", ")
        }
        Source::ActorEmails => {
            let keys = [AUTHOR, MISSING];
            let held = fetched_by::<source::ActorEmails>(sources, input(fetcher, &keys)).await;
            let failed =
                fetched_by::<source::ActorEmails>(sources, input(fetcher, &[FAILED])).await;
            format!(
                "{}, {}",
                shown(unanswered, &held, &keys),
                shown(unanswered, &failed, &[FAILED])
            )
        }
        Source::ActorPhones => {
            let keys = [AUTHOR, MISSING];
            let held = fetched_by::<source::ActorPhones>(sources, input(fetcher, &keys)).await;
            let failed =
                fetched_by::<source::ActorPhones>(sources, input(fetcher, &[FAILED])).await;
            format!(
                "{}, {}",
                shown(unanswered, &held, &keys),
                shown(unanswered, &failed, &[FAILED])
            )
        }
        Source::Flock => {
            let viewer = |graph, direction, destination| EdgeQuery {
                graph,
                direction,
                from: EdgeFrom::Viewer,
                destination_ids: vec![destination],
            };
            let sourced = |graph, destination| EdgeQuery {
                from: EdgeFrom::User(AUTHOR),
                ..EdgeQuery::forward(graph, vec![destination])
            };
            let mut held = vec![
                sourced(Graph::Blocks, FOLLOWER),
                sourced(Graph::Follows, FOLLOWER),
                sourced(Graph::Mutes, FOLLOWER),
            ];
            let mut failed = vec![sourced(Graph::Blocks, FAILED)];
            let viewer_id = match fetcher.owner {
                Owner::Request => {
                    held.extend(
                        EDGES.map(|(graph, direction, _)| viewer(graph, direction, FOLLOWER)),
                    );
                    held.push(viewer(Graph::Mutes, EdgeDirection::Forward, FOLLOWER));
                    failed.push(viewer(Graph::Follows, EdgeDirection::Forward, FAILED));
                    Some(VIEWER)
                }
                Owner::Related(_) => None,
            };
            let mut answers = vec![];
            for queries in [held, failed] {
                let select = Input {
                    viewer_id,
                    queries: &queries,
                    ..input(fetcher, &[])
                };
                let batches = fetched_by::<source::Flock>(sources, select).await;
                for (query, batch) in queries.iter().zip(&batches) {
                    answers.push(shown(unanswered, batch, &query.destination_ids));
                }
            }
            answers.join(", ")
        }
        Source::ViewerCountry => {
            let mut answers = vec![];
            for viewer in [VIEWER, MISSING, FAILED] {
                let viewers = [viewer];
                let call_input = Input {
                    viewer_id: Some(viewer),
                    ..input(fetcher, &viewers)
                };
                let country = fetched_by::<source::ViewerCountry>(sources, call_input).await;
                answers.push(shown(unanswered, &country, &viewers));
            }
            answers.join(", ")
        }
        Source::Wingman => {
            let keys = [ROOT, MISSING, FAILED, ABSENT];
            let paths = fetched_by::<source::Wingman>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &paths, &keys)
        }
        Source::CommunityModeration => {
            let posts = [TWEET, MISSING, FAILED].map(|tweet_id| CommunityPost { tweet_id, ..POST });
            let call_input = Input {
                community_posts: &posts,
                ..input(fetcher, &[])
            };
            let batch = fetched_by::<source::CommunityModeration>(sources, call_input).await;
            shown(unanswered, &batch, &posts.map(|post| post.tweet_id))
        }
        Source::CommunityModerator => {
            let keys = [COMMUNITY, MISSING, FAILED];
            let moderators =
                fetched_by::<source::CommunityModerator>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &moderators, &keys)
        }
        Source::CommunityViewerRemoved => {
            let keys = [COMMUNITY, MISSING, FAILED];
            let removals =
                fetched_by::<source::CommunityViewerRemoved>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &removals, &keys)
        }
        Source::ArticleMediaEntities => {
            let keys = [ARTICLE, MISSING, FAILED];
            let rows =
                fetched_by::<source::ArticleMediaEntities>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &rows, &keys)
        }
        Source::ArticleCoverMedia => {
            let keys = [ARTICLE, MISSING, FAILED];
            let rows =
                fetched_by::<source::ArticleCoverMedia>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &rows, &keys)
        }
        Source::ArticleLifecycle => {
            let keys = [ARTICLE, MISSING, FAILED];
            let lifecycles =
                fetched_by::<source::ArticleLifecycle>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &lifecycles, &keys)
        }
        Source::TrustedFriends => {
            let keys = [CIRCLE, MISSING, FAILED];
            let memberships =
                fetched_by::<source::TrustedFriends>(sources, input(fetcher, &keys)).await;
            shown(unanswered, &memberships, &keys)
        }
        Source::UserLocation => {
            let keys = [PLACE, MISSING];
            let located = fetched_by::<source::UserLocation>(sources, input(fetcher, &keys)).await;
            let failed = Input {
                viewer_id: Some(FAILED),
                ..input(fetcher, &[FAILED])
            };
            let failed = fetched_by::<source::UserLocation>(sources, failed).await;
            format!(
                "{}, {}",
                shown(unanswered, &located, &keys),
                shown(unanswered, &failed, &[FAILED])
            )
        }
    }
}

#[tokio::test]
async fn prod_sources_answer_what_their_backends_hold() {
    let (prod, held, defaults) = (prod_sources(), held(), defaults());
    let mut wrong = vec![];
    for fetcher in FetcherId::all() {
        let mut unanswered = vec![];
        let answer = fetch_shown(&prod, fetcher, &mut unanswered).await;
        let expected = fetch_shown(&held, fetcher, &mut vec![]).await;
        if answer != expected || answer == fetch_shown(&defaults, fetcher, &mut vec![]).await {
            wrong.push(format!("{fetcher:?}\n  prod {answer}\n  held {expected}"));
        }
        unanswered.retain(|key| ![FAILED, ABSENT].contains(key));
        if !unanswered.is_empty() {
            wrong.push(format!("{fetcher:?} answered {unanswered:?} unanswered"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
