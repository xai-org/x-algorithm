use crate::candidate_pipeline::following_candidate_pipeline::FollowingCandidatePipeline;
use crate::candidate_pipeline::for_you_candidate_pipeline::ForYouCandidatePipeline;
use crate::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
use crate::candidate_pipeline::phoenix_scores_pipeline::PhoenixScoresPipeline;
use crate::candidate_pipeline::ranked_following_candidate_pipeline::RankedFollowingCandidatePipeline;
use crate::clients::gizmoduck_client::{
    GizmoduckClient, MockGizmoduckClient, ProdGizmoduckClient, ViewerData,
};
use crate::clients::resurrection_date_client::{
    MockResurrectionDateClient, ProdResurrectionDateClient, ResurrectionDateClient,
};
use crate::following_feed_server::FollowingFeedServer;
use crate::for_you_server::ForYouFeedServer;
use crate::models::candidate::PostCandidate;
use crate::models::fs_recipient::FsRecipientInputs;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params;
use crate::phoenix_scores_server::{build_query_builder_input, PhoenixScoresServer};
use crate::ranked_following_feed_server::RankedFollowingFeedServer;
use crate::scored_posts_server::{build_debug_json, ScoredPostsServer};
use crate::util::strato_context;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::codec::CompressionEncoding;
use tonic::{Request, Response, Status};
use tracing::{info_span, Instrument};
use xai_candidate_pipeline::candidate_pipeline::PipelineResult;
use xai_candidate_pipeline::component_library::clients::{JetfuelClient, ProdJetfuelClient};
use xai_candidate_pipeline::component_library::utils::{
    generate_request_id, non_zero, resolve_request_id,
};
use xai_decider::{Decider, DeciderStore};
use xai_feature_switches::FeatureSwitches;
use xai_home_mixer_proto as pb;
use xai_home_mixer_proto::{
    DebugPhoenixScoresResponse, DebugScoredPostsResponse, FollowingFeedResponse,
    FollowingFeedUrtResponse, ForYouFeedResponse, ForYouFeedUrtResponse, PhoenixScoresResponse,
    RankedFollowingFeedResponse, RankedFollowingFeedUrtResponse, ScoredPost, ScoredPostsResponse,
};
use xai_pipeline_tracing::{extract_b3_info, B3RequestInfo};
use xai_recsys_proto::{network_type_string_to_enum, timezone_string_to_enum};
use xai_urt_thrift::cursor_utils;
use xai_urt_thrift::operation::CursorType;
use xai_visibility_filtering::tweet_safety_label::{
    ProdTweetSafetyLabelClient, TweetSafetyLabelClient,
};
use xai_visibility_filtering::vf_client::{VfClient, XaiVfClient};
use xai_x_rpc::wily_lookup_service::ShardCoordinate;

const VIEWER_ROLES_TIMEOUT_MS: u64 = 200;

pub struct RequestContext {
    pub b3_info: B3RequestInfo,
    pub query: ScoredPostsQuery,
    pub root_span: tracing::Span,
}

pub(crate) struct PipelineOutput {
    pub scored_posts: Vec<ScoredPost>,
    pub video_carousel: Option<pb::VideoCarouselModule>,
    pub pipeline_result: PipelineResult<ScoredPostsQuery, PostCandidate>,
    pub pipeline_trace: Option<pb::PipelineTrace>,
}

pub struct HomeMixerConfig {
    pub shard_coordinate: Option<ShardCoordinate>,
    pub phoenix_xds: crate::candidate_pipeline::PhoenixXdsConfig,
    pub vm_ranker_xds: crate::candidate_pipeline::VmRankerXdsConfig,
}

#[derive(Clone)]
pub struct QueryBuilder {
    feature_switches: Arc<FeatureSwitches>,
    decider: Decider,
    datacenter: String,
    gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
    resurrection_date_client: Arc<dyn ResurrectionDateClient>,
}

impl QueryBuilder {
    pub async fn build(
        &self,
        mut b3_info: B3RequestInfo,
        proto_query: pb::ScoredPostsQuery,
        fs_overrides: std::collections::HashMap<String, String>,
        span_name: &'static str,
        request_type: RequestType,
    ) -> Result<RequestContext, Status> {
        if proto_query.viewer_id == 0 {
            return Err(Status::invalid_argument("viewer_id must be specified"));
        }
        if params::TRACE_USER_IDS.contains(&proto_query.viewer_id) {
            b3_info.force_sample();
        }

        let (viewer_data, resurrection_time_ms) = tokio::join!(
            self.fetch_viewer_data(proto_query.viewer_id),
            self.fetch_resurrection_time(proto_query.viewer_id),
        );

        let in_network_only =
            proto_query.in_network_only || viewer_data.allow_for_you_recommendations == Some(false);

        let fs_recipient_inputs = FsRecipientInputs {
            user_id: proto_query.viewer_id,
            country_code: proto_query.country_code.clone(),
            language_code: proto_query.language_code.clone(),
            client_app_id: proto_query.client_app_id as i64,
            client_version: proto_query
                .device_status
                .as_ref()
                .map(|d| d.client_version.clone()),
            user_roles: viewer_data.roles.clone(),
            datacenter: self.datacenter.clone(),
            has_phone_number: viewer_data.has_phone_number,
            resurrection_time_ms,
            product: request_type.to_string(),
            now_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
            fs_overrides,
        };
        let params = self.evaluate_feature_switches(&fs_recipient_inputs);

        let push_to_home_post_id = non_zero(proto_query.push_to_home_post_id);
        let device_status = proto_query.device_status.unwrap_or_default();
        let prediction_id = generate_request_id();
        let request_id = resolve_request_id(proto_query.request_id);
        let is_shadow_traffic = crate::util::shadow::is_shadow_traffic(&params, prediction_id);
        let mut query = ScoredPostsQuery::new(
            proto_query.viewer_id,
            proto_query.client_app_id,
            proto_query.country_code,
            proto_query.language_code,
            proto_query.seen_ids,
            proto_query.served_ids,
            in_network_only,
            proto_query.is_bottom_request,
            params,
            self.decider.with_recipient(proto_query.viewer_id),
            viewer_data.roles,
            viewer_data.muted_keywords,
            viewer_data.follower_count,
            proto_query.topic_ids,
            proto_query.excluded_topic_ids,
            proto_query.exclude_videos,
            request_id,
            prediction_id,
            device_status.ip_address,
            device_status.user_agent,
            timezone_string_to_enum(device_status.time_zone.as_ref()),
            network_type_string_to_enum(device_status.device_network_type.as_ref()),
            device_status.client_version,
            device_status.device_id,
            device_status.mobile_device_id,
            device_status.mobile_device_ad_id,
            viewer_data.subscription_level,
            is_shadow_traffic,
            proto_query.is_preview,
            viewer_data.age_in_years,
            push_to_home_post_id,
        );
        query.request_type = request_type;
        query.resurrection_time_ms = resurrection_time_ms;
        query.fs_recipient_inputs = Some(fs_recipient_inputs);
        query.is_under_the_hood_request =
            proto_query.is_under_the_hood_request && query.params.get(params::EnableUnderTheHood);

        query.dsp_client_context = proto_query.dsp_client_context;

        let root_span = b3_info.root_span(info_span!(
            "request",
            endpoint = span_name,
            trace = %b3_info.trace_id_str,
            user = %query.user_id,
            b3 = %b3_info.b3_sampled,
        ));

        Ok(RequestContext {
            b3_info,
            query,
            root_span,
        })
    }

    fn evaluate_feature_switches(
        &self,
        inputs: &FsRecipientInputs,
    ) -> xai_feature_switches::Params {
        let mut results = self.feature_switches.match_recipient(&inputs.recipient());

        if !inputs.fs_overrides.is_empty() {
            for (key, value) in &inputs.fs_overrides {
                results.override_fs(key.clone(), value);
            }
            tracing::info!(
                "Applied {} FS overrides: {:?}",
                inputs.fs_overrides.len(),
                inputs.fs_overrides.keys().collect::<Vec<_>>()
            );
        }

        results.into()
    }

    async fn fetch_viewer_data(&self, viewer_id: u64) -> ViewerData {
        match tokio::time::timeout(
            std::time::Duration::from_millis(VIEWER_ROLES_TIMEOUT_MS),
            self.gizmoduck_client.get_viewer_data(viewer_id),
        )
        .await
        {
            Ok(Ok(data)) => data,
            Ok(Err(_)) | Err(_) => ViewerData::default(),
        }
    }

    async fn fetch_resurrection_time(&self, user_id: u64) -> Option<i64> {
        match tokio::time::timeout(
            std::time::Duration::from_millis(VIEWER_ROLES_TIMEOUT_MS),
            self.resurrection_date_client.fetch(user_id),
        )
        .await
        {
            Ok(Ok(ts)) => ts,
            Ok(Err(e)) => {
                tracing::warn!(user_id, error = %e, "failed to fetch resurrection date");
                None
            }
            Err(_) => {
                tracing::warn!(user_id, "resurrection date fetch timed out");
                None
            }
        }
    }

    pub fn mock() -> Self {
        Self {
            feature_switches: Arc::new(FeatureSwitches::new(vec![]).unwrap()),
            decider: Decider::new(DeciderStore::new(HashMap::new())),
            datacenter: "mock".to_string(),
            gizmoduck_client: Arc::new(MockGizmoduckClient::default()),
            resurrection_date_client: Arc::new(MockResurrectionDateClient),
        }
    }
}

pub struct HomeMixerServer {
    scored_posts: Arc<ScoredPostsServer>,
    for_you: Arc<ForYouFeedServer>,
    ranked_following: Arc<RankedFollowingFeedServer>,
    following: Arc<FollowingFeedServer>,
    phoenix_scores: Arc<PhoenixScoresServer>,
}

#[tonic::async_trait]
impl pb::scored_posts_service_server::ScoredPostsService for ScoredPostsServer {
    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_scored_posts(
        &self,
        request: Request<pb::ScoredPostsQuery>,
    ) -> Result<Response<ScoredPostsResponse>, Status> {
        let b3_info = extract_b3_info(request.metadata());
        let ctx = self
            .query_builder
            .build(
                b3_info,
                request.into_inner(),
                Default::default(),
                "scored_posts",
                RequestType::ScoredPosts,
            )
            .await?;
        let RequestContext {
            b3_info,
            query,
            root_span,
        } = ctx;
        let output = self.run_pipeline(query).instrument(root_span).await?;

        let mut response = Response::new(ScoredPostsResponse {
            scored_posts: output.scored_posts,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_debug_scored_posts(
        &self,
        request: Request<pb::DebugScoredPostsQuery>,
    ) -> Result<Response<DebugScoredPostsResponse>, Status> {
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let proto_query = debug_query.query.unwrap_or_default();

        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                fs_overrides,
                "debug_scored_posts",
                RequestType::ScoredPosts,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;
        query.return_backbone_scores = true;
        query.is_debug_request = true;
        let output = self.run_pipeline(query).instrument(root_span).await?;

        let debug_json = build_debug_json(&output.pipeline_result);

        let mut response = Response::new(DebugScoredPostsResponse {
            scored_posts: output.scored_posts,
            debug_json,
            pipeline_trace: output.pipeline_trace,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl pb::for_you_feed_service_server::ForYouFeedService for ForYouFeedServer {
    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_for_you_feed(
        &self,
        request: Request<pb::ForYouFeedQuery>,
    ) -> Result<Response<ForYouFeedResponse>, Status> {
        let b3_info = extract_b3_info(request.metadata());
        let feed_query = request.into_inner();
        let proto_query = feed_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                Default::default(),
                "for_you_feed",
                RequestType::ForYou,
            )
            .await?;
        let RequestContext {
            b3_info,
            query,
            root_span,
        } = ctx;
        let output = self.get_for_you_feed(query).instrument(root_span).await?;

        let mut response = Response::new(ForYouFeedResponse {
            items: output.items,
            pipeline_trace: output.pipeline_trace,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_for_you_feed_urt(
        &self,
        request: Request<pb::ForYouFeedQuery>,
    ) -> Result<Response<ForYouFeedUrtResponse>, Status> {
        let strato_ctx = strato_context::parse(request.metadata()).unwrap_or_default();
        let b3_info = extract_b3_info(request.metadata());
        let feed_query = request.into_inner();
        let proto_query = feed_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let cursor_str = proto_query.cursor.clone();
        let request_context = proto_query.request_context.clone();
        let is_polling = strato_ctx.is_polling || proto_query.is_polling;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                Default::default(),
                "for_you_feed_urt",
                RequestType::ForYou,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;

        query.request_context = request_context;
        query.is_polling = is_polling;
        query.mobile_device_id = strato_ctx.mobile_device_id;
        query.mobile_device_ad_id = strato_ctx.ad_id;
        if !cursor_str.is_empty() {
            match cursor_utils::decode_ordered_cursor(&cursor_str) {
                Ok(Some(c)) => {
                    query.is_bottom_request = c.cursor_type == Some(CursorType::BOTTOM);
                    query.is_top_request = c.cursor_type == Some(CursorType::TOP);
                    query.cursor = Some(c);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(cursor_str, error = %e, "failed to decode URT cursor, ignoring");
                }
            }
        }
        let urt = self
            .get_for_you_feed_urt(query)
            .instrument(root_span)
            .await?;

        let mut response = Response::new(ForYouFeedUrtResponse { urt: urt.into() });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_debug_for_you_feed(
        &self,
        request: Request<pb::DebugForYouFeedQuery>,
    ) -> Result<Response<ForYouFeedResponse>, Status> {
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let proto_query = debug_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                fs_overrides,
                "debug_for_you_feed",
                RequestType::ForYou,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;
        query.return_backbone_scores = true;
        let output = self.get_for_you_feed(query).instrument(root_span).await?;

        let mut response = Response::new(ForYouFeedResponse {
            items: output.items,
            pipeline_trace: output.pipeline_trace,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_debug_for_you_feed_urt(
        &self,
        request: Request<pb::DebugForYouFeedQuery>,
    ) -> Result<Response<ForYouFeedUrtResponse>, Status> {
        let strato_ctx = strato_context::parse(request.metadata()).unwrap_or_default();
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let proto_query = debug_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let cursor_str = proto_query.cursor.clone();
        let request_context = proto_query.request_context.clone();
        let is_polling = strato_ctx.is_polling || proto_query.is_polling;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                fs_overrides,
                "debug_for_you_feed_urt",
                RequestType::ForYou,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;

        query.return_backbone_scores = true;
        query.request_context = request_context;
        query.is_polling = is_polling;
        query.mobile_device_id = strato_ctx.mobile_device_id;
        query.mobile_device_ad_id = strato_ctx.ad_id;
        if !cursor_str.is_empty() {
            match cursor_utils::decode_ordered_cursor(&cursor_str) {
                Ok(Some(c)) => {
                    query.is_bottom_request = c.cursor_type == Some(CursorType::BOTTOM);
                    query.is_top_request = c.cursor_type == Some(CursorType::TOP);
                    query.cursor = Some(c);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(cursor_str, error = %e, "failed to decode URT cursor, ignoring");
                }
            }
        }
        let urt = self
            .get_for_you_feed_urt(query)
            .instrument(root_span)
            .await?;

        let mut response = Response::new(ForYouFeedUrtResponse { urt: urt.into() });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl pb::ranked_following_feed_service_server::RankedFollowingFeedService
    for RankedFollowingFeedServer
{
    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_ranked_following_feed_urt(
        &self,
        request: Request<pb::RankedFollowingFeedQuery>,
    ) -> Result<Response<RankedFollowingFeedUrtResponse>, Status> {
        let strato_ctx = strato_context::parse(request.metadata()).unwrap_or_default();
        let b3_info = extract_b3_info(request.metadata());
        let feed_query = request.into_inner();
        let proto_query = feed_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let cursor_str = proto_query.cursor.clone();
        let request_context = proto_query.request_context.clone();
        let is_polling = strato_ctx.is_polling || proto_query.is_polling;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                Default::default(),
                "ranked_following_feed_urt",
                RequestType::RankedFollowing,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;

        query.in_network_only = true;
        query.request_context = request_context;
        query.is_polling = is_polling;
        query.mobile_device_id = strato_ctx.mobile_device_id;
        query.mobile_device_ad_id = strato_ctx.ad_id;
        if !cursor_str.is_empty() {
            match cursor_utils::decode_ordered_cursor(&cursor_str) {
                Ok(Some(c)) => {
                    query.is_bottom_request = c.cursor_type == Some(CursorType::BOTTOM);
                    query.is_top_request = c.cursor_type == Some(CursorType::TOP);
                    query.cursor = Some(c);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(cursor_str, error = %e, "failed to decode URT cursor, ignoring");
                }
            }
        }
        let urt = self
            .get_ranked_following_feed_urt(query)
            .instrument(root_span)
            .await?;

        let mut response = Response::new(RankedFollowingFeedUrtResponse { urt: urt.into() });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_debug_ranked_following_feed(
        &self,
        request: Request<pb::DebugRankedFollowingFeedQuery>,
    ) -> Result<Response<RankedFollowingFeedResponse>, Status> {
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let proto_query = debug_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                fs_overrides,
                "debug_ranked_following_feed",
                RequestType::RankedFollowing,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;
        query.return_backbone_scores = true;
        query.in_network_only = true;
        let output = self
            .get_ranked_following_feed(query)
            .instrument(root_span)
            .await?;

        let mut response = Response::new(RankedFollowingFeedResponse {
            items: output.items,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl pb::following_feed_service_server::FollowingFeedService for FollowingFeedServer {
    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_following_feed_urt(
        &self,
        request: Request<pb::FollowingFeedQuery>,
    ) -> Result<Response<FollowingFeedUrtResponse>, Status> {
        let strato_ctx = strato_context::parse(request.metadata()).unwrap_or_default();
        let b3_info = extract_b3_info(request.metadata());
        let feed_query = request.into_inner();
        let proto_query = feed_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let cursor_str = proto_query.cursor.clone();
        let request_context = proto_query.request_context.clone();
        let is_polling = strato_ctx.is_polling || proto_query.is_polling;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                Default::default(),
                "following_feed_urt",
                RequestType::Following,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;

        query.in_network_only = true;
        query.request_context = request_context;
        query.is_polling = is_polling;
        query.mobile_device_id = strato_ctx.mobile_device_id;
        query.mobile_device_ad_id = strato_ctx.ad_id;
        if !cursor_str.is_empty() {
            match cursor_utils::decode_ordered_cursor(&cursor_str) {
                Ok(Some(c)) => {
                    query.is_bottom_request = c.cursor_type == Some(CursorType::BOTTOM);
                    query.is_top_request = c.cursor_type == Some(CursorType::TOP);
                    query.cursor = Some(c);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(cursor_str, error = %e, "failed to decode URT cursor, ignoring");
                }
            }
        }
        let urt = self
            .get_following_feed_urt(query)
            .instrument(root_span)
            .await?;

        let mut response = Response::new(FollowingFeedUrtResponse { urt: urt.into() });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_debug_following_feed(
        &self,
        request: Request<pb::DebugFollowingFeedQuery>,
    ) -> Result<Response<FollowingFeedResponse>, Status> {
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let proto_query = debug_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                fs_overrides,
                "debug_following_feed",
                RequestType::Following,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;
        query.return_backbone_scores = true;
        query.in_network_only = true;
        let output = self.get_following_feed(query).instrument(root_span).await?;

        let mut response = Response::new(FollowingFeedResponse {
            items: output.items,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl pb::phoenix_scores_service_server::PhoenixScoresService for PhoenixScoresServer {
    #[xai_stats_macro::receive_stats(latency=Bucket500To1000)]
    async fn get_phoenix_scores(
        &self,
        request: Request<pb::PhoenixScoresQuery>,
    ) -> Result<Response<PhoenixScoresResponse>, Status> {
        let b3_info = extract_b3_info(request.metadata());
        let phoenix_scores_query = request.into_inner();
        let candidate_tweet_ids = phoenix_scores_query.candidate_tweet_ids.clone();
        let proto_query = build_query_builder_input(phoenix_scores_query);
        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                Default::default(),
                "phoenix_scores",
                RequestType::PhoenixScores,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;
        query.seed_candidate_post_ids = candidate_tweet_ids;

        let output = self.run_pipeline(query).instrument(root_span).await?;

        let mut response = Response::new(PhoenixScoresResponse {
            scores: output.scores,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To1000)]
    async fn get_debug_phoenix_scores(
        &self,
        request: Request<pb::DebugPhoenixScoresQuery>,
    ) -> Result<Response<DebugPhoenixScoresResponse>, Status> {
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let phoenix_scores_query = debug_query.query.unwrap_or_default();
        let candidate_tweet_ids = phoenix_scores_query.candidate_tweet_ids.clone();
        let proto_query = build_query_builder_input(phoenix_scores_query);

        let ctx = self
            .query_builder
            .build(
                b3_info,
                proto_query,
                fs_overrides,
                "debug_phoenix_scores",
                RequestType::PhoenixScores,
            )
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;
        query.seed_candidate_post_ids = candidate_tweet_ids;

        let output = self.run_pipeline(query).instrument(root_span).await?;

        let debug_json = build_debug_json(&output.pipeline_result);

        let mut response = Response::new(DebugPhoenixScoresResponse {
            scores: output.scores,
            debug_json,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl xai_x_service_builder::XService for HomeMixerServer {
    type Config = HomeMixerConfig;

    async fn build(ctx: xai_x_service_builder::ServiceContext<HomeMixerConfig>) -> Self {
        let xai_x_service_builder::ServiceContext {
            feature_switches,
            decider,
            datacenter,
            config,
        } = ctx;

        let (xai_vf_client, vf_safety_labels_client) = tokio::join!(
            async {
                Arc::new(
                    XaiVfClient::connect(&datacenter)
                        .await
                        .expect("Failed to create XAI VF client"),
                ) as Arc<dyn VfClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdTweetSafetyLabelClient::new(&datacenter)
                        .await
                        .expect("Failed to create VF SafetyLabels client")
                        .with_timeout_ms(500)
                        .with_max_batch_size(100),
                ) as Arc<dyn TweetSafetyLabelClient>
            },
        );

        let (
            gizmoduck_client,
            resurrection_date_client,
            jetfuel_client,
            phoenix_candidate_pipeline,
            phoenix_scores_pipeline,
        ) = tokio::join!(
            async {
                Arc::new(
                    ProdGizmoduckClient::new(
                        config.shard_coordinate,
                        &datacenter,
                        Some("home-mixer.prod".to_string()),
                    )
                    .await
                    .expect("Failed to create Gizmoduck client"),
                ) as Arc<dyn GizmoduckClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdResurrectionDateClient::new(&datacenter)
                        .await
                        .expect("Failed to create ResurrectionDate client"),
                ) as Arc<dyn ResurrectionDateClient>
            },
            async {
                Arc::new(
                    ProdJetfuelClient::new(&datacenter)
                        .await
                        .expect("Failed to create Jetfuel client"),
                ) as Arc<dyn JetfuelClient>
            },
            async {
                Arc::new(
                    PhoenixCandidatePipeline::prod(
                        config.shard_coordinate,
                        &datacenter,
                        feature_switches.clone(),
                        &config.phoenix_xds,
                        &config.vm_ranker_xds,
                        Arc::clone(&xai_vf_client),
                        Arc::clone(&vf_safety_labels_client),
                    )
                    .await,
                )
            },
            async {
                Arc::new(
                    PhoenixScoresPipeline::prod(
                        config.shard_coordinate,
                        &datacenter,
                        &config.phoenix_xds,
                    )
                    .await,
                )
            },
        );

        let query_builder = QueryBuilder {
            feature_switches,
            decider,
            datacenter: datacenter.clone(),
            gizmoduck_client,
            resurrection_date_client,
        };

        let scored_posts = Arc::new(ScoredPostsServer::new(
            query_builder.clone(),
            phoenix_candidate_pipeline,
        ));

        let (for_you_pipeline, ranked_following_pipeline, following_pipeline) = tokio::join!(
            ForYouCandidatePipeline::new(
                Arc::clone(&scored_posts),
                Arc::clone(&jetfuel_client),
                config.shard_coordinate,
                &datacenter,
            ),
            RankedFollowingCandidatePipeline::new(Arc::clone(&scored_posts), &datacenter),
            FollowingCandidatePipeline::new(&datacenter, xai_vf_client, vf_safety_labels_client),
        );

        let for_you = Arc::new(ForYouFeedServer::new(
            query_builder.clone(),
            for_you_pipeline,
            jetfuel_client,
        ));

        let ranked_following = Arc::new(RankedFollowingFeedServer::new(
            query_builder.clone(),
            ranked_following_pipeline,
        ));

        let following = Arc::new(FollowingFeedServer::new(
            query_builder.clone(),
            following_pipeline,
        ));

        let phoenix_scores = Arc::new(PhoenixScoresServer::new(
            phoenix_scores_pipeline,
            query_builder,
        ));

        HomeMixerServer {
            scored_posts,
            for_you,
            ranked_following,
            following,
            phoenix_scores,
        }
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        routes.add_service(
            pb::scored_posts_service_server::ScoredPostsServiceServer::from_arc(Arc::clone(
                &self.scored_posts,
            ))
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
        routes.add_service(
            pb::for_you_feed_service_server::ForYouFeedServiceServer::from_arc(Arc::clone(
                &self.for_you,
            ))
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
        routes.add_service(
            pb::ranked_following_feed_service_server::RankedFollowingFeedServiceServer::from_arc(
                Arc::clone(&self.ranked_following),
            )
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
        routes.add_service(
            pb::following_feed_service_server::FollowingFeedServiceServer::from_arc(Arc::clone(
                &self.following,
            ))
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
        routes.add_service(
            pb::phoenix_scores_service_server::PhoenixScoresServiceServer::from_arc(Arc::clone(
                &self.phoenix_scores,
            ))
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
    }
}
