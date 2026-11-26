use crate::discovery::{
    build_vf_channel, VfChannel, VfChannelError, VfChannelParams, VfDiscovery, XdsTarget,
};
use crate::evaluated::EvaluationResult;
use crate::metrics::{ClientRpc, RequestMetrics};
use crate::models::{Action, DropReason, FilteredReason, SafetyResult};
use crate::tweet_safety_label::{proto_to_safety_label_map, SafetyLabelFailure};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TOutputProtocol, TStructIdentifier, TType,
};
use tonic::async_trait;
use tonic::codec::CompressionEncoding;
use tonic::transport::Channel;
use xai_safety_label_store::types::SafetyLabelMap;
use xai_strato::{
    decode, encode, MValCodec, StratoGrpc, StratoGrpcConfig, StratoResult, StratoValue,
};
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering_proto as vf_pb;
use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;

pub const FILTER_TWEETS_LATENCY_VM_METRIC: &str = "vf_client_filter_tweets_latency_ms_vm";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Hash, Eq, strum::FromRepr)]
#[serde(rename_all = "camelCase")]
pub enum SafetyLevel {
    #[default]
    FilterNone = 0,
    SearchTop = 6,
    SearchLatest = 7,
    TimelineHome = 8,
    TimelineHomeRecommendations = 61,
    SearchPhoto = 93,
    SearchVideo = 94,
    SearchTopQig = 165,
    SearchTopSafeSearchEnabled = 227,
    ExploreNsfwRecommendations = 236,
    ImmersiveExpandedRecommendations = 247,
}

impl SafetyLevel {
    pub fn from_tag(tag: i32) -> Option<Self> {
        usize::try_from(tag).ok().and_then(Self::from_repr)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct VisibilityFilteringLookupContext {
    pub safety_level: SafetyLevel,
    pub for_user_id: u64,
}

impl MValCodec for VisibilityFilteringLookupContext {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(_proto: &mut dyn TInputProtocol) -> Self {
        panic!("LookupContext should not be decoded from Thrift")
    }

    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        let struct_id = TStructIdentifier::new("VisibilityFilteringLookupContext");
        proto.write_struct_begin(&struct_id).unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("safety_level", TType::I32, 1))
            .unwrap();
        self.safety_level.to_thrift(proto);
        proto.write_field_end().unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("for_user_id", TType::I64, 2))
            .unwrap();
        proto.write_i64(self.for_user_id as i64).unwrap();
        proto.write_field_end().unwrap();
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }
}

impl MValCodec for SafetyLevel {
    fn thrift_type() -> TType {
        TType::I32
    }

    fn from_thrift(_proto: &mut dyn TInputProtocol) -> Self {
        panic!("QueryFields should not be decoded from Thrift")
    }

    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        proto.write_i32(self.clone() as i32).unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct TweetVisibility {
    pub action: Action,
    pub reason: Option<FilteredReason>,
    pub safety_labels: Result<SafetyLabelMap, SafetyLabelFailure>,
}

impl TweetVisibility {
    pub fn to_visibility_reason(&self) -> Option<FilteredReason> {
        match (&self.reason, &self.action) {
            (Some(reason @ FilteredReason::SafetyResult(_)), _) => Some(reason.clone()),
            (_, Action::Allow) => None,
            (_, Action::Interstitial) => Some(FilteredReason::SafetyResult(SafetyResult {
                reason: None,
                action: Action::Interstitial,
            })),
            (reason, _) => Some(reason.clone().unwrap_or(FilteredReason::UnspecifiedReason)),
        }
    }
}

#[async_trait]
pub trait VfClient {
    async fn get_result(
        &self,
        post_ids: Vec<u64>,
        safety_level: SafetyLevel,
        for_user_id: u64,
        context: Option<TwitterContextViewer>,
    ) -> HashMap<u64, Result<TweetVisibility>>;
}

pub struct StratoVfClient {
    pub grpc_client: Arc<StratoGrpc>,
}

impl StratoVfClient {
    pub async fn new(
        ca_cert_path: String,
        client_cert_path: String,
        client_key_path: String,
        client_id: String,
        datacenter: String,
    ) -> Result<Self, Box<dyn Error>> {
        let service_url = format!("", datacenter);
        let strato_grpc_config = StratoGrpcConfig {
            ca_cert_path,
            client_cert_path,
            client_key_path,
            aperture_size: Some(12),
            connect_timeout_ms: 400,
            request_timeout_ms: 990,
            client_id: Some(client_id),
            service_url,
            zone: datacenter,
            ..Default::default()
        };
        let grpc_client: StratoGrpc = StratoGrpc::new(strato_grpc_config).await?;
        Ok(Self {
            grpc_client: Arc::new(grpc_client),
        })
    }
}

fn action_from_strato_reason(reason: &Option<FilteredReason>) -> Action {
    match reason {
        None => Action::Allow,
        Some(FilteredReason::SafetyResult(safety_result)) => match safety_result.action {
            Action::NotEvaluated => Action::Allow,
            _ => safety_result.action.clone(),
        },
        Some(_) => Action::Drop(DropReason {}),
    }
}

#[async_trait]
impl VfClient for StratoVfClient {
    async fn get_result(
        &self,
        tweet_ids: Vec<u64>,
        safety_level: SafetyLevel,
        for_user_id: u64,
        context: Option<TwitterContextViewer>,
    ) -> HashMap<u64, Result<TweetVisibility>> {
        let client = &self.grpc_client;
        let view = VisibilityFilteringLookupContext {
            safety_level,
            for_user_id,
        };
        let calls: Vec<_> = tweet_ids
            .iter()
            .map(|tweet_id| {
                let args = vec![encode(&(*tweet_id, view.clone()))];
                (
                    "visibility/service/homeMixerFilteredReason.Tweet".to_string(),
                    "fetch".to_string(),
                    args,
                )
            })
            .collect::<Vec<(String, String, Vec<Vec<u8>>)>>();
        let result_batch = client.batch_call(calls, context.as_ref()).await;
        tweet_ids
            .iter()
            .zip(result_batch)
            .map(|(tweet_id, bytes_result)| {
                let item_result = bytes_result.and_then(|bytes| {
                    match decode::<StratoResult<StratoValue<FilteredReason>>>(&bytes) {
                        StratoResult::Ok(strato_value) => {
                            let reason = strato_value.v;
                            Ok(TweetVisibility {
                                action: action_from_strato_reason(&reason),
                                reason,
                                safety_labels: Err(SafetyLabelFailure::LookupFailed),
                            })
                        }
                        StratoResult::Err(err) => {
                            Err(anyhow!("Strato error code {}: {}", err.code, err.message))
                        }
                    }
                });
                (*tweet_id, item_result)
            })
            .collect()
    }
}

pub(crate) const XAI_VF_DEFAULT_TIMEOUT_MS: u64 = 400;
const XAI_VF_MAX_BATCH_SIZE: usize = 50;
const VF_FILTER_TWEETS_DISCOVERY_ENV: &str = "VF_FILTER_TWEETS_DISCOVERY";

pub struct XaiVfClient {
    client: VisibilityFilteringServiceClient<VfChannel>,
    timeout: Duration,
}

pub(crate) fn xds_channel_params<'a>(name: &'a str, dc: &'a str) -> VfChannelParams<'a> {
    VfChannelParams {
        name,
        dc,
        discovery: VfDiscovery::Xds,
        xds: XdsTarget::vf(dc),
    }
}

fn filter_tweets_channel_params(dc: &str) -> VfChannelParams<'_> {
    VfChannelParams {
        discovery: VfDiscovery::from_env(VF_FILTER_TWEETS_DISCOVERY_ENV, VfDiscovery::Xds),
        ..xds_channel_params("vf-filter-tweets", dc)
    }
}

pub(crate) async fn connect_vf_client(
    params: &VfChannelParams<'_>,
) -> Result<VisibilityFilteringServiceClient<VfChannel>> {
    let channel = build_vf_channel(params).await.map_err(|e| match e {
        VfChannelError::Config(e) | VfChannelError::Connect(e) => e,
    })?;
    Ok(make_vf_client(channel))
}

pub(crate) fn make_vf_client(channel: VfChannel) -> VisibilityFilteringServiceClient<VfChannel> {
    VisibilityFilteringServiceClient::new(channel)
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd)
}

impl XaiVfClient {
    pub async fn evaluate_tweets(
        &self,
        request: vf_pb::EvaluateTweetsRequest,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<Vec<EvaluationResult>> {
        let request_metrics = RequestMetrics::start(&ClientRpc::EVALUATE_TWEETS);
        self.evaluate_tweets_recording(request, metadata, request_metrics)
            .await
    }

    async fn evaluate_tweets_recording(
        &self,
        mut request: vf_pb::EvaluateTweetsRequest,
        metadata: &tonic::metadata::MetadataMap,
        mut request_metrics: RequestMetrics,
    ) -> Result<Vec<EvaluationResult>> {
        let tweets = std::mem::take(&mut request.tweets);
        let plan = plan_evaluate_chunks(&tweets);
        let chunks: Vec<_> = plan
            .iter()
            .map(|chunk| {
                let request = &request;
                let request_metrics = &request_metrics;
                let tweets = &tweets;
                async move {
                    let mut client = self.client.clone();
                    let mut rpc = tonic::Request::new(vf_pb::EvaluateTweetsRequest {
                        tweets: chunk.iter().map(|&i| tweets[i]).collect(),
                        ..request.clone()
                    });
                    *rpc.metadata_mut() = metadata.clone();
                    rpc.set_timeout(self.timeout);
                    let batch_start = Instant::now();
                    let response = client
                        .evaluate_tweets(rpc)
                        .await
                        .map_err(timeout_as_deadline_exceeded);
                    request_metrics.batch(batch_start.elapsed(), response.as_ref().err());
                    decode_response(chunk.len(), response?.into_inner())
                }
            })
            .collect();
        let results = within_deadline(self.timeout, futures::future::try_join_all(chunks)).await;
        match &results {
            Ok(_) => request_metrics.answered(0),
            Err(_) => request_metrics.failed(),
        }
        let mut ordered: Vec<Option<EvaluationResult>> =
            std::iter::repeat_with(|| None).take(tweets.len()).collect();
        for (chunk, results) in plan.iter().zip(results?) {
            for (&i, result) in chunk.iter().zip(results) {
                ordered[i].get_or_insert(result);
            }
        }
        ordered
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .context("VF evaluation plan missed a request index")
    }

    pub fn from_channel(channel: Channel) -> Self {
        Self {
            client: make_vf_client(VfChannel::Wily(channel)),
            timeout: Duration::from_millis(XAI_VF_DEFAULT_TIMEOUT_MS),
        }
    }

    pub async fn connect(datacenter: &str) -> Result<Self> {
        Ok(Self {
            client: connect_vf_client(&filter_tweets_channel_params(datacenter)).await?,
            timeout: Duration::from_millis(XAI_VF_DEFAULT_TIMEOUT_MS),
        })
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout = if timeout_ms == 0 {
            Duration::from_millis(XAI_VF_DEFAULT_TIMEOUT_MS)
        } else {
            Duration::from_millis(timeout_ms)
        };
        self
    }
}

const EVALUATION_DEADLINE_EXCEEDED: &str = "VF evaluation deadline exceeded";

pub(crate) async fn within_deadline<T, E: From<tonic::Status>>(
    timeout: Duration,
    call: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    tokio::time::timeout(timeout, call)
        .await
        .unwrap_or_else(|_| {
            Err(tonic::Status::deadline_exceeded(EVALUATION_DEADLINE_EXCEEDED).into())
        })
}

pub(crate) fn timeout_as_deadline_exceeded(status: tonic::Status) -> tonic::Status {
    if grpc_timeout_expired(&status) {
        tonic::Status::deadline_exceeded(EVALUATION_DEADLINE_EXCEEDED)
    } else {
        status
    }
}

pub(crate) fn grpc_timeout_expired(status: &tonic::Status) -> bool {
    let mut source = std::error::Error::source(status);
    while let Some(err) = source {
        if err.is::<tonic::TimeoutExpired>() {
            return true;
        }
        source = err.source();
    }
    false
}

fn plan_evaluate_chunks(tweets: &[vf_pb::TweetData]) -> Vec<Vec<usize>> {
    if tweets.iter().all(|tweet| tweet.quote_context.is_none()) {
        return (0..tweets.len())
            .step_by(XAI_VF_MAX_BATCH_SIZE)
            .map(|start| (start..tweets.len().min(start + XAI_VF_MAX_BATCH_SIZE)).collect())
            .collect();
    }
    let mut outers: HashMap<u64, (usize, Option<usize>)> = HashMap::new();
    for (i, tweet) in tweets.iter().enumerate() {
        if tweet.quote_context.is_none() {
            outers.entry(tweet.tweet_id).or_insert((i, None));
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, tweet) in tweets.iter().enumerate() {
        let outer = match tweet.quote_context {
            Some(quote) => outers.get_mut(&quote.outer_tweet_id),
            None => outers
                .get_mut(&tweet.tweet_id)
                .filter(|(index, _)| *index == i),
        };
        let Some((outer, group)) = outer else {
            groups.push((i, Vec::new()));
            continue;
        };
        let group = *group.get_or_insert_with(|| {
            groups.push((*outer, Vec::new()));
            groups.len() - 1
        });
        if i != *outer {
            groups[group].1.push(i);
        }
    }
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    for (outer, quotes) in groups {
        if quotes.len() >= XAI_VF_MAX_BATCH_SIZE {
            for part in quotes.chunks(XAI_VF_MAX_BATCH_SIZE - 1) {
                chunks.push(std::iter::once(outer).chain(part.iter().copied()).collect());
            }
            continue;
        }
        let size = 1 + quotes.len();
        let group = std::iter::once(outer).chain(quotes);
        match chunks.last_mut() {
            Some(last) if last.len() + size <= XAI_VF_MAX_BATCH_SIZE => last.extend(group),
            _ => chunks.push(group.collect()),
        }
    }
    chunks
}

fn decode_response(
    requested_count: usize,
    response: vf_pb::EvaluateTweetsResponse,
) -> Result<Vec<EvaluationResult>> {
    anyhow::ensure!(
        response.results.len() == requested_count,
        "VF tweet count mismatch"
    );
    response
        .results
        .into_iter()
        .map(|result| EvaluationResult::decode(result.outcome))
        .collect()
}

fn to_proto_safety_level(level: SafetyLevel) -> vf_pb::SafetyLevel {
    match level {
        SafetyLevel::TimelineHome => vf_pb::SafetyLevel::TimelineHome,
        SafetyLevel::TimelineHomeRecommendations => vf_pb::SafetyLevel::TimelineHomeRecommendations,
        SafetyLevel::ExploreNsfwRecommendations | SafetyLevel::ImmersiveExpandedRecommendations => {
            vf_pb::SafetyLevel::ImmersiveExpandedRecommendations
        }
        SafetyLevel::FilterNone
        | SafetyLevel::SearchTop
        | SafetyLevel::SearchLatest
        | SafetyLevel::SearchPhoto
        | SafetyLevel::SearchVideo
        | SafetyLevel::SearchTopQig
        | SafetyLevel::SearchTopSafeSearchEnabled => vf_pb::SafetyLevel::FilterAll,
    }
}

fn result_to_visibility(result: vf_pb::TweetVisibilityResult) -> TweetVisibility {
    let safety_labels = result
        .safety_labels
        .map(proto_to_safety_label_map)
        .ok_or(SafetyLabelFailure::LookupFailed);
    TweetVisibility {
        action: result.action.map(Action::from).unwrap_or_default(),
        reason: result.filtered_reason.map(FilteredReason::from),
        safety_labels,
    }
}

fn chunk_results(
    requested_tweet_ids: &[u64],
    results: Vec<vf_pb::TweetVisibilityResult>,
) -> Vec<(u64, TweetVisibility)> {
    let answered_in_order = results.len() == requested_tweet_ids.len()
        && results
            .iter()
            .zip(requested_tweet_ids)
            .all(|(r, &tweet_id)| r.tweet_id == tweet_id);
    let missing: Vec<u64> = if answered_in_order {
        Vec::new()
    } else {
        let answered: HashSet<u64> = results.iter().map(|r| r.tweet_id).collect();
        requested_tweet_ids
            .iter()
            .copied()
            .filter(|tweet_id| !answered.contains(tweet_id))
            .collect()
    };
    results
        .into_iter()
        .map(|r| (r.tweet_id, result_to_visibility(r)))
        .chain(missing.into_iter().map(|tweet_id| {
            (
                tweet_id,
                TweetVisibility {
                    action: Action::NotEvaluated,
                    reason: Some(FilteredReason::UnspecifiedReason),
                    safety_labels: Err(SafetyLabelFailure::LookupFailed),
                },
            )
        }))
        .collect()
}

#[async_trait]
impl VfClient for XaiVfClient {
    async fn get_result(
        &self,
        tweet_ids: Vec<u64>,
        safety_level: SafetyLevel,
        for_user_id: u64,
        context: Option<TwitterContextViewer>,
    ) -> HashMap<u64, Result<TweetVisibility>> {
        if tweet_ids.is_empty() {
            return HashMap::new();
        }

        let mut request_metrics = RequestMetrics::start(&ClientRpc::FILTER_TWEETS);
        let country_code = context
            .map(|v| v.request_country_code)
            .filter(|c| !c.is_empty());
        let safety_level_proto = to_proto_safety_level(safety_level) as i32;

        let chunks = tweet_ids.chunks(XAI_VF_MAX_BATCH_SIZE);
        let responses = futures::future::join_all(chunks.clone().map(|chunk| {
            let country_code = country_code.clone();
            let mut client = self.client.clone();
            async move {
                let tweets = chunk
                    .iter()
                    .map(|&tweet_id| vf_pb::TweetInput {
                        tweet_id,
                        author_id: None,
                    })
                    .collect();

                let mut request = tonic::Request::new(vf_pb::VisibilityFilterRequest {
                    safety_level: safety_level_proto,
                    tweets,
                    viewer_id: Some(for_user_id),
                    country_code,
                });
                request.set_timeout(self.timeout);

                let batch_start = Instant::now();
                let response = within_deadline(self.timeout, client.filter_tweets(request)).await;
                let elapsed = batch_start.elapsed();
                let answered = response.map(|r| chunk_results(chunk, r.into_inner().results));
                (answered, elapsed)
            }
        }))
        .await;

        let mut out = HashMap::with_capacity(tweet_ids.len());
        let mut failed_ids = 0;
        for (chunk, (answered, elapsed)) in chunks.zip(responses) {
            request_metrics.batch(elapsed, answered.as_ref().err());
            match answered {
                Ok(chunk_map) => out.extend(chunk_map.into_iter().map(|(id, v)| (id, Ok(v)))),
                Err(status) => {
                    failed_ids += chunk.len();
                    out.extend(chunk.iter().map(|&tweet_id| {
                        (
                            tweet_id,
                            Err(anyhow!("Rust VF filter_tweets error: {status}")),
                        )
                    }));
                }
            }
        }

        request_metrics.answered(failed_ids);
        out
    }
}

pub struct MockVfClient;

#[async_trait]
impl VfClient for MockVfClient {
    async fn get_result(
        &self,
        _user_ids: Vec<u64>,
        _safety_level: SafetyLevel,
        _for_user_id: u64,
        _context: Option<TwitterContextViewer>,
    ) -> HashMap<u64, Result<TweetVisibility>> {
        HashMap::new()
    }
}

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use crate::models::{Action, FilteredReason};

    const SAMPLE_VF_RESPONSE: &[u8] = &[
        12, 0, 4, 12, 9, 252, 12, 0, 118, 12, 105, 20, 12, 0, 11, 12, 0, 2, 12, 0, 15, 8, 0, 2, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 8, 193, 236, 255, 255, 255, 255, 0, 0, 0,
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_decode_simple() {
        use xai_strato::{decode, StratoResult, StratoValue};

        let result: StratoResult<StratoValue<FilteredReason>> = decode(SAMPLE_VF_RESPONSE);
        match result {
            StratoResult::Ok(strato_value) => {
                assert!(strato_value.v.is_some());
                let reason = strato_value.v.unwrap();
                match reason {
                    FilteredReason::SafetyResult(safety_result) => {
                        assert_eq!(safety_result.action, Action::Avoid);
                    }
                    _ => panic!("Unexpected reason: {:?}", reason),
                }
            }
            StratoResult::Err(x) => {
                panic!("Error decoding: {:?}", x);
            }
        }
    }
}

#[cfg(test)]
mod rust_vf_tests {
    use super::*;
    use crate::models::Action;
    use crate::test_support::RecordingReceiver;
    use tokio::sync::Mutex;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;
    use tonic::{Request, Response, Status};
    use xai_visibility_filtering_proto::visibility_filtering_service_server::{
        VisibilityFilteringService, VisibilityFilteringServiceServer,
    };

    #[test]
    fn safety_level_from_tag_reads_the_thrift_tags() {
        let tags = [0, 6, 7, 8, 61, 93, 94, 165, 227, 236, 247];
        for tag in tags {
            let level = SafetyLevel::from_tag(tag).expect("a known tag");
            assert_eq!(level as i32, tag);
        }
        for tag in [-1, 1, 62, 248, i32::MAX] {
            assert_eq!(SafetyLevel::from_tag(tag), None, "{tag}");
        }
    }

    #[test]
    fn decode_response_requires_one_decodable_outcome_per_requested_tweet() {
        use vf_pb::tweet_evaluation::Outcome;
        use xai_x_thrift::safety_result::FilteredReason;
        use xai_x_thrift::tweet_service::{TweetFieldsResultFiltered, TweetFieldsResultState};
        let bounced = TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(
            FilteredReason::TweetIsBounced(true),
        ));
        let evaluation = |outcome| vf_pb::TweetEvaluation {
            outcome: Some(outcome),
        };
        let results = vec![
            evaluation(Outcome::ResultStateThriftCompact(
                xai_x_thrift::serialize_compact(&bounced).unwrap().into(),
            )),
            evaluation(Outcome::NotEvaluated(vf_pb::NotEvaluated {})),
            evaluation(Outcome::Failed(vf_pb::Failed {})),
        ];
        let decode = |results| decode_response(3, vf_pb::EvaluateTweetsResponse { results });
        assert_eq!(
            decode(results.clone()).unwrap(),
            vec![
                EvaluationResult::Evaluated(Box::new(bounced)),
                EvaluationResult::NotEvaluated,
                EvaluationResult::Failed,
            ]
        );
        let mut missing = results.clone();
        missing.pop();
        let mut absent = results.clone();
        absent[0].outcome = None;
        let mut corrupt = results;
        corrupt[0].outcome = Some(Outcome::ResultStateThriftCompact(vec![0x4c].into()));
        for results in [missing, absent, corrupt] {
            assert!(decode(results).is_err());
        }
    }

    #[derive(Default)]
    struct EvaluateStub {
        requests: Arc<std::sync::Mutex<Vec<tonic::metadata::MetadataMap>>>,
        rpcs: Arc<std::sync::Mutex<Vec<Vec<vf_pb::TweetData>>>>,
        fail_chunks_smaller_than: usize,
        hang: bool,
    }

    #[tonic::async_trait]
    impl VisibilityFilteringService for EvaluateStub {
        async fn evaluate_tweets(
            &self,
            request: Request<vf_pb::EvaluateTweetsRequest>,
        ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
            self.requests
                .lock()
                .unwrap()
                .push(request.metadata().clone());
            if self.hang {
                std::future::pending::<()>().await;
            }
            let tweets = request.into_inner().tweets;
            self.rpcs.lock().unwrap().push(tweets.clone());
            if tweets.len() < self.fail_chunks_smaller_than {
                return Err(Status::unavailable("stub failure"));
            }
            Ok(Response::new(vf_pb::EvaluateTweetsResponse {
                results: tweets
                    .iter()
                    .map(|tweet| vf_pb::TweetEvaluation {
                        outcome: Some(outcome_for(tweet.tweet_id)),
                    })
                    .collect(),
            }))
        }
    }

    async fn serve_stub<S: VisibilityFilteringService>(
        stub: S,
    ) -> (Channel, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(
                    VisibilityFilteringServiceServer::new(stub)
                        .accept_compressed(CompressionEncoding::Zstd)
                        .send_compressed(CompressionEncoding::Zstd),
                )
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        (channel, handle)
    }

    async fn evaluate_against(
        stub: EvaluateStub,
        client_timeout_ms: u64,
        tweet_count: usize,
    ) -> (Result<Vec<EvaluationResult>>, Arc<RecordingReceiver>) {
        let (channel, handle) = serve_stub(stub).await;
        let client = XaiVfClient::from_channel(channel).with_timeout_ms(client_timeout_ms);
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert("twittercontext", "ctx".parse().unwrap());
        let sr = Arc::new(RecordingReceiver::default());
        let request_metrics =
            RequestMetrics::with_receiver(&ClientRpc::EVALUATE_TWEETS, Some(sr.clone()));
        let result = tokio::spawn(async move {
            client
                .evaluate_tweets_recording(
                    vf_pb::EvaluateTweetsRequest {
                        safety_level: 8,
                        tweets: (0..tweet_count)
                            .map(|id| vf_pb::TweetData {
                                tweet_id: id as u64,
                                quote_context: None,
                            })
                            .collect(),
                        ..Default::default()
                    },
                    &metadata,
                    request_metrics,
                )
                .await
        })
        .await
        .unwrap();
        handle.abort();
        let _ = handle.await;
        (result, sr)
    }

    #[tokio::test]
    async fn evaluate_tweets_chunks_and_fails_whole_call_on_any_chunk_error_or_deadline() {
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (results, sr) = evaluate_against(
            EvaluateStub {
                requests: requests.clone(),
                ..Default::default()
            },
            1_000,
            XAI_VF_MAX_BATCH_SIZE + 1,
        )
        .await;
        assert_eq!(results.unwrap().len(), XAI_VF_MAX_BATCH_SIZE + 1);
        assert_eq!(
            sr.series(),
            [
                "incr vf_client_evaluate_tweets|requests=completed",
                "observe_vm vf_client_evaluate_tweets_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_evaluate_tweets_latency_ms_vm",
            ]
        );
        {
            let captured = requests.lock().unwrap();
            assert_eq!(captured.len(), 2);
            for metadata in captured.iter() {
                assert_eq!(metadata.get("twittercontext").unwrap(), "ctx");
                assert_eq!(metadata.get("grpc-timeout").unwrap(), "1000000u");
            }
        }

        let (failed, sr) = evaluate_against(
            EvaluateStub {
                requests: requests.clone(),
                fail_chunks_smaller_than: 2,
                ..Default::default()
            },
            1_000,
            XAI_VF_MAX_BATCH_SIZE + 1,
        )
        .await;
        assert!(failed.is_err());
        assert_eq!(sr.counter("vf_client_evaluate_tweets|requests=failed"), 1);
        assert_eq!(
            sr.counter("vf_client_evaluate_tweets_batch_error|code=unavailable"),
            1
        );
        let (deadline, sr) = evaluate_against(
            EvaluateStub {
                requests,
                hang: true,
                ..Default::default()
            },
            50,
            XAI_VF_MAX_BATCH_SIZE + 1,
        )
        .await;
        assert_eq!(
            deadline
                .unwrap_err()
                .downcast_ref::<tonic::Status>()
                .map(tonic::Status::code),
            Some(tonic::Code::DeadlineExceeded)
        );
        assert_eq!(sr.counter("vf_client_evaluate_tweets|requests=failed"), 1);
        assert!(sr
            .series()
            .contains(&"observe_vm vf_client_evaluate_tweets_latency_ms_vm".to_string()));
    }

    #[tokio::test]
    async fn a_batch_past_its_grpc_timeout_counts_as_deadline_exceeded() {
        let (channel, handle) = serve_stub(EvaluateStub {
            hang: true,
            ..Default::default()
        })
        .await;
        let mut rpc = tonic::Request::new(vf_pb::EvaluateTweetsRequest::default());
        rpc.set_timeout(Duration::from_millis(50));
        let status = make_vf_client(VfChannel::Wily(channel))
            .evaluate_tweets(rpc)
            .await
            .unwrap_err();
        handle.abort();
        let sr = Arc::new(RecordingReceiver::default());
        RequestMetrics::with_receiver(&ClientRpc::EVALUATE_TWEETS, Some(sr.clone()))
            .batch(Duration::ZERO, Some(&status));
        assert_eq!(
            sr.counter("vf_client_evaluate_tweets_batch_error|code=deadline_exceeded"),
            1
        );
    }

    fn outcome_for(tweet_id: u64) -> vf_pb::tweet_evaluation::Outcome {
        if tweet_id.is_multiple_of(2) {
            vf_pb::tweet_evaluation::Outcome::NotEvaluated(vf_pb::NotEvaluated {})
        } else {
            vf_pb::tweet_evaluation::Outcome::Failed(vf_pb::Failed {})
        }
    }

    fn expected_for(tweet_id: u64) -> EvaluationResult {
        if tweet_id.is_multiple_of(2) {
            EvaluationResult::NotEvaluated
        } else {
            EvaluationResult::Failed
        }
    }

    async fn evaluate_capturing(
        tweets: Vec<vf_pb::TweetData>,
    ) -> (Vec<EvaluationResult>, Vec<Vec<vf_pb::TweetData>>) {
        let rpcs = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (channel, handle) = serve_stub(EvaluateStub {
            rpcs: rpcs.clone(),
            ..Default::default()
        })
        .await;
        let results = XaiVfClient::from_channel(channel)
            .with_timeout_ms(10_000)
            .evaluate_tweets(
                vf_pb::EvaluateTweetsRequest {
                    safety_level: 8,
                    tweets,
                    ..Default::default()
                },
                &tonic::metadata::MetadataMap::new(),
            )
            .await
            .unwrap();
        handle.abort();
        let rpcs = rpcs.lock().unwrap().clone();
        (results, rpcs)
    }

    fn standalone_tweet(tweet_id: u64) -> vf_pb::TweetData {
        vf_pb::TweetData {
            tweet_id,
            quote_context: None,
        }
    }

    fn quote_tweet(tweet_id: u64, outer_tweet_id: u64) -> vf_pb::TweetData {
        vf_pb::TweetData {
            tweet_id,
            quote_context: Some(vf_pb::QuoteContext { outer_tweet_id }),
        }
    }

    fn assert_each_quote_has_its_outer(rpcs: &[Vec<vf_pb::TweetData>]) {
        for rpc in rpcs {
            assert!(rpc.len() <= XAI_VF_MAX_BATCH_SIZE);
            for tweet in rpc {
                if let Some(quote) = tweet.quote_context {
                    assert!(
                        rpc.contains(&standalone_tweet(quote.outer_tweet_id)),
                        "quote {} sent without its outer {}",
                        tweet.tweet_id,
                        quote.outer_tweet_id
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn evaluate_tweets_sends_each_quote_in_the_rpc_of_its_outer() {
        let quotes = (0..60u64).map(|k| quote_tweet(1_000 + k, 5_000 + 3 * k));
        let outers = (0..60u64).map(|k| standalone_tweet(5_000 + 3 * k));
        let tweets: Vec<_> = quotes.chain(outers).collect();
        let expected: Vec<_> = tweets.iter().map(|t| expected_for(t.tweet_id)).collect();

        let (results, rpcs) = evaluate_capturing(tweets.clone()).await;

        assert_eq!(results, expected);
        assert_each_quote_has_its_outer(&rpcs);
        let mut sent: Vec<_> = rpcs.into_iter().flatten().collect();
        sent.sort_by_key(|t| t.tweet_id);
        let mut requested = tweets;
        requested.sort_by_key(|t| t.tweet_id);
        assert_eq!(sent, requested, "each entry is sent once");
    }

    #[tokio::test]
    async fn evaluate_tweets_repeats_an_outer_in_each_rpc_of_its_quotes() {
        let outer = 7;
        let tweets: Vec<_> = std::iter::once(standalone_tweet(outer))
            .chain((0..120u64).map(|k| quote_tweet(1_000 + k, outer)))
            .collect();
        let expected: Vec<_> = tweets.iter().map(|t| expected_for(t.tweet_id)).collect();

        let (results, rpcs) = evaluate_capturing(tweets).await;

        assert_eq!(results, expected, "the outer has one result, at its index");
        assert_eq!(rpcs.len(), 3);
        assert_each_quote_has_its_outer(&rpcs);
    }

    #[test]
    fn evaluate_tweets_without_quotes_chunks_in_request_order() {
        let tweets: Vec<_> = (0..2 * XAI_VF_MAX_BATCH_SIZE as u64 + 1)
            .map(standalone_tweet)
            .collect();
        let indices: Vec<usize> = (0..tweets.len()).collect();

        assert_eq!(
            plan_evaluate_chunks(&tweets),
            indices
                .chunks(XAI_VF_MAX_BATCH_SIZE)
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn safety_level_maps_to_proto() {
        assert_eq!(
            to_proto_safety_level(SafetyLevel::TimelineHome),
            vf_pb::SafetyLevel::TimelineHome
        );
        assert_eq!(
            to_proto_safety_level(SafetyLevel::TimelineHomeRecommendations),
            vf_pb::SafetyLevel::TimelineHomeRecommendations
        );
        assert_eq!(
            to_proto_safety_level(SafetyLevel::ExploreNsfwRecommendations),
            vf_pb::SafetyLevel::ImmersiveExpandedRecommendations
        );
        assert_eq!(
            to_proto_safety_level(SafetyLevel::ImmersiveExpandedRecommendations),
            vf_pb::SafetyLevel::ImmersiveExpandedRecommendations
        );
        assert_eq!(
            to_proto_safety_level(SafetyLevel::FilterNone),
            vf_pb::SafetyLevel::FilterAll
        );
    }

    #[test]
    fn chunk_results_converts_actions_and_fails_closed() {
        let results = vec![
            vf_pb::TweetVisibilityResult {
                tweet_id: 1,
                action: Some(vf_pb::Action {
                    kind: Some(vf_pb::action::Kind::Allow(true)),
                }),
                filtered_reason: None,
                safety_labels: None,
            },
            vf_pb::TweetVisibilityResult {
                tweet_id: 2,
                action: Some(vf_pb::Action {
                    kind: Some(vf_pb::action::Kind::Drop(vf_pb::DropReason {})),
                }),
                filtered_reason: Some(vf_pb::FilteredReason {
                    reason: Some(vf_pb::filtered_reason::Reason::AuthorIsUnsafe(true)),
                }),
                safety_labels: None,
            },
            vf_pb::TweetVisibilityResult {
                tweet_id: 3,
                action: None,
                filtered_reason: Some(vf_pb::FilteredReason {
                    reason: Some(vf_pb::filtered_reason::Reason::SafetyResult(
                        vf_pb::SafetyResult {
                            action: Some(vf_pb::Action {
                                kind: Some(vf_pb::action::Kind::Drop(vf_pb::DropReason {})),
                            }),
                        },
                    )),
                }),
                safety_labels: None,
            },
            vf_pb::TweetVisibilityResult {
                tweet_id: 7,
                action: Some(vf_pb::Action {
                    kind: Some(vf_pb::action::Kind::Interstitial(true)),
                }),
                filtered_reason: Some(vf_pb::FilteredReason {
                    reason: Some(vf_pb::filtered_reason::Reason::ContainNsfwMedia(true)),
                }),
                safety_labels: None,
            },
        ];

        let map: HashMap<u64, Result<TweetVisibility>> = chunk_results(&[1, 2, 3, 4, 7], results)
            .into_iter()
            .map(|(id, v)| (id, Ok(v)))
            .collect();

        assert!(matches!(
            map.get(&1),
            Some(Ok(TweetVisibility {
                action: Action::Allow,
                reason: None,
                ..
            }))
        ));
        assert!(matches!(
            map.get(&2),
            Some(Ok(TweetVisibility {
                action: Action::Drop(_),
                reason: Some(FilteredReason::AuthorIsUnsafe),
                ..
            }))
        ));
        match map.get(&3) {
            Some(Ok(TweetVisibility {
                action: Action::NotEvaluated,
                reason: Some(FilteredReason::SafetyResult(sr)),
                ..
            })) => {
                assert!(matches!(sr.action, Action::Drop(_)));
            }
            other => panic!("unexpected mapping for id 3: {other:?}"),
        }
        assert!(
            matches!(
                map.get(&4),
                Some(Ok(TweetVisibility {
                    action: Action::NotEvaluated,
                    reason: Some(FilteredReason::UnspecifiedReason),
                    ..
                }))
            ),
            "missing response ids surface no verdict"
        );
        assert!(matches!(
            map.get(&7),
            Some(Ok(TweetVisibility {
                action: Action::Interstitial,
                reason: Some(FilteredReason::ContainNsfwMedia),
                ..
            }))
        ));
    }

    #[test]
    fn strato_absent_and_bare_reasons_keep_their_policy() {
        assert_eq!(action_from_strato_reason(&None), Action::Allow);
        assert_eq!(
            action_from_strato_reason(&Some(FilteredReason::AuthorBlockViewer)),
            Action::Drop(DropReason {})
        );
    }

    #[test]
    fn response_projection_preserves_rust_reason_shape() {
        for (action, reason, expected) in [
            (Action::Allow, Some(FilteredReason::ContainNsfwMedia), None),
            (
                Action::Interstitial,
                Some(FilteredReason::ContainNsfwMedia),
                Some(FilteredReason::SafetyResult(SafetyResult {
                    reason: None,
                    action: Action::Interstitial,
                })),
            ),
            (
                Action::Drop(DropReason {}),
                Some(FilteredReason::AuthorIsUnsafe),
                Some(FilteredReason::AuthorIsUnsafe),
            ),
            (
                Action::NotEvaluated,
                None,
                Some(FilteredReason::UnspecifiedReason),
            ),
        ] {
            let visibility = TweetVisibility {
                action,
                reason,
                safety_labels: Err(SafetyLabelFailure::LookupFailed),
            };
            assert_eq!(visibility.to_visibility_reason(), expected);
        }
    }

    #[test]
    fn chunk_results_converts_labels_and_fails_closed() {
        use xai_x_thrift::tweet_safety_label::SafetyLabelType;

        let results = vec![
            vf_pb::TweetVisibilityResult {
                tweet_id: 1,
                action: Some(vf_pb::Action {
                    kind: Some(vf_pb::action::Kind::Allow(true)),
                }),
                filtered_reason: None,
                safety_labels: Some(vf_pb::SafetyLabelMap {
                    labels: HashMap::from([(
                        i32::from(SafetyLabelType::NSFW_HIGH_PRECISION),
                        vf_pb::SafetyLabel {
                            source: Some("some rule".to_string()),
                            ..Default::default()
                        },
                    )]),
                }),
            },
            vf_pb::TweetVisibilityResult {
                tweet_id: 2,
                action: Some(vf_pb::Action {
                    kind: Some(vf_pb::action::Kind::Allow(true)),
                }),
                filtered_reason: None,
                safety_labels: None,
            },
        ];

        let map: HashMap<u64, Result<TweetVisibility>> = chunk_results(&[1, 2, 3], results)
            .into_iter()
            .map(|(id, v)| (id, Ok(v)))
            .collect();

        match map.get(&1) {
            Some(Ok(r)) => {
                assert_eq!(r.reason, None);
                let labels = r.safety_labels.as_ref().expect("labels present");
                let label = labels
                    .get(&SafetyLabelType::NSFW_HIGH_PRECISION)
                    .expect("label converted");
                assert_eq!(label.source.as_deref(), Some("some rule"));
            }
            other => panic!("unexpected mapping for id 1: {other:?}"),
        }
        match map.get(&2) {
            Some(Ok(r)) => {
                assert_eq!(r.reason, None);
                assert!(r.safety_labels.is_err(), "absent map stays unavailable");
            }
            other => panic!("unexpected mapping for id 2: {other:?}"),
        }
        match map.get(&3) {
            Some(Ok(r)) => {
                assert_eq!(r.reason, Some(FilteredReason::UnspecifiedReason));
                assert!(r.safety_labels.is_err(), "missing ids fail closed");
            }
            other => panic!("unexpected mapping for id 3: {other:?}"),
        }
    }

    #[test]
    fn chunk_results_marks_only_unanswered_ids_not_evaluated() {
        let allowed = |tweet_id| vf_pb::TweetVisibilityResult {
            tweet_id,
            action: Some(vf_pb::Action {
                kind: Some(vf_pb::action::Kind::Allow(true)),
            }),
            filtered_reason: None,
            safety_labels: None,
        };
        let actions = |results| -> Vec<(u64, Action)> {
            chunk_results(&[1, 2, 2], results)
                .into_iter()
                .map(|(id, v)| (id, v.action))
                .collect()
        };

        assert_eq!(
            actions(vec![allowed(1), allowed(2), allowed(2)]),
            vec![(1, Action::Allow), (2, Action::Allow), (2, Action::Allow)]
        );
        assert_eq!(
            actions(vec![allowed(2), allowed(1)]),
            vec![(2, Action::Allow), (1, Action::Allow)]
        );
        assert_eq!(
            actions(vec![allowed(2)]),
            vec![(2, Action::Allow), (1, Action::NotEvaluated)]
        );
    }

    #[derive(Clone, Default)]
    struct StubVfService {
        requests: Arc<Mutex<Vec<vf_pb::VisibilityFilterRequest>>>,
    }

    #[tonic::async_trait]
    impl VisibilityFilteringService for StubVfService {
        async fn filter_tweets(
            &self,
            request: Request<vf_pb::VisibilityFilterRequest>,
        ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
            let req = request.into_inner();
            self.requests.lock().await.push(req.clone());
            let results = req
                .tweets
                .iter()
                .enumerate()
                .map(|(index, t)| vf_pb::TweetVisibilityResult {
                    tweet_id: t.tweet_id,
                    action: Some(vf_pb::Action {
                        kind: Some(vf_pb::action::Kind::Drop(vf_pb::DropReason {})),
                    }),
                    filtered_reason: Some(
                        if index == 0 {
                            FilteredReason::AuthorBlockViewer
                        } else {
                            FilteredReason::ViewerBlocksAuthor
                        }
                        .into(),
                    ),
                    safety_labels: None,
                })
                .collect();
            Ok(Response::new(vf_pb::VisibilityFilterResponse { results }))
        }
    }

    async fn start_stub_server() -> (
        XaiVfClient,
        tokio::task::JoinHandle<()>,
        Arc<Mutex<Vec<vf_pb::VisibilityFilterRequest>>>,
    ) {
        let service = StubVfService::default();
        let requests = service.requests.clone();
        let (channel, handle) = serve_stub(service).await;
        (XaiVfClient::from_channel(channel), handle, requests)
    }

    #[tokio::test]
    async fn get_result_keeps_block_directions_distinct() {
        let (client, handle, _requests) = start_stub_server().await;

        let result = client
            .get_result(vec![10, 20], SafetyLevel::TimelineHome, 99, None)
            .await;

        assert_eq!(result.len(), 2);
        assert!(matches!(
            result.get(&10),
            Some(Ok(TweetVisibility {
                action: Action::Drop(_),
                reason: Some(FilteredReason::AuthorBlockViewer),
                ..
            }))
        ));
        assert!(matches!(
            result.get(&20),
            Some(Ok(TweetVisibility {
                action: Action::Drop(_),
                reason: Some(FilteredReason::ViewerBlocksAuthor),
                ..
            }))
        ));
        handle.abort();
    }

    #[tokio::test]
    async fn get_result_forwards_country_code() {
        let (client, handle, requests) = start_stub_server().await;

        let with_country = TwitterContextViewer {
            request_country_code: "DE".to_string(),
            ..Default::default()
        };
        client
            .get_result(vec![10], SafetyLevel::TimelineHome, 99, Some(with_country))
            .await;

        let empty_country = TwitterContextViewer {
            request_country_code: String::new(),
            ..Default::default()
        };
        client
            .get_result(vec![11], SafetyLevel::TimelineHome, 99, Some(empty_country))
            .await;

        client
            .get_result(vec![12], SafetyLevel::TimelineHome, 99, None)
            .await;

        let captured = requests.lock().await;
        assert_eq!(captured.len(), 3);
        assert_eq!(captured[0].country_code.as_deref(), Some("DE"));
        assert_eq!(captured[1].country_code, None);
        assert_eq!(captured[2].country_code, None);
        handle.abort();
    }

    #[tokio::test]
    async fn get_result_chunks_large_batches() {
        let (client, handle, requests) = start_stub_server().await;

        let tweet_ids: Vec<u64> = (1..=(XAI_VF_MAX_BATCH_SIZE as u64 + 1)).collect();
        let result = client
            .get_result(tweet_ids.clone(), SafetyLevel::TimelineHome, 99, None)
            .await;

        assert_eq!(result.len(), tweet_ids.len());
        let captured = requests.lock().await;
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].tweets.len(), XAI_VF_MAX_BATCH_SIZE);
        assert_eq!(captured[1].tweets.len(), 1);
        handle.abort();
    }

    #[tokio::test]
    async fn a_chunk_waiting_for_an_endpoint_fails_at_the_client_deadline() {
        let (channel, _endpoints) = Channel::balance_channel::<u8>(1);
        let client = XaiVfClient::from_channel(channel).with_timeout_ms(50);

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            client.get_result(vec![1, 2], SafetyLevel::TimelineHome, 99, None),
        )
        .await
        .expect("the client deadline bounds a chunk");

        assert_eq!(result.len(), 2);
        for id in [1, 2] {
            let err = result[&id].as_ref().unwrap_err().to_string();
            assert!(err.contains(EVALUATION_DEADLINE_EXCEEDED), "{err}");
        }
    }

    #[tokio::test]
    async fn get_result_partial_success_fills_missing_ids() {
        #[derive(Clone, Default)]
        struct PartialVfService;

        #[tonic::async_trait]
        impl VisibilityFilteringService for PartialVfService {
            async fn filter_tweets(
                &self,
                request: Request<vf_pb::VisibilityFilterRequest>,
            ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
                let req = request.into_inner();
                let first = req.tweets.first().map(|t| t.tweet_id).unwrap_or(0);
                Ok(Response::new(vf_pb::VisibilityFilterResponse {
                    results: vec![vf_pb::TweetVisibilityResult {
                        tweet_id: first,
                        action: Some(vf_pb::Action {
                            kind: Some(vf_pb::action::Kind::Drop(vf_pb::DropReason {})),
                        }),
                        filtered_reason: Some(vf_pb::FilteredReason {
                            reason: Some(vf_pb::filtered_reason::Reason::AuthorIsUnsafe(true)),
                        }),
                        safety_labels: None,
                    }],
                }))
            }
        }

        let (channel, handle) = serve_stub(PartialVfService).await;
        let client = XaiVfClient::from_channel(channel);

        let result = client
            .get_result(vec![10, 20], SafetyLevel::TimelineHome, 99, None)
            .await;

        assert_eq!(result.len(), 2);
        assert!(matches!(
            result.get(&10),
            Some(Ok(TweetVisibility {
                reason: Some(FilteredReason::AuthorIsUnsafe),
                ..
            }))
        ));
        assert!(matches!(
            result.get(&20),
            Some(Ok(TweetVisibility {
                reason: Some(FilteredReason::UnspecifiedReason),
                ..
            }))
        ));
        handle.abort();
    }

    #[tokio::test]
    async fn get_result_multi_chunk_partial_rpc_failure() {
        #[derive(Clone, Default)]
        struct FailSecondChunkService;

        #[tonic::async_trait]
        impl VisibilityFilteringService for FailSecondChunkService {
            async fn filter_tweets(
                &self,
                request: Request<vf_pb::VisibilityFilterRequest>,
            ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
                let req = request.into_inner();
                if req
                    .tweets
                    .iter()
                    .any(|t| t.tweet_id > XAI_VF_MAX_BATCH_SIZE as u64)
                {
                    return Err(Status::unavailable("simulated chunk failure"));
                }
                let results = req
                    .tweets
                    .iter()
                    .map(|t| vf_pb::TweetVisibilityResult {
                        tweet_id: t.tweet_id,
                        action: Some(vf_pb::Action {
                            kind: Some(vf_pb::action::Kind::Allow(true)),
                        }),
                        filtered_reason: None,
                        safety_labels: None,
                    })
                    .collect();
                Ok(Response::new(vf_pb::VisibilityFilterResponse { results }))
            }
        }

        let (channel, handle) = serve_stub(FailSecondChunkService).await;
        let client = XaiVfClient::from_channel(channel);

        let tweet_ids: Vec<u64> = (1..=(XAI_VF_MAX_BATCH_SIZE as u64 + 1)).collect();
        let result = client
            .get_result(tweet_ids.clone(), SafetyLevel::TimelineHome, 99, None)
            .await;

        assert_eq!(result.len(), tweet_ids.len());
        for id in 1..=XAI_VF_MAX_BATCH_SIZE as u64 {
            assert!(
                matches!(
                    result.get(&id),
                    Some(Ok(TweetVisibility { reason: None, .. }))
                ),
                "chunk-1 id {id} should Allow"
            );
        }
        let failed_id = XAI_VF_MAX_BATCH_SIZE as u64 + 1;
        assert!(
            matches!(result.get(&failed_id), Some(Err(_))),
            "failed chunk id should be Err"
        );

        handle.abort();
    }
}
