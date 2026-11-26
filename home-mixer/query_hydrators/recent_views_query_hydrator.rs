use crate::clients::user_action_aggregation_client::{
    UserActionAggregationClient, UserActionsCluster,
};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    self as p, PostTopicsTimeoutMs, RecentViewsRows, RecentViewsSequenceLength, UasSourceDataType,
    UseXdsForUas, UserActionsClusterId, VMRankerSendPostTopics,
};
use crate::util::post_annotations::fetch_post_topics;
use arrow::array::{Array, BooleanArray, FixedSizeListArray, Int32Array, Int64Array};
use arrow::ipc::reader::StreamReader;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::{
    ResponseFormat, UaasModelType, UserActionAggregationType, UserActionSequenceSourceDataType,
};
use xai_vm_ranker_proto::experiment::RecentView;

const COL_TWEET_ID: &str = "tweetId";
const COL_IMPRESSED_TIME_MS: &str = "impressedTimeMs";
const COL_PRODUCT_SURFACE: &str = "productSurface";
const COL_ACTION_NAME_MULTI_HOT: &str = "actionNameMultiHot";
const COL_DWELL_TIME_MS: &str = "dwellTimeMs";

pub struct RecentViewsQueryHydrator {
    pub user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

fn newest_views(columnar: &bytes::Bytes, max_rows: usize) -> Option<Vec<RecentView>> {
    let reader = StreamReader::try_new(std::io::Cursor::new(columnar.as_ref()), None).ok()?;
    let mut views = Vec::new();
    for batch in reader {
        let batch = batch.ok()?;
        let tweet_ids = batch
            .column_by_name(COL_TWEET_ID)?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let impressed = batch
            .column_by_name(COL_IMPRESSED_TIME_MS)?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let surfaces = batch
            .column_by_name(COL_PRODUCT_SURFACE)?
            .as_any()
            .downcast_ref::<Int32Array>()?;
        let dwell = batch
            .column_by_name(COL_DWELL_TIME_MS)?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let multi_hot = batch
            .column_by_name(COL_ACTION_NAME_MULTI_HOT)?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()?;
        let hot_values = multi_hot.values().as_any().downcast_ref::<BooleanArray>()?;
        let vocab = (multi_hot.value_length() as usize).min(64);
        let stride = multi_hot.value_length() as usize;
        for row in 0..batch.num_rows() {
            if tweet_ids.is_null(row) || tweet_ids.value(row) <= 0 {
                continue;
            }
            let action_bits = (0..vocab)
                .filter(|&i| hot_values.value(row * stride + i))
                .fold(0u64, |bits, i| bits | (1u64 << i));
            views.push(RecentView {
                post_id: tweet_ids.value(row) as u64,
                impressed_time_ms: impressed.value(row),
                product_surface: surfaces.value(row),
                action_bits,
                dwell_time_ms: dwell.value(row),
            });
        }
    }
    views.sort_by(|a, b| b.impressed_time_ms.cmp(&a.impressed_time_ms));
    views.truncate(max_rows);
    Some(views)
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for RecentViewsQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(VMRankerSendPostTopics)
            && !query.in_network_only
            && !query.is_topic_request()
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let cluster = UserActionsCluster::parse(&query.params.get(UserActionsClusterId));
        let source_data_type =
            UserActionSequenceSourceDataType::from_str_name(&query.params.get(UasSourceDataType))
                .unwrap_or(UserActionSequenceSourceDataType::Arrow);
        let result = self
            .user_action_aggregation_client
            .fetch_aggregated_sequence(
                cluster,
                query.user_id,
                p::UAS_WINDOW_TIME_MS as u32,
                query.params.get(RecentViewsSequenceLength),
                UserActionAggregationType::DenseWithRecentImpressions,
                source_data_type,
                ResponseFormat::Arrow,
                None,
                query.params.get(UseXdsForUas),
                UaasModelType::Ranking,
            )
            .await
            .map_err(|e| format!("Recent views fetch failed: {}", e))?;
        let recent_views = result
            .columnar_bytes
            .as_ref()
            .and_then(|bytes| newest_views(bytes, query.params.get(RecentViewsRows)))
            .unwrap_or_default();
        let post_ids: Vec<u64> = recent_views.iter().map(|v| v.post_id).collect();
        let recent_view_topics = fetch_post_topics(
            self.strato_client.as_ref(),
            &post_ids,
            Duration::from_millis(query.params.get(PostTopicsTimeoutMs)),
        )
        .await
        .unwrap_or_default();
        Ok(ScoredPostsQuery {
            recent_views,
            recent_view_topics,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.recent_views = hydrated.recent_views;
        query.recent_view_topics = hydrated.recent_view_topics;
    }
}
