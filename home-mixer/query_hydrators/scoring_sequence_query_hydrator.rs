use crate::clients::user_action_aggregation_client::{
    UserActionAggregationClient, UserActionsCluster,
};
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{
    self as p, EnableScoringSequenceHydration, HistoryTopicsEnabled, HistoryTopicsWindowDays,
    MaxSeqLengthScoring, PhoenixAggregationType, UasSourceDataType, UseXdsForUas,
    UserActionsClusterId,
};
use crate::util::viewer_history::engaged_history_topics;
use bytes::Bytes;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::{
    ResponseFormat, UaasModelType, UserActionAggregationType, UserActionSequence,
    UserActionSequenceSourceDataType,
};

const MS_PER_DAY: i64 = 86_400_000;
const STATS_HISTORY_WINDOW_DAYS: i64 = 28;

pub struct ScoringSequenceQueryHydrator {
    pub user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
}

impl ScoringSequenceQueryHydrator {
    pub fn new(
        user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
    ) -> Self {
        Self {
            user_action_aggregation_client,
        }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for ScoringSequenceQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableScoringSequenceHydration)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let aggregation_type =
            UserActionAggregationType::from_str_name(&query.params.get(PhoenixAggregationType))
                .unwrap_or(UserActionAggregationType::DenseWithFavVqvCleaned);
        let request_id = (is_prod() && query.request_type != RequestType::PhoenixScores)
            .then_some(query.prediction_id as i64);
        let (scoring_sequence, columnar_scoring_sequence) = fetch_scoring_sequence(
            self.user_action_aggregation_client.as_ref(),
            query,
            aggregation_type,
            request_id,
        )
        .await?;
        let (history_topics, stats_history_topics) = if query.params.get(HistoryTopicsEnabled) {
            let topics_since = |days: i64| {
                columnar_scoring_sequence.as_ref().and_then(|sequence| {
                    engaged_history_topics(sequence, query.request_time_ms - days * MS_PER_DAY)
                })
            };
            let window_days = query.params.get(HistoryTopicsWindowDays);
            let history_topics = topics_since(window_days);
            let stats_history_topics = if window_days == STATS_HISTORY_WINDOW_DAYS {
                history_topics.clone()
            } else {
                topics_since(STATS_HISTORY_WINDOW_DAYS)
            };
            (history_topics, stats_history_topics)
        } else {
            (None, None)
        };

        Ok(ScoredPostsQuery {
            scoring_sequence,
            columnar_scoring_sequence,
            history_topics,
            stats_history_topics,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.scoring_sequence = hydrated.scoring_sequence;
        query.columnar_scoring_sequence = hydrated.columnar_scoring_sequence;
        query.history_topics = hydrated.history_topics;
        query.stats_history_topics = hydrated.stats_history_topics;
    }
}

pub async fn fetch_scoring_sequence(
    client: &(dyn UserActionAggregationClient + Send + Sync),
    query: &ScoredPostsQuery,
    aggregation_type: UserActionAggregationType,
    request_id: Option<i64>,
) -> Result<(Option<UserActionSequence>, Option<Bytes>), String> {
    let cluster = UserActionsCluster::parse(&query.params.get(UserActionsClusterId));
    let source_data_type =
        UserActionSequenceSourceDataType::from_str_name(&query.params.get(UasSourceDataType))
            .unwrap_or(UserActionSequenceSourceDataType::Arrow);
    let result = client
        .fetch_aggregated_sequence(
            cluster,
            query.user_id,
            p::UAS_WINDOW_TIME_MS as u32,
            query.params.get(MaxSeqLengthScoring),
            aggregation_type,
            source_data_type,
            ResponseFormat::Arrow,
            request_id,
            query.params.get(UseXdsForUas),
            UaasModelType::Ranking,
        )
        .await
        .map_err(|e| format!("Aggregation service call failed: {}", e))?;
    Ok((Some(result.sequence), result.columnar_bytes))
}
