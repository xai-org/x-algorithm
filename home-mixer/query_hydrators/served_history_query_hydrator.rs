use crate::clients::served_history_client::{ServedHistoryClient, TimelineType};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableUrtMigrationComponents, ExcludeServedTweetIdsDuration, ExcludeServedTweetIdsNumber,
    FeedSurveyFatigueMinutes, VideoCarouselFatigueMinutes, WhoToFollowFatigueHours,
};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::client_utils::app_id_to_served_history_id;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_x_thrift::served_history::{EntityIdType, ServedHistory};

pub struct ServedHistoryQueryHydrator {
    client: Arc<dyn ServedHistoryClient>,
}

impl ServedHistoryQueryHydrator {
    pub fn from_client(client: Arc<dyn ServedHistoryClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for ServedHistoryQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableUrtMigrationComponents)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let platform = app_id_to_served_history_id(query.client_app_id);
        let timeline_type = TimelineType::for_request(query.request_type);
        let entries = self
            .client
            .get_recent(query.user_id, timeline_type, platform)
            .await
            .map_err(|e| e.to_string())?;

        let who_to_follow_eligible = is_module_eligible(
            &entries,
            EntityIdType::WHO_TO_FOLLOW,
            query.params.get(WhoToFollowFatigueHours) as i64 * 3_600_000,
            query.request_time_ms,
        );

        let feed_survey_eligible = is_module_eligible(
            &entries,
            EntityIdType::ANNOTATION,
            query.params.get(FeedSurveyFatigueMinutes) as i64 * 60_000,
            query.request_time_ms,
        );

        let video_carousel_eligible = is_module_eligible(
            &entries,
            EntityIdType::VIDEO_CAROUSEL_MODULE,
            query.params.get(VideoCarouselFatigueMinutes) as i64 * 60_000,
            query.request_time_ms,
        );

        let served_ids = recently_served_ids(
            &entries,
            query.request_time_ms,
            query.params.get(ExcludeServedTweetIdsDuration),
            query.params.get(ExcludeServedTweetIdsNumber),
        );

        Ok(ScoredPostsQuery {
            served_history: entries,
            served_ids,
            who_to_follow_eligible,
            feed_survey_eligible,
            video_carousel_eligible,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.served_history = hydrated.served_history;
        query.served_ids = hydrated.served_ids;
        query.who_to_follow_eligible = hydrated.who_to_follow_eligible;
        query.feed_survey_eligible = hydrated.feed_survey_eligible;
        query.video_carousel_eligible = hydrated.video_carousel_eligible;
    }
}

fn recently_served_ids(
    history: &[ServedHistory],
    now_ms: i64,
    duration_minutes: u32,
    max_items: usize,
) -> Vec<u64> {
    let min_time_ms = now_ms - (duration_minutes as i64 * 60_000);

    history
        .iter()
        .filter(|sh| sh.served_time_ms.is_some_and(|t| t >= min_time_ms))
        .flat_map(|sh| {
            sh.entries
                .iter()
                .filter(|entry| entry.entity_type != EntityIdType::VIDEO_CAROUSEL_MODULE)
                .flat_map(|entry| {
                    entry.item_ids.iter().flatten().flat_map(|ids| {
                        [ids.tweet_id, ids.source_tweet_id]
                            .into_iter()
                            .flatten()
                            .map(|id| id as u64)
                    })
                })
        })
        .take(max_items)
        .collect()
}

fn is_module_eligible(
    history: &[ServedHistory],
    entity_type: EntityIdType,
    fatigue_interval_ms: i64,
    now_ms: i64,
) -> bool {
    let last_served_ms = history
        .iter()
        .filter(|sh| sh.entries.iter().any(|e| e.entity_type == entity_type))
        .filter_map(|sh| sh.served_time_ms)
        .max();

    match last_served_ms {
        Some(ts) => (now_ms - ts) >= fatigue_interval_ms,
        None => true,
    }
}
