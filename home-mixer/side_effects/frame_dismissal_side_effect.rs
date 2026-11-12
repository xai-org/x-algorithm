use std::sync::Arc;

use futures::future::join_all;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::{feed_item, FeedItem};

use crate::frames;
use crate::models::query::ScoredPostsQuery;

pub struct FrameDismissalSideEffect {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

fn served_fatigues(query: &ScoredPostsQuery, selected: &[FeedItem]) -> Vec<(&'static str, i64)> {
    let mut fatigues: Vec<(&'static str, i64)> = selected
        .iter()
        .filter_map(|item| match &item.item {
            Some(feed_item::Item::Frame(frame)) => frames::fatigue_for_route(&frame.route),
            _ => None,
        })
        .filter_map(|fatigue| {
            fatigue
                .window_ms(query)
                .map(|window_ms| (fatigue.served_key, query.request_time_ms + window_ms))
        })
        .collect();
    fatigues.sort_unstable();
    fatigues.dedup_by_key(|(key, _)| *key);
    fatigues
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for FrameDismissalSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && !frames::fatigue_keys(&query).is_empty()
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let puts = served_fatigues(query, &input.selected_candidates)
            .into_iter()
            .map(|(frame_key, expires_at_ms)| async move {
                self.strato_client
                    .put_user_frame_dismissal(query.user_id, frame_key, expires_at_ms)
                    .await
                    .map_err(|e| e.to_string())
            });
        let failures: Vec<String> = join_all(puts)
            .await
            .into_iter()
            .filter_map(Result::err)
            .collect();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{GROKBOT_SUGGESTED_TASKS_ROUTE, GROKBOT_SUGGESTED_TASKS_SERVED_KEY};
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};
    use xai_home_mixer_proto::Frame;

    const NOW_MS: i64 = 1_000_000_000;

    fn query(fatigue_hours: u32) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_fatigue_hours".to_string(),
            &fatigue_hours.to_string(),
        );
        ScoredPostsQuery {
            user_id: 7,
            request_time_ms: NOW_MS,
            params: results.into(),
            ..Default::default()
        }
    }

    fn frame(route: &str) -> FeedItem {
        FeedItem {
            position: 3,
            item: Some(feed_item::Item::Frame(Frame {
                route: route.to_string(),
                ..Default::default()
            })),
        }
    }

    #[test]
    fn a_served_grokbot_frame_is_hidden_for_its_fatigue_window() {
        assert_eq!(
            served_fatigues(&query(6), &[frame(GROKBOT_SUGGESTED_TASKS_ROUTE)]),
            vec![(GROKBOT_SUGGESTED_TASKS_SERVED_KEY, NOW_MS + 6 * 3_600_000)]
        );
    }

    #[test]
    fn nothing_is_written_without_fatigue() {
        assert!(served_fatigues(&query(0), &[frame(GROKBOT_SUGGESTED_TASKS_ROUTE)]).is_empty());
        assert!(served_fatigues(&query(6), &[frame("/cards/soccer/timelineCarousel")]).is_empty());
    }
}
