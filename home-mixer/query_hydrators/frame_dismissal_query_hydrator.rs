use std::sync::Arc;

use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

use crate::frames;
use crate::models::query::ScoredPostsQuery;

pub struct FrameDismissalQueryHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for FrameDismissalQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !frames::fatigue_keys(query).is_empty()
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let frame_dismissals = self
            .strato_client
            .get_user_frame_dismissals(query.user_id, &frames::fatigue_keys(query))
            .await
            .map_err(|e| e.to_string())?;
        Ok(ScoredPostsQuery {
            frame_dismissals: Some(frame_dismissals),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.frame_dismissals = hydrated.frame_dismissals;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{
        GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY, GROKBOT_SUGGESTED_TASKS_SERVED_KEY,
    };
    use std::collections::HashMap;
    use xai_candidate_pipeline::component_library::clients::MockStratoClient;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};

    fn home_query(fatigue_hours: u32) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_jetfuel_frames".to_string(), "true");
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_enabled".to_string(),
            "true",
        );
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_fatigue_hours".to_string(),
            &fatigue_hours.to_string(),
        );
        ScoredPostsQuery {
            user_id: 7,
            params: results.into(),
            ..Default::default()
        }
    }

    fn hydrator(strato: &Arc<MockStratoClient>) -> FrameDismissalQueryHydrator {
        FrameDismissalQueryHydrator {
            strato_client: strato.clone(),
        }
    }

    #[tokio::test]
    async fn reads_the_viewers_dismissal_for_each_fatigued_frame() {
        let strato = Arc::new(MockStratoClient {
            user_frame_dismissals: HashMap::from([(
                GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY.to_string(),
                123,
            )]),
            ..Default::default()
        });
        let hydrated = hydrator(&strato).hydrate(&home_query(6)).await.unwrap();

        assert_eq!(
            hydrated.frame_dismissals,
            Some(HashMap::from([(
                GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY.to_string(),
                123
            )]))
        );
        assert_eq!(
            *strato.user_frame_dismissal_lookups.lock().unwrap(),
            vec![(
                7,
                vec![
                    GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY.to_string(),
                    GROKBOT_SUGGESTED_TASKS_SERVED_KEY.to_string()
                ]
            )]
        );
    }

    #[test]
    fn skipped_when_fatigue_is_off_or_the_frame_is_not_eligible() {
        let strato = Arc::new(MockStratoClient::default());
        assert!(hydrator(&strato).enable(&home_query(6)));
        assert!(!hydrator(&strato).enable(&home_query(0)));
        let scroll_down = ScoredPostsQuery {
            is_bottom_request: true,
            ..home_query(6)
        };
        assert!(!hydrator(&strato).enable(&scroll_down));
    }
}
