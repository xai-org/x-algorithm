use std::sync::Arc;

use futures::future::join_all;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::{JetfuelClient, JetfuelView};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::FeedItem;

use crate::frames::{self, Fetch, FrameDef};
use crate::models::query::ScoredPostsQuery;

pub struct JetfuelDirectFrameSource {
    pub jetfuel_client: Arc<dyn JetfuelClient>,
}

fn via_jetfuel(frame: &FrameDef) -> bool {
    matches!(frame.fetch, Fetch::Jetfuel { .. })
}

fn view(query: &ScoredPostsQuery) -> JetfuelView {
    JetfuelView {
        client_app_id: i64::from(query.client_app_id),
        client_version: query.client_version.clone(),
        locale: query.language_code.clone(),
        country_code: query.country_code.clone(),
        user_id: (query.user_id != 0).then_some(query.user_id),
        ..Default::default()
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for JetfuelDirectFrameSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        frames::eligible(query).any(via_jetfuel)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let view = view(query);
        let view = &view;
        let fetched = join_all(frames::requests(query, via_jetfuel).into_iter().map(
            |request| async move {
                let Fetch::Jetfuel { height } = request.frame.fetch else {
                    return (request, Ok(None));
                };
                let payload = self
                    .jetfuel_client
                    .fetch_payload(&request.fetch_route, view)
                    .await
                    .map(|payload| payload.map(|payload| (payload, height)))
                    .map_err(|e| e.to_string());
                (request, payload)
            },
        ))
        .await;
        frames::collect("JetfuelDirectFrameSource", fetched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::GROKBOT_SUGGESTED_TASKS_ROUTE;
    use xai_candidate_pipeline::component_library::clients::MockJetfuelClient;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};
    use xai_home_mixer_proto::{feed_item, Frame};

    const VIEWER: u64 = 42;

    fn home_query(grokbot_enabled: bool) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_jetfuel_frames".to_string(), "true");
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_enabled".to_string(),
            if grokbot_enabled { "true" } else { "false" },
        );
        ScoredPostsQuery {
            user_id: VIEWER,
            params: results.into(),
            ..Default::default()
        }
    }

    fn source(client: &Arc<MockJetfuelClient>) -> JetfuelDirectFrameSource {
        JetfuelDirectFrameSource {
            jetfuel_client: client.clone(),
        }
    }

    fn frames_of(items: &[FeedItem]) -> Vec<&Frame> {
        items
            .iter()
            .filter_map(|item| match &item.item {
                Some(feed_item::Item::Frame(frame)) => Some(frame),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn fetches_the_grokbot_frame_from_jetfuel_as_the_viewer() {
        let client = Arc::new(MockJetfuelClient {
            payload: Some("RENDERED_BINARY".to_string()),
            ..Default::default()
        });
        let items = source(&client).source(&home_query(true)).await.unwrap();

        assert_eq!(
            client.requests(),
            vec![(GROKBOT_SUGGESTED_TASKS_ROUTE.to_string(), Some(VIEWER))]
        );
        let frames = frames_of(&items);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].route, GROKBOT_SUGGESTED_TASKS_ROUTE);
        assert_eq!(frames[0].payload, "RENDERED_BINARY");
        assert!(!frames[0].is_dismissible);
    }

    #[test]
    fn disabled_when_no_direct_frame_is_eligible() {
        let client = Arc::new(MockJetfuelClient::default());
        assert!(!source(&client).enable(&home_query(false)));
        assert!(source(&client).enable(&home_query(true)));
    }

    #[tokio::test]
    async fn a_missing_payload_emits_nothing_and_a_failure_is_an_error() {
        let missing = Arc::new(MockJetfuelClient::default());
        assert!(source(&missing)
            .source(&home_query(true))
            .await
            .unwrap()
            .is_empty());

        let failing = Arc::new(MockJetfuelClient {
            fail: true,
            ..Default::default()
        });
        assert!(source(&failing).source(&home_query(true)).await.is_err());
    }
}
