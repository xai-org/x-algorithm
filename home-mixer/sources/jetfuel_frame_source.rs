use std::sync::Arc;

use futures::future::join_all;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::FeedItem;
use xai_twittercontext_proto::GetTwitterContextViewer;

use crate::frames::{self, Fetch, FrameDef};
use crate::models::query::ScoredPostsQuery;

pub struct JetfuelFrameSource {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

fn via_strato(frame: &FrameDef) -> bool {
    frame.fetch == Fetch::Strato
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for JetfuelFrameSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        frames::eligible(query).any(via_strato)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let viewer = query.get_viewer();
        let viewer = viewer.as_ref();
        let fetched = join_all(frames::requests(query, via_strato).into_iter().map(
            |request| async move {
                let payload = self
                    .strato_client
                    .get_jetfuel_payload(&request.fetch_route, viewer)
                    .await
                    .map_err(|e| e.to_string());
                (request, payload)
            },
        ))
        .await;
        frames::collect("JetfuelFrameSource", fetched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{BASEBALL_TOPIC_ID, MLB_TOPIC_ID, NFL_TOPIC_ID, SOCCER_TOPIC_ID};
    use xai_candidate_pipeline::component_library::clients::MockStratoClient;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};
    use xai_home_mixer_proto::{feed_item, Frame};
    use xai_x_thrift::served_history::{
        EntryWithItemIds, ItemIds, RequestType as ServedRequestType, ServedHistory,
    };

    fn params(soccer_enabled: bool) -> Params {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_jetfuel_frames".to_string(), "true");
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_soccer_enabled".to_string(),
            if soccer_enabled { "true" } else { "false" },
        );
        results.into()
    }

    fn query(soccer_enabled: bool, request_topics: Vec<i64>) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_id: 1,
            topic_ids: request_topics,
            params: params(soccer_enabled),
            ..Default::default()
        }
    }

    fn query_with(overrides: &[(&str, bool)], topic_ids: Vec<i64>) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_jetfuel_frames".to_string(), "true");
        for (key, value) in overrides {
            results.override_fs(key.to_string(), if *value { "true" } else { "false" });
        }
        ScoredPostsQuery {
            user_id: 1,
            topic_ids,
            params: results.into(),
            ..Default::default()
        }
    }

    fn nfl_query(nfl_enabled: bool) -> ScoredPostsQuery {
        query_with(
            &[("rust_home_mixer_jetfuel_frame_nfl_enabled", nfl_enabled)],
            vec![NFL_TOPIC_ID],
        )
    }

    fn source(payload: Option<(String, i32)>) -> JetfuelFrameSource {
        JetfuelFrameSource {
            strato_client: Arc::new(MockStratoClient {
                jetfuel_payload: payload,
                ..Default::default()
            }),
        }
    }

    fn frames_of(items: &[FeedItem]) -> Vec<&Frame> {
        items
            .iter()
            .filter_map(|i| match &i.item {
                Some(feed_item::Item::Frame(f)) => Some(f),
                _ => None,
            })
            .collect()
    }

    async fn emitted_routes(query: &ScoredPostsQuery) -> Vec<String> {
        let items = source(Some(("RENDERED_BINARY".to_string(), 220)))
            .source(query)
            .await
            .unwrap_or_default();
        frames_of(&items).iter().map(|f| f.route.clone()).collect()
    }

    #[test]
    fn enabled_when_configured_topic_is_requested() {
        let q = query(true, vec![SOCCER_TOPIC_ID]);
        assert!(source(None).enable(&q));
    }

    #[test]
    fn frames_fetched_directly_from_jetfuel_are_left_to_their_own_source() {
        let q = query_with(
            &[(
                "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_enabled",
                true,
            )],
            vec![],
        );
        assert!(frames::eligible(&q).next().is_some());
        assert!(!source(None).enable(&q));
    }

    #[test]
    fn disabled_when_frame_fs_off() {
        let q = query(false, vec![SOCCER_TOPIC_ID]);
        assert!(!source(None).enable(&q));
    }

    #[test]
    fn disabled_when_topic_not_requested() {
        let q = query(true, vec![42]);
        assert!(!source(None).enable(&q));
    }

    #[test]
    fn disabled_on_scroll_down_when_first_response_only() {
        let scroll_down = ScoredPostsQuery {
            is_bottom_request: true,
            ..query(true, vec![SOCCER_TOPIC_ID])
        };
        assert!(!source(None).enable(&scroll_down));
    }

    #[test]
    fn nfl_timeline_disabled_by_default_emits_nothing() {
        assert!(!source(None).enable(&nfl_query(false)));
    }

    #[tokio::test]
    async fn nfl_timeline_emits_the_pinned_card() {
        let q = query_with(
            &[
                ("rust_home_mixer_jetfuel_frame_nfl_enabled", true),
                (
                    "rust_home_mixer_jetfuel_frame_nfl_timeline_pinned_enabled",
                    true,
                ),
            ],
            vec![NFL_TOPIC_ID],
        );
        assert_eq!(emitted_routes(&q).await, vec!["/cards/nfl/timeline/pinned"]);
    }

    #[tokio::test]
    async fn the_rotating_card_is_fetched_once_per_occurrence() {
        let q = query_with(
            &[
                ("rust_home_mixer_jetfuel_frame_nfl_enabled", true),
                (
                    "rust_home_mixer_jetfuel_frame_nfl_timeline_rotation_enabled",
                    true,
                ),
            ],
            vec![NFL_TOPIC_ID],
        );
        let strato = Arc::new(MockStratoClient {
            jetfuel_payload: Some(("RENDERED_BINARY".to_string(), 220)),
            ..Default::default()
        });
        let items = JetfuelFrameSource {
            strato_client: strato.clone(),
        }
        .source(&q)
        .await
        .unwrap_or_default();

        let frames = frames_of(&items);
        assert!(frames.len() > 1, "expected several rotating occurrences");
        assert!(frames.iter().all(|f| f.route == "/cards/nfl/timeline/card"));
        assert_eq!(
            frames.iter().map(|f| f.occurrence).collect::<Vec<_>>(),
            (0..frames.len() as u32).collect::<Vec<_>>()
        );
        assert_eq!(strato.call_count(), frames.len());
    }

    #[tokio::test]
    async fn only_the_rotating_card_carries_a_rotation_index() {
        let q = query_with(
            &[
                ("rust_home_mixer_jetfuel_frame_nfl_enabled", true),
                (
                    "rust_home_mixer_jetfuel_frame_nfl_timeline_pinned_enabled",
                    true,
                ),
                (
                    "rust_home_mixer_jetfuel_frame_nfl_timeline_rotation_enabled",
                    true,
                ),
            ],
            vec![NFL_TOPIC_ID],
        );
        let strato = Arc::new(MockStratoClient {
            jetfuel_payload: Some(("RENDERED_BINARY".to_string(), 220)),
            ..Default::default()
        });
        JetfuelFrameSource {
            strato_client: strato.clone(),
        }
        .source(&q)
        .await
        .unwrap_or_default();

        let routes = strato.jetfuel_routes();
        let (pinned, rotating): (Vec<&String>, Vec<&String>) = routes
            .iter()
            .partition(|r| r.starts_with("/cards/nfl/timeline/pinned"));

        assert_eq!(pinned, vec!["/cards/nfl/timeline/pinned"]);
        assert!(rotating.len() > 1, "expected several rotating fetches");

        let indices: Vec<u32> = rotating
            .iter()
            .map(|r| {
                let (path, index) = r.split_once("?i=").unwrap_or_else(|| {
                    panic!("rotating fetch {r} is missing its ?i= rotation index")
                });
                assert_eq!(path, "/cards/nfl/timeline/card");
                index.parse().expect("rotation index should be a number")
            })
            .collect();

        let base = *indices.iter().min().expect("at least one rotating fetch");
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (base..base + indices.len() as u32).collect::<Vec<_>>(),
            "each occurrence should ask jetfuel for the next card in the rotation"
        );
    }

    #[tokio::test]
    async fn the_pinned_and_rotating_cards_are_emitted_together() {
        let q = query_with(
            &[
                ("rust_home_mixer_jetfuel_frame_nfl_enabled", true),
                (
                    "rust_home_mixer_jetfuel_frame_nfl_timeline_pinned_enabled",
                    true,
                ),
                (
                    "rust_home_mixer_jetfuel_frame_nfl_timeline_rotation_enabled",
                    true,
                ),
            ],
            vec![NFL_TOPIC_ID],
        );
        let routes = emitted_routes(&q).await;
        assert_eq!(
            routes.first().map(String::as_str),
            Some("/cards/nfl/timeline/pinned")
        );
        assert!(routes.contains(&"/cards/nfl/timeline/card".to_string()));
    }

    const NFL_TIMELINE_ON: &[(&str, bool)] = &[
        ("rust_home_mixer_jetfuel_frame_nfl_enabled", true),
        (
            "rust_home_mixer_jetfuel_frame_nfl_timeline_pinned_enabled",
            true,
        ),
        (
            "rust_home_mixer_jetfuel_frame_nfl_timeline_rotation_enabled",
            true,
        ),
    ];

    fn nfl_timeline_page(cards_already_served: &[u32]) -> ScoredPostsQuery {
        let entries = cards_already_served
            .iter()
            .map(|index| EntryWithItemIds {
                entity_type: frames::FRAME_ENTITY_TYPE,
                sort_index: None,
                size: None,
                item_ids: Some(vec![ItemIds {
                    tweet_id: None,
                    source_tweet_id: None,
                    quote_tweet_id: None,
                    source_author_id: None,
                    quote_author_id: None,
                    in_reply_to_tweet_id: None,
                    in_reply_to_author_id: None,
                    article_id: None,
                    tweet_score: None,
                    entry_id_to_replace: None,
                    user_id: None,
                    impression_id: Some(format!("/cards/nfl/timeline/card?i={index}")),
                }]),
            })
            .collect();

        ScoredPostsQuery {
            served_history: vec![ServedHistory {
                request_type: ServedRequestType::OLDER,
                served_id: None,
                served_time_ms: None,
                entries,
            }],
            ..query_with(NFL_TIMELINE_ON, vec![NFL_TOPIC_ID])
        }
    }

    async fn rotation_indices(query: &ScoredPostsQuery) -> Vec<u32> {
        let strato = Arc::new(MockStratoClient {
            jetfuel_payload: Some(("RENDERED_BINARY".to_string(), 220)),
            ..Default::default()
        });
        JetfuelFrameSource {
            strato_client: strato.clone(),
        }
        .source(query)
        .await
        .unwrap_or_default();

        let mut indices: Vec<u32> = strato
            .jetfuel_routes()
            .iter()
            .filter_map(|route| route.split_once("?i=").map(|(_, index)| index.to_string()))
            .map(|index| index.parse().expect("rotation index should be a number"))
            .collect();
        indices.sort_unstable();
        indices
    }

    #[tokio::test]
    async fn the_rotation_picks_up_after_the_last_card_already_served() {
        let indices = rotation_indices(&nfl_timeline_page(&[40, 41, 42])).await;
        assert_eq!(indices[0], 43);
    }

    #[tokio::test]
    async fn three_pages_of_scrolling_never_ask_for_the_same_card_twice() {
        let mut served: Vec<u32> = Vec::new();
        let mut asked: Vec<u32> = Vec::new();

        for _ in 0..3 {
            let page = rotation_indices(&nfl_timeline_page(&served)).await;
            served.extend(&page);
            asked.extend(&page);
        }

        let start = asked[0];
        assert_eq!(
            asked,
            (start..start + asked.len() as u32).collect::<Vec<_>>(),
            "each page should continue the rotation: {asked:?}"
        );
    }

    #[tokio::test]
    async fn each_frame_carries_the_route_it_was_fetched_with() {
        let items = source(Some(("RENDERED_BINARY".to_string(), 220)))
            .source(&nfl_timeline_page(&[]))
            .await
            .unwrap_or_default();

        for frame in frames_of(&items) {
            let (path, params) = frame
                .fetch_route
                .split_once('?')
                .unwrap_or((frame.fetch_route.as_str(), ""));
            assert_eq!(path, frame.route);
            if frame.route == "/cards/nfl/timeline/card" {
                assert!(
                    params.starts_with("i="),
                    "rotating card should record its ?i= index, got {:?}",
                    frame.fetch_route
                );
            } else {
                assert!(params.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn baseball_topic_emits_mlb_carousel() {
        let q = query_with(
            &[("rust_home_mixer_jetfuel_frame_mlb_carousel_enabled", true)],
            vec![BASEBALL_TOPIC_ID],
        );
        assert_eq!(
            emitted_routes(&q).await,
            vec!["/cards/mlb/timelineCarousel"]
        );
    }

    const EVERY_FRAME_ON: &[(&str, bool)] = &[
        ("rust_home_mixer_jetfuel_frame_soccer_enabled", true),
        ("rust_home_mixer_jetfuel_frame_nfl_enabled", true),
        (
            "rust_home_mixer_jetfuel_frame_nfl_timeline_pinned_enabled",
            true,
        ),
        (
            "rust_home_mixer_jetfuel_frame_nfl_timeline_rotation_enabled",
            true,
        ),
        ("rust_home_mixer_jetfuel_frame_mlb_carousel_enabled", true),
    ];

    const EVERY_TOPIC: &[i64] = &[
        SOCCER_TOPIC_ID,
        NFL_TOPIC_ID,
        MLB_TOPIC_ID,
        BASEBALL_TOPIC_ID,
    ];

    fn query_with_master_switch(enabled: bool, topic_id: i64) -> ScoredPostsQuery {
        let mut overrides = EVERY_FRAME_ON.to_vec();
        overrides.push(("rust_home_mixer_enable_jetfuel_frames", enabled));
        query_with(&overrides, vec![topic_id])
    }

    #[tokio::test]
    async fn the_master_switch_stops_every_jetfuel_fetch() {
        for topic_id in EVERY_TOPIC {
            let strato = Arc::new(MockStratoClient {
                jetfuel_payload: Some(("RENDERED_BINARY".to_string(), 220)),
                ..Default::default()
            });
            let items = JetfuelFrameSource {
                strato_client: strato.clone(),
            }
            .source(&query_with_master_switch(false, *topic_id))
            .await
            .unwrap_or_default();
            assert!(items.is_empty());
            assert_eq!(strato.call_count(), 0);
        }
    }

    #[tokio::test]
    async fn source_emits_rendered_payload() {
        let q = query(true, vec![SOCCER_TOPIC_ID]);
        let items = source(Some(("RENDERED_BINARY".to_string(), 220)))
            .source(&q)
            .await
            .unwrap_or_default();
        let frames = frames_of(&items);
        assert_eq!(frames.len(), 1);
        for frame in frames {
            assert_eq!(frame.payload, "RENDERED_BINARY");
            assert_eq!(frame.height, 220);
            assert_eq!(frame.route, "/cards/soccer/timelineCarousel");
        }
    }

    #[tokio::test]
    async fn source_emits_nothing_when_payload_missing() {
        let q = query(true, vec![SOCCER_TOPIC_ID]);
        assert!(source(None).source(&q).await.unwrap_or_default().is_empty());
    }

    #[tokio::test]
    async fn source_emits_nothing_when_disabled() {
        let q = query(false, vec![SOCCER_TOPIC_ID]);
        let items = source(Some(("RENDERED_BINARY".to_string(), 220)))
            .source(&q)
            .await
            .unwrap_or_default();
        assert!(items.is_empty());
    }
}
