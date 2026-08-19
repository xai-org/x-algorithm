use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::EnableUrtMigrationComponents;
use crate::util::tweet_type_metrics::{TWEET_TYPE_PREDICATES, VIDEO, bitset_get};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;

use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    CLIENT_EVENT_TOPIC, KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient,
};
use xai_candidate_pipeline::component_library::utils::client_utils::{
    ClientPlatform, RequestContext,
};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::{FeedItem, ScoredPost, ServedType, feed_item};
use xai_x_thrift::log_event::{EventNamespace, LogBase, LogEvent};
use xai_x_thrift::serialize_binary;

pub struct ClientEventsKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl ClientEventsKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }

    pub async fn prod() -> Self {
        Self::new(Arc::new(
            ProdKafkaPublisherClient::new(CLIENT_EVENT_TOPIC, KafkaCluster::ClientEvents).await,
        ))
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for ClientEventsKafkaSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnableUrtMigrationComponents)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let items = &input.selected_candidates;

        let posts: Vec<&ScoredPost> = items
            .iter()
            .filter_map(|i| match &i.item {
                Some(feed_item::Item::Post(p)) => Some(p),
                _ => None,
            })
            .collect();
        let ad_count = items
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
            .count() as i64;
        let wtf_count = items
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::WhoToFollow(_))))
            .count() as i64;
        let post_count = posts.iter().map(|p| conversation_post_count(p)).sum();

        let base = ClientEventParams {
            query,
            client_name: ClientPlatform::from_app_id(query.client_app_id).client_name(),
            section: section_for(query.request_type),
            component: None,
            element: None,
            action: "served_tweets",
            value: 0,
        };

        let mut events = Vec::new();
        events.extend(build_served_events(&base, post_count, ad_count, wtf_count));
        events.extend(build_tweet_type_events(&base, &posts));
        events.extend(build_served_type_events(&base, &posts));
        events.extend(build_video_events(&base, &posts));
        events.extend(build_empty_timeline_events(&base, post_count));
        events.extend(build_query_events(&base, post_count));

        let payloads: Vec<Vec<u8>> = events
            .iter()
            .filter(|e| e.event_value.unwrap_or(0) > 0)
            .map(|e| serialize_binary(e).map_err(|e| format!("Thrift serialization failed: {e}")))
            .collect::<Result<_, _>>()?;

        let futs: Vec<_> = payloads
            .iter()
            .map(|bytes| self.kafka_client.send(bytes))
            .collect();
        let results = futures::future::join_all(futs).await;

        if let Some(err) = results.into_iter().find_map(|r| r.err()) {
            return Err(format!("Client-event Kafka publish failed: {err}"));
        }

        Ok(())
    }
}

fn conversation_post_count(post: &ScoredPost) -> i64 {
    let mut count = 1;
    if let Some(&root) = post.ancestors.iter().min() {
        if !post.tombstone_ancestor_ids.contains(&root) {
            count += 1;
        }
        if let Some(&parent) = post.ancestors.iter().max()
            && parent != root
            && !post.tombstone_ancestor_ids.contains(&parent)
        {
            count += 1;
        }
    }
    count
}

fn section_for(request_type: RequestType) -> &'static str {
    match request_type {
        RequestType::RankedFollowing => "ranked_following",
        RequestType::Following => "latest",
        _ => "home",
    }
}

fn build_served_events(
    base: &ClientEventParams,
    post_count: i64,
    ad_count: i64,
    wtf_count: i64,
) -> Vec<LogEvent> {
    vec![
        build_log_event(&ClientEventParams {
            value: post_count + ad_count,
            ..*base
        }),
        build_log_event(&ClientEventParams {
            component: Some("injected"),
            value: post_count,
            ..*base
        }),
        build_log_event(&ClientEventParams {
            component: Some("promoted"),
            value: ad_count,
            ..*base
        }),
        build_log_event(&ClientEventParams {
            component: Some("who_to_follow"),
            action: "served_users",
            value: wtf_count,
            ..*base
        }),
    ]
}

fn build_tweet_type_events(base: &ClientEventParams, posts: &[&ScoredPost]) -> Vec<LogEvent> {
    TWEET_TYPE_PREDICATES
        .iter()
        .map(|&(bit_idx, name)| {
            let count = posts
                .iter()
                .filter(|p| bitset_get(&p.tweet_type_metrics, bit_idx))
                .count();
            build_log_event(&ClientEventParams {
                component: Some("injected"),
                element: Some(name),
                value: count as i64,
                ..*base
            })
        })
        .collect()
}

fn build_served_type_events(base: &ClientEventParams, posts: &[&ScoredPost]) -> Vec<LogEvent> {
    let mut counts: HashMap<i32, i64> = HashMap::new();
    for post in posts {
        *counts.entry(post.served_type).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(st, count)| {
            let name = ServedType::try_from(st)
                .map(|t| t.as_str_name().to_ascii_lowercase())
                .unwrap_or_else(|_| format!("unknown_{st}"));
            base_build_log_event(base, Some(name), None, base.action, count)
        })
        .collect()
}

fn build_video_events(base: &ClientEventParams, posts: &[&ScoredPost]) -> Vec<LogEvent> {
    let num_videos = posts
        .iter()
        .filter(|p| bitset_get(&p.tweet_type_metrics, VIDEO))
        .count();
    vec![build_log_event(&ClientEventParams {
        component: Some("with_video_duration"),
        element: Some("num_videos"),
        value: num_videos as i64,
        ..*base
    })]
}

fn build_empty_timeline_events(base: &ClientEventParams, post_count: i64) -> Vec<LogEvent> {
    if post_count > 0 {
        return Vec::new();
    }
    vec![build_log_event(&ClientEventParams {
        action: "empty",
        element: Some("served_non_promoted_tweet"),
        ..*base
    })]
}

fn build_query_events(base: &ClientEventParams, post_count: i64) -> Vec<LogEvent> {
    let query = base.query;
    let ctx = RequestContext::parse(&query.request_context);

    let query_predicates: Vec<(&str, bool)> = vec![
        ("request", true),
        ("get_older", query.is_bottom_request),
        ("get_newer", query.is_top_request),
        ("get_initial", query.cursor.is_none()),
        ("pull_to_refresh", ctx == RequestContext::PullToRefresh),
        ("request_context_launch", ctx == RequestContext::Launch),
        (
            "request_context_foreground",
            ctx == RequestContext::Foreground,
        ),
    ];

    let size_predicates: Vec<(&str, bool)> = vec![
        ("size_is_empty", post_count == 0),
        ("size_at_most_5", post_count <= 5),
        ("size_at_most_10", post_count <= 10),
        ("size_at_most_35", post_count <= 35),
    ];

    let mut events = Vec::new();
    for &(query_name, query_match) in &query_predicates {
        if !query_match {
            continue;
        }
        for &(size_name, size_match) in &size_predicates {
            if !size_match {
                continue;
            }
            events.push(build_log_event(&ClientEventParams {
                component: Some(size_name),
                action: query_name,
                element: None,
                value: 1,
                ..*base
            }));
        }
    }
    events
}

#[derive(Clone, Copy)]
struct ClientEventParams<'a> {
    query: &'a ScoredPostsQuery,
    client_name: &'a str,
    section: &'a str,
    component: Option<&'a str>,
    element: Option<&'a str>,
    action: &'a str,
    value: i64,
}

fn build_log_event(p: &ClientEventParams) -> LogEvent {
    base_build_log_event(
        p,
        p.component.map(String::from),
        p.element.map(String::from),
        p.action,
        p.value,
    )
}

fn base_build_log_event(
    p: &ClientEventParams,
    component: Option<String>,
    element: Option<String>,
    action: &str,
    value: i64,
) -> LogEvent {
    let now_ms = p.query.request_time_ms;

    let event_name = [
        p.client_name,
        "home",
        p.section,
        component.as_deref().unwrap_or(""),
        action,
    ]
    .join(":");

    LogEvent {
        log_base: Some(LogBase {
            transaction_id: String::new(),
            ip_address: p.query.ip_address.clone(),
            user_id: Some(p.query.user_id as i64),
            timestamp: now_ms,
            language: Some(p.query.language_code.clone()),
            client_app_id: Some(p.query.client_app_id as i64),
            device_id: Some(p.query.device_id.clone()),
            country: Some(p.query.country_code.clone()),
            timezone: None,
        }),
        event_name: Some(event_name),
        event_value: Some(value),
        event_details: None,
        event_namespace: Some(EventNamespace {
            client: Some(p.client_name.to_string()),
            page: Some("home".to_string()),
            section: Some(p.section.to_string()),
            component,
            element,
            action: Some(action.to_string()),
        }),
        notification_details: None,
    }
}
