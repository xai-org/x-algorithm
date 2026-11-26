use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAdsFunnelTrace;
use std::sync::Arc;
use tonic::async_trait;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_ads_proto::funnel_trace::LiFunnelCounters;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient, FUNNEL_TRACE_TOPIC,
};
use xai_candidate_pipeline::component_library::utils::funnel_trace::{
    self, FunnelCounts, DROPPED_BLENDER_KEY, SERVED_KEY,
};
use xai_candidate_pipeline::component_library::utils::{current_timestamp_millis, is_prod};
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::{feed_item, FeedItem};
use xai_recsys_proto::AdIndexInfo;

#[derive(Clone)]
pub struct FunnelTraceSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
    pod_id: String,
}

impl FunnelTraceSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self {
            kafka_client,
            pod_id: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        }
    }

    pub async fn prod() -> Self {
        static FUNNEL_TRACE: tokio::sync::OnceCell<FunnelTraceSideEffect> =
            tokio::sync::OnceCell::const_new();
        FUNNEL_TRACE
            .get_or_init(|| async {
                Self::new(Arc::new(
                    ProdKafkaPublisherClient::new(FUNNEL_TRACE_TOPIC, KafkaCluster::Ads).await,
                ))
            })
            .await
            .clone()
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for FunnelTraceSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnableAdsFunnelTrace)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let counts = input.query.ad_drops.read(|drops| {
            funnel_counts(
                &input.selected_candidates,
                &input.non_selected_candidates,
                drops,
            )
        });
        if counts.is_empty() {
            return Ok(());
        }
        let flush_timestamp_ms = current_timestamp_millis().unwrap_or_default() as i64;
        let messages: Vec<LiFunnelCounters> = counts
            .into_iter()
            .map(|(line_item_id, counts)| LiFunnelCounters {
                line_item_id,
                counts,
                flush_timestamp_ms,
                pod_id: self.pod_id.clone(),
            })
            .collect();
        funnel_trace::publish(self.kafka_client.as_ref(), messages).await
    }
}

fn drop_key(reason: AdDropReason) -> &'static str {
    match reason {
        AdDropReason::TooFewPosts => "dropped.too_few_posts",
        AdDropReason::SpacingCap => "dropped.spacing_cap",
        AdDropReason::NoEligibleNeighbours => "dropped.no_eligible_neighbours",
        AdDropReason::LowRiskNeighbour => "dropped.low_risk_neighbour",
        AdDropReason::ExcludedHandle => "dropped.excluded_handle",
        AdDropReason::ExcludedKeyword => "dropped.excluded_keyword",
        AdDropReason::ServedNeighbour => "dropped.served_neighbour",
        AdDropReason::Truncated => "dropped.truncated",
        AdDropReason::Unspecified => DROPPED_BLENDER_KEY,
    }
}

fn funnel_counts(
    selected: &[FeedItem],
    non_selected: &[FeedItem],
    drops: &[DroppedAd],
) -> FunnelCounts {
    let any_traced = selected
        .iter()
        .chain(non_selected)
        .any(|item| traced_ad(item).is_some())
        || drops.iter().any(|drop| traced_drop(drop).is_some());
    if !any_traced {
        return FunnelCounts::new();
    }
    let reason_key = |impression_id: i64| {
        drops
            .iter()
            .find(|drop| drop.impression_id == impression_id as u64)
            .map_or(DROPPED_BLENDER_KEY, |drop| drop_key(drop.drop_reason()))
    };
    let selector_ad_ids: Vec<u64> = selected
        .iter()
        .chain(non_selected)
        .filter_map(as_ad)
        .map(|ad| ad.impression_id as u64)
        .collect();
    let served = selected
        .iter()
        .filter_map(traced_ad)
        .map(|ad| (ad.line_item_id, SERVED_KEY));
    let dropped_by_selector = non_selected
        .iter()
        .filter_map(traced_ad)
        .map(|ad| (ad.line_item_id, reason_key(ad.impression_id)));
    let dropped_after_selection = drops
        .iter()
        .filter(|drop| !selector_ad_ids.contains(&drop.impression_id))
        .filter_map(|drop| {
            traced_drop(drop).map(|ad| (ad.line_item_id, drop_key(drop.drop_reason())))
        });
    let mut counts = FunnelCounts::new();
    for (line_item_id, key) in served
        .chain(dropped_by_selector)
        .chain(dropped_after_selection)
    {
        *counts
            .entry(line_item_id)
            .or_default()
            .entry(key.to_string())
            .or_default() += 1;
    }
    counts
}

fn as_ad(item: &FeedItem) -> Option<&AdIndexInfo> {
    match &item.item {
        Some(feed_item::Item::Ad(ad)) => Some(ad),
        _ => None,
    }
}

fn traced_ad(item: &FeedItem) -> Option<&AdIndexInfo> {
    as_ad(item).filter(|ad| ad.funnel_traced)
}

fn traced_drop(drop: &DroppedAd) -> Option<&AdIndexInfo> {
    drop.ad_info.as_ref().filter(|ad| ad.funnel_traced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ads::drops::dropped_ad;

    fn ad(line_item_id: i64, impression_id: i64, funnel_traced: bool) -> AdIndexInfo {
        AdIndexInfo {
            line_item_id,
            impression_id,
            funnel_traced,
            ..Default::default()
        }
    }

    fn item(ad: AdIndexInfo) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(ad)),
        }
    }

    #[test]
    fn traced_served_ad_counts_as_served() {
        let counts = funnel_counts(&[item(ad(10, 1, true))], &[], &[]);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&10].len(), 1);
        assert_eq!(counts[&10]["served"], 1);
    }

    #[test]
    fn traced_non_selected_ad_counts_under_its_reason() {
        let drops = vec![dropped_ad(&ad(10, 1, true), AdDropReason::SpacingCap)];
        let counts = funnel_counts(&[], &[item(ad(10, 1, true))], &drops);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&10].len(), 1);
        assert_eq!(counts[&10]["dropped.spacing_cap"], 1);
    }

    #[test]
    fn traced_non_selected_ad_without_reason_counts_as_blender_drop() {
        let counts = funnel_counts(&[], &[item(ad(10, 1, true))], &[]);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&10].len(), 1);
        assert_eq!(counts[&10]["dropped.blender"], 1);
    }

    #[test]
    fn traced_ad_dropped_after_selection_counts_under_its_reason() {
        let drops = vec![dropped_ad(&ad(10, 1, true), AdDropReason::ServedNeighbour)];
        let counts = funnel_counts(&[], &[], &drops);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&10].len(), 1);
        assert_eq!(counts[&10]["dropped.served_neighbour"], 1);
    }

    #[test]
    fn same_line_item_served_and_dropped_keeps_both_keys() {
        let drops = vec![dropped_ad(&ad(10, 2, true), AdDropReason::SpacingCap)];
        let counts = funnel_counts(&[item(ad(10, 1, true))], &[item(ad(10, 2, true))], &drops);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&10].len(), 2);
        assert_eq!(counts[&10]["served"], 1);
        assert_eq!(counts[&10]["dropped.spacing_cap"], 1);
    }

    #[test]
    fn untraced_ads_produce_nothing() {
        let drops = vec![dropped_ad(&ad(10, 3, false), AdDropReason::ServedNeighbour)];
        let counts = funnel_counts(&[item(ad(10, 1, false))], &[item(ad(10, 2, false))], &drops);
        assert!(counts.is_empty());
    }
}
