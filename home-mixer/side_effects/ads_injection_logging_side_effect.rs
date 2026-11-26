use crate::ads::drops::logged_ad_info;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{AdsBlenderType, EnableAdsBrandSafetyVerdictV2, EnableAdsInjectionLogging};
use prost::Message;
use std::sync::Arc;
use tonic::async_trait;
use xai_ads_injection_proto::ads_injected_timeline::{
    AdsInjectedTimeline, Counts, DroppedAd, SubscriptionLevel, TimelineEntry,
};
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient, ADS_INJECTION_TOPIC,
};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_core_entities::entities::SubscriptionLevel as CoreSubscriptionLevel;
use xai_home_mixer_proto::{feed_item, FeedItem};
use xai_recsys_proto::ProductSurface;
use xai_served_impression_proto::served_impression::DisplayLocation;

#[derive(Clone)]
pub struct AdsInjectionLoggingSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl AdsInjectionLoggingSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }

    pub async fn prod() -> Self {
        Self::new(Arc::new(
            ProdKafkaPublisherClient::new(ADS_INJECTION_TOPIC, KafkaCluster::Ads).await,
        ))
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for AdsInjectionLoggingSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnableAdsInjectionLogging)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let items = &input.selected_candidates;

        let fetched_posts = count_posts(items) + count_posts(&input.non_selected_candidates);
        let fetched_ads = count_ads(items) + count_ads(&input.non_selected_candidates);

        let request_time_ms = query.request_time_ms;

        let entries: Vec<TimelineEntry> = items
            .iter()
            .enumerate()
            .map(|(pos, item)| build_timeline_entry(item, pos))
            .collect();

        if entries.is_empty() {
            return Ok(());
        }

        let display_location = match query.request_type {
            RequestType::RankedFollowing | RequestType::Following => {
                DisplayLocation::TimelineHomeReverseChron
            }
            _ => DisplayLocation::TimelineHome,
        };
        let product_surface = match query.request_type {
            RequestType::RankedFollowing => ProductSurface::HomeTimelineRankedFollowing,
            RequestType::Following => ProductSurface::HomeTimelineLatest,
            _ => ProductSurface::HomeTimelineRanking,
        };

        let timeline = AdsInjectedTimeline {
            user_id: query.user_id,
            entries,
            request_time_ms,
            display_location: display_location.into(),
            country_code: query.country_code.clone(),
            request_id: query.request_id,
            subscription_level: query
                .subscription_level
                .map(sub_level_to_proto)
                .unwrap_or(SubscriptionLevel::LevelUnspecified)
                .into(),
            client_app_id: query.client_app_id as i64,
            counts: Some(Counts {
                retrieved_posts: fetched_posts as i32,
                retrieved_ads: fetched_ads as i32,
                response_posts: count_posts(items) as i32,
                response_ads: count_ads(items) as i32,
            }),
            ip_address: query.ip_address.clone(),
            user_agent: query.user_agent.clone(),
            ads_injection_experiment_bucket: String::new(),
            ddg_experiment_bucket: query
                .params
                .experiment_buckets(EnableAdsBrandSafetyVerdictV2)
                .first()
                .map(|b| format!("{}:{}", b.experiment, b.bucket))
                .unwrap_or_default(),
            product_surface: product_surface.into(),
            ads_blender: match query.request_type {
                RequestType::Following => "following".to_string(),
                _ => query.params.get(AdsBlenderType),
            },
            dropped_ads: query.ad_drops.read(<[DroppedAd]>::to_vec),
        };

        let bytes = timeline.encode_to_vec();
        self.kafka_client
            .send(&bytes)
            .await
            .map_err(|e| format!("Ads Kafka publish failed: {e}"))?;
        Ok(())
    }
}

fn build_timeline_entry(item: &FeedItem, position: usize) -> TimelineEntry {
    match &item.item {
        Some(feed_item::Item::Post(post)) => TimelineEntry {
            tweet_id: post.tweet_id,
            author_id: post.author_id,
            position: position as i32,
            promoted: false,
            impression_id: 0,
            brand_safety_verdict: post.brand_safety_verdict,
            ad_info: None,
            safety_labels: post.safety_label_types.clone(),
            nsfw_author_ads: post.nsfw_author_ads,
        },
        Some(feed_item::Item::Ad(ad)) => TimelineEntry {
            tweet_id: ad.post_id as u64,
            author_id: ad.author_id as u64,
            position: position as i32,
            promoted: true,
            impression_id: ad.impression_id as u64,
            brand_safety_verdict: 0,
            ad_info: Some(logged_ad_info(ad)),
            safety_labels: vec![],
            nsfw_author_ads: false,
        },
        Some(feed_item::Item::WhoToFollow(_))
        | Some(feed_item::Item::Prompt(_))
        | Some(feed_item::Item::PushToHome(_))
        | Some(feed_item::Item::Frame(_))
        | Some(feed_item::Item::FeedSurvey(_))
        | Some(feed_item::Item::VideoCarousel(_))
        | None => TimelineEntry {
            position: position as i32,
            ..Default::default()
        },
    }
}

fn sub_level_to_proto(level: CoreSubscriptionLevel) -> SubscriptionLevel {
    match level {
        CoreSubscriptionLevel::Basic => SubscriptionLevel::Basic,
        CoreSubscriptionLevel::Premium => SubscriptionLevel::Premium,
        CoreSubscriptionLevel::PremiumPlus => SubscriptionLevel::PremiumPlus,
    }
}

fn count_posts(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
        .count()
}

fn count_ads(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo};

    #[test]
    fn ad_entry_logs_line_item_campaign_and_objective() {
        let ad = AdIndexInfo {
            post_id: 10,
            line_item_id: 57018926,
            author_id: 20,
            action_probabilities: vec![0.1, 0.2],
            account_id: 18000000000,
            campaign_id: 42769584,
            line_item_objective: 5,
            impression_id: 30,
            insert_position: 4,
            ad_adjacency_control: Some(AdAdjacencyControl {
                keywords: vec!["kw".to_string()],
                brand_safety_risk: 3,
                handles: vec![7],
            }),
            ..Default::default()
        };
        let item = FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(ad)),
        };

        let entry = build_timeline_entry(&item, 2);

        assert_eq!(
            entry.ad_info,
            Some(AdIndexInfo {
                account_id: 18000000000,
                line_item_id: 57018926,
                campaign_id: 42769584,
                line_item_objective: 5,
                insert_position: 4,
                ad_adjacency_control: Some(AdAdjacencyControl {
                    brand_safety_risk: 3,
                    ..Default::default()
                }),
                ..Default::default()
            })
        );
    }
}
