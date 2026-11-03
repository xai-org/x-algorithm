use super::util::{Block, BlockReason};
use std::sync::{Mutex, PoisonError};
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, FeedItem};
use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo};

#[derive(Debug, Default)]
pub struct AdDrops(Mutex<Vec<DroppedAd>>);

impl AdDrops {
    pub fn extend(&self, drops: impl IntoIterator<Item = DroppedAd>) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(drops);
    }

    pub fn read<R>(&self, f: impl FnOnce(&[DroppedAd]) -> R) -> R {
        f(&self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

pub(crate) fn logged_ad_info(ad: &AdIndexInfo) -> AdIndexInfo {
    AdIndexInfo {
        account_id: ad.account_id,
        line_item_id: ad.line_item_id,
        campaign_id: ad.campaign_id,
        line_item_objective: ad.line_item_objective,
        insert_position: ad.insert_position,
        ad_adjacency_control: ad
            .ad_adjacency_control
            .as_ref()
            .map(|c| AdAdjacencyControl {
                brand_safety_risk: c.brand_safety_risk,
                ..Default::default()
            }),
        ..Default::default()
    }
}

pub(crate) fn dropped_ad(ad: &AdIndexInfo, reason: AdDropReason) -> DroppedAd {
    DroppedAd {
        tweet_id: ad.post_id as u64,
        impression_id: ad.impression_id as u64,
        drop_reason: reason.into(),
        ad_info: Some(AdIndexInfo {
            funnel_traced: ad.funnel_traced,
            ..logged_ad_info(ad)
        }),
        ..Default::default()
    }
}

pub(crate) fn blocked_ad(ad: &AdIndexInfo, reason: AdDropReason, tweet_id: u64) -> DroppedAd {
    DroppedAd {
        blocking_tweet_id: tweet_id,
        ..dropped_ad(ad, reason)
    }
}

pub(crate) fn adjacency_drop(ad: &AdIndexInfo, block: Block) -> DroppedAd {
    let blocked = |reason| blocked_ad(ad, reason, block.tweet_id);
    match block.reason {
        BlockReason::LowRiskNeighbour => blocked(AdDropReason::LowRiskNeighbour),
        BlockReason::ExcludedHandle(user_id) => DroppedAd {
            matched_user_id: user_id,
            ..blocked(AdDropReason::ExcludedHandle)
        },
        BlockReason::ExcludedKeyword(keyword) => DroppedAd {
            matched_keyword: keyword,
            ..blocked(AdDropReason::ExcludedKeyword)
        },
    }
}

pub(crate) fn truncate_recording_drops(
    items: &mut Vec<FeedItem>,
    len: usize,
    drops: &mut Vec<DroppedAd>,
) {
    let mut removed: Vec<FeedItem> = items.split_off(len.min(items.len()));
    if matches!(items.last(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        removed.extend(items.pop());
    }
    drops.extend(removed.iter().filter_map(|item| match &item.item {
        Some(feed_item::Item::Ad(ad)) => Some(dropped_ad(ad, AdDropReason::Truncated)),
        _ => None,
    }));
}
