use super::util::{is_bsr_low_ad, BlockReason};
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, FeedItem};
use xai_stats_receiver::global_stats_receiver;

pub(crate) struct HealthMetrics {
    pub(crate) enforcement: &'static str,
    pub(crate) slot_outcome: &'static str,
    pub(crate) serving_limitation: &'static str,
}

impl HealthMetrics {
    pub(crate) fn emit_serving_limitation(&self, factor: &'static str) {
        if let Some(receiver) = global_stats_receiver() {
            receiver.incr(self.serving_limitation, &[("factor", factor)], 1);
        }
    }

    pub(crate) fn emit_enforcement(
        &self,
        bsr_drop: u64,
        bsr_ok: u64,
        handle_drop: u64,
        keyword_drop: u64,
    ) {
        let Some(receiver) = global_stats_receiver() else {
            return;
        };
        for (action, count) in [
            ("drop", bsr_drop),
            ("ok", bsr_ok),
            ("handle_drop", handle_drop),
            ("keyword_drop", keyword_drop),
        ] {
            if count > 0 {
                receiver.incr(self.enforcement, &[("action", action)], count);
            }
        }
    }

    pub(crate) fn emit_slot_outcome(&self, filled: u64, unfilled: u64, stuck: &SlotRejections) {
        let Some(receiver) = global_stats_receiver() else {
            return;
        };
        if filled > 0 {
            receiver.incr(self.slot_outcome, &[("outcome", "filled")], filled);
        }
        if unfilled == 0 {
            return;
        }
        receiver.incr(self.slot_outcome, &[("outcome", stuck.stuck_outcome())], 1);
        if unfilled > 1 {
            receiver.incr(
                self.slot_outcome,
                &[("outcome", "unfilled_no_ads")],
                unfilled - 1,
            );
        }
    }

    pub(crate) fn emit_page(&self, offered: usize, items: &[FeedItem], drops: &[DroppedAd]) {
        let mut placed: u64 = 0;
        let mut sensitive_placed: u64 = 0;
        for item in items {
            if let Some(feed_item::Item::Ad(ad)) = &item.item {
                placed += 1;
                sensitive_placed += u64::from(is_bsr_low_ad(ad));
            }
        }
        let mut rejections = SlotRejections::default();
        for drop in drops {
            match drop.drop_reason() {
                AdDropReason::LowRiskNeighbour => rejections.bsr_low += 1,
                AdDropReason::ExcludedHandle => rejections.handle += 1,
                AdDropReason::ExcludedKeyword => rejections.keyword += 1,
                _ => {}
            }
        }
        self.emit_enforcement(
            rejections.bsr_low,
            sensitive_placed,
            rejections.handle,
            rejections.keyword,
        );
        self.emit_slot_outcome(placed, (offered as u64).saturating_sub(placed), &rejections);
    }
}

pub(crate) fn serving_limitation(
    ads_supply: usize,
    from_spacing: usize,
    from_safe: usize,
) -> &'static str {
    let budget = ads_supply.min(from_spacing).min(from_safe);
    if budget == ads_supply {
        "ads_supply"
    } else if budget == from_spacing {
        "spacing"
    } else {
        "safe_posts"
    }
}

#[derive(Default)]
pub(crate) struct SlotRejections {
    pub(crate) bsr_low: u64,
    pub(crate) handle: u64,
    pub(crate) keyword: u64,
}

impl SlotRejections {
    pub(crate) fn count(&mut self, reason: &BlockReason) {
        match reason {
            BlockReason::LowRiskNeighbour => self.bsr_low += 1,
            BlockReason::ExcludedHandle(_) => self.handle += 1,
            BlockReason::ExcludedKeyword(_) => self.keyword += 1,
        }
    }

    pub(crate) fn stuck_outcome(&self) -> &'static str {
        if self.bsr_low == 0 && self.handle == 0 && self.keyword == 0 {
            "unfilled_no_ads"
        } else if self.bsr_low >= self.keyword && self.bsr_low >= self.handle {
            "unfilled_bsr_low"
        } else if self.keyword >= self.handle {
            "unfilled_keyword"
        } else {
            "unfilled_handle"
        }
    }
}
