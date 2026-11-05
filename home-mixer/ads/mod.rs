pub mod drops;
mod following_ad_blender;
mod health_metrics;
mod least_movement_blender;
mod multi_risk_blender;
mod neighbours;
mod partition_organic_blender;
mod safe_gap_blender;
mod slot_spread_blender;
#[cfg(test)]
mod tests;
mod time_gap_blender;
pub(crate) mod util;

pub use following_ad_blender::FollowingAdBlender;
pub use least_movement_blender::LeastMovementAdsBlender;
pub use multi_risk_blender::MultiRiskAdsBlender;
pub use partition_organic_blender::PartitionOrganicAdsBlender;
pub use safe_gap_blender::SafeGapAdsBlender;
pub use slot_spread_blender::SlotSpreadAdsBlender;
pub use time_gap_blender::{TimeGapAdsBlender, TimeGapConfig};

use util::{record_ad_risk_stats, record_post_verdict_stats};
use xai_ads_injection_proto::ads_injected_timeline::DroppedAd;
use xai_home_mixer_proto::{FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

pub trait AdsBlender: Send + Sync {
    fn blend_inner(&self, scored_posts: Vec<ScoredPost>, ads: Vec<AdIndexInfo>) -> Vec<FeedItem> {
        self.blend_inner_with_drops(scored_posts, ads, &mut Vec::new())
    }

    fn blend_inner_with_drops(
        &self,
        scored_posts: Vec<ScoredPost>,
        ads: Vec<AdIndexInfo>,
        _drops: &mut Vec<DroppedAd>,
    ) -> Vec<FeedItem> {
        self.blend_inner(scored_posts, ads)
    }

    fn blend(
        &self,
        scored_posts: Vec<ScoredPost>,
        ads: Vec<AdIndexInfo>,
        drops: &mut Vec<DroppedAd>,
    ) -> Vec<FeedItem> {
        record_post_verdict_stats(&scored_posts);
        record_ad_risk_stats(&ads);
        self.blend_inner_with_drops(scored_posts, ads, drops)
    }
}
