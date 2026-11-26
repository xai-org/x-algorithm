use super::{
    assert_brand_safe, drop_summary, mixed_feed, organic_moves, served_ads, unplaceable_ad_feed,
};
use crate::ads::slot_spread_blender::blend_with_drops;
use xai_ads_injection_proto::ads_injected_timeline::AdDropReason::{LowRiskNeighbour, SpacingCap};

#[test]
fn serves_the_top_ads_at_prod_positions_on_a_mixed_feed() {
    let (posts, ads) = mixed_feed();
    let mut drops = Vec::new();
    let items = blend_with_drops(posts, ads, 5, &mut drops);

    assert_eq!(
        served_ads(&items),
        [
            (1, 100),
            (6, 101),
            (11, 102),
            (16, 103),
            (21, 104),
            (26, 105),
            (32, 106)
        ]
    );
    assert_eq!(
        drop_summary(&drops),
        [
            (107, SpacingCap, 0),
            (108, SpacingCap, 0),
            (109, SpacingCap, 0)
        ]
    );
    assert_brand_safe(&items);
    assert_eq!(organic_moves(&items), 8);
}

#[test]
fn names_the_first_block_for_an_ad_it_cannot_place() {
    let (posts, ads) = unplaceable_ad_feed();
    let mut drops = Vec::new();
    let items = blend_with_drops(posts, ads, 5, &mut drops);

    assert_eq!(served_ads(&items), [(1, 101)]);
    assert_eq!(drop_summary(&drops), [(100, LowRiskNeighbour, 3)]);
}
