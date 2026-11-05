use crate::ads::following_ad_blender::*;
use crate::ads::util::MIN_POSTS_FOR_ADS;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo, BrandSafetyRiskLevel};

fn make_post(tweet_id: u64) -> ScoredPost {
    make_post_with(tweet_id, BrandSafetyVerdict::Safe, 0, "")
}

fn make_post_with(
    tweet_id: u64,
    verdict: BrandSafetyVerdict,
    author_id: u64,
    text: &str,
) -> ScoredPost {
    ScoredPost {
        tweet_id,
        brand_safety_verdict: verdict.into(),
        author_id,
        tweet_text: text.to_string(),
        ..Default::default()
    }
}

fn make_reply(tweet_id: u64, ancestors: &[u64], verdict: BrandSafetyVerdict) -> ScoredPost {
    ScoredPost {
        tweet_id,
        in_reply_to_tweet_id: ancestors.first().copied().unwrap_or(0),
        ancestors: ancestors.to_vec(),
        brand_safety_verdict: verdict.into(),
        ..Default::default()
    }
}

fn make_ad(post_id: i64) -> AdIndexInfo {
    make_ad_at(post_id, 0)
}

fn make_ad_at(post_id: i64, insert_position: i32) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        insert_position,
        ..Default::default()
    }
}

fn make_ad_with(
    post_id: i64,
    insert_position: i32,
    risk: Option<BrandSafetyRiskLevel>,
    handles: Vec<i64>,
    keywords: Vec<&str>,
) -> AdIndexInfo {
    let control = if risk.is_none() && handles.is_empty() && keywords.is_empty() {
        None
    } else {
        Some(AdAdjacencyControl {
            brand_safety_risk: risk.unwrap_or(BrandSafetyRiskLevel::BsrUnknown).into(),
            handles,
            keywords: keywords.into_iter().map(str::to_string).collect(),
        })
    };
    AdIndexInfo {
        post_id,
        insert_position,
        ad_adjacency_control: control,
        ..Default::default()
    }
}

fn is_ad(item: &FeedItem) -> bool {
    matches!(item.item, Some(feed_item::Item::Ad(_)))
}

fn organic_ids(items: &[FeedItem]) -> Vec<u64> {
    items
        .iter()
        .filter_map(|item| match &item.item {
            Some(feed_item::Item::Post(p)) => Some(p.tweet_id),
            _ => None,
        })
        .collect()
}

fn ad_ids(items: &[FeedItem]) -> Vec<i64> {
    items
        .iter()
        .filter_map(|item| match &item.item {
            Some(feed_item::Item::Ad(a)) => Some(a.post_id),
            _ => None,
        })
        .collect()
}

fn gaps_between_ads(items: &[FeedItem]) -> Vec<usize> {
    let ad_positions: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| is_ad(item))
        .map(|(i, _)| i)
        .collect();
    let mut prev = -1isize;
    ad_positions
        .into_iter()
        .map(|pos| {
            let gap = (pos as isize - prev - 1) as usize;
            prev = pos as isize;
            gap
        })
        .collect()
}

fn blend(
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    spacing: usize,
    result_size: usize,
) -> Vec<FeedItem> {
    blend_with_spacing(
        posts,
        ads,
        MIN_POSTS_FOR_ADS,
        result_size,
        spacing,
        &mut Vec::new(),
    )
}

fn blend_recording_drops(
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    spacing: usize,
    result_size: usize,
) -> (Vec<i64>, Vec<DroppedAd>) {
    let mut drops = Vec::new();
    let out = blend_with_spacing(
        posts,
        ads,
        MIN_POSTS_FOR_ADS,
        result_size,
        spacing,
        &mut drops,
    );
    (ad_ids(&out), drops)
}

fn drop_reasons(drops: &[DroppedAd]) -> Vec<(u64, AdDropReason)> {
    drops
        .iter()
        .map(|drop| (drop.tweet_id, drop.drop_reason()))
        .collect()
}

#[test]
fn records_adjacency_drops_with_blocking_post() {
    let posts: Vec<_> = (1..=12)
        .map(|i| make_post_with(i, BrandSafetyVerdict::LowRisk, 777, "crypto scandal"))
        .collect();
    let ads = vec![
        make_ad_with(101, 0, None, vec![777], vec![]),
        make_ad_with(102, 0, None, vec![], vec!["crypto"]),
        make_ad_with(103, 0, Some(BrandSafetyRiskLevel::BsrLow), vec![], vec![]),
        make_ad(104),
    ];
    let (served, drops) = blend_recording_drops(posts, ads, 3, 50);
    assert_eq!(served, vec![104]);
    assert_eq!(
        drop_reasons(&drops),
        vec![
            (101, AdDropReason::ExcludedHandle),
            (102, AdDropReason::ExcludedKeyword),
            (103, AdDropReason::LowRiskNeighbour),
        ]
    );
    assert_eq!(drops[0].matched_user_id, 777);
    assert_eq!(drops[1].matched_keyword, "crypto");
    assert!(drops.iter().all(|drop| drop.blocking_tweet_id == 11));
}

#[test]
fn records_no_room_drops() {
    let few: Vec<_> = (1..=4).map(make_post).collect();
    let (served, drops) = blend_recording_drops(few, vec![make_ad(101)], 3, 50);
    assert!(served.is_empty());
    assert_eq!(drop_reasons(&drops), vec![(101, AdDropReason::TooFewPosts)]);

    let posts: Vec<_> = (1..=8).map(make_post).collect();
    let ads: Vec<_> = (101..=104).map(make_ad).collect();
    let (served, drops) = blend_recording_drops(posts, ads, 3, 50);
    assert_eq!(served, vec![101, 102, 103]);
    assert_eq!(drop_reasons(&drops), vec![(104, AdDropReason::SpacingCap)]);

    let mut posts: Vec<_> = (1..=8).map(make_post).collect();
    for post in &mut posts[2..] {
        post.brand_safety_verdict = BrandSafetyVerdict::MediumRisk.into();
    }
    let (served, drops) = blend_recording_drops(posts, vec![make_ad(101), make_ad(102)], 3, 50);
    assert_eq!(served, vec![101]);
    assert_eq!(
        drop_reasons(&drops),
        vec![(102, AdDropReason::NoEligibleNeighbours)]
    );

    let posts: Vec<_> = (1..=20).map(make_post).collect();
    let ads: Vec<_> = (101..=105).map(make_ad).collect();
    let (served, drops) = blend_recording_drops(posts, ads, 2, 10);
    assert_eq!(served, vec![101, 102, 103]);
    assert_eq!(
        drop_reasons(&drops),
        vec![
            (104, AdDropReason::Truncated),
            (105, AdDropReason::Truncated)
        ]
    );
}

#[test]
fn blends_with_spacing_and_never_first_or_last() {
    let posts: Vec<_> = (1..=30).map(make_post).collect();
    let ads: Vec<_> = (101..=104).map(make_ad).collect();
    let result = blend(posts, ads, 3, 50);

    assert_eq!(ad_ids(&result).len(), 4);
    assert_eq!(organic_ids(&result), (1..=30).collect::<Vec<_>>());
    assert!(!is_ad(&result[0]));
    assert!(!is_ad(result.last().unwrap()));
    let gaps = gaps_between_ads(&result);
    assert_eq!(gaps[0], 1);
    assert!(gaps[1..].iter().all(|&g| g == 3));
}

#[test]
fn spacing_and_first_position_from_insert_positions() {
    let premium = vec![make_ad_at(101, 1), make_ad_at(102, 9), make_ad_at(103, 17)];
    assert_eq!(compute_spacing(&premium), 8);
    assert_eq!(compute_first_position(&premium, 8), 1);
    assert_eq!(compute_spacing(&[make_ad(101)]), 3);
    assert_eq!(compute_first_position(&[make_ad_at(101, 3)], 4), 3);

    let posts: Vec<_> = (1..=30).map(make_post).collect();
    let result = blend_impl(posts, premium, MIN_POSTS_FOR_ADS, 50);
    assert_eq!(result.iter().position(is_ad), Some(1));
    assert!(gaps_between_ads(&result)[1..].iter().all(|&g| g == 8));
}

#[test]
fn medium_risk_blocks_and_can_widen_gap() {
    let mut posts: Vec<_> = (1..=20).map(make_post).collect();
    posts[4] = make_post_with(5, BrandSafetyVerdict::MediumRisk, 0, "");
    posts[5] = make_post_with(6, BrandSafetyVerdict::MediumRisk, 0, "");
    let ads: Vec<_> = (101..=104).map(make_ad).collect();
    let result = blend(posts, ads, 3, 50);

    let gaps = gaps_between_ads(&result);
    assert_eq!(gaps[0], 1);
    assert!(gaps.iter().skip(1).all(|&g| g >= 3));
    assert!(gaps.iter().skip(1).any(|&g| g > 3));
    let ad_pos = result.iter().position(is_ad).unwrap();
    for neighbour in [&result[ad_pos - 1], &result[ad_pos + 1]] {
        match &neighbour.item {
            Some(feed_item::Item::Post(p)) => {
                assert_eq!(p.brand_safety_verdict(), BrandSafetyVerdict::Safe);
            }
            _ => panic!("ad neighbour must be a post"),
        }
    }
}

#[test]
fn defers_incompatible_ads() {
    let mut posts = vec![
        make_post_with(1, BrandSafetyVerdict::Safe, 777, ""),
        make_post_with(2, BrandSafetyVerdict::LowRisk, 0, "crypto scandal"),
        make_post_with(3, BrandSafetyVerdict::LowRisk, 0, ""),
    ];
    posts.extend((4..=12).map(make_post));

    let bsr_low = make_ad_with(101, 0, Some(BrandSafetyRiskLevel::BsrLow), vec![], vec![]);
    let handle_ad = make_ad_with(102, 0, None, vec![777], vec![]);
    let keyword_ad = make_ad_with(103, 0, None, vec![], vec!["crypto"]);
    let clean = make_ad(104);
    let result = blend(posts, vec![bsr_low, handle_ad, keyword_ad, clean], 2, 50);

    let ids = ad_ids(&result);
    assert_eq!(ids.len(), 4);
    assert_eq!(ids[0], 104);
}

#[test]
fn truncates_without_trailing_ad() {
    let posts: Vec<_> = (1..=20).map(make_post).collect();
    let ads: Vec<_> = (101..=105).map(make_ad).collect();
    let result = blend(posts, ads, 2, 10);

    assert!(result.len() <= 10);
    assert!(!is_ad(result.last().unwrap()));
}

#[test]
fn conversation_units_and_min_posts() {
    let short = vec![
        make_post(10),
        make_reply(11, &[10], BrandSafetyVerdict::Safe),
        make_post(20),
        make_reply(21, &[20], BrandSafetyVerdict::Safe),
        make_post(30),
        make_reply(31, &[30], BrandSafetyVerdict::Safe),
    ];
    assert!(ad_ids(&blend(short, vec![make_ad(201)], 3, 50)).is_empty());

    let posts = vec![
        make_post(1),
        make_reply(101, &[100], BrandSafetyVerdict::Safe),
        make_post(100),
        make_post(2),
        make_post(3),
        make_post(4),
        make_post(5),
        make_post(6),
    ];
    assert_eq!(
        blend_units(&posts),
        vec![0..1, 1..3, 3..4, 4..5, 5..6, 6..7, 7..8]
    );
    let result = blend(posts, vec![make_ad(201)], 2, 50);
    let i101 = result
        .iter()
        .position(|item| matches!(&item.item, Some(feed_item::Item::Post(p)) if p.tweet_id == 101))
        .unwrap();
    let i100 = result
        .iter()
        .position(|item| matches!(&item.item, Some(feed_item::Item::Post(p)) if p.tweet_id == 100))
        .unwrap();
    assert_eq!(i100, i101 + 1);
    assert!(!is_ad(&result[i101 + 1]));
}

#[test]
fn empty_or_short_feeds_skip_ads() {
    let few: Vec<_> = (1..=4).map(make_post).collect();
    assert!(ad_ids(&blend(few, vec![make_ad(101)], 3, 50)).is_empty());

    let posts: Vec<_> = (1..=10).map(make_post).collect();
    assert!(ad_ids(&blend(posts, vec![], 3, 50)).is_empty());
}
