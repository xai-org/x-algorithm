use std::sync::LazyLock;
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_post_text::{TokenSequence, TweetTokenizer};
use xai_recsys_proto::{AdIndexInfo, BrandSafetyRiskLevel};
use xai_stats_receiver::global_stats_receiver;

static TWEET_TOKENIZER: LazyLock<TweetTokenizer> = LazyLock::new(TweetTokenizer::new);

pub(crate) const MIN_POSTS_FOR_ADS: usize = 5;

pub(crate) const MIN_REQUESTED_GAP: usize = 3;

pub(crate) const DEFAULT_SPACING: AdSpacing = AdSpacing {
    requested: 3,
    min: 2,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdSpacing {
    pub(crate) requested: usize,
    pub(crate) min: usize,
}

pub(crate) fn has_avoid(post: &ScoredPost) -> bool {
    matches!(
        post.brand_safety_verdict(),
        BrandSafetyVerdict::MediumRisk | BrandSafetyVerdict::HighRisk
    )
}

pub(crate) fn is_high_risk(post: &ScoredPost) -> bool {
    post.brand_safety_verdict() == BrandSafetyVerdict::HighRisk
}

pub(crate) fn is_medium_risk(post: &ScoredPost) -> bool {
    post.brand_safety_verdict() == BrandSafetyVerdict::MediumRisk
}

pub(crate) fn find_safe_gaps(scored_posts: &[ScoredPost]) -> Vec<usize> {
    let n = scored_posts.len();
    let mut safe = Vec::new();
    for g in 1..n {
        if has_avoid(&scored_posts[g - 1]) {
            continue;
        }
        if g < n && has_avoid(&scored_posts[g]) {
            continue;
        }
        safe.push(g);
    }
    safe
}

pub(crate) fn compute_spacing(ads: &[AdIndexInfo]) -> AdSpacing {
    if ads.len() < 2 {
        return DEFAULT_SPACING;
    }

    let mut positions: Vec<i32> = ads.iter().take(4).map(|a| a.insert_position).collect();
    positions.sort_unstable();

    let min_diff = positions
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .filter(|&d| d > 0)
        .min();

    match min_diff {
        Some(requested) if requested >= MIN_REQUESTED_GAP => AdSpacing {
            requested,
            min: requested.div_ceil(2),
        },
        _ => DEFAULT_SPACING,
    }
}

pub(crate) fn is_bsr_low_ad(ad: &AdIndexInfo) -> bool {
    let risk = ad
        .ad_adjacency_control
        .as_ref()
        .map(|c| c.brand_safety_risk())
        .unwrap_or(BrandSafetyRiskLevel::BsrUnknown);
    matches!(
        risk,
        BrandSafetyRiskLevel::BsrLow | BrandSafetyRiskLevel::BsrIas
    )
}

pub(crate) fn is_bsr_high_ad(ad: &AdIndexInfo) -> bool {
    ad.ad_adjacency_control
        .as_ref()
        .map(|c| c.brand_safety_risk())
        .unwrap_or(BrandSafetyRiskLevel::BsrUnknown)
        == BrandSafetyRiskLevel::BsrHigh
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BlockReason {
    LowRiskNeighbour,
    ExcludedHandle(u64),
    ExcludedKeyword(String),
}

#[derive(Debug)]
pub(crate) struct Block {
    pub(crate) tweet_id: u64,
    pub(crate) reason: BlockReason,
}

pub(crate) type SlotTokens = Option<[Option<TokenSequence>; 2]>;

pub(crate) fn low_risk_block(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> Option<Block> {
    if !is_bsr_low_ad(ad) {
        return None;
    }
    [above, below]
        .into_iter()
        .flatten()
        .find(|p| p.brand_safety_verdict() == BrandSafetyVerdict::LowRisk)
        .map(|post| Block {
            tweet_id: post.tweet_id,
            reason: BlockReason::LowRiskNeighbour,
        })
}

pub(crate) fn handle_block(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> Option<Block> {
    let handles = match ad.ad_adjacency_control.as_ref() {
        Some(ctrl) if !ctrl.handles.is_empty() => &ctrl.handles,
        _ => return None,
    };
    [above, below].into_iter().flatten().find_map(|post| {
        [post.author_id, post.retweeted_user_id, post.quoted_user_id]
            .iter()
            .chain(&post.ancestor_users)
            .find(|&&id| id != 0 && handles.contains(&(id as i64)))
            .map(|&id| Block {
                tweet_id: post.tweet_id,
                reason: BlockReason::ExcludedHandle(id),
            })
    })
}

pub(crate) fn keyword_block(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
    slot_tokens: &mut SlotTokens,
) -> Option<Block> {
    let keywords = ad_keywords(ad)?;
    let tokens = slot_tokens.get_or_insert_with(|| {
        [above, below].map(|post| post.map(|p| tokenize_tweet_text(&p.tweet_text)))
    });
    [above, below]
        .into_iter()
        .zip(tokens.iter())
        .find_map(|(post, tokens)| {
            let (post, tokens) = (post?, tokens.as_ref()?);
            keywords
                .iter()
                .find(|(_, keyword)| tokens.contains_keyword_sequence(keyword))
                .map(|&(keyword, _)| Block {
                    tweet_id: post.tweet_id,
                    reason: BlockReason::ExcludedKeyword(keyword.to_string()),
                })
        })
}

pub(crate) fn should_drop_bsr_low(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> bool {
    low_risk_block(ad, above, below).is_some()
}

pub(crate) fn should_drop_handle(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> bool {
    handle_block(ad, above, below).is_some()
}

pub(crate) fn should_drop_keyword(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> bool {
    keyword_block(ad, above, below, &mut None).is_some()
}

pub(crate) fn tokenize_tweet_text(text: &str) -> TokenSequence {
    TWEET_TOKENIZER.tokenize(text)
}

pub(crate) fn tokenize_ad_keywords(ad: &AdIndexInfo) -> Option<Vec<TokenSequence>> {
    ad_keywords(ad).map(|keywords| keywords.into_iter().map(|(_, tokens)| tokens).collect())
}

pub(crate) fn ad_keywords(ad: &AdIndexInfo) -> Option<Vec<(&str, TokenSequence)>> {
    let keywords: Vec<_> = ad
        .ad_adjacency_control
        .as_ref()?
        .keywords
        .iter()
        .map(|kw| (kw.as_str(), TWEET_TOKENIZER.tokenize(kw)))
        .filter(|(_, tokens)| !tokens.is_empty())
        .collect();
    (!keywords.is_empty()).then_some(keywords)
}

pub(crate) fn tokens_match_any_keyword(
    tweet_tokens: &TokenSequence,
    keywords: &[TokenSequence],
) -> bool {
    if tweet_tokens.is_empty() {
        return false;
    }
    keywords
        .iter()
        .any(|kw_tokens| tweet_tokens.contains_keyword_sequence(kw_tokens))
}

pub(crate) fn posts_to_feed_items(scored_posts: Vec<ScoredPost>) -> Vec<FeedItem> {
    scored_posts
        .into_iter()
        .enumerate()
        .map(|(i, post)| FeedItem {
            position: i as i32,
            item: Some(feed_item::Item::Post(post)),
        })
        .collect()
}

pub(crate) fn interleave_and_finalize(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    placements: &[usize],
    result_size: usize,
) -> Vec<FeedItem> {
    let n = scored_posts.len();
    let mut items: Vec<FeedItem> = Vec::with_capacity(n + placements.len());
    let mut ads_iter = ads.into_iter();
    let mut pi = 0;

    for (i, post) in scored_posts.into_iter().enumerate() {
        if pi < placements.len() && placements[pi] == i {
            items.push(FeedItem {
                position: 0,
                item: Some(feed_item::Item::Ad(ads_iter.next().unwrap())),
            });
            pi += 1;
        }
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(post)),
        });
    }

    items.truncate(result_size);
    if matches!(items.last(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        items.pop();
    }

    for (i, item) in items.iter_mut().enumerate() {
        item.position = i as i32;
    }

    items
}

const VERDICT_METRIC: &str = "AdsBlender.post_brand_safety_verdict";
const RISK_METRIC: &str = "AdsBlender.ad_brand_safety_risk";

pub(crate) fn record_post_verdict_stats(posts: &[ScoredPost]) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };

    for post in posts {
        let label = post.brand_safety_verdict().as_str_name();
        receiver.incr(VERDICT_METRIC, &[("verdict", label)], 1);
    }
}

pub(crate) fn record_ad_risk_stats(ads: &[AdIndexInfo]) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };

    for ad in ads {
        let risk_level = ad
            .ad_adjacency_control
            .as_ref()
            .map(|c| c.brand_safety_risk())
            .unwrap_or(BrandSafetyRiskLevel::BsrUnknown);

        receiver.incr(RISK_METRIC, &[("risk", risk_level.as_str_name())], 1);
    }
}
