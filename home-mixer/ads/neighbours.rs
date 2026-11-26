use super::drops::{adjacency_drop, dropped_ad, truncate_recording_drops};
use super::util::{
    ad_keywords, handle_block, is_bsr_high_ad, is_bsr_low_ad, is_high_risk, is_medium_risk,
    low_risk_block, tokenize_tweet_text, Block, BlockReason,
};
use rustc_hash::FxBuildHasher;
use std::cell::OnceCell;
use std::hash::BuildHasher;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_post_text::TokenSequence;
use xai_recsys_proto::AdIndexInfo;

pub(crate) enum Fit {
    Compatible,
    Risk,
    Blocked(Block),
}

pub(crate) struct AdRules<'a> {
    pub(crate) ad: &'a AdIndexInfo,
    pub(crate) sensitive: bool,
    pub(crate) bsr_high: bool,
    keywords: OnceCell<Option<KeywordIndex<'a>>>,
}

impl<'a> AdRules<'a> {
    pub(crate) fn new(ad: &'a AdIndexInfo) -> Self {
        Self {
            ad,
            sensitive: is_bsr_low_ad(ad),
            bsr_high: is_bsr_high_ad(ad),
            keywords: OnceCell::new(),
        }
    }

    fn keywords(&self) -> Option<&KeywordIndex<'a>> {
        self.keywords
            .get_or_init(|| KeywordIndex::new(self.ad))
            .as_ref()
    }
}

struct KeywordIndex<'a> {
    keywords: Vec<(&'a str, TokenSequence)>,
    by_first_token: Vec<(u64, usize)>,
}

impl<'a> KeywordIndex<'a> {
    fn new(ad: &'a AdIndexInfo) -> Option<Self> {
        let keywords = ad_keywords(ad)?;
        let mut by_first_token: Vec<(u64, usize)> = keywords
            .iter()
            .enumerate()
            .map(|(i, (_, tokens))| (token_hash(tokens.token(0)), i))
            .collect();
        by_first_token.sort_unstable();
        Some(Self {
            keywords,
            by_first_token,
        })
    }

    fn first_match(&self, tokens: &TokenSequence) -> Option<&'a str> {
        let mut best: Option<usize> = None;
        for offset in 0..tokens.len() {
            let token = tokens.token(offset);
            let unprefixed = token.strip_prefix(['#', '@']);
            for key in std::iter::once(token).chain(unprefixed) {
                let hash = token_hash(key);
                let start = self.by_first_token.partition_point(|&(h, _)| h < hash);
                let found = self.by_first_token[start..]
                    .iter()
                    .take_while(|&&(h, i)| h == hash && best.is_none_or(|b| i < b))
                    .map(|&(_, i)| i)
                    .find(|&i| matches_at(tokens, offset, &self.keywords[i].1));
                if found.is_some() {
                    best = found;
                }
            }
        }
        best.map(|i| self.keywords[i].0)
    }
}

fn token_hash(token: &str) -> u64 {
    FxBuildHasher.hash_one(token)
}

fn matches_at(tokens: &TokenSequence, offset: usize, keyword: &TokenSequence) -> bool {
    offset + keyword.len() <= tokens.len()
        && (0..keyword.len())
            .all(|j| TokenSequence::token_matches(tokens.token(offset + j), keyword.token(j)))
}

pub(crate) struct NeighbourChecker<'a> {
    posts: &'a [ScoredPost],
    tokens: Vec<Option<TokenSequence>>,
}

impl<'a> NeighbourChecker<'a> {
    pub(crate) fn new(posts: &'a [ScoredPost]) -> Self {
        Self {
            posts,
            tokens: posts.iter().map(|_| None).collect(),
        }
    }

    pub(crate) fn fit(&mut self, rules: &AdRules, post: usize) -> Fit {
        let candidate = &self.posts[post];
        if is_high_risk(candidate) || (is_medium_risk(candidate) && !rules.bsr_high) {
            return Fit::Risk;
        }
        if let Some(block) = low_risk_block(rules.ad, Some(candidate), None)
            .or_else(|| handle_block(rules.ad, Some(candidate), None))
        {
            return Fit::Blocked(block);
        }
        let Some(keywords) = rules.keywords() else {
            return Fit::Compatible;
        };
        let tokens =
            self.tokens[post].get_or_insert_with(|| tokenize_tweet_text(&candidate.tweet_text));
        match keywords.first_match(tokens) {
            Some(keyword) => Fit::Blocked(Block {
                tweet_id: candidate.tweet_id,
                reason: BlockReason::ExcludedKeyword(keyword.to_string()),
            }),
            None => Fit::Compatible,
        }
    }

    pub(crate) fn compatible(&mut self, rules: &AdRules, post: usize) -> bool {
        matches!(self.fit(rules, post), Fit::Compatible)
    }

    pub(crate) fn class(&self, post: usize) -> PostClass {
        PostClass::of(&self.posts[post])
    }

    pub(crate) fn tweet_id(&self, post: usize) -> u64 {
        self.posts[post].tweet_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostClass {
    Safe,
    Low,
    Medium,
    High,
}

impl PostClass {
    pub(crate) fn of(post: &ScoredPost) -> Self {
        match post.brand_safety_verdict() {
            BrandSafetyVerdict::HighRisk => Self::High,
            BrandSafetyVerdict::MediumRisk => Self::Medium,
            BrandSafetyVerdict::LowRisk => Self::Low,
            _ => Self::Safe,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ClassCounts {
    pub(crate) safe: usize,
    pub(crate) low: usize,
    pub(crate) medium: usize,
}

impl ClassCounts {
    pub(crate) fn of(posts: &[ScoredPost]) -> Self {
        let mut counts = Self::default();
        for post in posts {
            match PostClass::of(post) {
                PostClass::Safe => counts.safe += 1,
                PostClass::Low => counts.low += 1,
                PostClass::Medium => counts.medium += 1,
                PostClass::High => {}
            }
        }
        counts
    }
}

pub(crate) fn assemble(
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    order: &[usize],
    ad_before: &[(usize, usize)],
    result_size: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    let mut posts: Vec<Option<ScoredPost>> = posts.into_iter().map(Some).collect();
    let mut ads: Vec<Option<AdIndexInfo>> = ads.into_iter().map(Some).collect();
    let mut ad_at: Vec<Option<usize>> = vec![None; order.len()];
    for &(index, ad) in ad_before {
        match ad_at.get_mut(index) {
            Some(slot) => *slot = Some(ad),
            None => {
                if let Some(cut) = ads[ad].take() {
                    drops.push(dropped_ad(&cut, AdDropReason::Truncated));
                }
            }
        }
    }
    let mut items: Vec<FeedItem> = Vec::with_capacity(order.len() + ad_before.len());
    for (index, &post) in order.iter().enumerate() {
        if let Some(ad) = ad_at[index].and_then(|ad| ads[ad].take()) {
            items.push(FeedItem {
                position: 0,
                item: Some(feed_item::Item::Ad(ad)),
            });
        }
        if let Some(post) = posts[post].take() {
            items.push(FeedItem {
                position: 0,
                item: Some(feed_item::Item::Post(post)),
            });
        }
    }
    truncate_recording_drops(&mut items, result_size, drops);
    for (i, item) in items.iter_mut().enumerate() {
        item.position = i as i32;
    }
    items
}

pub(crate) fn page_capacity(spacing: usize, posts: usize, result_size: usize) -> usize {
    if posts < 2 || result_size < 3 || spacing == 0 {
        return 0;
    }
    ((posts - 2) / spacing + 1).min((result_size - 2 + spacing) / (spacing + 1))
}

pub(crate) fn unplaceable_drop(ad: &AdIndexInfo, fits: impl IntoIterator<Item = Fit>) -> DroppedAd {
    let mut eligible = 0;
    let mut first: [Option<Block>; 3] = [None, None, None];
    for fit in fits {
        match fit {
            Fit::Risk => continue,
            Fit::Compatible => {}
            Fit::Blocked(block) => {
                let rank = match block.reason {
                    BlockReason::LowRiskNeighbour => 0,
                    BlockReason::ExcludedHandle(_) => 1,
                    BlockReason::ExcludedKeyword(_) => 2,
                };
                first[rank].get_or_insert(block);
            }
        }
        eligible += 1;
    }
    match first.into_iter().flatten().next() {
        Some(block) if eligible >= 2 => adjacency_drop(ad, block),
        _ => dropped_ad(ad, AdDropReason::NoEligibleNeighbours),
    }
}

pub(crate) fn safe_slot_estimate(posts: &[ScoredPost], ads: &[AdIndexInfo]) -> usize {
    let counts = ClassCounts::of(posts);
    let normal_ads = ads.iter().filter(|ad| !is_bsr_low_ad(ad)).count();
    let high_ads = ads.iter().filter(|ad| is_bsr_high_ad(ad)).count();
    ((counts.safe + counts.low) / 2).min(counts.safe / 2 + normal_ads)
        + high_ads.min(counts.medium / 2)
}
