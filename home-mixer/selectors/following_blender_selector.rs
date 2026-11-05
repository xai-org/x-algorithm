use super::build_non_selected;
use crate::ads::{AdsBlender, FollowingAdBlender};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    FOLLOWING_MAX_RESULT_SIZE, FOLLOWING_WHO_TO_FOLLOW_MIN_POSTS, PROMPTS_POSITION,
    WHO_TO_FOLLOW_POSITION,
};
use xai_candidate_pipeline::selector::{SelectResult, Selector};
use xai_home_mixer_proto::{feed_item, FeedItem, Prompt, ScoredPost, WhoToFollowModule};
use xai_recsys_proto::AdIndexInfo;

pub struct FollowingBlenderSelector;

impl FollowingBlenderSelector {
    pub fn new() -> Self {
        Self
    }
}

impl Selector<ScoredPostsQuery, FeedItem> for FollowingBlenderSelector {
    fn select(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<FeedItem>,
    ) -> SelectResult<FeedItem> {
        let PartitionedFeedItems {
            posts,
            ads,
            wtf_modules,
            prompts,
        } = partition_feed_items(candidates);

        let input_post_count = posts.len();
        let input_ads = ads.clone();

        let blender = FollowingAdBlender {
            result_size: FOLLOWING_MAX_RESULT_SIZE,
        };
        let mut drops = Vec::new();
        let mut blended = blender.blend(posts, ads, &mut drops);
        query.ad_drops.extend(drops);

        insert_prompts(&mut blended, prompts);
        if input_post_count > FOLLOWING_WHO_TO_FOLLOW_MIN_POSTS {
            insert_who_to_follow(&mut blended, wtf_modules);
        }

        let output_post_count = blended
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
            .count();
        let dropped_posts = input_post_count.saturating_sub(output_post_count);
        let non_selected = build_non_selected(dropped_posts, input_ads, &blended);

        SelectResult {
            selected: blended,
            non_selected,
        }
    }

    fn score(&self, _candidate: &FeedItem) -> f64 {
        0.0
    }
}

fn insert_prompts(blended: &mut Vec<FeedItem>, prompts: Vec<Prompt>) {
    for (i, prompt) in prompts.into_iter().enumerate() {
        blended.insert(
            i,
            FeedItem {
                position: PROMPTS_POSITION,
                item: Some(feed_item::Item::Prompt(prompt)),
            },
        );
    }
}

fn insert_who_to_follow(blended: &mut Vec<FeedItem>, wtf_modules: Vec<WhoToFollowModule>) {
    let Some(wtf) = wtf_modules.into_iter().next() else {
        return;
    };
    let insert_idx = WHO_TO_FOLLOW_POSITION.saturating_sub(1).min(blended.len());
    blended.insert(
        insert_idx,
        FeedItem {
            position: WHO_TO_FOLLOW_POSITION as i32,
            item: Some(feed_item::Item::WhoToFollow(wtf)),
        },
    );
}

struct PartitionedFeedItems {
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    wtf_modules: Vec<WhoToFollowModule>,
    prompts: Vec<Prompt>,
}

fn partition_feed_items(items: Vec<FeedItem>) -> PartitionedFeedItems {
    let mut posts = Vec::new();
    let mut ads = Vec::new();
    let mut wtf_modules = Vec::new();
    let mut prompts = Vec::new();
    for item in items {
        match item.item {
            Some(feed_item::Item::Post(post)) => posts.push(post),
            Some(feed_item::Item::Ad(ad)) => ads.push(ad),
            Some(feed_item::Item::WhoToFollow(wtf)) => wtf_modules.push(wtf),
            Some(feed_item::Item::Prompt(prompt)) => prompts.push(prompt),
            _ => {}
        }
    }
    PartitionedFeedItems {
        posts,
        ads,
        wtf_modules,
        prompts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_home_mixer_proto::BrandSafetyVerdict;
    use xai_recsys_proto::{AdAdjacencyControl, BrandSafetyRiskLevel};

    fn low_risk_post(tweet_id: u64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost {
                tweet_id,
                brand_safety_verdict: BrandSafetyVerdict::LowRisk.into(),
                ..Default::default()
            })),
        }
    }

    fn ad(post_id: i64, impression_id: i64, risk: BrandSafetyRiskLevel) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo {
                post_id,
                impression_id,
                ad_adjacency_control: Some(AdAdjacencyControl {
                    brand_safety_risk: risk.into(),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        }
    }

    fn ad_ids(items: &[FeedItem]) -> Vec<i64> {
        items
            .iter()
            .filter_map(|item| match &item.item {
                Some(feed_item::Item::Ad(ad)) => Some(ad.post_id),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn non_selected_carries_the_earlier_ad_the_blender_skipped() {
        let mut candidates: Vec<FeedItem> = (1..=12).map(low_risk_post).collect();
        candidates.push(ad(101, 1, BrandSafetyRiskLevel::BsrLow));
        candidates.push(ad(102, 2, BrandSafetyRiskLevel::BsrNormal));
        let result =
            FollowingBlenderSelector::new().select(&ScoredPostsQuery::default(), candidates);
        assert_eq!(ad_ids(&result.selected), vec![102]);
        assert_eq!(ad_ids(&result.non_selected), vec![101]);
    }
}
