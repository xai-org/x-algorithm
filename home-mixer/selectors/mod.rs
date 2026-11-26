mod blender_selector;
mod following_blender_selector;
mod passthrough_selector;
mod top_k_score_selector;

pub use blender_selector::BlenderSelector;
pub use following_blender_selector::FollowingBlenderSelector;
pub use passthrough_selector::PassthroughSelector;
pub use top_k_score_selector::TopKScoreSelector;

use xai_candidate_pipeline::selector::SelectResult;
use xai_home_mixer_proto::{FeedItem, ScoredPost, feed_item};
use xai_recsys_proto::AdIndexInfo;

fn build_non_selected(
    dropped_posts: usize,
    mut dropped_ads: Vec<AdIndexInfo>,
    selected: &[FeedItem],
) -> Vec<FeedItem> {
    for item in selected {
        if let Some(feed_item::Item::Ad(ad)) = &item.item
            && let Some(index) = dropped_ads
                .iter()
                .position(|dropped| dropped.impression_id == ad.impression_id)
        {
            dropped_ads.remove(index);
        }
    }
    let mut non_selected = Vec::with_capacity(dropped_posts + dropped_ads.len());
    for _ in 0..dropped_posts {
        non_selected.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost::default())),
        });
    }
    non_selected.extend(dropped_ads.into_iter().map(|ad| FeedItem {
        position: 0,
        item: Some(feed_item::Item::Ad(ad)),
    }));
    non_selected
}

fn select_ads_only(candidates: Vec<FeedItem>) -> SelectResult<FeedItem> {
    let mut dropped_posts = 0;
    let mut ads = Vec::new();
    for candidate in candidates {
        match candidate.item {
            Some(feed_item::Item::Ad(ad)) => ads.push(ad),
            Some(feed_item::Item::Post(_)) => dropped_posts += 1,
            _ => {}
        }
    }
    let selected: Vec<FeedItem> = ads
        .into_iter()
        .enumerate()
        .map(|(position, ad)| FeedItem {
            position: position as i32,
            item: Some(feed_item::Item::Ad(ad)),
        })
        .collect();
    SelectResult {
        non_selected: build_non_selected(dropped_posts, Vec::new(), &selected),
        selected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_home_mixer_proto::{Prompt, PushToHomePost, WhoToFollowModule};

    fn item(item: feed_item::Item) -> FeedItem {
        FeedItem {
            position: 7,
            item: Some(item),
        }
    }

    fn ad(post_id: i64) -> FeedItem {
        item(feed_item::Item::Ad(AdIndexInfo {
            post_id,
            ..Default::default()
        }))
    }

    #[test]
    fn ads_only_selection_serves_every_ad_in_order_and_nothing_else() {
        let candidates = vec![
            item(feed_item::Item::Post(ScoredPost::default())),
            ad(1),
            item(feed_item::Item::WhoToFollow(WhoToFollowModule::default())),
            item(feed_item::Item::PushToHome(PushToHomePost::default())),
            ad(2),
            item(feed_item::Item::Prompt(Prompt::default())),
            item(feed_item::Item::Post(ScoredPost::default())),
        ];
        let result = select_ads_only(candidates);
        let served: Vec<(i32, i64)> = result
            .selected
            .iter()
            .map(|served| match &served.item {
                Some(feed_item::Item::Ad(ad)) => (served.position, ad.post_id),
                other => panic!("served a non-ad item: {other:?}"),
            })
            .collect();
        assert_eq!(served, vec![(0, 1), (1, 2)]);
        assert_eq!(result.non_selected.len(), 2);
    }
}
