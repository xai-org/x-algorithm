use super::client_event::{post_client_event_info, served_type_component};
use super::post_marshaller::{make_tweet_helper, make_tweet_item, ENTRY_NAMESPACE_TWEET};
use std::collections::HashSet;
use xai_home_mixer_proto::ScoredPost;
use xai_urt_thrift::entry::{TimelineEntry, TimelineEntryContent};
use xai_urt_thrift::metadata::{ClientEventInfo, FeedbackInfo};
use xai_urt_thrift::timeline_module::{
    ModuleConversationMetadata, ModuleDisplayType, ModuleItem, ModuleMetadata, TimelineModule,
};
use xai_urt_thrift::tweet::{TweetFacepile, TweetFacepileDisplayType};

const ENTRY_NAMESPACE_CONVERSATION: &str = "home-conversation";
const NUM_INTERMEDIATE_ANCESTORS: usize = 1;

pub(super) fn marshal_conversation_module(
    post: &ScoredPost,
    sort_index: i64,
    feedback_info: Option<FeedbackInfo>,
) -> TimelineEntry {
    let focal_id = post.tweet_id;
    let component = served_type_component(post.served_type);
    let tombstone_ids: HashSet<u64> = post.tombstone_ancestor_ids.iter().copied().collect();

    let mut parent_tweets: Vec<u64> = post.ancestors.clone();
    if parent_tweets.is_empty()
        && post.in_reply_to_tweet_id != 0
        && post.in_reply_to_tweet_id != focal_id
    {
        parent_tweets.push(post.in_reply_to_tweet_id);
    }
    parent_tweets.sort_unstable();
    parent_tweets.reverse();

    let root = parent_tweets.last().copied();
    let intermediates: Vec<u64> = if parent_tweets.len() > 1 {
        parent_tweets[..parent_tweets.len() - 1]
            .iter()
            .take(NUM_INTERMEDIATE_ANCESTORS)
            .rev()
            .copied()
            .collect()
    } else {
        vec![]
    };

    let all_tweet_ids: Vec<i64> = {
        let mut ids: Vec<u64> = parent_tweets.clone();
        ids.push(focal_id);
        ids.sort_unstable();
        ids.iter().map(|&id| id as i64).collect()
    };

    let item_ids: Vec<u64> = root
        .into_iter()
        .chain(intermediates.iter().copied())
        .chain(std::iter::once(focal_id))
        .collect();

    let cei = || post_client_event_info(&component, post, sort_index as i32);
    let facepile = following_replied_facepile(post);

    let module_items: Vec<ModuleItem> = item_ids
        .into_iter()
        .filter(|id| *id == focal_id || !tombstone_ids.contains(id))
        .map(|id| {
            let fi = if id == focal_id {
                feedback_info.clone()
            } else {
                None
            };
            ModuleItem {
                entry_id: format!("{}-{}", ENTRY_NAMESPACE_TWEET, id),
                item: make_tweet_item(
                    make_tweet_helper(id, facepile.clone(), None),
                    Some(cei()),
                    fi,
                ),
                dispensable: None,
                tree_display: None,
                pill_group: None,
            }
        })
        .collect();

    let module_cei = ClientEventInfo {
        component: Some(component.to_string()),
        element: None,
        details: None,
        action: None,
        entity_token: None,
    };

    TimelineEntry {
        entry_id: format!("{}-{}", ENTRY_NAMESPACE_CONVERSATION, focal_id),
        sort_index,
        content: TimelineEntryContent::TimelineModule(TimelineModule {
            items: module_items,
            display_type: ModuleDisplayType::VERTICAL_CONVERSATION,
            header: None,
            footer: None,
            client_event_info: Some(module_cei),
            feedback_info: None,
            metadata: Some(ModuleMetadata {
                ads_metadata: None,
                conversation_metadata: Some(ModuleConversationMetadata {
                    all_tweet_ids: Some(all_tweet_ids),
                    social_context: None,
                    enable_deduplication: Some(true),
                }),
                grid_carousel_metadata: None,
                vertical_metadata: None,
                pill_group_metadata: None,
            }),
            show_more_behavior: None,
        }),
        expiry_time: None,
    }
}

fn following_replied_facepile(post: &ScoredPost) -> Option<TweetFacepile> {
    if post.following_replied_user_ids.is_empty() {
        return None;
    }
    Some(TweetFacepile {
        user_ids: post
            .following_replied_user_ids
            .iter()
            .map(|&id| id as i64)
            .collect(),
        display_type: Some(TweetFacepileDisplayType::MUTUAL_REPLIED),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_module_with_ancestors() {
        let post = ScoredPost {
            tweet_id: 300,
            ancestors: vec![100, 200],
            in_reply_to_tweet_id: 200,
            ..Default::default()
        };
        let entry = marshal_conversation_module(&post, 1, None);
        if let TimelineEntryContent::TimelineModule(module) = entry.content {
            assert_eq!(module.items.len(), 3);
            assert_eq!(module.items[0].entry_id, "tweet-100");
            assert_eq!(module.items[1].entry_id, "tweet-200");
            assert_eq!(module.items[2].entry_id, "tweet-300");
        } else {
            panic!("Expected TimelineModule");
        }
    }

    #[test]
    fn test_conversation_module_fallback_when_ancestors_empty() {
        let post = ScoredPost {
            tweet_id: 200,
            ancestors: vec![],
            in_reply_to_tweet_id: 100,
            ..Default::default()
        };
        let entry = marshal_conversation_module(&post, 1, None);
        if let TimelineEntryContent::TimelineModule(module) = entry.content {
            assert_eq!(module.items.len(), 2);
            assert_eq!(module.items[0].entry_id, "tweet-100");
            assert_eq!(module.items[1].entry_id, "tweet-200");
        } else {
            panic!("Expected TimelineModule");
        }
    }
}

