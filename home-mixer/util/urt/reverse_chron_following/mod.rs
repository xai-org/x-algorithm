mod cursors;

use super::ad_marshaller::marshal_ad;
use super::convo_marshaller::marshal_conversation_module;
use super::new_tweets_pill::build_new_tweets_pill_instruction;
use super::post_marshaller::marshal_post;
use super::prompt_marshaller::{marshal_prompt, PromptUrtResult};
use super::wtf_marshaller::marshal_wtf;
use crate::models::query::FollowingPaginationMeta;
use cursors::{build_cursors, initial_sort_index as compute_initial_sort_index};
use xai_home_mixer_proto::feed_item::Item as FeedItemKind;
use xai_home_mixer_proto::FeedItem;
use xai_urt_thrift::cursor::UrtOrderedCursor;
use xai_urt_thrift::entry::TimelineEntry;
use xai_urt_thrift::instruction::{AddEntries, TimelineInstruction};
use xai_urt_thrift::response::{
    Timeline, TimelineMetadata, TimelineResponse, TimelineScribeConfig, TimelineState,
};

pub(crate) fn make_urt_timeline(
    items: &[FeedItem],
    cursor: Option<&UrtOrderedCursor>,
    request_context: &str,
    client_app_id: i32,
    viewer_id: u64,
    language_code: &str,
    country_code: Option<&str>,
    pagination_meta: &FollowingPaginationMeta,
) -> TimelineResponse {
    let initial_sort_index = compute_initial_sort_index(cursor);
    let mut instructions: Vec<TimelineInstruction> = build_new_tweets_pill_instruction(
        request_context,
        items,
        viewer_id,
        cursor,
        language_code,
        country_code,
        false,
    )
    .into_iter()
    .collect();

    let mut entries: Vec<TimelineEntry> = items
        .iter()
        .filter_map(|feed_item| match &feed_item.item {
            Some(FeedItemKind::Post(post)) => {
                if post.ancestors.is_empty() {
                    Some(marshal_post(post, feed_item.position as i64, None))
                } else {
                    Some(marshal_conversation_module(
                        post,
                        feed_item.position as i64,
                        None,
                    ))
                }
            }
            Some(FeedItemKind::Ad(ad)) => Some(marshal_ad(ad, feed_item.position as i64)),
            Some(FeedItemKind::WhoToFollow(wtf)) => {
                marshal_wtf(wtf, feed_item.position as i64, client_app_id)
            }
            Some(FeedItemKind::Prompt(module)) => {
                match marshal_prompt(module, feed_item.position as i64) {
                    Some(PromptUrtResult::Entry(entry)) => Some(entry),
                    Some(PromptUrtResult::Instruction(instr)) => {
                        instructions.push(instr);
                        None
                    }
                    None => None,
                }
            }
            _ => None,
        })
        .collect();

    for (i, entry) in entries.iter_mut().enumerate() {
        entry.sort_index = initial_sort_index - i as i64;
    }

    let (top_post_id, bottom_post_id) = cursor_post_ids(items);
    let (top_cursor, bottom_or_gap_cursor) = build_cursors(
        &entries,
        initial_sort_index,
        top_post_id,
        bottom_post_id,
        cursor,
        items,
        pagination_meta,
    );

    let mut all_entries = Vec::with_capacity(entries.len() + 2);
    all_entries.push(top_cursor);
    all_entries.append(&mut entries);
    all_entries.push(bottom_or_gap_cursor);
    instructions.push(TimelineInstruction::AddEntries(AddEntries {
        entries: all_entries,
    }));

    TimelineResponse {
        state: TimelineState::OK,
        timeline: Timeline {
            id: "following".to_string(),
            instructions,
            response_objects: None,
            metadata: Some(TimelineMetadata {
                title: None,
                scribe_config: Some(TimelineScribeConfig {
                    page: Some("following".to_string()),
                    section: None,
                    entity_token: None,
                }),
                reader_mode_config: None,
                display_type_config: None,
            }),
        },
    }
}

fn cursor_post_ids(items: &[FeedItem]) -> (Option<i64>, Option<i64>) {
    let mut ids = items.iter().filter_map(|feed_item| match &feed_item.item {
        Some(FeedItemKind::Post(post)) if post.tweet_id > 0 => Some(post.tweet_id as i64),
        _ => None,
    });
    let top = ids.next();
    let bottom = ids.next_back().or(top);
    (top, bottom)
}
