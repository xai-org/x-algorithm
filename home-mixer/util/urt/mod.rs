mod ad_marshaller;
mod client_event;
mod controller_data;
mod convo_marshaller;
mod cursors;
mod feed_survey_marshaller;
mod feedback;
mod frame_marshaller;
mod navigation;
mod new_tweets_pill;
mod post_marshaller;
mod prompt_marshaller;
mod push_to_home_marshaller;
pub(crate) mod reverse_chron_following;
mod wtf_marshaller;
use crate::models::query::RequestType;
use std::collections::BTreeMap;
use xai_home_mixer_proto::FeedItem;
use xai_home_mixer_proto::feed_item::Item as FeedItemKind;
use xai_urt_thrift::cursor::UrtOrderedCursor;
use xai_urt_thrift::entry::TimelineEntry;
use xai_urt_thrift::instruction::{AddEntries, TimelineInstruction};
use xai_urt_thrift::metadata::FeedbackAction;
use xai_urt_thrift::response::{
    ResponseObjects, Timeline, TimelineMetadata, TimelineResponse, TimelineScribeConfig,
    TimelineState,
};

pub(crate) fn make_urt_timeline(
    items: &[FeedItem],
    cursor: Option<&UrtOrderedCursor>,
    request_context: &str,
    client_app_id: i32,
    viewer_id: u64,
    language_code: &str,
    country_code: Option<&str>,
    entry_id_nonce: Option<u64>,
    request_type: RequestType,
    request_id: u64,
) -> TimelineResponse {
    let page = urt_page(request_type);
    let initial_sort_index = cursors::initial_sort_index(cursor);

    let mut all_feedback_actions: BTreeMap<String, FeedbackAction> = BTreeMap::new();
    let mut instructions =
        navigation::build_navigation_instructions(request_context, client_app_id, items);
    instructions.extend(new_tweets_pill::build_new_tweets_pill_instruction(
        request_context,
        items,
        viewer_id,
        cursor,
        language_code,
        country_code,
        true,
    ));

    let mut entries: Vec<TimelineEntry> = items
        .iter()
        .filter_map(|feed_item| match &feed_item.item {
            Some(FeedItemKind::Post(post)) => {
                let fb = feedback::build_dont_like_feedback(
                    post,
                    viewer_id,
                    language_code,
                    country_code,
                );
                let feedback_info = fb.as_ref().map(|f| f.feedback_info.clone());
                if let Some(f) = fb {
                    all_feedback_actions.extend(f.actions);
                }

                let mut entry = if post.ancestors.is_empty()
                    && post.following_replied_user_ids.is_empty()
                {
                    post_marshaller::marshal_post(post, feed_item.position as i64, feedback_info)
                } else {
                    convo_marshaller::marshal_conversation_module(
                        post,
                        feed_item.position as i64,
                        feedback_info,
                    )
                };
                if let Some(nonce) = entry_id_nonce {
                    entry.entry_id = format!("{}-{}", entry.entry_id, nonce);
                }
                Some(entry)
            }
            Some(FeedItemKind::PushToHome(post)) => Some(
                push_to_home_marshaller::marshal_push_to_home(post, feed_item.position as i64),
            ),
            Some(FeedItemKind::Ad(ad)) => {
                Some(ad_marshaller::marshal_ad(ad, feed_item.position as i64))
            }
            Some(FeedItemKind::WhoToFollow(wtf)) => {
                wtf_marshaller::marshal_wtf(wtf, feed_item.position as i64, client_app_id)
            }
            Some(FeedItemKind::Prompt(module)) => {
                match prompt_marshaller::marshal_prompt(module, feed_item.position as i64) {
                    Some(prompt_marshaller::PromptUrtResult::Entry(entry)) => Some(entry),
                    Some(prompt_marshaller::PromptUrtResult::Instruction(instr)) => {
                        instructions.push(instr);
                        None
                    }
                    None => None,
                }
            }
            Some(FeedItemKind::Frame(f)) => Some(frame_marshaller::marshal_frame(
                f,
                feed_item.position as i64,
                initial_sort_index,
            )),
            Some(FeedItemKind::FeedSurvey(_)) => Some(feed_survey_marshaller::marshal_feed_survey(
                feed_item.position as i64,
                language_code,
                request_id,
            )),
            _ => None,
        })
        .collect();

    for (i, entry) in entries.iter_mut().enumerate() {
        entry.sort_index = initial_sort_index - i as i64;
    }

    let (top_post_id, bottom_post_id) = cursor_post_ids(items);
    let (top_cursor, bottom_cursor) =
        cursors::build_cursors(&entries, initial_sort_index, top_post_id, bottom_post_id);

    let mut all_entries = Vec::with_capacity(entries.len() + 2);
    all_entries.push(top_cursor);
    all_entries.append(&mut entries);
    all_entries.push(bottom_cursor);

    instructions.push(TimelineInstruction::AddEntries(AddEntries {
        entries: all_entries,
    }));

    let response_objects = if all_feedback_actions.is_empty() {
        None
    } else {
        Some(ResponseObjects {
            feedback_actions: Some(all_feedback_actions),
            immediate_reactions: None,
        })
    };

    TimelineResponse {
        state: TimelineState::OK,
        timeline: Timeline {
            id: page.to_string(),
            instructions,
            response_objects,
            metadata: Some(TimelineMetadata {
                title: None,
                scribe_config: Some(TimelineScribeConfig {
                    page: Some(page.to_string()),
                    section: None,
                    entity_token: None,
                }),
                reader_mode_config: None,
                display_type_config: None,
            }),
        },
    }
}

fn urt_page(request_type: RequestType) -> &'static str {
    match request_type {
        RequestType::ForYou => "for_you",
        RequestType::RankedFollowing => "ranked_following",
        RequestType::Following => "following",
        RequestType::ScoredPosts => "scored_posts",
        RequestType::PhoenixScores => "phoenix_scores",
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

#[cfg(test)]
mod tests {
    use super::*;
    use xai_home_mixer_proto::ScoredPost;
    use xai_recsys_proto::AdIndexInfo;
    use xai_urt_thrift::cursor_utils::decode_ordered_cursor;
    use xai_urt_thrift::entry::TimelineEntryContent;
    use xai_urt_thrift::operation::{CursorType, TimelineOperation};

    fn post_item(tweet_id: u64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(FeedItemKind::Post(ScoredPost {
                tweet_id,
                ..Default::default()
            })),
        }
    }

    fn ad_item(post_id: i64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(FeedItemKind::Ad(AdIndexInfo {
                post_id,
                ..Default::default()
            })),
        }
    }

    fn rendered_entries(items: &[FeedItem], entry_id_nonce: Option<u64>) -> Vec<TimelineEntry> {
        let resp = make_urt_timeline(
            items,
            None,
            "foreground",
            0,
            1,
            "en",
            None,
            entry_id_nonce,
            RequestType::Following,
            0,
        );
        resp.timeline
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                TimelineInstruction::AddEntries(add) => Some(add.entries.clone()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    fn rendered_entry_ids(items: &[FeedItem], entry_id_nonce: Option<u64>) -> Vec<String> {
        rendered_entries(items, entry_id_nonce)
            .into_iter()
            .map(|e| e.entry_id)
            .collect()
    }

    fn cursor_post_id(entry: &TimelineEntry) -> Option<i64> {
        let TimelineEntryContent::Operation(TimelineOperation::Cursor(cursor)) = &entry.content
        else {
            return None;
        };
        decode_ordered_cursor(&cursor.value)
            .ok()
            .flatten()
            .and_then(|c| c.id)
    }

    #[test]
    fn without_nonce_entry_ids_are_unchanged() {
        let ids = rendered_entry_ids(&[post_item(100), ad_item(200)], None);
        assert!(ids.contains(&"tweet-100".to_string()), "{ids:?}");
        assert!(ids.contains(&"promoted-tweet-200".to_string()), "{ids:?}");
    }

    #[test]
    fn nonce_suffixes_posts_but_not_ads_or_cursors() {
        let ids = rendered_entry_ids(&[post_item(100), ad_item(200)], Some(42));
        assert!(ids.contains(&"tweet-100-42".to_string()), "{ids:?}");
        assert!(!ids.contains(&"tweet-100".to_string()), "{ids:?}");
        assert!(ids.contains(&"promoted-tweet-200".to_string()), "{ids:?}");
        assert!(
            ids.iter().any(|id| id.starts_with("cursor-top-")),
            "{ids:?}"
        );
    }

    #[test]
    fn cursors_use_first_and_last_post_ids_skipping_ads() {
        let entries = rendered_entries(
            &[
                post_item(111),
                ad_item(999),
                post_item(222),
                ad_item(998),
                post_item(333),
            ],
            None,
        );
        let top = entries
            .iter()
            .find(|e| e.entry_id.starts_with("cursor-top-"))
            .expect("top cursor");
        let bottom = entries
            .iter()
            .find(|e| e.entry_id.starts_with("cursor-bottom-"))
            .expect("bottom cursor");

        assert_eq!(cursor_post_id(top), Some(111));
        assert_eq!(cursor_post_id(bottom), Some(333));

        let TimelineEntryContent::Operation(TimelineOperation::Cursor(top_c)) = &top.content else {
            panic!("expected top cursor op");
        };
        let TimelineEntryContent::Operation(TimelineOperation::Cursor(bottom_c)) = &bottom.content
        else {
            panic!("expected bottom cursor op");
        };
        assert_eq!(top_c.cursor_type, CursorType::TOP);
        assert_eq!(bottom_c.cursor_type, CursorType::BOTTOM);
    }

    #[test]
    fn cursor_post_ids_single_post() {
        assert_eq!(
            cursor_post_ids(&[post_item(42), ad_item(1)]),
            (Some(42), Some(42))
        );
    }
}
