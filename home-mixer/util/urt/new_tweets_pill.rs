use xai_candidate_pipeline::component_library::utils::client_utils::RequestContext;
use xai_home_mixer_proto::feed_item::Item as FeedItemKind;
use xai_home_mixer_proto::FeedItem;
use xai_urt_thrift::color::RosettaColor;
use xai_urt_thrift::cursor::UrtOrderedCursor;
use xai_urt_thrift::instruction::{
    AlertType, ShowAlert, ShowAlertColorConfiguration, ShowAlertDisplayLocation, ShowAlertIcon,
    ShowAlertIconDisplayInfo, TimelineInstruction,
};
use xai_urt_thrift::metadata::ClientEventInfo;
use xai_urt_thrift::operation::CursorType;
use xai_urt_thrift::richtext::RichText;

const NUM_AVATARS: usize = 3;
const TRIGGER_DELAY_MS: i32 = 4 * 60 * 1000;
const DISPLAY_DURATION_MS: i32 = -1;
const SHOW_ALERT_ELEMENT: &str = "showAlert";
const TWEETED_KEY: &str = "Tweeted";

pub(super) fn build_new_tweets_pill_instruction(
    request_context: &str,
    items: &[FeedItem],
    viewer_id: u64,
    cursor: Option<&UrtOrderedCursor>,
    language: &str,
    country: Option<&str>,
    require_full_facepile: bool,
) -> Option<TimelineInstruction> {
    let has_top_cursor = cursor
        .and_then(|c| c.cursor_type.as_ref())
        .is_some_and(|ct| *ct == CursorType::TOP);
    if !has_top_cursor {
        return None;
    }

    let ctx = RequestContext::parse(request_context);
    if ctx == RequestContext::PullToRefresh {
        return None;
    }

    let user_ids = extract_pill_user_ids(items, viewer_id).filter(|ids| ids.len() >= NUM_AVATARS);
    if require_full_facepile && user_ids.is_none() {
        return None;
    }

    let trigger_delay_ms = match ctx {
        RequestContext::TweetSelfThread | RequestContext::ManualRefresh => 0,
        _ => TRIGGER_DELAY_MS,
    };

    let color_config = ShowAlertColorConfiguration {
        background: RosettaColor::TWITTER_BLUE,
        text: RosettaColor::WHITE,
        border: Some(RosettaColor::WHITE),
    };

    let icon_display_info = ShowAlertIconDisplayInfo {
        icon: ShowAlertIcon::UP_ARROW,
        tint: RosettaColor::WHITE,
    };

    let client_event_info = ClientEventInfo {
        component: None,
        element: Some(SHOW_ALERT_ELEMENT.to_string()),
        details: None,
        action: None,
        entity_token: None,
    };

    Some(TimelineInstruction::ShowAlert(ShowAlert {
        alert_type: AlertType::NEW_TWEETS,
        text: None,
        trigger_delay_ms: Some(trigger_delay_ms),
        display_duration_ms: Some(DISPLAY_DURATION_MS),
        client_event_info: Some(client_event_info),
        collapse_delay_ms: None,
        user_ids,
        rich_text: pill_text(language, country),
        icon_display_info: Some(icon_display_info),
        color_config,
        display_location: ShowAlertDisplayLocation::TOP,
        navigation_metadata: None,
    }))
}

fn pill_text(language: &str, country: Option<&str>) -> Option<RichText> {
    let text = xai_stringcenter::global()?.get(TWEETED_KEY, language, country)?;
    Some(RichText {
        text,
        entities: vec![],
        rtl: None,
        alignment: None,
    })
}

fn extract_pill_user_ids(items: &[FeedItem], viewer_id: u64) -> Option<Vec<i64>> {
    let mut authors = Vec::with_capacity(NUM_AVATARS);
    for item in items {
        if let Some(FeedItemKind::Post(post)) = &item.item {
            let is_retweet = post.retweeted_tweet_id != 0;
            let author = if is_retweet {
                post.retweeted_user_id
            } else {
                post.author_id
            };
            if author != viewer_id && author != 0 {
                let id = author as i64;
                if !authors.contains(&id) {
                    authors.push(id);
                    if authors.len() == NUM_AVATARS {
                        break;
                    }
                }
            }
        }
    }
    if authors.is_empty() {
        None
    } else {
        Some(authors)
    }
}
