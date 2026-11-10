use super::client_event::{
    video_carousel_item_client_event_info, video_carousel_module_client_event_info,
};
use super::post_marshaller::{make_tweet, make_tweet_item, ENTRY_NAMESPACE_TWEET};
use xai_home_mixer_proto::{ScoredPost, VideoCarouselModule};
use xai_urt_thrift::entry::{TimelineEntry, TimelineEntryContent};
use xai_urt_thrift::icon::HorizonIcon;
use xai_urt_thrift::timeline_module::{
    ModuleDisplayType, ModuleHeader, ModuleHeaderDisplayType, ModuleItem, TimelineModule,
};
use xai_urt_thrift::tweet::TweetDisplayType;

const ENTRY_NAMESPACE_VIDEO_CAROUSEL: &str = "video-carousel";
const HEADER_KEY: &str = "VideoCarouselModule.header";

pub(super) fn marshal_video_carousel(
    carousel: &VideoCarouselModule,
    sort_index: i64,
    module_id: i64,
    language: &str,
    country: Option<&str>,
    header_icon: bool,
) -> Option<TimelineEntry> {
    if carousel.videos.is_empty() {
        return None;
    }
    let module_entry_id = format!("{ENTRY_NAMESPACE_VIDEO_CAROUSEL}-{module_id}");

    let items: Vec<ModuleItem> = carousel
        .videos
        .iter()
        .enumerate()
        .map(|(index, video)| {
            let video = as_source_tweet(video);
            let mut tweet = make_tweet(video.tweet_id);
            tweet.display_type = TweetDisplayType::MEDIA_SHORT;
            ModuleItem {
                entry_id: format!(
                    "{module_entry_id}-{ENTRY_NAMESPACE_TWEET}-{}",
                    video.tweet_id
                ),
                item: make_tweet_item(
                    tweet,
                    Some(video_carousel_item_client_event_info(&video, index as i32)),
                    None,
                ),
                dispensable: None,
                tree_display: None,
                pill_group: None,
            }
        })
        .collect();

    Some(TimelineEntry {
        entry_id: module_entry_id,
        sort_index,
        content: TimelineEntryContent::TimelineModule(TimelineModule {
            items,
            display_type: ModuleDisplayType::COMPACT_CAROUSEL,
            header: header(lookup(HEADER_KEY, language, country), header_icon),
            footer: None,
            client_event_info: Some(video_carousel_module_client_event_info()),
            feedback_info: None,
            metadata: None,
            show_more_behavior: None,
        }),
        expiry_time: None,
    })
}

fn as_source_tweet(video: &ScoredPost) -> ScoredPost {
    if video.retweeted_tweet_id == 0 {
        return video.clone();
    }
    ScoredPost {
        tweet_id: video.retweeted_tweet_id,
        author_id: video.retweeted_user_id,
        retweeted_tweet_id: 0,
        retweeted_user_id: 0,
        ..video.clone()
    }
}

fn header(text: Option<String>, show_icon: bool) -> Option<ModuleHeader> {
    text.map(|text| ModuleHeader {
        text,
        sticky: Some(false),
        context: None,
        social_context: None,
        icon: show_icon.then_some(HorizonIcon::VIDEO_PLAYER_LIST),
        button: None,
        custom_icon: None,
        display_type: Some(ModuleHeaderDisplayType::CLASSIC),
        landing_url: None,
    })
}

fn lookup(key: &str, language: &str, country: Option<&str>) -> Option<String> {
    xai_stringcenter::global()?.get(key, language, country)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_home_mixer_proto::ServedType;
    use xai_urt_thrift::item::TimelineItemContent;
    use xai_urt_thrift::metadata::ClientEventInfo;

    fn injection_type(info: &ClientEventInfo) -> Option<&str> {
        info.details
            .as_ref()?
            .timelines_details
            .as_ref()?
            .injection_type
            .as_deref()
    }

    #[test]
    fn marshals_a_compact_carousel_of_media_short_videos() {
        let carousel = VideoCarouselModule {
            videos: vec![
                ScoredPost {
                    tweet_id: 11,
                    author_id: 1,
                    served_type: ServedType::ForYouPhoenixRetrieval as i32,
                    ..Default::default()
                },
                ScoredPost {
                    tweet_id: 22,
                    author_id: 2,
                    retweeted_tweet_id: 33,
                    retweeted_user_id: 3,
                    served_type: ServedType::ForYouInNetwork as i32,
                    ..Default::default()
                },
            ],
        };

        let entry = marshal_video_carousel(&carousel, 3, 900, "en", None, false).unwrap();
        let TimelineEntryContent::TimelineModule(module) = &entry.content else {
            panic!("expected a timeline module");
        };

        assert_eq!(entry.entry_id, "video-carousel-900");
        assert_eq!(module.display_type, ModuleDisplayType::COMPACT_CAROUSEL);
        assert!(module.feedback_info.is_none());
        let items: Vec<(&str, Option<&str>, Option<&str>)> = module
            .items
            .iter()
            .map(|item| {
                let TimelineItemContent::Tweet(tweet) = &item.item.content else {
                    panic!("expected a tweet item");
                };
                assert_eq!(tweet.display_type, TweetDisplayType::MEDIA_SHORT);
                assert!(item.item.feedback_info.is_none());
                let info = item.item.client_event_info.as_ref();
                (
                    item.entry_id.as_str(),
                    info.and_then(|info| info.component.as_deref()),
                    info.and_then(injection_type),
                )
            })
            .collect();
        assert_eq!(
            items,
            vec![
                (
                    "video-carousel-900-tweet-11",
                    Some("video_carousel"),
                    Some("VideoCarousel")
                ),
                (
                    "video-carousel-900-tweet-33",
                    Some("video_carousel"),
                    Some("VideoCarousel")
                ),
            ]
        );
        let module_event = module.client_event_info.as_ref().unwrap();
        assert_eq!(module_event.component.as_deref(), Some("video_carousel"));
        assert_eq!(injection_type(module_event), Some("VideoCarousel"));
        let with_icon = header(Some("Videos for you".to_string()), true).unwrap();
        assert!(with_icon.landing_url.is_none());
        assert_eq!(with_icon.icon, Some(HorizonIcon::VIDEO_PLAYER_LIST));
        let without_icon = header(Some("Videos for you".to_string()), false).unwrap();
        assert_eq!(without_icon.icon, None);
    }
}
