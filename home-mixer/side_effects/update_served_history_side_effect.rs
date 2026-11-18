use crate::clients::served_history_client::{ServedHistoryClient, TimelineType};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableUrtMigrationComponents;
use crate::util::string_case::upper_snake_to_pascal;
use std::sync::Arc;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::client_utils::app_id_to_served_history_id;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::feed_item::Item as FeedItemKind;
use xai_home_mixer_proto::{
    FeedItem, Frame, PushToHomePost, ScoredPost, ServedType, VideoCarouselModule, WhoToFollowModule,
};
use xai_recsys_proto::AdIndexInfo;
use xai_urt_thrift::operation::CursorType;
use xai_x_thrift::served_history::{
    EntityIdType, EntryWithItemIds, ItemIds, RequestType, ServedHistory, TweetScore, TweetScoreV1,
};

pub struct UpdateServedHistorySideEffect {
    client: Arc<dyn ServedHistoryClient>,
}

impl UpdateServedHistorySideEffect {
    pub fn new(client: Arc<dyn ServedHistoryClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for UpdateServedHistorySideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnableUrtMigrationComponents)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let items = &input.selected_candidates;
        if items.is_empty() {
            return Ok(());
        }

        let query = &input.query;
        let now_ms = query.request_time_ms;

        let entries: Vec<EntryWithItemIds> = items
            .iter()
            .enumerate()
            .flat_map(|(position, item)| build_entries(item, position as i64))
            .collect();

        if entries.is_empty() {
            return Ok(());
        }

        let history = ServedHistory {
            request_type: request_type(query),
            entries,
            served_id: Some(query.request_id as i64),
            served_time_ms: None,
        };

        let platform = app_id_to_served_history_id(query.client_app_id);
        self.client
            .put(
                query.user_id,
                TimelineType::for_request(query.request_type),
                platform,
                now_ms,
                &history,
            )
            .await
            .map_err(|e| e.to_string())
    }
}

pub(crate) fn request_type(query: &ScoredPostsQuery) -> RequestType {
    if query.is_polling {
        return RequestType::POLLING;
    }
    match query.cursor.as_ref() {
        None => RequestType::INITIAL,
        Some(c) => match c.cursor_type.as_ref() {
            Some(ct) if *ct == CursorType::TOP => RequestType::NEWER,
            Some(ct) if *ct == CursorType::BOTTOM => RequestType::OLDER,
            Some(ct) if *ct == CursorType::GAP => RequestType::MIDDLE,
            _ => RequestType::OTHER,
        },
    }
}

fn build_entries(feed_item: &FeedItem, position: i64) -> Vec<EntryWithItemIds> {
    match feed_item.item.as_ref() {
        Some(FeedItemKind::Post(post)) => build_post_entries(post, position),
        Some(FeedItemKind::Ad(ad)) => vec![build_ad_entry(ad, position)],
        Some(FeedItemKind::WhoToFollow(wtf)) => {
            build_wtf_entry(wtf, position).into_iter().collect()
        }
        Some(FeedItemKind::Prompt(_)) => vec![build_prompt_entry(position)],
        Some(FeedItemKind::PushToHome(pth)) => build_push_to_home_entries(pth, position),
        Some(FeedItemKind::Frame(frame)) => {
            build_frame_entry(frame, position).into_iter().collect()
        }
        Some(FeedItemKind::FeedSurvey(_)) => vec![build_feed_survey_entry(position)],
        Some(FeedItemKind::VideoCarousel(carousel)) => {
            build_video_carousel_entry(carousel, position)
                .into_iter()
                .collect()
        }
        None => vec![],
    }
}

fn build_post_entries(post: &ScoredPost, position: i64) -> Vec<EntryWithItemIds> {
    if post.ancestors.is_empty() {
        vec![build_tweet_entry(post, post.tweet_id, position)]
    } else {
        let mut tweet_ids: Vec<u64> = post.ancestors.clone();
        tweet_ids.push(post.tweet_id);
        tweet_ids.sort_unstable();
        tweet_ids
            .into_iter()
            .map(|tweet_id| build_tweet_entry(post, tweet_id, position))
            .collect()
    }
}

fn nonzero(v: u64) -> Option<i64> {
    if v != 0 {
        Some(v as i64)
    } else {
        None
    }
}

fn build_tweet_entry(post: &ScoredPost, tweet_id: u64, position: i64) -> EntryWithItemIds {
    let source_tweet_id = nonzero(post.retweeted_tweet_id);
    let source_author_id = Some(nonzero(post.retweeted_user_id).unwrap_or(post.author_id as i64));
    let in_reply_to_tweet_id = nonzero(post.in_reply_to_tweet_id);
    let prediction_request_id = nonzero(post.prediction_request_id);

    let served_type_name = upper_snake_to_pascal(
        ServedType::try_from(post.served_type)
            .unwrap_or(ServedType::Undefined)
            .as_str_name(),
    );

    let tweet_score = TweetScore::TweetScoreV1(TweetScoreV1 {
        score: OrderedFloat::from(post.score as f64),
        served_type: Some(served_type_name),
        debug_info: None,
        prediction_request_id,
        topics: None,
        tags: None,
        predicted_scores: None,
    });

    EntryWithItemIds {
        entity_type: EntityIdType::TWEET,
        sort_index: Some(position),
        size: None,
        item_ids: Some(vec![ItemIds {
            tweet_id: Some(tweet_id as i64),
            source_tweet_id,
            quote_tweet_id: None,
            source_author_id,
            quote_author_id: None,
            in_reply_to_tweet_id,
            in_reply_to_author_id: None,
            article_id: None,
            tweet_score: Some(tweet_score),
            entry_id_to_replace: None,
            user_id: None,
            impression_id: None,
        }]),
    }
}

fn build_ad_entry(ad: &AdIndexInfo, position: i64) -> EntryWithItemIds {
    let impression_string = format!("{:x}", ad.impression_id);
    EntryWithItemIds {
        entity_type: EntityIdType::PROMOTED_TWEET,
        sort_index: Some(position),
        size: None,
        item_ids: Some(vec![ItemIds {
            tweet_id: Some(ad.post_id),
            source_tweet_id: None,
            quote_tweet_id: None,
            source_author_id: Some(ad.author_id),
            quote_author_id: None,
            in_reply_to_tweet_id: None,
            in_reply_to_author_id: None,
            article_id: None,
            tweet_score: None,
            entry_id_to_replace: None,
            user_id: None,
            impression_id: Some(impression_string),
        }]),
    }
}

fn build_feed_survey_entry(position: i64) -> EntryWithItemIds {
    EntryWithItemIds {
        entity_type: EntityIdType::ANNOTATION,
        sort_index: Some(position),
        size: None,
        item_ids: None,
    }
}

fn build_frame_entry(frame: &Frame, position: i64) -> Option<EntryWithItemIds> {
    if frame.fetch_route.is_empty() {
        return None;
    }
    Some(EntryWithItemIds {
        entity_type: crate::frames::FRAME_ENTITY_TYPE,
        sort_index: Some(position),
        size: None,
        item_ids: Some(vec![ItemIds {
            tweet_id: None,
            source_tweet_id: None,
            quote_tweet_id: None,
            source_author_id: None,
            quote_author_id: None,
            in_reply_to_tweet_id: None,
            in_reply_to_author_id: None,
            article_id: None,
            tweet_score: None,
            entry_id_to_replace: None,
            user_id: None,
            impression_id: Some(frame.fetch_route.clone()),
        }]),
    })
}

fn build_prompt_entry(position: i64) -> EntryWithItemIds {
    EntryWithItemIds {
        entity_type: EntityIdType::PROMPT,
        sort_index: Some(position),
        size: None,
        item_ids: None,
    }
}

fn build_push_to_home_entries(pth: &PushToHomePost, position: i64) -> Vec<EntryWithItemIds> {
    let in_reply_to_tweet_id = nonzero(pth.in_reply_to_tweet_id);

    let served_type_name = upper_snake_to_pascal(
        ServedType::try_from(pth.served_type)
            .unwrap_or(ServedType::Undefined)
            .as_str_name(),
    );

    let tweet_score = TweetScore::TweetScoreV1(TweetScoreV1 {
        score: OrderedFloat::from(0.0),
        served_type: Some(served_type_name),
        debug_info: None,
        prediction_request_id: None,
        topics: None,
        tags: None,
        predicted_scores: None,
    });

    vec![EntryWithItemIds {
        entity_type: EntityIdType::TWEET,
        sort_index: Some(position),
        size: None,
        item_ids: Some(vec![ItemIds {
            tweet_id: Some(pth.tweet_id as i64),
            source_tweet_id: None,
            quote_tweet_id: None,
            source_author_id: Some(pth.author_id as i64),
            quote_author_id: None,
            in_reply_to_tweet_id,
            in_reply_to_author_id: None,
            article_id: None,
            tweet_score: Some(tweet_score),
            entry_id_to_replace: None,
            user_id: None,
            impression_id: None,
        }]),
    }]
}

fn build_wtf_entry(wtf: &WhoToFollowModule, position: i64) -> Option<EntryWithItemIds> {
    let resp = wtf.who_to_follow_response.as_ref()?;
    if resp.user_recommendations.is_empty() {
        return None;
    }
    let size = resp.user_recommendations.len() as i16;
    let item_ids: Vec<ItemIds> = resp
        .user_recommendations
        .iter()
        .map(|rec| ItemIds {
            tweet_id: None,
            source_tweet_id: None,
            quote_tweet_id: None,
            source_author_id: None,
            quote_author_id: None,
            in_reply_to_tweet_id: None,
            in_reply_to_author_id: None,
            article_id: None,
            tweet_score: None,
            entry_id_to_replace: None,
            user_id: Some(rec.user_id),
            impression_id: None,
        })
        .collect();
    Some(EntryWithItemIds {
        entity_type: EntityIdType::WHO_TO_FOLLOW,
        sort_index: Some(position),
        size: Some(size),
        item_ids: Some(item_ids),
    })
}

fn build_video_carousel_entry(
    carousel: &VideoCarouselModule,
    position: i64,
) -> Option<EntryWithItemIds> {
    if carousel.videos.is_empty() {
        return None;
    }
    let item_ids: Vec<ItemIds> = carousel
        .videos
        .iter()
        .filter_map(|video| build_tweet_entry(video, video.tweet_id, position).item_ids)
        .flatten()
        .collect();
    Some(EntryWithItemIds {
        entity_type: EntityIdType::VIDEO_CAROUSEL_MODULE,
        sort_index: Some(position),
        size: Some(carousel.videos.len() as i16),
        item_ids: Some(item_ids),
    })
}
