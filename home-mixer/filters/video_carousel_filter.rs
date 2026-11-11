use crate::models::brand_safety::BrandSafetyVerdict;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableVideoCarouselAuthorDedupe, EnableVideoCarouselEmbeddingDedupe,
    EnableVideoCarouselHighRiskFilter, EnableVideoCarouselMediumRiskFilter,
    EnableVideoCarouselTakeFromFeed, VideoCarouselEmbeddingDedupeThreshold, VideoCarouselPosition,
    VideoCarouselSemanticIdDedupeLevels, RESULT_SIZE,
};
use crate::util::video_carousel::{self, CarouselScoring};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct VideoCarouselFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for VideoCarouselFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (mut removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| candidate.video_carousel_extra);
        let take_from_feed = query.params.get(EnableVideoCarouselTakeFromFeed);
        let pool_start = if take_from_feed {
            (query.params.get(VideoCarouselPosition) as usize).saturating_sub(1)
        } else {
            RESULT_SIZE
        };
        let posts_above_pool = &kept[..pool_start.min(kept.len())];
        let drop_high_risk = query.params.get(EnableVideoCarouselHighRiskFilter);
        let drop_medium_risk = query.params.get(EnableVideoCarouselMediumRiskFilter);
        let carousel = video_carousel::select_videos(
            kept.iter()
                .skip(pool_start)
                .chain(&removed)
                .filter(|candidate| {
                    !video_carousel::was_seen(&query.seen_carousel_video_ids, candidate)
                })
                .filter(|candidate| match candidate.brand_safety_verdict {
                    Some(BrandSafetyVerdict::HighRisk) => !drop_high_risk,
                    Some(BrandSafetyVerdict::MediumRisk) => !drop_medium_risk,
                    Some(_) => true,
                    None => !drop_high_risk && !drop_medium_risk,
                }),
            posts_above_pool,
            &CarouselScoring::from_query(query),
            query.params.get(EnableVideoCarouselAuthorDedupe),
            query.params.get(VideoCarouselSemanticIdDedupeLevels) as usize,
            query
                .params
                .get(EnableVideoCarouselEmbeddingDedupe)
                .then(|| query.params.get(VideoCarouselEmbeddingDedupeThreshold)),
        );
        let kept = if take_from_feed {
            let (taken, kept): (Vec<_>, Vec<_>) = kept
                .into_iter()
                .partition(|post| video_carousel::shows_carousel_video(post, &carousel));
            removed.extend(taken);
            kept
        } else {
            kept
        };
        let _ = query.video_carousel.set(carousel);
        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::PhoenixScores;
    use std::collections::HashSet;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};

    #[test]
    fn carousel_skips_videos_seen_in_a_carousel() {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_video_carousel".to_string(), "true");
        let params: Params = results.into();
        let query = ScoredPostsQuery {
            params,
            video_carousel_eligible: true,
            seen_carousel_video_ids: HashSet::from([101, 103]),
            ..Default::default()
        };
        let main_feed = (1..=RESULT_SIZE as u64).map(|tweet_id| PostCandidate {
            tweet_id,
            ..Default::default()
        });
        let videos = (101..=106).map(|tweet_id| PostCandidate {
            tweet_id,
            author_id: tweet_id,
            has_video: Some(true),
            video_aspect_ratio: Some(0.5),
            brand_safety_verdict: Some(BrandSafetyVerdict::Safe),
            phoenix_scores: PhoenixScores {
                video_open_score: Some(1.0 / tweet_id as f64),
                ..Default::default()
            },
            ..Default::default()
        });

        VideoCarouselFilter.filter(&query, main_feed.chain(videos).collect());

        let carousel_ids: HashSet<u64> = query
            .video_carousel
            .get()
            .unwrap()
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect();
        assert_eq!(carousel_ids, HashSet::from([102, 104, 105, 106]));
    }

    #[test]
    fn takes_videos_below_the_carousel_from_the_feed() {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_video_carousel".to_string(), "true");
        results.override_fs(
            "rust_home_mixer_enable_video_carousel_take_from_feed".to_string(),
            "true",
        );
        let query = ScoredPostsQuery {
            params: results.into(),
            video_carousel_eligible: true,
            ..Default::default()
        };
        let video = |tweet_id, media_id, aspect_ratio, extra| PostCandidate {
            tweet_id,
            author_id: tweet_id,
            has_video: Some(true),
            video_aspect_ratio: Some(aspect_ratio),
            video_media_id: Some(media_id),
            video_carousel_extra: extra,
            brand_safety_verdict: Some(BrandSafetyVerdict::Safe),
            phoenix_scores: PhoenixScores {
                video_open_score: Some(if extra { 0.1 } else { 0.9 }),
                ..Default::default()
            },
            ..Default::default()
        };
        let feed = (1..=40).map(|tweet_id| match tweet_id {
            3 | 11 => video(tweet_id, tweet_id, 0.5, false),
            21 => video(21, 11, 1.5, false),
            _ => PostCandidate {
                tweet_id,
                ..Default::default()
            },
        });
        let extras = (101..=104).map(|tweet_id| video(tweet_id, tweet_id, 0.5, true));

        let result = VideoCarouselFilter.filter(&query, feed.chain(extras).collect());

        let ids =
            |posts: &[PostCandidate]| posts.iter().map(|post| post.tweet_id).collect::<Vec<_>>();
        assert_eq!(
            ids(query.video_carousel.get().unwrap()),
            [11, 101, 102, 103, 104]
        );
        let kept = ids(&result.kept);
        assert_eq!(kept[..5], [1, 2, 3, 4, 5]);
        assert!(!kept.contains(&11) && !kept.contains(&21));
    }
}
