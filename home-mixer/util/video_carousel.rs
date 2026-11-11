use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{
    EnableVideoCarousel, VIDEO_CAROUSEL_MAX_VIDEOS, VIDEO_CAROUSEL_MIN_VIDEOS,
    VideoCarouselImmersiveVqvWeight, VideoCarouselNotInterestedWeight,
    VideoCarouselOpenTimesImmersiveDwellWeight, VideoCarouselVideoContinuationWeight,
    VideoCarouselVideoOpenWeight, VideoCarouselWeightedScoreWeight,
};
use crate::util::video_carousel_ctr_model::CtrModel;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

const MAX_VERTICAL_ASPECT_RATIO: f32 = 1.0;

pub(crate) fn is_enabled(query: &ScoredPostsQuery) -> bool {
    query.request_type == RequestType::ForYou
        && query.params.get(EnableVideoCarousel)
        && query.video_carousel_eligible
        && !query.is_bottom_request
}

pub(crate) fn was_seen(seen: &HashSet<u64>, candidate: &PostCandidate) -> bool {
    seen.contains(&candidate.tweet_id) || seen.contains(&candidate.get_original_tweet_id())
}

pub(crate) fn shows_carousel_video(post: &PostCandidate, carousel: &[PostCandidate]) -> bool {
    carousel.iter().any(|video| {
        video.tweet_id == post.tweet_id
            || video
                .video_media_id
                .is_some_and(|id| post.video_media_id == Some(id))
            || md5_hashes(video).any(|hash| md5_hashes(post).any(|other| other == hash))
    })
}

pub(crate) fn ranks_by_immersive_scores(query: &ScoredPostsQuery) -> bool {
    query.request_type == RequestType::ForYou
        && query.params.get(EnableVideoCarousel)
        && CarouselScoring::from_query(query).needs_immersive_scores()
        && !query.is_bottom_request
}

#[derive(Clone, Debug)]
pub(crate) struct CarouselScoring {
    video_open_weight: f64,
    immersive_vqv_weight: f64,
    open_times_immersive_dwell_weight: f64,
    weighted_score_weight: f64,
    video_continuation_weight: f64,
    not_interested_weight: f64,
    ctr_model: Option<Arc<CtrModel>>,
}

impl CarouselScoring {
    pub(crate) fn from_query(query: &ScoredPostsQuery) -> Self {
        Self {
            video_open_weight: query.params.get(VideoCarouselVideoOpenWeight),
            immersive_vqv_weight: query.params.get(VideoCarouselImmersiveVqvWeight),
            open_times_immersive_dwell_weight: query
                .params
                .get(VideoCarouselOpenTimesImmersiveDwellWeight),
            weighted_score_weight: query.params.get(VideoCarouselWeightedScoreWeight),
            video_continuation_weight: query.params.get(VideoCarouselVideoContinuationWeight),
            not_interested_weight: query.params.get(VideoCarouselNotInterestedWeight),
            ctr_model: query.video_carousel_ctr_model.get().cloned(),
        }
    }

    pub(crate) fn score(&self, candidate: &PostCandidate) -> f64 {
        if let Some(model) = &self.ctr_model {
            return model.score(
                video_open_score(candidate),
                candidate.video_embedding.as_deref(),
            );
        }
        self.video_open_weight * video_open_score(candidate)
            + self.immersive_vqv_weight * immersive_vqv_score(candidate)
            + self.open_times_immersive_dwell_weight
                * video_open_score(candidate)
                * immersive_dwell_time(candidate)
            + self.weighted_score_weight * candidate.weighted_score.unwrap_or(0.0)
            + self.video_continuation_weight
                * video_open_score(candidate)
                * video_continuation_secs(candidate)
            + self.not_interested_weight * not_interested_score(candidate)
    }

    pub(crate) fn needs_immersive_scores(&self) -> bool {
        self.immersive_vqv_weight != 0.0 || self.open_times_immersive_dwell_weight != 0.0
    }

    fn rank(&self, a: &PostCandidate, b: &PostCandidate) -> Ordering {
        self.score(b)
            .total_cmp(&self.score(a))
            .then_with(|| video_open_score(b).total_cmp(&video_open_score(a)))
    }
}

pub(crate) fn select_videos<'a>(
    candidates: impl IntoIterator<Item = &'a PostCandidate>,
    main_feed: &'a [PostCandidate],
    scoring: &CarouselScoring,
    dedupe_by_author: bool,
    semantic_id_dedupe_levels: usize,
    embedding_dedupe_threshold: Option<f64>,
) -> Vec<PostCandidate> {
    let mut videos: Vec<&PostCandidate> = candidates
        .into_iter()
        .filter(|candidate| is_carousel_video(candidate) && is_vertical(candidate))
        .collect();
    videos.sort_by(|a, b| scoring.rank(a, b));
    let videos = take_distinct(
        videos,
        main_feed,
        dedupe_by_author,
        semantic_id_dedupe_levels,
        embedding_dedupe_threshold,
    );
    if videos.len() < VIDEO_CAROUSEL_MIN_VIDEOS {
        return Vec::new();
    }
    videos.into_iter().cloned().collect()
}

fn take_distinct<'a>(
    ranked: Vec<&'a PostCandidate>,
    main_feed: &'a [PostCandidate],
    dedupe_by_author: bool,
    semantic_id_dedupe_levels: usize,
    embedding_dedupe_threshold: Option<f64>,
) -> Vec<&'a PostCandidate> {
    let mut authors = HashSet::new();
    let mut prefixes: HashSet<&[i32]> = HashSet::new();
    let mut embeddings: Vec<&[f32]> = Vec::new();
    let mut media_ids: HashSet<u64> = main_feed
        .iter()
        .filter_map(|post| post.video_media_id)
        .collect();
    let mut hashes: HashSet<&str> = main_feed.iter().flat_map(md5_hashes).collect();
    let mut kept = Vec::with_capacity(VIDEO_CAROUSEL_MAX_VIDEOS);
    for video in ranked {
        if kept.len() == VIDEO_CAROUSEL_MAX_VIDEOS {
            break;
        }
        if video
            .video_media_id
            .is_some_and(|id| media_ids.contains(&id))
            || md5_hashes(video).any(|hash| hashes.contains(hash))
        {
            continue;
        }
        let author = video.get_original_author_id();
        if dedupe_by_author && authors.contains(&author) {
            continue;
        }
        let prefix = video.semantic_id_prefix(semantic_id_dedupe_levels);
        if prefix.is_some_and(|prefix| prefixes.contains(prefix)) {
            continue;
        }
        let embedding = match (embedding_dedupe_threshold, &video.video_embedding) {
            (Some(_), Some(embedding)) if video.quoted_tweet_id.is_none() => {
                Some(embedding.as_slice())
            }
            _ => None,
        };
        if let (Some(threshold), Some(embedding)) = (embedding_dedupe_threshold, embedding)
            && embeddings
                .iter()
                .any(|picked| cosine_similarity(picked, embedding) >= threshold)
        {
            continue;
        }
        authors.insert(author);
        prefixes.extend(prefix);
        embeddings.extend(embedding);
        media_ids.extend(video.video_media_id);
        hashes.extend(md5_hashes(video));
        kept.push(video);
    }
    kept
}

fn md5_hashes(post: &PostCandidate) -> impl Iterator<Item = &str> {
    post.video_md5_hashes.iter().flatten().map(String::as_str)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b)
        .fold((0.0, 0.0, 0.0), |(d, na, nb), (&x, &y)| {
            let (x, y) = (f64::from(x), f64::from(y));
            (d + x * y, na + x * x, nb + y * y)
        });
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub(crate) fn take_videos_by_carousel_score(
    candidates: &mut Vec<PostCandidate>,
    count: usize,
    scoring: &CarouselScoring,
    seen: &HashSet<u64>,
) -> Vec<PostCandidate> {
    let (mut videos, rest): (Vec<_>, Vec<_>) = std::mem::take(candidates)
        .into_iter()
        .partition(|candidate| is_carousel_video(candidate) && !was_seen(seen, candidate));
    videos.sort_by(|a, b| scoring.rank(a, b));
    let overflow = videos.split_off(count.min(videos.len()));
    *candidates = rest;
    candidates.extend(overflow);
    for video in &mut videos {
        video.video_carousel_extra = true;
    }
    videos
}

pub(crate) fn is_carousel_video(candidate: &PostCandidate) -> bool {
    candidate.has_video == Some(true)
        && candidate.in_reply_to_tweet_id.is_none()
        && candidate.phoenix_scores.video_open_score.is_some()
}

fn is_vertical(candidate: &PostCandidate) -> bool {
    candidate
        .video_aspect_ratio
        .is_some_and(|ratio| ratio < MAX_VERTICAL_ASPECT_RATIO)
}

fn video_open_score(candidate: &PostCandidate) -> f64 {
    candidate.phoenix_scores.video_open_score.unwrap_or(0.0)
}

fn immersive_vqv_score(candidate: &PostCandidate) -> f64 {
    candidate
        .immersive_scores
        .as_ref()
        .and_then(|scores| scores.vqv_score)
        .unwrap_or(0.0)
}

fn immersive_dwell_time(candidate: &PostCandidate) -> f64 {
    candidate
        .immersive_scores
        .as_ref()
        .and_then(|scores| scores.dwell_time)
        .unwrap_or(0.0)
}

fn video_continuation_secs(candidate: &PostCandidate) -> f64 {
    candidate
        .phoenix_scores
        .home_video_continuation_secs
        .unwrap_or(0.0)
}

fn not_interested_score(candidate: &PostCandidate) -> f64 {
    candidate.phoenix_scores.not_interested_score.unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scoring(video_open_weight: f64, immersive_vqv_weight: f64) -> CarouselScoring {
        CarouselScoring {
            video_open_weight,
            immersive_vqv_weight,
            open_times_immersive_dwell_weight: 0.0,
            weighted_score_weight: 0.0,
            video_continuation_weight: 0.0,
            not_interested_weight: 0.0,
            ctr_model: None,
        }
    }
    use crate::models::candidate::PhoenixScores;

    fn video(tweet_id: u64, aspect_ratio: f32, video_open: f64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            has_video: Some(true),
            video_aspect_ratio: Some(aspect_ratio),
            phoenix_scores: PhoenixScores {
                video_open_score: Some(video_open),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn tweet_ids(candidates: &[PostCandidate]) -> Vec<u64> {
        candidates
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect()
    }

    #[test]
    fn keeps_vertical_videos_ranked_by_video_open() {
        let candidates = vec![
            video(1, 0.5625, 0.1),
            video(2, 1.7778, 0.9),
            video(3, 1.0, 0.8),
            video(4, 0.5625, 0.5),
            video(5, 0.75, 0.3),
            PostCandidate {
                in_reply_to_tweet_id: Some(9),
                ..video(6, 0.5625, 0.7)
            },
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &scoring(1.0, 0.0),
                false,
                0,
                None
            )),
            vec![4, 5, 1]
        );
    }

    fn vertical(tweet_id: u64, author_id: u64, sid: &[i32], video_open: f64) -> PostCandidate {
        PostCandidate {
            author_id,
            semantic_ids: (!sid.is_empty()).then(|| sid.to_vec()),
            ..video(tweet_id, 0.5625, video_open)
        }
    }

    #[test]
    fn author_dedupe_keeps_the_top_video_per_original_author() {
        let candidates = vec![
            vertical(1, 10, &[], 0.9),
            PostCandidate {
                retweeted_user_id: Some(10),
                ..vertical(2, 20, &[], 0.8)
            },
            vertical(3, 10, &[], 0.7),
            vertical(4, 30, &[], 0.6),
            vertical(5, 40, &[], 0.5),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &scoring(1.0, 0.0),
                true,
                0,
                None
            )),
            vec![1, 4, 5]
        );
    }

    #[test]
    fn semantic_id_dedupe_matches_full_prefixes_only() {
        let candidates = vec![
            vertical(1, 10, &[1, 2, 3], 0.9),
            vertical(2, 20, &[1, 2, 9], 0.8),
            vertical(3, 30, &[1, 5, 3], 0.7),
            vertical(4, 40, &[], 0.6),
            vertical(5, 50, &[1], 0.5),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &scoring(1.0, 0.0),
                false,
                2,
                None
            )),
            vec![1, 3, 4, 5]
        );
    }

    #[test]
    fn dedupe_fills_from_lower_ranks_up_to_the_max() {
        let candidates: Vec<_> = (1..=8)
            .map(|id| {
                vertical(
                    id,
                    if id <= 3 { 10 } else { id },
                    &[],
                    1.0 - id as f64 / 10.0,
                )
            })
            .collect();

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &scoring(1.0, 0.0),
                true,
                0,
                None
            )),
            vec![1, 4, 5, 6, 7]
        );
    }

    #[test]
    fn dedupe_below_the_minimum_drops_the_carousel() {
        let candidates = vec![
            vertical(1, 10, &[], 0.9),
            vertical(2, 10, &[], 0.8),
            vertical(3, 20, &[], 0.7),
        ];

        assert!(select_videos(&candidates, &[], &scoring(1.0, 0.0), true, 0, None).is_empty());
    }

    fn with_immersive_vqv(candidate: PostCandidate, vqv: f64) -> PostCandidate {
        PostCandidate {
            immersive_scores: Some(PhoenixScores {
                vqv_score: Some(vqv),
                ..Default::default()
            }),
            ..candidate
        }
    }

    fn with_not_interested(mut candidate: PostCandidate, score: f64) -> PostCandidate {
        candidate.phoenix_scores.not_interested_score = Some(score);
        candidate
    }

    #[test]
    fn ranks_by_weighted_sum_with_missing_scores_as_zero() {
        let candidates = vec![
            with_immersive_vqv(vertical(1, 10, &[], 0.9), 0.1),
            with_not_interested(with_immersive_vqv(vertical(2, 20, &[], 0.2), 0.9), 0.02),
            vertical(3, 30, &[], 0.5),
            with_immersive_vqv(vertical(4, 40, &[], 0.3), 0.95),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &CarouselScoring {
                    not_interested_weight: -25.0,
                    ..scoring(1.0, 1.0)
                },
                false,
                0,
                None
            )),
            vec![4, 1, 2, 3]
        );
    }

    fn with_continuation(mut candidate: PostCandidate, seconds: f64) -> PostCandidate {
        candidate.phoenix_scores.home_video_continuation_secs = Some(seconds);
        candidate
    }

    #[test]
    fn ranks_by_open_times_continuation_with_missing_continuation_as_zero() {
        let candidates = vec![
            with_continuation(vertical(1, 10, &[], 0.9), 2.0),
            with_continuation(vertical(2, 20, &[], 0.1), 10.0),
            with_continuation(vertical(3, 30, &[], 0.5), 3.0),
            vertical(4, 40, &[], 0.8),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &CarouselScoring {
                    video_continuation_weight: 1.0,
                    ..scoring(0.0, 0.0)
                },
                false,
                0,
                None
            )),
            vec![1, 3, 2, 4]
        );
    }

    fn with_embedding(mut candidate: PostCandidate, embedding: &[f32]) -> PostCandidate {
        candidate.video_embedding = Some(embedding.to_vec());
        candidate
    }

    #[test]
    fn embedding_dedupe_skips_near_duplicates_except_quotes() {
        let candidates = vec![
            with_embedding(vertical(1, 10, &[], 0.9), &[1.0, 0.0]),
            with_embedding(vertical(2, 20, &[], 0.8), &[0.99, 0.1]),
            with_embedding(vertical(3, 30, &[], 0.7), &[0.0, 1.0]),
            PostCandidate {
                quoted_tweet_id: Some(99),
                ..with_embedding(vertical(4, 40, &[], 0.6), &[1.0, 0.0])
            },
            vertical(5, 50, &[], 0.5),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &scoring(1.0, 0.0),
                false,
                0,
                Some(0.72)
            )),
            vec![1, 3, 4, 5]
        );
    }

    fn with_media(candidate: PostCandidate, media_id: Option<u64>, md5: &[&str]) -> PostCandidate {
        PostCandidate {
            video_media_id: media_id,
            video_md5_hashes: (!md5.is_empty())
                .then(|| md5.iter().map(|hash| hash.to_string()).collect()),
            ..candidate
        }
    }

    #[test]
    fn media_dedupe_skips_repeated_media_ids_and_md5_hashes() {
        let candidates = vec![
            with_media(vertical(1, 10, &[], 0.9), Some(100), &["a"]),
            with_media(vertical(2, 20, &[], 0.8), Some(100), &[]),
            with_media(vertical(3, 30, &[], 0.7), Some(300), &["b", "a"]),
            with_media(vertical(4, 40, &[], 0.6), Some(400), &["c"]),
            vertical(5, 50, &[], 0.5),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &[],
                &scoring(1.0, 0.0),
                false,
                0,
                None
            )),
            vec![1, 4, 5]
        );
    }

    #[test]
    fn media_dedupe_skips_videos_already_in_the_main_feed() {
        let main_feed = vec![
            with_media(video(90, 1.7778, 0.9), Some(100), &[]),
            with_media(video(91, 0.5625, 0.8), None, &["a"]),
        ];
        let candidates = vec![
            with_media(vertical(1, 10, &[], 0.9), Some(100), &[]),
            with_media(vertical(2, 20, &[], 0.8), Some(200), &["a"]),
            vertical(3, 30, &[], 0.7),
            vertical(4, 40, &[], 0.6),
            vertical(5, 50, &[], 0.5),
        ];

        assert_eq!(
            tweet_ids(&select_videos(
                &candidates,
                &main_feed,
                &scoring(1.0, 0.0),
                false,
                0,
                None
            )),
            vec![3, 4, 5]
        );
    }
}
