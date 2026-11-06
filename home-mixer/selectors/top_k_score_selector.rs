use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{self, VideoCarouselExtraCandidates};
use crate::util::video_carousel::{self, CarouselScoring};
use xai_candidate_pipeline::selector::{SelectResult, Selector};

pub struct TopKScoreSelector;

impl Selector<ScoredPostsQuery, PostCandidate> for TopKScoreSelector {
    fn select(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> SelectResult<PostCandidate> {
        let mut selected = self.sort(candidates);
        let mut non_selected =
            selected.split_off(params::TOP_K_CANDIDATES_TO_SELECT.min(selected.len()));
        if video_carousel::is_enabled(query) {
            selected.extend(video_carousel::take_videos_by_carousel_score(
                &mut non_selected,
                query.params.get(VideoCarouselExtraCandidates),
                &CarouselScoring::from_query(query),
                &query.seen_carousel_video_ids,
            ));
        }
        SelectResult {
            selected,
            non_selected,
        }
    }

    fn score(&self, candidate: &PostCandidate) -> f64 {
        candidate.score.unwrap_or(f64::NEG_INFINITY)
    }

    fn size(&self) -> Option<usize> {
        Some(params::TOP_K_CANDIDATES_TO_SELECT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::PhoenixScores;
    use std::collections::HashSet;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};

    fn carousel_query(video_open_weight: &str, immersive_vqv_weight: &str) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_video_carousel".to_string(), "true");
        results.override_fs(
            "rust_home_mixer_video_carousel_video_open_weight".to_string(),
            video_open_weight,
        );
        results.override_fs(
            "rust_home_mixer_video_carousel_immersive_vqv_weight".to_string(),
            immersive_vqv_weight,
        );
        let params: Params = results.into();
        ScoredPostsQuery {
            params,
            video_carousel_eligible: true,
            ..Default::default()
        }
    }

    fn post(tweet_id: u64, score: f64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            score: Some(score),
            ..Default::default()
        }
    }

    fn video(tweet_id: u64, video_open: f64) -> PostCandidate {
        PostCandidate {
            has_video: Some(true),
            phoenix_scores: PhoenixScores {
                video_open_score: Some(video_open),
                ..Default::default()
            },
            ..post(tweet_id, 0.0)
        }
    }

    fn candidates() -> Vec<PostCandidate> {
        let top_k = params::TOP_K_CANDIDATES_TO_SELECT as u64;
        let mut candidates: Vec<PostCandidate> = (1..=top_k)
            .map(|id| post(id, (top_k + 1 - id) as f64))
            .collect();
        candidates.extend([
            video(101, 0.2),
            PostCandidate {
                has_video: Some(false),
                ..video(102, 0.9)
            },
            video(103, 0.8),
            PostCandidate {
                in_reply_to_tweet_id: Some(1),
                ..video(104, 0.95)
            },
            PostCandidate {
                phoenix_scores: PhoenixScores::default(),
                ..video(105, 0.0)
            },
            video(106, 0.5),
        ]);
        candidates
    }

    fn tweet_ids(candidates: &[PostCandidate]) -> Vec<u64> {
        candidates
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect()
    }

    #[test]
    fn carousel_adds_videos_by_video_open_after_the_top_k() {
        let result = TopKScoreSelector.select(&carousel_query("1.0", "0.0"), candidates());

        let mut expected: Vec<u64> = (1..=params::TOP_K_CANDIDATES_TO_SELECT as u64).collect();
        expected.extend([103, 106, 101]);
        assert_eq!(tweet_ids(&result.selected), expected);
        assert_eq!(tweet_ids(&result.non_selected), vec![102, 104, 105]);
        let (top_k, extras) = result.selected.split_at(params::TOP_K_CANDIDATES_TO_SELECT);
        assert!(top_k
            .iter()
            .all(|candidate| !candidate.video_carousel_extra));
        assert!(extras
            .iter()
            .all(|candidate| candidate.video_carousel_extra));
    }

    #[test]
    fn carousel_extras_follow_the_carousel_score_weights() {
        let immersive_vqv = |tweet_id| match tweet_id {
            101 => Some(0.9),
            103 => Some(0.1),
            106 => Some(0.5),
            _ => None,
        };
        let candidates = candidates()
            .into_iter()
            .map(|candidate| PostCandidate {
                immersive_scores: immersive_vqv(candidate.tweet_id).map(|vqv| PhoenixScores {
                    vqv_score: Some(vqv),
                    ..Default::default()
                }),
                ..candidate
            })
            .collect();

        let result = TopKScoreSelector.select(&carousel_query("0.0", "1.0"), candidates);

        let (_, extras) = result.selected.split_at(params::TOP_K_CANDIDATES_TO_SELECT);
        assert_eq!(tweet_ids(extras), vec![101, 106, 103]);
    }

    #[test]
    fn carousel_extras_skip_videos_seen_in_a_carousel() {
        let query = ScoredPostsQuery {
            seen_carousel_video_ids: HashSet::from([101, 103]),
            ..carousel_query("1.0", "0.0")
        };

        let result = TopKScoreSelector.select(&query, candidates());

        let (_, extras) = result.selected.split_at(params::TOP_K_CANDIDATES_TO_SELECT);
        assert_eq!(tweet_ids(extras), vec![106]);
    }
}
