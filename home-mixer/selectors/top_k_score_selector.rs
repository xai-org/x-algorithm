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
        let non_selected =
            selected.split_off(params::TOP_K_CANDIDATES_TO_SELECT.min(selected.len()));
        if video_carousel::is_enabled(query) {
            selected.extend(video_carousel::top_videos_by_carousel_score(
                &non_selected,
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
        assert_eq!(
            tweet_ids(&result.non_selected),
            vec![101, 102, 103, 104, 105, 106]
        );
        assert!(result
            .non_selected
            .iter()
            .all(|candidate| !candidate.video_carousel_extra));
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

    fn plain_query() -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs("rust_home_mixer_enable_video_carousel".to_string(), "false");
        ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        }
    }

    fn populated_candidates(count: usize) -> Vec<PostCandidate> {
        (0..count)
            .map(|i| PostCandidate {
                author_id: 1000 + (i % 600) as u64,
                tweet_text: format!(
                    "Post {i}: a realistic timeline text with discussion, links and Unicode café. {}",
                    "Shared content and contextual details. ".repeat(3)
                ),
                author_screen_name: Some(format!("author_{}", i % 600)),
                ancestors: if i % 7 == 0 { vec![10_000 + i as u64] } else { vec![] },
                semantic_ids: Some(vec![(i % 32) as i32, (i % 128) as i32, (i % 512) as i32]),
                video_embedding: (i % 10 == 0).then(|| vec![0.25; 64]),
                phoenix_scores: PhoenixScores {
                    favorite_score: Some((i % 100) as f64 / 100.0),
                    reply_score: Some((i % 25) as f64 / 100.0),
                    video_open_score: Some((i % 50) as f64 / 100.0),
                    ..Default::default()
                },
                ..post(1 + i as u64, ((i * 691) % count) as f64)
            })
            .collect()
    }

    #[test]
    fn selection_preserves_stable_ties_and_complete_candidate_payloads() {
        let query = plain_query();
        let mut candidates = populated_candidates(3000);
        for candidate in &mut candidates {
            candidate.score = Some((candidate.tweet_id % 5) as f64);
        }
        let originals: std::collections::HashMap<_, _> =
            candidates.iter().map(|c| (c.tweet_id, c.clone())).collect();
        let mut expected = Vec::new();
        for score in (0..5).rev() {
            expected.extend(
                candidates
                    .iter()
                    .filter(|c| c.score == Some(score as f64))
                    .map(|c| c.tweet_id),
            );
        }
        let result = TopKScoreSelector.select(&query, candidates);
        let actual: Vec<_> = result.selected.iter().chain(&result.non_selected).collect();
        assert_eq!(
            actual.iter().map(|c| c.tweet_id).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(result.selected.len(), params::TOP_K_CANDIDATES_TO_SELECT);
        for candidate in actual {
            let original = &originals[&candidate.tweet_id];
            assert_eq!(
                serde_json::to_value(candidate).unwrap(),
                serde_json::to_value(original).unwrap()
            );
            assert_eq!(candidate.video_embedding, original.video_embedding);
        }
    }

    #[test]
    fn selection_handles_missing_scores_infinities_and_signed_zero_stably() {
        let query = plain_query();
        let candidates = vec![
            post(1, -0.0),
            post(2, f64::INFINITY),
            PostCandidate {
                score: None,
                ..post(3, 0.0)
            },
            post(4, 0.0),
            post(5, f64::NEG_INFINITY),
        ];
        let result = TopKScoreSelector.select(&query, candidates.clone());
        let expected: Vec<_> = [1, 0, 3, 2, 4]
            .into_iter()
            .map(|i| candidates[i].tweet_id)
            .collect();
        assert_eq!(tweet_ids(&result.selected), expected);
        assert!(result.non_selected.is_empty());
        assert!(TopKScoreSelector.select(&query, vec![]).is_empty());
    }

    #[test]
    fn large_batch_carousel_keeps_score_order_and_appends_only_requested_extras() {
        let mut query = carousel_query("1.0", "0.0");
        let fs = query.params.0.as_mut().unwrap();
        for key in [
            "rust_home_mixer_video_carousel_open_times_immersive_dwell_weight",
            "rust_home_mixer_video_carousel_weighted_score_weight",
            "rust_home_mixer_video_carousel_video_continuation_weight",
        ] {
            fs.override_fs(key.to_string(), "0.0");
        }
        fs.override_fs(
            "rust_home_mixer_video_carousel_extra_candidates".to_string(),
            "3",
        );
        let base = populated_candidates(3000);
        let mut expected = base
            .iter()
            .map(|c| (c.score.unwrap() as usize, c.tweet_id))
            .collect::<Vec<_>>();
        expected.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        let mut input = base;
        let extras = [video(3001, 0.9), video(3002, 0.2), video(3003, 0.5)];
        input.extend(extras.iter().cloned());
        let result = TopKScoreSelector.select(&query, input);
        let top_k = params::TOP_K_CANDIDATES_TO_SELECT;
        assert_eq!(
            tweet_ids(&result.selected[..top_k]),
            expected[..top_k]
                .iter()
                .map(|(_, id)| *id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            tweet_ids(&result.selected[top_k..]),
            [extras[0].tweet_id, extras[2].tweet_id, extras[1].tweet_id]
        );
        assert_eq!(
            tweet_ids(&result.non_selected),
            expected[top_k..]
                .iter()
                .map(|(_, id)| *id)
                .chain(extras.iter().map(|c| c.tweet_id))
                .collect::<Vec<_>>()
        );
        assert!(result.selected[..top_k]
            .iter()
            .all(|c| !c.video_carousel_extra));
        assert!(result.selected[top_k..]
            .iter()
            .all(|c| c.video_carousel_extra));
    }

    #[test]
    fn benchmark_selector_populated_3000() {
        let query = plain_query();
        let candidates = populated_candidates(3000);
        let mut elapsed = Vec::new();
        for iteration in 0..35 {
            let input = candidates.clone();
            let start = std::time::Instant::now();
            let result = std::hint::black_box(TopKScoreSelector.select(&query, input));
            let ns = start.elapsed().as_nanos();
            let scores: Vec<_> = result
                .selected
                .iter()
                .chain(&result.non_selected)
                .map(|c| c.score.unwrap())
                .collect();
            assert_eq!(
                scores,
                (0..3000).rev().map(|i| i as f64).collect::<Vec<_>>()
            );
            assert_eq!(
                result.selected.len() + result.non_selected.len(),
                candidates.len()
            );
            if iteration >= 5 {
                elapsed.push(ns);
                println!("selector_populated_3000 sample={} ns={ns}", iteration - 5);
            }
        }
        elapsed.sort_unstable();
        let median = elapsed[elapsed.len() / 2];
        let mean = elapsed.iter().sum::<u128>() as f64 / elapsed.len() as f64;
        println!(
            "selector_populated_3000 candidates=3000 warmup=5 samples=30 median_ns={median} mean_ns={mean:.0} requests_per_second={:.2}",
            1_000_000_000.0 / mean
        );
    }
}
