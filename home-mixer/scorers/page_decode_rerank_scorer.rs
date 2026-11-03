use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    ClickWeight, ContDwellTimeWeight, EnablePageDecodeRerank, FavoriteWeight, MinVideoDurationMs,
    NotDwelledWeight, NotInterestedWeight, PageDecodePoolN, PageDecodeRewardHead, PageDecodeSteps,
    PageDecodeTemperature, ProfileClickWeight, QuoteWeight, ReplyWeight, ReportWeight,
    RetweetWeight, ShareWeight, VqvWeight,
};
use std::cmp::Ordering;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_recsys_proto::{PageDecodeParams, RankingScoreWeights};
use xai_stats_receiver::global_stats_receiver;

const REQUESTS_SENT_METRIC: &str = "PageDecodeRerank.requests_sent";
const PICKS_RETURNED_METRIC: &str = "PageDecodeRerank.picks_returned";
const PICKS_APPLIED_METRIC: &str = "PageDecodeRerank.picks_applied";
const FAIL_OPEN_METRIC: &str = "PageDecodeRerank.fail_open";

pub struct PageDecodeRerankScorer;

pub(crate) fn page_decode_params(query: &ScoredPostsQuery) -> PageDecodeParams {
    PageDecodeParams {
        pool_n: query.params.get(PageDecodePoolN),
        steps: query.params.get(PageDecodeSteps),
        temperature: query.params.get(PageDecodeTemperature) as f32,
        ranking_score_weights: Some(ranking_score_weights(query)),
        diversity_weight: None,
        reward_head: Some(query.params.get(PageDecodeRewardHead)),
    }
}

fn ranking_score_weights(query: &ScoredPostsQuery) -> RankingScoreWeights {
    let params = &query.params;
    RankingScoreWeights {
        favorite_weight: params.get(FavoriteWeight) as f32,
        reply_weight: params.get(ReplyWeight) as f32,
        retweet_weight: params.get(RetweetWeight) as f32,
        click_weight: params.get(ClickWeight) as f32,
        profile_click_weight: params.get(ProfileClickWeight) as f32,
        vqv_weight: params.get(VqvWeight) as f32,
        share_weight: params.get(ShareWeight) as f32,
        quote_weight: params.get(QuoteWeight) as f32,
        cont_dwell_time_weight: params.get(ContDwellTimeWeight) as f32,
        not_interested_weight: params.get(NotInterestedWeight) as f32,
        report_weight: params.get(ReportWeight) as f32,
        not_dwelled_weight: params.get(NotDwelledWeight) as f32,
        min_video_duration_ms: params.get(MinVideoDurationMs) as f32,
    }
}

pub(crate) fn record_decode_request_sent() {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(REQUESTS_SENT_METRIC, &[], 1);
    }
}

pub(crate) fn record_decode_picks_returned(pick_count: usize) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(PICKS_RETURNED_METRIC, &[], pick_count as u64);
    }
}

fn record_rerank(metric: &str, query: &ScoredPostsQuery, value: u64) {
    if let Some(receiver) = global_stats_receiver() {
        let request = if query.has_cached_posts {
            "cached"
        } else {
            "fresh"
        };
        receiver.incr(metric, &[("request", request)], value);
    }
}

fn by_score_desc(a: Option<f64>, b: Option<f64>) -> Ordering {
    let a = a.unwrap_or(f64::NEG_INFINITY);
    let b = b.unwrap_or(f64::NEG_INFINITY);
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

pub(crate) fn decoded_on_top(
    scores: &[Option<f64>],
    slots: &[Option<u32>],
) -> Option<Vec<Option<f64>>> {
    let mut picks: Vec<(u32, usize)> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.map(|slot| (slot, index)))
        .collect();
    if picks.is_empty() {
        return None;
    }
    picks.sort_unstable();

    let mut by_score: Vec<usize> = (0..scores.len()).collect();
    by_score.sort_unstable_by(|&a, &b| by_score_desc(scores[a], scores[b]).then(a.cmp(&b)));

    let mut is_picked = vec![false; scores.len()];
    for &(_, index) in &picks {
        is_picked[index] = true;
    }
    let order = picks
        .iter()
        .map(|&(_, index)| index)
        .chain(by_score.iter().copied().filter(|&index| !is_picked[index]));
    let mut reranked = vec![None; scores.len()];
    for (rank, index) in order.enumerate() {
        reranked[index] = scores[by_score[rank]];
    }
    Some(reranked)
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PageDecodeRerankScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePageDecodeRerank)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let scores: Vec<Option<f64>> = candidates.iter().map(|c| c.score).collect();
        let slots: Vec<Option<u32>> = candidates.iter().map(|c| c.page_decode_slot).collect();
        let reranked = match decoded_on_top(&scores, &slots) {
            Some(reranked) => {
                record_rerank(
                    PICKS_APPLIED_METRIC,
                    query,
                    slots.iter().flatten().count() as u64,
                );
                reranked
            }
            None => {
                record_rerank(FAIL_OPEN_METRIC, query, 1);
                scores
            }
        };
        reranked
            .into_iter()
            .map(|score| {
                Ok(PostCandidate {
                    score,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query_with(overrides: &[(&str, &str)]) -> ScoredPostsQuery {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        for (key, value) in overrides {
            results.override_fs(key.to_string(), value);
        }
        ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        }
    }

    fn candidate(tweet_id: u64, score: f64, slot: Option<u32>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            score: Some(score),
            page_decode_slot: slot,
            page_decode_prob: slot.map(|_| 0.5),
            ..Default::default()
        }
    }

    fn sorted(scores: impl IntoIterator<Item = Option<f64>>) -> Vec<f64> {
        let mut scores: Vec<f64> = scores.into_iter().map(|s| s.unwrap()).collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        scores
    }

    async fn rerank(query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> Vec<Option<f64>> {
        PageDecodeRerankScorer
            .score(query, candidates)
            .await
            .into_iter()
            .map(|r| r.unwrap().score)
            .collect()
    }

    #[tokio::test]
    async fn fresh_decode_takes_the_top_scores_in_slot_order() {
        let query = query_with(&[("rust_home_mixer_enable_page_decode_rerank", "true")]);
        let candidates = vec![
            candidate(1, 9.0, Some(2)),
            candidate(2, 8.0, None),
            candidate(3, 7.0, Some(0)),
            candidate(4, 6.0, None),
            candidate(5, 5.0, Some(1)),
            candidate(6, 4.0, None),
        ];

        let reranked = rerank(&query, &candidates).await;

        assert_eq!(
            reranked,
            vec![
                Some(7.0),
                Some(6.0),
                Some(9.0),
                Some(5.0),
                Some(8.0),
                Some(4.0)
            ]
        );
        assert_eq!(sorted(reranked), sorted(candidates.iter().map(|c| c.score)));
    }

    #[tokio::test]
    async fn cached_decode_keeps_surviving_picks_on_top_after_filtering() {
        let mut query = query_with(&[("rust_home_mixer_enable_page_decode_rerank", "true")]);
        query.has_cached_posts = true;
        let survivors = vec![
            candidate(10, 3.0, None),
            candidate(11, 2.5, Some(5)),
            candidate(12, 9.5, None),
            candidate(13, 1.0, Some(2)),
            candidate(14, 4.0, None),
            candidate(15, 0.5, Some(7)),
        ];

        let reranked = rerank(&query, &survivors).await;

        let at = |tweet_id: u64| {
            reranked[survivors
                .iter()
                .position(|c| c.tweet_id == tweet_id)
                .unwrap()]
        };
        assert_eq!([13, 11, 15].map(at), [Some(9.5), Some(4.0), Some(3.0)]);
        assert_eq!([12, 14, 10].map(at), [Some(2.5), Some(1.0), Some(0.5)]);
        assert_eq!(sorted(reranked), sorted(survivors.iter().map(|c| c.score)));
    }

    #[tokio::test]
    async fn no_decoded_candidates_leaves_scores_untouched() {
        let query = query_with(&[("rust_home_mixer_enable_page_decode_rerank", "true")]);
        let candidates = vec![
            candidate(1, 1.0, None),
            candidate(2, 3.0, None),
            candidate(3, 2.0, None),
        ];

        let reranked = rerank(&query, &candidates).await;

        assert_eq!(reranked, vec![Some(1.0), Some(3.0), Some(2.0)]);
        assert_eq!(decoded_on_top(&[Some(1.0), Some(3.0)], &[None, None]), None);
    }

    #[test]
    fn decode_params_carry_the_current_value_model_weights() {
        let query = query_with(&[
            ("rust_home_mixer_page_decode_pool_n", "180"),
            ("rust_home_mixer_page_decode_steps", "64"),
            ("rust_home_mixer_page_decode_temperature", "0.3"),
            ("rust_home_mixer_favorite_weight", "0.7"),
            ("rust_home_mixer_reply_weight", "4.0"),
            ("rust_home_mixer_report_weight", "-200.0"),
            ("rust_home_mixer_not_dwelled_weight", "-1.5"),
            ("rust_home_mixer_min_video_duration_ms", "8000"),
        ]);

        let params = page_decode_params(&query);

        assert_eq!((params.pool_n, params.steps), (180, 64));
        assert!((params.temperature - 0.3).abs() < 1e-6);
        let weights = params.ranking_score_weights.expect("weights");
        assert!((weights.favorite_weight - 0.7).abs() < 1e-6);
        assert_eq!(weights.reply_weight, 4.0);
        assert_eq!(weights.report_weight, -200.0);
        assert_eq!(weights.not_dwelled_weight, -1.5);
        assert_eq!(weights.min_video_duration_ms, 8000.0);
    }
}
