use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePageDecodeRerank, EnablePageRlExploration, PageRlExplorationTemperature,
    PageRlExplorationTopK,
};
use rand::Rng;
use std::cmp::Ordering;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_stats_receiver::global_stats_receiver;

const REQUESTS_METRIC: &str = "PageRlExploration.requests";
const MIN_TEMPERATURE: f64 = 1e-6;

pub struct PageRlExplorationScorer;

#[derive(Debug, PartialEq)]
pub(crate) struct ExploredPage {
    pub scores: Vec<Option<f64>>,
    pub slot_probs: Vec<Option<f64>>,
}

struct SlotPick {
    pool_index: usize,
    prob: f64,
}

fn sample_plackett_luce<R: Rng>(logits: &[f64], rng: &mut R) -> Vec<SlotPick> {
    let mut remaining: Vec<usize> = (0..logits.len()).collect();
    let mut picks = Vec::with_capacity(logits.len());

    while !remaining.is_empty() {
        let max_logit = remaining
            .iter()
            .map(|&i| logits[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = remaining
            .iter()
            .map(|&i| (logits[i] - max_logit).exp())
            .collect();
        let total: f64 = weights.iter().sum();

        let mut draw = rng.random::<f64>() * total;
        let mut chosen = remaining.len() - 1;
        for (j, w) in weights.iter().enumerate() {
            draw -= w;
            if draw <= 0.0 {
                chosen = j;
                break;
            }
        }

        picks.push(SlotPick {
            pool_index: remaining[chosen],
            prob: weights[chosen] / total,
        });
        remaining.remove(chosen);
    }

    picks
}

fn top_k_by_score(scores: &[Option<f64>], k: usize) -> Vec<usize> {
    let cmp = |a: &usize, b: &usize| {
        let sa = scores[*a].unwrap_or(f64::NEG_INFINITY);
        let sb = scores[*b].unwrap_or(f64::NEG_INFINITY);
        sb.partial_cmp(&sa)
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    };
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    if indices.len() > k {
        indices.select_nth_unstable_by(k, cmp);
        indices.truncate(k);
    }
    indices.sort_unstable_by(cmp);
    indices
}

pub(crate) fn explore_top_k<R: Rng>(
    scores: &[Option<f64>],
    top_k: usize,
    temperature: f64,
    rng: &mut R,
) -> Option<ExploredPage> {
    let pool = top_k_by_score(scores, top_k);
    if pool.len() < 2 {
        return None;
    }

    let tau = temperature.max(MIN_TEMPERATURE);
    let logits: Vec<f64> = pool
        .iter()
        .map(|&i| scores[i].unwrap_or(0.0) / tau)
        .collect();
    let picks = sample_plackett_luce(&logits, rng);

    let mut explored = ExploredPage {
        scores: scores.to_vec(),
        slot_probs: vec![None; scores.len()],
    };
    for (slot, pick) in picks.iter().enumerate() {
        let candidate_index = pool[pick.pool_index];
        explored.scores[candidate_index] = scores[pool[slot]];
        explored.slot_probs[candidate_index] = Some(pick.prob);
    }
    Some(explored)
}

fn record_request(outcome: &str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(REQUESTS_METRIC, &[("outcome", outcome)], 1);
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PageRlExplorationScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePageRlExploration)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let scores: Vec<Option<f64>> = candidates.iter().map(|c| c.score).collect();
        let decoded = query.params.get(EnablePageDecodeRerank)
            && candidates.iter().any(|c| c.page_decode_slot.is_some());
        if decoded {
            record_request("decoded");
            return scores
                .into_iter()
                .map(|score| {
                    Ok(PostCandidate {
                        score,
                        ..Default::default()
                    })
                })
                .collect();
        }
        let explored = explore_top_k(
            &scores,
            query.params.get(PageRlExplorationTopK),
            query.params.get(PageRlExplorationTemperature),
            &mut rand::rng(),
        );
        let explored = match explored {
            Some(explored) => {
                record_request("sampled");
                explored
            }
            None => {
                record_request("pool_too_small");
                ExploredPage {
                    slot_probs: vec![None; scores.len()],
                    scores,
                }
            }
        };
        explored
            .scores
            .into_iter()
            .zip(explored.slot_probs)
            .map(|(score, page_rl_exploration_prob)| {
                Ok(PostCandidate {
                    score,
                    page_rl_exploration_prob,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
        candidate.page_rl_exploration_prob = scored.page_rl_exploration_prob;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn descending(scores: impl IntoIterator<Item = Option<f64>>) -> Vec<f64> {
        let mut scores: Vec<f64> = scores.into_iter().map(|s| s.unwrap()).collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        scores
    }

    #[test]
    fn seeded_sampling_permutes_only_the_top_k() {
        let scores: Vec<Option<f64>> = [0.31, 0.92, 0.15, 0.77, 0.64, 0.08, 0.55, 0.43]
            .into_iter()
            .map(Some)
            .collect();
        let top_k = 5;
        let top_k_indices = top_k_by_score(&scores, top_k);

        let explored = explore_top_k(&scores, top_k, 0.2, &mut StdRng::seed_from_u64(11))
            .expect("pool is large enough");

        assert_eq!(top_k_indices, vec![1, 3, 4, 6, 7]);
        assert_eq!(
            descending(top_k_indices.iter().map(|&i| explored.scores[i])),
            descending(top_k_indices.iter().map(|&i| scores[i]))
        );
        for (i, score) in scores.iter().enumerate() {
            if top_k_indices.contains(&i) {
                let prob = explored.slot_probs[i].expect("placed candidates carry a probability");
                assert!(prob > 0.0 && prob <= 1.0, "prob {prob} out of (0, 1]");
            } else {
                assert_eq!(explored.scores[i], *score);
                assert_eq!(explored.slot_probs[i], None);
            }
        }
        assert_eq!(
            explore_top_k(&scores, top_k, 0.2, &mut StdRng::seed_from_u64(11)),
            Some(explored)
        );
    }

    #[test]
    fn near_zero_temperature_reproduces_score_order() {
        let scores: Vec<Option<f64>> = [2.0, 5.0, 1.0, 4.0, 3.0].into_iter().map(Some).collect();

        let explored = explore_top_k(&scores, 4, 0.0, &mut StdRng::seed_from_u64(3))
            .expect("pool is large enough");

        assert_eq!(explored.scores, scores);
        assert_eq!(
            explored.slot_probs,
            vec![Some(1.0), Some(1.0), None, Some(1.0), Some(1.0)]
        );
    }

    #[tokio::test]
    async fn scorer_records_probabilities_and_keeps_the_score_multiset() {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_page_rl_exploration".to_string(),
            "true",
        );
        results.override_fs(
            "rust_home_mixer_page_rl_exploration_temperature".to_string(),
            "0.2",
        );
        results.override_fs("rust_home_mixer_page_rl_exploration_top_k".to_string(), "3");
        let query = ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        };
        let candidates: Vec<PostCandidate> = [0.4, 0.9, 0.1, 0.7, 0.3]
            .into_iter()
            .map(|score| PostCandidate {
                score: Some(score),
                page_rl_exploration_prob: Some(0.99),
                ..Default::default()
            })
            .collect();

        let scored: Vec<PostCandidate> = PageRlExplorationScorer
            .score(&query, &candidates)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(
            descending(scored.iter().map(|c| c.score)),
            descending(candidates.iter().map(|c| c.score))
        );
        let placed: Vec<usize> = (0..scored.len())
            .filter(|&i| scored[i].page_rl_exploration_prob.is_some())
            .collect();
        assert_eq!(placed, vec![0, 1, 3]);
        assert_eq!(scored[2].score, Some(0.1));
        assert_eq!(scored[4].score, Some(0.3));
    }
}
