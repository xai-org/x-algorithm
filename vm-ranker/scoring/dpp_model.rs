use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use half::f16;
use log::info;
use rand::Rng;
use xai_vm_ranker_proto::{RankCandidate, RankRequest};

use super::DppContext;
use crate::dpp::{self, DppInput};
use crate::metrics::{DPP_SEED_CONTEXT, DPP_WINDOW_HISTORY_ITEMS};

const HISTORY_MAX_AGE_MS: i64 = 10 * 60 * 1000;

fn l2_norm(v: &[f16]) -> f64 {
    v.iter()
        .map(|x| {
            let xf = x.to_f32();
            xf * xf
        })
        .sum::<f32>()
        .sqrt() as f64
}

fn random_unit_embedding(dim: usize) -> Arc<Vec<f16>> {
    let mut rng = rand::rng();
    let v: Vec<f32> = (0..dim)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let emb = if norm > 0.0 {
        v.into_iter().map(|x| f16::from_f32(x / norm)).collect()
    } else {
        let mut u = vec![f16::ZERO; dim];
        u[0] = f16::ONE;
        u
    };
    Arc::new(emb)
}

#[derive(Default)]
pub struct EmbeddingMemo(HashMap<u64, (Arc<Vec<f16>>, bool)>);

impl EmbeddingMemo {
    fn resolve(&mut self, embedding_id: u64, ctx: &DppContext) -> (Arc<Vec<f16>>, bool) {
        let (embedding, missing) = self.0.entry(embedding_id).or_insert_with(|| {
            match ctx.store.client.get(embedding_id) {
                Some(arc) => (arc, false),
                None => (random_unit_embedding(ctx.store.dim()), true),
            }
        });
        (Arc::clone(embedding), *missing)
    }
}

fn build_dpp_inputs(
    req: &RankRequest,
    scores: &[Option<f64>],
    ctx: &DppContext,
    memo: &mut EmbeddingMemo,
) -> Vec<DppInput> {
    let max_rank = ctx.config.max_selected_rank;

    let mut scored: Vec<(usize, &RankCandidate, f64)> = req
        .candidates
        .iter()
        .zip(scores)
        .enumerate()
        .filter_map(|(i, (c, s))| s.map(|s| (i, c, s)))
        .collect();

    scored.sort_unstable_by(|(i_a, _, s_a), (i_b, _, s_b)| {
        s_b.partial_cmp(s_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| i_a.cmp(i_b))
    });

    scored
        .into_iter()
        .take(max_rank)
        .map(|(_, c, score)| {
            let embedding_id = if c.retweeted_tweet_id != 0 {
                c.retweeted_tweet_id
            } else {
                c.tweet_id
            };
            let (embedding, embedding_missing) = memo.resolve(embedding_id, ctx);
            let norm = if embedding_missing {
                1.0
            } else {
                l2_norm(&embedding)
            };
            DppInput {
                id: c.tweet_id,
                score,
                embedding,
                norm,
                embedding_missing,
            }
        })
        .collect()
}

fn build_seed_input(req: &RankRequest, ctx: &DppContext) -> Option<DppInput> {
    if req.seed_tweet_id == 0 {
        return None;
    }
    match ctx.store.client.get(req.seed_tweet_id) {
        Some(embedding) => {
            DPP_SEED_CONTEXT.with_label_values(&["pinned"]).inc();
            let norm = l2_norm(&embedding);
            Some(DppInput {
                id: req.seed_tweet_id,
                score: 0.0,
                embedding,
                norm,
                embedding_missing: false,
            })
        }
        None => {
            DPP_SEED_CONTEXT
                .with_label_values(&["embedding_missing"])
                .inc();
            None
        }
    }
}

fn build_history_inputs(
    req: &RankRequest,
    size: usize,
    ctx: &DppContext,
    memo: &mut EmbeddingMemo,
) -> Vec<DppInput> {
    let now_ms = req
        .viewer
        .as_ref()
        .map(|v| v.now_ms)
        .filter(|&t| t > 0)
        .or_else(|| {
            i64::try_from(req.request_timestamp_ms)
                .ok()
                .filter(|&t| t > 0)
        })
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as i64)
        });
    let fresh = req
        .dpp_history
        .iter()
        .take_while(|item| now_ms.saturating_sub(item.served_time_ms) <= HISTORY_MAX_AGE_MS)
        .count();
    req.dpp_history[..fresh]
        .chunk_by(|a, b| a.served_time_ms == b.served_time_ms)
        .flat_map(|response| response.iter().rev())
        .take(size)
        .map(|item| {
            let (embedding, embedding_missing) = memo.resolve(item.tweet_id, ctx);
            DppInput {
                id: item.tweet_id,
                score: 0.0,
                norm: l2_norm(&embedding),
                embedding,
                embedding_missing,
            }
        })
        .collect()
}

pub fn rank(req: &RankRequest, scores: &[Option<f64>], ctx: &DppContext) -> Vec<f64> {
    rank_with_memo(req, scores, ctx, &mut EmbeddingMemo::default())
}

pub fn rank_with_memo(
    req: &RankRequest,
    scores: &[Option<f64>],
    ctx: &DppContext,
    memo: &mut EmbeddingMemo,
) -> Vec<f64> {
    let inputs = build_dpp_inputs(req, scores, ctx, memo);
    let seed = build_seed_input(req, ctx);
    let history = match ctx.config.window {
        Some(window) => {
            let history = build_history_inputs(req, window.size, ctx, memo);
            DPP_WINDOW_HISTORY_ITEMS.observe(history.len() as f64);
            history
        }
        None => Vec::new(),
    };

    let results = dpp::rescore(&inputs, seed.as_ref(), &history, &ctx.config, req.viewer_id);

    let debug = ctx.config.debug_viewer_id != 0 && req.viewer_id == ctx.config.debug_viewer_id;
    let served: HashMap<u64, f64> = results.iter().map(|r| (r.id, r.score)).collect();

    if debug {
        let mut sorted_inputs: Vec<&DppInput> = inputs.iter().collect();
        sorted_inputs.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let dropped: Vec<(usize, u64)> = sorted_inputs
            .iter()
            .enumerate()
            .filter(|(_, inp)| !served.contains_key(&inp.id))
            .map(|(rank, inp)| (rank + 1, inp.id))
            .collect();

        if !dropped.is_empty() {
            let dropped_ids: Vec<String> = dropped.iter().map(|(_, id)| id.to_string()).collect();
            info!(
                "DPP filtered {}/{} posts. Dropped (score_rank, postId): {:?}\nDropped postIds: {}",
                dropped.len(),
                inputs.len(),
                dropped,
                dropped_ids.join(","),
            );
        }

        let limit = sorted_inputs.len().min(50);
        let mut before_log = format!("Top {} BEFORE DPP (rank, postId, score):\n", limit);
        for (rank, inp) in sorted_inputs.iter().enumerate().take(limit) {
            before_log.push_str(&format!(
                "  #{:<3} postId={:<20} score={:.6}\n",
                rank + 1,
                inp.id,
                inp.score
            ));
        }
        info!("{}", before_log);

        let after_limit = results.len().min(50);
        let mut after_log = format!("Top {} AFTER DPP (rank, postId, score):\n", after_limit);
        for (rank, r) in results.iter().enumerate().take(after_limit) {
            after_log.push_str(&format!(
                "  #{:<3} postId={:<20} score={:.6}\n",
                rank + 1,
                r.id,
                r.score
            ));
        }
        info!("{}", after_log);
    }

    req.candidates
        .iter()
        .map(|c| served.get(&c.tweet_id).copied().unwrap_or(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpp::{DppConfig, SlidingWindow};
    use crate::embedding_store::EmbeddingStore;
    use xai_vm_ranker_proto::DppHistoryItem;

    #[test]
    fn greedy_order_scores_reach_the_served_candidates() {
        let embeddings = [
            (1, [1.0, 0.0, 0.0]),
            (2, [0.8, 0.0, 0.6]),
            (3, [0.0, 1.0, 0.0]),
            (4, [0.0, 1.0, 0.0]),
            (5, [1.0, 0.0, 0.0]),
        ];
        let req = |age_ms: i64| RankRequest {
            request_timestamp_ms: (1_000_000 + age_ms) as u64,
            candidates: embeddings[..3]
                .iter()
                .map(|&(tweet_id, _)| RankCandidate {
                    tweet_id,
                    ..Default::default()
                })
                .collect(),
            dpp_history: [4, 5]
                .map(|tweet_id| DppHistoryItem {
                    tweet_id,
                    served_time_ms: 1_000_000,
                    carousel: false,
                })
                .to_vec(),
            ..Default::default()
        };
        let scores = [Some(10.0), Some(9.0), Some(5.0)];
        let served = |req: &RankRequest, greedy_order: bool, window: Option<SlidingWindow>| {
            let ctx = DppContext {
                store: Arc::new(EmbeddingStore::in_process(3)),
                config: DppConfig {
                    top_k: 3,
                    theta: 0.5,
                    max_selected_rank: 10,
                    debug_viewer_id: 0,
                    greedy_order,
                    window,
                },
            };
            let mut memo = EmbeddingMemo::default();
            for (id, embedding) in embeddings {
                let embedding = embedding.iter().map(|&x| f16::from_f32(x)).collect();
                memo.0.insert(id, (Arc::new(embedding), false));
            }
            rank_with_memo(req, &scores, &ctx, &mut memo)
        };

        let recent = req(60_000);
        assert_eq!(served(&recent, false, None), [10.0, 9.0, 5.0]);
        assert_eq!(served(&recent, true, None), [10.0, 5.0, 9.0]);
        let window = SlidingWindow {
            size: 1,
            half_distance: 5.0,
        };
        assert_eq!(served(&recent, true, Some(window)), [9.0, 5.0, 10.0]);
        assert_eq!(
            served(&req(11 * 60_000), true, Some(window)),
            [10.0, 5.0, 9.0]
        );
    }

    #[test]
    fn history_window_compares_with_the_posts_right_above_the_page() {
        let t2 = 2_000_000i64;
        let item = |tweet_id, served_time_ms| DppHistoryItem {
            tweet_id,
            served_time_ms,
            carousel: false,
        };
        let mut dpp_history: Vec<DppHistoryItem> = (21..=25).map(|id| item(id, t2)).collect();
        dpp_history.extend((11..=15).map(|id| item(id, t2 - 60_000)));
        let req = RankRequest {
            request_timestamp_ms: (t2 + 30_000) as u64,
            candidates: [1, 2, 3, 4]
                .map(|tweet_id| RankCandidate {
                    tweet_id,
                    ..Default::default()
                })
                .to_vec(),
            dpp_history,
            ..Default::default()
        };
        let ctx = DppContext {
            store: Arc::new(EmbeddingStore::in_process(16)),
            config: DppConfig {
                top_k: 4,
                theta: 0.5,
                max_selected_rank: 10,
                debug_viewer_id: 0,
                greedy_order: true,
                window: Some(SlidingWindow {
                    size: 3,
                    half_distance: 1.0,
                }),
            },
        };
        let basis = |i: usize| {
            let mut v = vec![f16::ZERO; 16];
            v[i] = f16::ONE;
            Arc::new(v)
        };
        let mut memo = EmbeddingMemo::default();
        for (i, id) in [21, 22, 23, 24, 25, 11, 12, 13, 14, 15]
            .into_iter()
            .enumerate()
        {
            memo.0.insert(id, (basis(i), false));
        }
        for (id, i) in [(1, 4), (2, 9), (3, 2), (4, 15)] {
            memo.0.insert(id, (basis(i), false));
        }
        assert_eq!(
            rank_with_memo(
                &req,
                &[Some(10.0), Some(5.0), Some(9.0), Some(1.0)],
                &ctx,
                &mut memo
            ),
            [5.0, 9.0, 10.0, 1.0]
        );
    }
}
