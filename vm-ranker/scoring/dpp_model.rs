use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use half::f16;
use log::info;
use rand::Rng;
use xai_value_model::compare_scores_desc;
use xai_vm_ranker_proto::{RankCandidate, RankRequest};

use super::DppContext;
use crate::dpp::{self, DppInput};
use crate::metrics::DPP_SEED_CONTEXT;

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
        compare_scores_desc(*s_a, *s_b).then_with(|| i_a.cmp(i_b))
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

    let results = dpp::rescore(&inputs, seed.as_ref(), &ctx.config, req.viewer_id);

    let debug = ctx.config.debug_viewer_id != 0 && req.viewer_id == ctx.config.debug_viewer_id;
    let selected_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();

    if debug {
        let mut sorted_inputs: Vec<&DppInput> = inputs.iter().collect();
        sorted_inputs.sort_unstable_by(|a, b| compare_scores_desc(a.score, b.score));

        let dropped: Vec<(usize, u64)> = sorted_inputs
            .iter()
            .enumerate()
            .filter(|(_, inp)| !selected_ids.contains(&inp.id))
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
        .zip(scores)
        .map(|(c, s)| {
            if selected_ids.contains(&c.tweet_id) {
                s.unwrap_or(0.0)
            } else {
                0.0
            }
        })
        .collect()
}
