use std::collections::HashSet;
use std::sync::Arc;

use half::f16;
use log::info;
use xai_vm_ranker_proto::{RankCandidate, RankRequest, RankedCandidate};

use super::DppContext;
use crate::dpp::{self, DppInput};

fn l2_norm(v: &[f16]) -> f64 {
    v.iter()
        .map(|x| {
            let xf = x.to_f32();
            xf * xf
        })
        .sum::<f32>()
        .sqrt() as f64
}

fn mix_u64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Builds a stable fallback whose expected cosine similarity with other embeddings is zero.
///
/// DPP treats embeddings as directions, so a dense hash-derived unit vector preserves the old
/// random fallback's approximately orthogonal geometry without changing between requests. Keying
/// by the embedding ID also gives multiple candidates for the same original post the same vector.
fn deterministic_unit_embedding(embedding_id: u64, dim: usize) -> (Arc<Vec<f16>>, f64) {
    if dim == 0 {
        return (Arc::new(Vec::new()), 0.0);
    }

    let seed = mix_u64(embedding_id ^ (dim as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let v: Vec<f32> = (0..dim)
        .map(|coordinate| {
            let bits = mix_u64(seed ^ (coordinate as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let unit_interval = (bits >> 40) as f32 / (1u32 << 24) as f32;
            unit_interval.mul_add(2.0, -1.0)
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let emb: Vec<f16> = if norm.is_finite() && norm > f32::EPSILON {
        v.into_iter().map(|x| f16::from_f32(x / norm)).collect()
    } else {
        let mut u = vec![f16::ZERO; dim];
        u[(mix_u64(seed) % dim as u64) as usize] = f16::ONE;
        u
    };
    let quantized_norm = l2_norm(&emb);
    (Arc::new(emb), quantized_norm)
}

fn build_dpp_inputs(req: &RankRequest, ctx: &DppContext) -> Vec<DppInput> {
    let dim = ctx.store.dim();
    let max_rank = ctx.config.max_selected_rank;

    let mut scored: Vec<(usize, &RankCandidate, f64)> = req
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.score.map(|s| (i, c, s)))
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
            let fetched = ctx.store.client.get(embedding_id);
            let embedding_missing = fetched.is_none();
            let (embedding, norm) = match fetched {
                Some(arc) => {
                    let n = l2_norm(&arc);
                    (arc, n)
                }
                None => deterministic_unit_embedding(embedding_id, dim),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_embedding_fallback_is_repeatable() {
        let (first, first_norm) = deterministic_unit_embedding(42, 128);
        let (second, second_norm) = deterministic_unit_embedding(42, 128);
        let (different_id, _) = deterministic_unit_embedding(43, 128);

        assert_eq!(first, second);
        assert_eq!(first_norm, second_norm);
        assert_ne!(first, different_id);
    }

    #[test]
    fn missing_embedding_fallback_is_finite_and_normalized() {
        for embedding_id in [0, 1, 42, u64::MAX] {
            for dim in [1, 2, 128, 1024] {
                let (embedding, norm) = deterministic_unit_embedding(embedding_id, dim);

                assert_eq!(embedding.len(), dim);
                assert!(
                    embedding.iter().all(|value| value.to_f32().is_finite()),
                    "embedding_id={embedding_id}, dim={dim} contains a non-finite value"
                );
                assert!(norm.is_finite(), "embedding_id={embedding_id}, dim={dim}");
                assert!(
                    (norm - 1.0).abs() < 1e-3,
                    "embedding_id={embedding_id}, dim={dim}, norm={norm}"
                );
                assert!((l2_norm(&embedding) - norm).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn missing_embedding_fallback_handles_empty_dimensions() {
        let (embedding, norm) = deterministic_unit_embedding(42, 0);
        assert!(embedding.is_empty());
        assert_eq!(norm, 0.0);
    }
}

pub fn rank(req: &RankRequest, ctx: &DppContext) -> Vec<RankedCandidate> {
    let inputs = build_dpp_inputs(req, ctx);

    let results = dpp::rescore(&inputs, &ctx.config, req.viewer_id);

    let debug = ctx.config.debug_viewer_id != 0 && req.viewer_id == ctx.config.debug_viewer_id;
    let selected_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();

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
        .map(|c| RankedCandidate {
            tweet_id: c.tweet_id,
            score: if selected_ids.contains(&c.tweet_id) {
                c.score.unwrap_or(0.0)
            } else {
                0.0
            },
        })
        .collect()
}
