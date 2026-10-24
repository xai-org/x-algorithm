use std::collections::HashSet;
use std::sync::Arc;

use half::f16;
use xai_value_model::compare_scores_desc;

use crate::helpers::dot_product;
use crate::metrics::{
    DPP_AVG_SIMILARITY, DPP_EMBEDDING_MISS_RATIO, DPP_LOG_DET, DPP_POOL_SCORES, DPP_POOL_SIZE,
    DPP_RESCALING, DPP_SELECTED_COUNT, DPP_TERMINAL_CV, DPP_TOP_K_OVERLAP,
};

const EPSILON: f64 = 1e-6;

#[derive(Clone, Copy)]
pub struct DppConfig {
    pub top_k: usize,
    pub theta: f64,
    pub max_selected_rank: usize,
    pub debug_viewer_id: u64,
}

pub struct DppInput {
    pub id: u64,
    pub score: f64,
    pub embedding: Arc<Vec<f16>>,
    pub norm: f64,
    pub embedding_missing: bool,
}

pub struct DppResult {
    pub id: u64,
    pub score: f64,
}

pub fn rescore(
    inputs: &[DppInput],
    seed: Option<&DppInput>,
    config: &DppConfig,
    _viewer_id: u64,
) -> Vec<DppResult> {
    let n = inputs.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sorted: Vec<usize> = (0..n).filter(|&i| inputs[i].score.is_finite()).collect();
    sorted.sort_unstable_by(|&a, &b| compare_scores_desc(inputs[a].score, inputs[b].score));

    let top_count = config.max_selected_rank.min(sorted.len());
    if top_count == 0 {
        return Vec::new();
    }
    let top_indices: &[usize] = &sorted[..top_count];
    let m: usize = top_count;

    DPP_POOL_SIZE
        .with_label_values(&[&config.max_selected_rank.to_string()])
        .observe(m as f64);

    let miss_count = top_indices
        .iter()
        .filter(|&&i| inputs[i].embedding_missing)
        .count();
    DPP_EMBEDDING_MISS_RATIO
        .with_label_values(&[&m.to_string()])
        .observe(miss_count as f64 / m as f64);

    {
        let top_score = inputs[top_indices[0]].score;
        let bot_score = inputs[top_indices[m - 1]].score;
        let med_score = inputs[top_indices[m / 2]].score;
        DPP_POOL_SCORES
            .with_label_values(&["max"])
            .observe(top_score);
        DPP_POOL_SCORES
            .with_label_values(&["min"])
            .observe(bot_score);
        DPP_POOL_SCORES
            .with_label_values(&["median"])
            .observe(med_score);
        let safe_max = top_score.max(EPSILON);
        DPP_POOL_SCORES
            .with_label_values(&["q_min"])
            .observe(bot_score / safe_max);
        DPP_POOL_SCORES
            .with_label_values(&["q_median"])
            .observe(med_score / safe_max);
    }

    let max_score = inputs[top_indices[0]].score.max(EPSILON);

    let theta = config.theta.clamp(0.0, 1.0 - EPSILON);
    let alpha = theta / (2.0 * (1.0 - theta));

    let total = m + usize::from(seed.is_some());
    let item_at = |ki: usize| -> &DppInput {
        if ki < m {
            &inputs[top_indices[ki]]
        } else {
            seed.expect("index m only exists when seed is present")
        }
    };

    let qf: Vec<f64> = (0..total)
        .map(|ki| {
            let q = if ki < m {
                item_at(ki).score / max_score
            } else {
                1.0
            };
            (alpha * q).exp()
        })
        .collect();

    let norms: Vec<f64> = (0..total).map(|ki| item_at(ki).norm).collect();

    let mut kernel = vec![0.0f64; total * total];
    for i in 0..total {
        kernel[i * total + i] = qf[i] * qf[i];
        for j in (i + 1)..total {
            let denom = norms[i] * norms[j];
            let cos = if denom > EPSILON {
                dot_product(&item_at(i).embedding, &item_at(j).embedding) / denom
            } else {
                0.0
            };
            let val = qf[i] * qf[j] * cos;
            kernel[i * total + j] = val;
            kernel[j * total + i] = val;
        }
    }

    let pinned = seed.map(|_| m);
    let dpp_result = greedy_dpp(&kernel, total, config.top_k, pinned);
    let selected: Vec<(usize, f64)> = dpp_result
        .selected
        .iter()
        .copied()
        .filter(|&(ki, _)| ki < m)
        .collect();
    let selected = &selected;

    let reason = if dpp_result.rank_exhausted {
        "rank_exhausted"
    } else {
        "top_k_reached"
    };
    DPP_TERMINAL_CV
        .with_label_values(&[reason])
        .observe(dpp_result.terminal_cv);

    let mut dpp_selected: Vec<usize> = selected.iter().map(|&(ki, _)| top_indices[ki]).collect();
    dpp_selected.sort_unstable_by(|&a, &b| compare_scores_desc(inputs[a].score, inputs[b].score));

    let results: Vec<DppResult> = dpp_selected
        .iter()
        .map(|&orig_idx| {
            let score = inputs[orig_idx].score;
            DppResult {
                id: inputs[orig_idx].id,
                score,
            }
        })
        .collect();

    let k = results.len();
    let top_k_label = config.top_k.to_string();

    DPP_SELECTED_COUNT
        .with_label_values(&[&top_k_label])
        .observe(k as f64);

    if let Some(&(_, log_det)) = selected.last() {
        DPP_LOG_DET
            .with_label_values(&[&top_k_label])
            .observe(log_det);
    }

    if k >= 2 {
        let baseline_top_k = &sorted[..k];
        DPP_AVG_SIMILARITY
            .with_label_values(&["before"])
            .observe(avg_pairwise_cosine(inputs, baseline_top_k));

        let dpp_top_k: Vec<usize> = dpp_selected.clone();
        DPP_AVG_SIMILARITY
            .with_label_values(&["after"])
            .observe(avg_pairwise_cosine(inputs, &dpp_top_k));
    }

    DPP_RESCALING
        .with_label_values(&["unchanged"])
        .observe(k as f64);

    let original_top_k: HashSet<u64> = sorted[..k.min(n)]
        .iter()
        .map(|&idx| inputs[idx].id)
        .collect();
    let overlap = results
        .iter()
        .filter(|r| original_top_k.contains(&r.id))
        .count();
    DPP_TOP_K_OVERLAP
        .with_label_values(&[&k.to_string()])
        .observe(overlap as f64);

    results
}

struct GreedyDppResult {
    selected: Vec<(usize, f64)>,
    terminal_cv: f64,
    rank_exhausted: bool,
}

fn greedy_dpp(kernel: &[f64], n: usize, k: usize, pinned_first: Option<usize>) -> GreedyDppResult {
    if n == 0 || k == 0 {
        return GreedyDppResult {
            selected: Vec::new(),
            terminal_cv: 0.0,
            rank_exhausted: true,
        };
    }
    let pinned = pinned_first.filter(|&p| kernel[p * n + p] > EPSILON);
    let max_items = match pinned {
        Some(_) => (k + 1).min(n),
        None => k.min(n),
    };
    let mut selected: Vec<(usize, f64)> = Vec::with_capacity(max_items);
    let mut available = vec![true; n];

    let (first, first_val) = match pinned {
        Some(p) => (p, kernel[p * n + p]),
        None => {
            let mut first = 0;
            let mut first_val = f64::NEG_INFINITY;
            for i in 0..n {
                let d = kernel[i * n + i];
                if d > first_val {
                    first_val = d;
                    first = i;
                }
            }
            (first, first_val)
        }
    };
    let mut log_det = first_val.ln();
    selected.push((first, log_det));
    available[first] = false;

    let mut cholesky = vec![0.0f64; max_items * n];
    let sqrt_first = first_val.sqrt();
    if sqrt_first > EPSILON {
        for j in 0..n {
            cholesky[j] = kernel[first * n + j] / sqrt_first;
        }
    }

    let mut terminal_cv = first_val;
    let mut rank_exhausted = false;

    for s in 1..max_items {
        let mut best_idx = 0usize;
        let mut best_vol = f64::NEG_INFINITY;

        for i in 0..n {
            if !available[i] {
                continue;
            }
            let mut cv = kernel[i * n + i];
            for j in 0..s {
                let f = cholesky[j * n + i];
                cv -= f * f;
            }
            if cv > best_vol {
                best_vol = cv;
                best_idx = i;
            }
        }

        if best_vol <= EPSILON {
            terminal_cv = best_vol;
            rank_exhausted = true;
            break;
        }

        terminal_cv = best_vol;
        log_det += best_vol.ln();
        selected.push((best_idx, log_det));
        available[best_idx] = false;

        let sqrt_bv = best_vol.sqrt();
        for i in 0..n {
            let mut sum = kernel[best_idx * n + i];
            for j in 0..s {
                sum -= cholesky[j * n + best_idx] * cholesky[j * n + i];
            }
            cholesky[s * n + i] = sum / sqrt_bv;
        }
    }

    GreedyDppResult {
        selected,
        terminal_cv,
        rank_exhausted,
    }
}

fn avg_pairwise_cosine(inputs: &[DppInput], indices: &[usize]) -> f64 {
    if indices.len() < 2 {
        return 0.0;
    }
    let mut sim_sum = 0.0f64;
    let mut count = 0u64;
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let a = &inputs[indices[i]];
            let b = &inputs[indices[j]];
            let denom = a.norm * b.norm;
            let cos = if denom > EPSILON {
                dot_product(&a.embedding, &b.embedding) / denom
            } else {
                0.0
            };
            sim_sum += cos;
            count += 1;
        }
    }
    if count > 0 {
        sim_sum / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(id: u64, score: f64, embedding: &[f32]) -> DppInput {
        let emb: Vec<f16> = embedding.iter().map(|&v| f16::from_f32(v)).collect();
        let norm = emb
            .iter()
            .map(|x| {
                let xf = x.to_f32();
                xf * xf
            })
            .sum::<f32>()
            .sqrt() as f64;
        DppInput {
            id,
            score,
            embedding: Arc::new(emb),
            norm,
            embedding_missing: false,
        }
    }

    fn default_config() -> DppConfig {
        DppConfig {
            top_k: 10,
            theta: 0.5,
            max_selected_rank: 50,
            debug_viewer_id: 0,
        }
    }

    #[test]
    fn empty_input() {
        let result = rescore(&[], None, &default_config(), 0);
        assert!(result.is_empty());
    }

    #[test]
    fn finite_candidates_fill_pool_before_invalid_scores() {
        let inputs = vec![
            make_input(1, 5.0, &[1.0, 0.0]),
            make_input(2, f64::NAN, &[1.0, 0.0]),
            make_input(3, 10.0, &[0.0, 1.0]),
            make_input(4, f64::INFINITY, &[1.0, 0.0]),
            make_input(5, f64::NEG_INFINITY, &[1.0, 0.0]),
        ];
        let config = DppConfig {
            top_k: 2,
            max_selected_rank: 2,
            ..default_config()
        };
        let result = rescore(&inputs, None, &config, 0);
        assert_eq!(
            result.iter().map(|r| (r.id, r.score)).collect::<Vec<_>>(),
            vec![(3, 10.0), (1, 5.0)],
        );
    }

    #[test]
    fn invalid_scores_cannot_displace_finite_candidates() {
        let inputs = vec![
            make_input(1, 10.0, &[1.0, 0.0, 0.0, 0.0]),
            make_input(2, f64::INFINITY, &[0.0, 1.0, 0.0, 0.0]),
            make_input(3, 5.0, &[0.0, 0.0, 1.0, 0.0]),
            make_input(4, f64::NAN, &[0.0, 0.0, 0.0, 1.0]),
        ];
        let result = rescore(&inputs, None, &default_config(), 0);
        assert_eq!(result.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn all_invalid_scores_select_nothing() {
        let inputs = vec![
            make_input(1, f64::NAN, &[1.0, 0.0]),
            make_input(2, f64::INFINITY, &[0.0, 1.0]),
            make_input(3, f64::NEG_INFINITY, &[1.0, 1.0]),
        ];
        let seed = make_input(99, 0.0, &[1.0, 0.0]);
        assert!(rescore(&inputs, None, &default_config(), 0).is_empty());
        assert!(rescore(&inputs, Some(&seed), &default_config(), 0).is_empty());
    }

    #[test]
    fn single_candidate() {
        let inputs = vec![make_input(1, 5.0, &[1.0, 0.0, 0.0])];
        let result = rescore(&inputs, None, &default_config(), 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
        assert!((result[0].score - 5.0).abs() < 1e-9);
    }

    #[test]
    fn identical_embeddings_select_top_scorer_only() {
        let emb = vec![1.0, 0.0, 0.0];
        let inputs = vec![
            make_input(1, 10.0, &emb),
            make_input(2, 5.0, &emb),
            make_input(3, 1.0, &emb),
        ];
        let config = DppConfig {
            top_k: 3,
            theta: 0.5,
            max_selected_rank: 10,

            debug_viewer_id: 0,
        };
        let result = rescore(&inputs, None, &config, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn orthogonal_embeddings_promote_diversity() {
        let inputs = vec![
            make_input(1, 10.0, &[1.0, 0.0]),
            make_input(2, 9.5, &[1.0, 0.0]),
            make_input(3, 5.0, &[0.0, 1.0]),
        ];
        let config = DppConfig {
            top_k: 3,
            theta: 0.5,
            max_selected_rank: 10,

            debug_viewer_id: 0,
        };
        let result = rescore(&inputs, None, &config, 0);
        let ids: Vec<u64> = result.iter().map(|r| r.id).collect();
        assert!(ids.contains(&3), "orthogonal candidate should be selected");
        assert!(
            !ids.contains(&2),
            "duplicate candidate should not be selected"
        );
    }

    #[test]
    fn filter_only_preserves_original_scores() {
        let inputs: Vec<DppInput> = (0..20)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::PI / 10.0;
                make_input(i as u64, 10.0 - i as f64 * 0.4, &[angle.cos(), angle.sin()])
            })
            .collect();
        let config = DppConfig {
            top_k: 10,
            theta: 0.5,
            max_selected_rank: 20,

            debug_viewer_id: 0,
        };
        let result = rescore(&inputs, None, &config, 0);
        assert!(result.len() <= 10, "at most top_k items returned");
        assert!(!result.is_empty(), "should select at least one item");
        for w in result.windows(2) {
            assert!(w[0].score >= w[1].score, "descending");
        }
        for r in &result {
            let orig = &inputs[r.id as usize];
            assert!(
                (r.score - orig.score).abs() < 1e-9,
                "id={} score={} should match original={}",
                r.id,
                r.score,
                orig.score
            );
        }
    }

    #[test]
    fn max_selected_rank_limits_pool() {
        let inputs: Vec<DppInput> = (0..100)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::PI / 100.0;
                make_input(i, 100.0 - i as f64, &[angle.cos(), angle.sin()])
            })
            .collect();
        let config = DppConfig {
            top_k: 5,
            theta: 0.5,
            max_selected_rank: 10,

            debug_viewer_id: 0,
        };
        let result = rescore(&inputs, None, &config, 0);
        assert!(result.len() <= 5, "at most top_k items");
        assert!(!result.is_empty());
        for r in &result {
            assert!(r.id < 10, "from top-10 pool");
        }
    }

    #[test]
    fn seed_suppresses_near_duplicate() {
        let seed = make_input(999, 0.0, &[1.0, 0.0]);
        let inputs = vec![
            make_input(1, 10.0, &[1.0, 0.0]),
            make_input(2, 5.0, &[0.0, 1.0]),
        ];
        let config = DppConfig {
            top_k: 2,
            theta: 0.5,
            max_selected_rank: 10,
            debug_viewer_id: 0,
        };

        let ids: Vec<u64> = rescore(&inputs, None, &config, 0)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(ids.contains(&1) && ids.contains(&2));

        let result = rescore(&inputs, Some(&seed), &config, 0);
        let ids: Vec<u64> = result.iter().map(|r| r.id).collect();
        assert!(
            !ids.contains(&1),
            "seed near-duplicate should be suppressed, got {ids:?}"
        );
        assert!(ids.contains(&2), "orthogonal candidate should survive");
        assert!(!ids.contains(&999), "seed itself never returned");
    }

    #[test]
    fn seed_does_not_consume_top_k_budget() {
        let seed = make_input(999, 0.0, &[0.0, 0.0, 1.0]);
        let inputs = vec![
            make_input(1, 10.0, &[1.0, 0.0, 0.0]),
            make_input(2, 5.0, &[0.0, 1.0, 0.0]),
        ];
        let config = DppConfig {
            top_k: 2,
            theta: 0.5,
            max_selected_rank: 10,
            debug_viewer_id: 0,
        };
        let result = rescore(&inputs, Some(&seed), &config, 0);
        let ids: Vec<u64> = result.iter().map(|r| r.id).collect();
        assert_eq!(ids.len(), 2, "seed must not eat top_k budget, got {ids:?}");
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    fn f16_vec(vals: &[f32]) -> Vec<f16> {
        vals.iter().map(|&v| f16::from_f32(v)).collect()
    }

    #[test]
    fn dot_product_identical_unit_vectors() {
        let v = f16_vec(&[0.0, 1.0]);
        let d = dot_product(&v, &v);
        assert!((d - 1.0).abs() < 1e-3, "identical unit vectors: dot=1.0");
    }

    #[test]
    fn dot_product_opposite_vectors() {
        let a = f16_vec(&[1.0, 0.0]);
        let b = f16_vec(&[-1.0, 0.0]);
        let d = dot_product(&a, &b);
        assert!((d - (-1.0)).abs() < 1e-3, "opposite vectors: dot=-1.0");
    }

    #[test]
    fn dot_product_orthogonal_vectors() {
        let a = f16_vec(&[1.0, 0.0]);
        let b = f16_vec(&[0.0, 1.0]);
        let d = dot_product(&a, &b);
        assert!(d.abs() < 1e-3, "orthogonal vectors: dot=0.0");
    }

    #[test]
    fn dot_product_zero_vector() {
        let a = f16_vec(&[1.0, 0.0]);
        let b = f16_vec(&[0.0, 0.0]);
        let d = dot_product(&a, &b);
        assert!(d.abs() < 1e-3, "zero vector: dot=0.0");
    }
}
