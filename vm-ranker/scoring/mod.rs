pub mod dpp_model;

use std::sync::Arc;

use log::error;

use xai_vm_ranker_proto::{DppParams, RankRequest, RankedCandidate};

use crate::dpp::DppConfig;
use crate::embedding_store::EmbeddingStore;

#[derive(Clone)]
pub struct DppContext {
    pub store: Arc<EmbeddingStore>,
    pub config: DppConfig,
}

fn apply_request_dpp_params(config: &mut DppConfig, params: &DppParams) {
    if params.theta != 0.0 {
        config.theta = params.theta;
    }
    if params.max_selected_rank != 0 {
        config.max_selected_rank =
            (params.max_selected_rank as usize).min(config.max_selected_rank);
    }
}

pub async fn rank(
    req: RankRequest,
    dpp: Option<&DppContext>,
) -> Result<Vec<RankedCandidate>, String> {
    if let Some(ctx) = dpp {
        if req.value_model_id != "dpp" {
            error!(
                "DPP context provided but value_model_id='{}' is not 'dpp'",
                req.value_model_id
            );
        }
        let mut ctx = ctx.clone();

        if let Some(params) = &req.dpp_params {
            apply_request_dpp_params(&mut ctx.config, params);
        }

        let dpp_result = tokio::task::spawn_blocking(move || dpp_model::rank(&req, &ctx))
            .await
            .map_err(|e| format!("DPP spawn_blocking failed: {e}"))?;
        return Ok(dpp_result);
    }

    Ok(req
        .candidates
        .iter()
        .map(|c| RankedCandidate {
            tweet_id: c.tweet_id,
            score: c.score.unwrap_or(0.0),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_pool_limit(max_selected_rank: usize) -> DppConfig {
        DppConfig {
            top_k: 50,
            theta: 0.5,
            max_selected_rank,
            debug_viewer_id: 0,
        }
    }

    #[test]
    fn request_cannot_raise_server_dpp_pool_limit() {
        let mut config = config_with_pool_limit(100);
        let params = DppParams {
            theta: 0.0,
            max_selected_rank: u32::MAX,
        };

        apply_request_dpp_params(&mut config, &params);

        assert_eq!(config.max_selected_rank, 100);
    }

    #[test]
    fn request_can_select_a_smaller_dpp_pool() {
        let mut config = config_with_pool_limit(100);
        let params = DppParams {
            theta: 0.0,
            max_selected_rank: 40,
        };

        apply_request_dpp_params(&mut config, &params);

        assert_eq!(config.max_selected_rank, 40);
    }

    #[test]
    fn zero_request_value_keeps_server_dpp_pool_limit() {
        let mut config = config_with_pool_limit(100);
        let params = DppParams {
            theta: 0.0,
            max_selected_rank: 0,
        };

        apply_request_dpp_params(&mut config, &params);

        assert_eq!(config.max_selected_rank, 100);
    }
}
