use log::{debug, info, warn};
use rand::Rng;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tonic::{Request, Response, Status};
use xai_value_model::compare_scores_desc;

use xai_vm_ranker_proto::{
    vm_ranker_service_server::{VmRankerService, VmRankerServiceServer},
    RankRequest, RankResponse, RankedCandidate,
};

use crate::metrics::{
    Timer, IN_FLIGHT_REQUESTS, REJECTED_REQUESTS, SCORE_CANDIDATES_IN, SCORE_DURATION,
    SCORE_ERRORS, SCORE_REQUESTS,
};
use crate::ranking_config::RankingConfig;
use crate::scoring::{DebiasModels, DppContext, RankContext};
use crate::user_debias_store::UserDebiasStore;

pub struct VMRankerServiceImpl {
    request_semaphore: Arc<Semaphore>,
    dpp: Option<DppContext>,
    user_debias: Option<Arc<UserDebiasStore>>,
    config: Option<Arc<RankingConfig>>,
}

impl VMRankerServiceImpl {
    pub fn new(
        max_concurrent_requests: usize,
        dpp: Option<DppContext>,
        user_debias: Option<Arc<UserDebiasStore>>,
        config: Option<Arc<RankingConfig>>,
    ) -> Self {
        let enabled = |on: bool| if on { "enabled" } else { "disabled" };
        info!(
            "Initializing VMRankerService with max_concurrent_requests={}, dpp={}, user_debias={}, ranking_config={}",
            max_concurrent_requests,
            enabled(dpp.is_some()),
            enabled(user_debias.is_some()),
            enabled(config.is_some()),
        );
        Self {
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            dpp,
            user_debias,
            config,
        }
    }

    pub fn server(self) -> VmRankerServiceServer<Self> {
        VmRankerServiceServer::new(self)
            .max_decoding_message_size(20 * 1024 * 1024)
            .max_encoding_message_size(20 * 1024 * 1024)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .accept_compressed(tonic::codec::CompressionEncoding::Zstd)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Zstd)
    }
}

#[tonic::async_trait]
impl VmRankerService for VMRankerServiceImpl {
    async fn rank(&self, request: Request<RankRequest>) -> Result<Response<RankResponse>, Status> {
        let req = request.into_inner();
        let value_model_id = req.value_model_id.clone();
        let model_id = if value_model_id.is_empty() {
            "unknown"
        } else {
            &value_model_id
        };

        SCORE_REQUESTS.with_label_values(&[model_id]).inc();

        let viewer_id = req.viewer_id;

        let _permit = match self.request_semaphore.try_acquire() {
            Ok(permit) => {
                IN_FLIGHT_REQUESTS.inc();
                permit
            }
            Err(_) => {
                REJECTED_REQUESTS.with_label_values(&[model_id]).inc();
                return Err(Status::resource_exhausted(
                    "Server at capacity, please retry",
                ));
            }
        };

        struct InFlightGuard;
        impl Drop for InFlightGuard {
            fn drop(&mut self) {
                IN_FLIGHT_REQUESTS.dec();
            }
        }
        let _guard = InFlightGuard;
        let _timer = Timer::new(SCORE_DURATION.with_label_values(&[model_id]).clone());

        let num_candidates = req.candidates.len();
        SCORE_CANDIDATES_IN
            .with_label_values(&[model_id])
            .observe(num_candidates as f64);
        let ctx = RankContext {
            dpp: self.dpp.clone(),
            debias: DebiasModels {
                user: self.user_debias.as_ref().and_then(|store| store.model()),
                user_changed_at_ms: self
                    .user_debias
                    .as_ref()
                    .map_or(0, |store| store.changed_at_ms()),
            },
            config: self.config.clone(),
        };
        let result = crate::scoring::rank(req, ctx).await.map_err(|e| {
            warn!(
                "rank failed: viewer={} value_model_id='{}': {}",
                viewer_id, model_id, e
            );
            SCORE_ERRORS.with_label_values(&[model_id]).inc();
            Status::internal(e)
        })?;

        if rand::rng().random_ratio(1, 1000) {
            info!(
                "rank: viewer={} model={} candidates={}",
                viewer_id, model_id, num_candidates
            );
        }

        let mut top: Vec<&RankedCandidate> = result.iter().collect();
        top.sort_by(|a, b| compare_scores_desc(a.score, b.score));
        for s in top.iter().take(50) {
            debug!("  tweet={} score={:.6}", s.tweet_id, s.score);
        }

        Ok(Response::new(RankResponse { candidates: result }))
    }
}
