use crate::clients::vm_ranker_client::{VMRankerClient, VMRankerCluster};
use crate::models::candidate::{PostCandidate, SlateContext};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::scorers::author_cold_start::{AuthorColdStart, ColdStartOutcome};
use crate::scorers::value_model;
use crate::scorers::vm_ranker_request::RequestShape;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_stats_receiver::global_stats_receiver;
use xai_vm_ranker_proto::{RankRequest, RankResponse, RankedCandidate};

const METRIC_PREFIX: &str = "VMRanker";

pub struct VMRanker {
    pub client: Arc<dyn VMRankerClient>,
    pub xds_client: Option<Arc<dyn VMRankerClient>>,
    pub author_cold_start: AuthorColdStart,
}

struct LocalScores {
    weighted: Vec<f64>,
    cold_start: ColdStartOutcome,
}

impl VMRanker {
    async fn rank(
        &self,
        query: &ScoredPostsQuery,
        cluster: VMRankerCluster,
        request: RankRequest,
    ) -> Result<RankResponse, String> {
        let use_xds = self.xds_client.is_some()
            && crate::util::xds::use_xds_for_vm_ranker_cluster(query, &cluster.gate_name());

        if use_xds {
            let xds = self.xds_client.as_ref().expect("checked is_some above");
            match xds.rank(cluster, request.clone()).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    if !query.params.get(VMRankerEnableFallback) {
                        return Err(format!(
                            "VMRanker xDS gRPC call failed (fallback disabled): {e}"
                        ));
                    }
                    tracing::warn!(cluster = ?cluster, error = %e, "VMRanker xDS rank failed; falling back to DNS");
                }
            }
        }

        self.client
            .rank(cluster, request)
            .await
            .map_err(|e| format!("VMRanker gRPC call failed: {e}"))
    }

    fn local_scores(&self, query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> LocalScores {
        let weights = value_model::weights_for(query);
        let weighted: Vec<f64> = candidates
            .iter()
            .map(|c| match c.weighted_score {
                Some(cached) => cached,
                None => value_model::weighted_score(query, &weights, c),
            })
            .collect();
        let cold_start = self
            .author_cold_start
            .apply_with_decisions(query, candidates, &weighted);
        LocalScores {
            weighted,
            cold_start,
        }
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for VMRanker {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableRanking)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let shape = RequestShape::from_query(query);
        let local = self.local_scores(query, candidates);
        let slate_contexts = slate_contexts(query, candidates);

        let mut scored: Vec<PostCandidate> = (0..candidates.len())
            .map(|i| PostCandidate {
                weighted_score: Some(local.weighted[i]),
                score: Some(local.cold_start.scores[i]),
                cold_start_lift_to_rank: local.cold_start.lift_to_rank(i),
                slate_context: slate_contexts.as_ref().map(|contexts| contexts[i]),
                ..Default::default()
            })
            .collect();

        let cluster = VMRankerCluster::parse(&query.params.get(VMRankerClusterId));
        let request = shape.build(query, candidates, &scored);
        record_request(&cluster);

        let response = match self.rank(query, cluster, request).await {
            Ok(resp) => resp,
            Err(msg) => {
                tracing::warn!(error = %msg, "VMRanker rank failed; serving local weighted scores");
                record_fallback("rpc_error", candidates.len());
                return scored.into_iter().map(Ok).collect();
            }
        };

        let mut returned: FxHashMap<u64, RankedCandidate> = response
            .candidates
            .into_iter()
            .map(|sc| (sc.tweet_id, sc))
            .collect();

        let mut missing = 0;
        for (c, out) in candidates.iter().zip(scored.iter_mut()) {
            match returned.remove(&c.tweet_id) {
                Some(sc) => {
                    out.score = Some(sc.score);
                    out.weighted_score = sc.weighted_score.or(out.weighted_score);
                    out.top_heads = sc
                        .top_heads
                        .into_iter()
                        .map(|h| (h.head, h.contribution, h.share))
                        .collect();
                }
                None => missing += 1,
            }
        }
        if missing > 0 {
            record_fallback("missing_candidate", missing);
        }
        scored.into_iter().map(Ok).collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
        candidate.score = scored.score;
        candidate.cold_start_lift_to_rank = scored.cold_start_lift_to_rank;
        candidate.slate_context = scored.slate_context;
        candidate.top_heads = scored.top_heads;
    }
}

fn slate_contexts(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
) -> Option<Vec<SlateContext>> {
    let served: Option<Vec<SlateContext>> =
        candidates.iter().map(|c| c.served_slate_context).collect();
    served.or_else(|| {
        query
            .has_cached_posts
            .then(|| candidates.iter().map(|c| c.slate_context).collect())
            .flatten()
    })
}

fn record_request(cluster: &VMRankerCluster) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(
            &format!("{METRIC_PREFIX}.request"),
            &[("vm_cluster", &format!("{cluster:?}"))],
            1,
        );
    }
}

fn record_fallback(reason: &str, candidate_count: usize) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(
            &format!("{METRIC_PREFIX}.local_score_fallback"),
            &[("reason", reason)],
            candidate_count as u64,
        );
    }
}
