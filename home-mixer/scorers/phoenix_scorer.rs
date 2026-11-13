use crate::models::candidate::CandidateHelpers;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableHomeExcursionScores, EnablePageDecodeRerank, PhoenixInferenceClusterId,
    PhoenixRankerNewUserHistoryThreshold, PhoenixRankerNewUserInferenceClusterId, RerankerHeadTag,
};
use crate::scorers::page_decode_rerank_scorer::{
    page_decode_params, record_decode_picks_returned, record_decode_request_sent,
};
use crate::util::egress::PredictionDispatch;
use crate::util::phoenix_request::build_prediction_request;
use crate::util::post_engagers::record_sent_post_engagers;
use std::collections::HashMap;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PredictNextActions,
};

use xai_candidate_pipeline::component_library::models::PhoenixScores;
use xai_candidate_pipeline::component_library::utils::current_timestamp_millis;
use xai_candidate_pipeline::scorer::Scorer;
use xai_recsys_proto::{ContinuousActionName, ProductSurface};

pub const PHOENIX_RANKER_KILL_SWITCH_DECIDER: &str = "disable_home_mixer_phoenix_ranker";

pub struct PhoenixScorer {
    pub dispatch: PredictionDispatch,
}

impl PhoenixScorer {
    pub(crate) fn resolve_cluster(query: &ScoredPostsQuery) -> PhoenixCluster {
        let configured_cluster =
            PhoenixCluster::parse(&query.params.get(PhoenixInferenceClusterId));

        let threshold: u64 = query.params.get(PhoenixRankerNewUserHistoryThreshold);
        if threshold > 0 {
            let action_count = query
                .scoring_sequence
                .as_ref()
                .and_then(|s| s.metadata.as_ref())
                .map(|m| m.length)
                .unwrap_or(0);

            if action_count < threshold {
                return PhoenixCluster::parse(
                    &query.params.get(PhoenixRankerNewUserInferenceClusterId),
                );
            }
        }

        if let Some(decider) = &query.decider {
            let is_prod = matches!(
                configured_cluster,
                PhoenixCluster::Experiment1Fou | PhoenixCluster::Experiment2Fou
            );
            if is_prod {
                if decider.enabled("override_qf_use_experiment2_fou") {
                    return PhoenixCluster::Experiment2Fou;
                }
                if decider.enabled("override_qf_use_experiment1_fou") {
                    return PhoenixCluster::Experiment1Fou;
                }
            }
        }

        configured_cluster
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PhoenixScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        if query.has_cached_posts {
            return false;
        }
        let killed = query
            .decider
            .as_ref()
            .is_some_and(|d| d.enabled(PHOENIX_RANKER_KILL_SWITCH_DECIDER));
        !killed
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let last_scored_at_ms = current_timestamp_millis();
        let product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        if query.scoring_sequence.is_none() {
            return vec![Ok(PostCandidate::default()); candidates.len()];
        };

        let cluster = Self::resolve_cluster(query);
        let serve_time_ms = last_scored_at_ms.map_or(query.request_time_ms, |ms| ms as i64);
        let mut request =
            build_prediction_request(query, candidates, product_surface, serve_time_ms);
        record_sent_post_engagers(candidates, &request.candidate_sets[0].candidates);
        if query.params.get(EnablePageDecodeRerank) {
            request.page_decode = Some(page_decode_params(query));
            record_decode_request_sent();
        }

        let predictions = self
            .dispatch
            .predict(query, cluster, request)
            .await
            .map_err(|e| format!("Phoenix prediction failed: {}", e));

        let predictions = match predictions {
            Ok(predictions) => predictions,
            Err(err) => return vec![Err(err); candidates.len()],
        };

        let page_decode_picks = predictions.page_decode_picks();
        if query.params.get(EnablePageDecodeRerank) {
            record_decode_picks_returned(page_decode_picks.len());
        }
        let decode_slots: HashMap<u64, (u32, f64)> = page_decode_picks
            .iter()
            .enumerate()
            .map(|(slot, pick)| (pick.tweet_id, (slot as u32, pick.prob)))
            .collect();

        let home_excursion_scores = query.params.get(EnableHomeExcursionScores);
        candidates
            .iter()
            .map(|c| {
                let decode_slot = decode_slots.get(&c.get_original_tweet_id());
                PostCandidate {
                    phoenix_scores: if home_excursion_scores {
                        with_home_excursion_heads(&predictions, c.get_original_tweet_id())
                    } else {
                        predictions.candidate_scores(&c.get_original_tweet_id())
                    },
                    backbone_scores: predictions
                        .candidate_backbone_scores(&c.get_original_tweet_id())
                        .map(Box::new),
                    served_slate_context: predictions
                        .candidate_slate_context(&c.get_original_tweet_id())
                        .map(|context| Box::new(context.into())),
                    prediction_request_id: Some(query.prediction_id),
                    last_scored_at_ms,
                    reranker_head_tag: Some(query.params.get(RerankerHeadTag) as u32),
                    page_decode_slot: decode_slot.map(|&(slot, _)| slot),
                    page_decode_prob: decode_slot.map(|&(_, prob)| prob),
                    ..Default::default()
                }
            })
            .map(Ok)
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.phoenix_scores = scored.phoenix_scores;
        candidate.backbone_scores = scored.backbone_scores;
        candidate.served_slate_context = scored.served_slate_context;
        candidate.prediction_request_id = scored.prediction_request_id;
        candidate.last_scored_at_ms = scored.last_scored_at_ms;
        candidate.reranker_head_tag = scored.reranker_head_tag;
        candidate.page_decode_slot = scored.page_decode_slot;
        candidate.page_decode_prob = scored.page_decode_prob;
    }
}

fn with_home_excursion_heads(predictions: &PredictNextActions, tweet_id: u64) -> PhoenixScores {
    PhoenixScores {
        home_video_continuation_secs: predictions.candidate_continuous_value(
            &tweet_id,
            ContinuousActionName::HomeVideoContinuationSecs as usize,
        ),
        home_profile_visit_secs: predictions.candidate_continuous_value(
            &tweet_id,
            ContinuousActionName::HomeProfileVisitSecs as usize,
        ),
        ..predictions.candidate_scores(&tweet_id)
    }
}
