use crate::clients::vm_ranker_client::{VMRankerClient, VMRankerCluster};
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::scorers::author_cold_start::AuthorColdStart;
use crate::util::candidates_util;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_vm_ranker_proto::{DppParams, PhoenixScores, RankCandidate, RankRequest, SlateContext};

const AUTHOR_DIVERSITY_VALUE_MODEL_ID: &str = "author_diversity";
const MPN_FOLD_MULTIPLIER_MAX: f64 = 10.0;

pub struct VMRanker {
    pub client: Arc<dyn VMRankerClient>,
    pub xds_client: Option<Arc<dyn VMRankerClient>>,
    pub author_cold_start: AuthorColdStart,
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for VMRanker {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableVMRanker)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let cluster = VMRankerCluster::parse(&query.params.get(VMRankerClusterId));
        let request = build_request(query, candidates);

        let use_xds = self.xds_client.is_some()
            && crate::util::xds::use_xds_for_vm_ranker_cluster(query, &cluster.gate_name());

        let response = if use_xds {
            let xds = self.xds_client.as_ref().expect("checked is_some above");
            match xds.rank(cluster, request.clone()).await {
                Ok(resp) => resp,
                Err(e) => {
                    let enable_fallback = query.params.get(VMRankerEnableFallback);
                    if !enable_fallback {
                        let msg = format!("VMRanker xDS gRPC call failed (fallback disabled): {e}");
                        return vec![Err(msg); candidates.len()];
                    }
                    tracing::warn!(cluster = ?cluster, error = %e, "VMRanker xDS rank failed; falling back to DNS");
                    match self.client.rank(cluster, request).await {
                        Ok(resp) => resp,
                        Err(e) => {
                            let msg = format!("VMRanker gRPC call failed: {e}");
                            return vec![Err(msg); candidates.len()];
                        }
                    }
                }
            }
        } else {
            match self.client.rank(cluster, request).await {
                Ok(resp) => resp,
                Err(e) => {
                    let msg = format!("VMRanker gRPC call failed: {e}");
                    return vec![Err(msg); candidates.len()];
                }
            }
        };

        let score_map: FxHashMap<u64, f64> = response
            .candidates
            .iter()
            .map(|sc| (sc.tweet_id, sc.score))
            .collect();

        let fold_weights = (query.params.get(EnableMpnScoring)
            && query.params.get(VMRankerValueModelId) == AUTHOR_DIVERSITY_VALUE_MODEL_ID)
            .then(|| crate::scorers::ranking_scorer::ScoringWeights::from_params(&query.params));

        let mut scored: Vec<Option<f64>> = candidates
            .iter()
            .map(|c| {
                let returned = score_map.get(&c.tweet_id).copied();
                match (&fold_weights, c.mpn_parts, returned, c.score) {
                    (Some(weights), Some(parts), Some(resp), Some(sent)) if sent > 0.0 => {
                        let multiplier = (resp / sent).clamp(0.0, MPN_FOLD_MULTIPLIER_MAX);
                        let model_net = multiplier * parts.pos - parts.neg;
                        let scaled = if model_net >= 0.0 {
                            parts.scalar_multiplier * model_net
                        } else {
                            model_net
                        };
                        Some(crate::scorers::ranking_scorer::RankingScorer::offset_score(
                            scaled, weights,
                        ))
                    }
                    _ if fold_weights.is_some() => c.score,
                    _ => returned.or(c.score),
                }
            })
            .collect();

        if fold_weights.is_some() {
            let scores: Vec<f64> = scored.iter().map(|s| s.unwrap_or(0.0)).collect();
            let effective = self.author_cold_start.apply(query, candidates, &scores);
            for (slot, value) in scored.iter_mut().zip(effective) {
                if slot.is_some() {
                    *slot = Some(value);
                }
            }
        }

        scored
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

fn build_request(query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> RankRequest {
    let min_video_duration_ms = query.params.get(MinVideoDurationMs);
    let vqv_weight_value = query.params.get(VqvWeight);
    let request_timestamp_ms = query.request_time_ms as u64;
    let scoring_weights = query
        .params
        .get(VMRankerSendHeadWeights)
        .then(|| crate::scorers::ranking_scorer::ScoringWeights::from_params(&query.params));

    let proto_candidates: Vec<RankCandidate> = candidates
        .iter()
        .map(|c| {
            let phoenix_scores = Some(PhoenixScores {
                favorite_score: c.phoenix_scores.favorite_score,
                reply_score: c.phoenix_scores.reply_score,
                retweet_score: c.phoenix_scores.retweet_score,
                photo_expand_score: c.phoenix_scores.photo_expand_score,
                click_score: c.phoenix_scores.click_score,
                profile_click_score: c.phoenix_scores.profile_click_score,
                vqv_score: c.phoenix_scores.vqv_score,
                share_score: c.phoenix_scores.share_score,
                share_via_dm_score: c.phoenix_scores.share_via_dm_score,
                share_via_copy_link_score: c.phoenix_scores.share_via_copy_link_score,
                dwell_score: c.phoenix_scores.dwell_score,
                quote_score: c.phoenix_scores.quote_score,
                quoted_click_score: c.phoenix_scores.quoted_click_score,
                follow_author_score: c.phoenix_scores.follow_author_score,
                not_interested_score: c.phoenix_scores.not_interested_score,
                block_author_score: c.phoenix_scores.block_author_score,
                mute_author_score: c.phoenix_scores.mute_author_score,
                report_score: c.phoenix_scores.report_score,
                not_dwelled_score: c.phoenix_scores.not_dwelled_score,
                dwell_time: c.phoenix_scores.dwell_time,
                click_dwell_time: c.phoenix_scores.click_dwell_time,
                video_open_score: scoring_weights
                    .as_ref()
                    .and(c.phoenix_scores.video_open_score),
                open_link_score: scoring_weights
                    .as_ref()
                    .and(c.phoenix_scores.open_link_score),
                quoted_vqv_score: scoring_weights
                    .as_ref()
                    .and(c.phoenix_scores.quoted_vqv_score),
                post_unexplored_score: scoring_weights
                    .as_ref()
                    .and(c.phoenix_scores.post_unexplored_score),
                active_secs_5m_residual_norm: scoring_weights
                    .as_ref()
                    .and(c.phoenix_scores.active_secs_5m_residual_norm),
            });

            let vqv_weight =
                candidates_util::vqv_weight(query, c, min_video_duration_ms, vqv_weight_value);

            RankCandidate {
                tweet_id: c.tweet_id,
                author_id: c.author_id,
                in_network: c.in_network.unwrap_or(false),
                is_retweet: c.retweeted_tweet_id.is_some(),
                is_reply: c.in_reply_to_tweet_id.is_some(),
                author_followers_count: c.author_followers_count.unwrap_or(0),
                vqv_ineligible: vqv_weight == 0.0,
                retweeted_tweet_id: c.retweeted_tweet_id.unwrap_or(0),
                score: c.score,
                phoenix_scores,
                slate_context: c.slate_context.as_ref().map(|s| SlateContext {
                    k: s.k,
                    pool_rank: s.pool_rank,
                    pool_rank_gap: s.pool_rank_gap,
                    sid_known: s.sid_known,
                    sid_k1: s.sid_k_l1,
                    sid_k2: s.sid_k_l2,
                    sid_k3: s.sid_k_l3,
                    sid_gap1: s.sid_gap_l1,
                    sid_gap2: s.sid_gap_l2,
                    sid_gap3: s.sid_gap_l3,
                }),
                head_weights: scoring_weights
                    .as_ref()
                    .map(|w| w.effective_head_weights(query, c)),
                weighted_score: scoring_weights.as_ref().and(c.weighted_score),
            }
        })
        .collect();

    let dpp_theta = query.params.get(VMRankerDppTheta);
    let dpp_max_selected_rank = query.params.get(VMRankerDppMaxSelectedRank);

    let dpp_params = if dpp_theta > 0.0 || dpp_max_selected_rank > 0 {
        Some(DppParams {
            theta: dpp_theta,
            max_selected_rank: dpp_max_selected_rank,
        })
    } else {
        None
    };

    RankRequest {
        viewer_id: query.user_id,
        request_timestamp_ms,
        candidates: proto_candidates,
        value_model_id: query.params.get(VMRankerValueModelId),
        viewer_following_count: query.user_features.followed_user_ids.len() as u32,
        dpp_params,
        new_user_age_threshold_secs: Some(query.params.get(NewUserAgeThresholdSecs)),
    }
}
