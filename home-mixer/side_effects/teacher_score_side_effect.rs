use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixRequestCacheSideEffect, EnableTeacherScoreSideEffect, TeacherActionIndices,
    TeacherPhoenixCluster, TeacherSampleBps, TeacherTimeoutMs, TeacherTopK, TeacherTtlSeconds,
};
use crate::side_effects::phoenix_request_cache_side_effect::{
    candidate_cache_key, encode_cache_value, product_surface, teacher_cache_key, write_to_both_dcs,
};
use crate::util::phoenix_request::{build_prediction_request, build_tweet_infos};
use crate::util::post_engagers::clear_post_engagers;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PhoenixPredictionClient, PredictNextActions,
};
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::utils::{is_prod, is_sampled};
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::Params;
use xai_recsys_proto::{CandidateSet, ProductSurface, ScoreInfo, TweetInfo};
use xai_stats_receiver::global_stats_receiver;

const CALLS_METRIC: &str = "TeacherScoreSideEffect.calls";

pub struct TeacherScoreSideEffect {
    phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
    teacher_cache_atla_client: Arc<dyn RedisClient>,
    teacher_cache_pdxa_client: Arc<dyn RedisClient>,
}

impl TeacherScoreSideEffect {
    pub fn new(
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        teacher_cache_atla_client: Arc<dyn RedisClient>,
        teacher_cache_pdxa_client: Arc<dyn RedisClient>,
    ) -> Self {
        Self {
            phoenix_client,
            teacher_cache_atla_client,
            teacher_cache_pdxa_client,
        }
    }
}

fn teacher_cluster(params: &Params) -> Option<PhoenixCluster> {
    params.get(TeacherPhoenixCluster).parse().ok()
}

fn teacher_action_indices(params: &Params) -> Vec<u32> {
    params
        .get(TeacherActionIndices)
        .into_iter()
        .filter_map(|index| u32::try_from(index).ok())
        .collect()
}

pub(crate) fn teacher_candidates<'a>(
    selected: &'a [PostCandidate],
    non_selected: &'a [PostCandidate],
    top_k: usize,
) -> Vec<&'a PostCandidate> {
    let mut ranked: Vec<&PostCandidate> = non_selected
        .iter()
        .filter(|c| c.weighted_score.is_some_and(|s| s > 0.0))
        .collect();
    ranked.sort_by(|a, b| {
        b.weighted_score
            .unwrap_or_default()
            .total_cmp(&a.weighted_score.unwrap_or_default())
    });
    let mut seen = HashSet::new();
    selected
        .iter()
        .chain(ranked)
        .filter(|c| seen.insert(c.get_original_tweet_id()))
        .take(top_k)
        .collect()
}

pub(crate) fn teacher_entry(
    user_id: u64,
    product_surface: ProductSurface,
    candidates: &[PostCandidate],
    predictions: &PredictNextActions,
    action_indices: &[u32],
) -> CandidateSet {
    let scored = candidates
        .iter()
        .filter_map(|c| {
            let original_tweet_id = c.get_original_tweet_id();
            let prediction_values = action_indices
                .iter()
                .map(|&index| {
                    predictions
                        .candidate_action_prob(&original_tweet_id, index as usize)
                        .map(|prob| prob as f32)
                })
                .collect::<Option<Vec<f32>>>()?;
            Some(TweetInfo {
                tweet_id: original_tweet_id,
                retweeting_tweet_id: if c.retweeted_tweet_id.is_some() {
                    c.tweet_id
                } else {
                    0
                },
                score_info: Some(ScoreInfo {
                    prediction_heads: action_indices.to_vec(),
                    prediction_values,
                    ..Default::default()
                }),
                ..Default::default()
            })
        })
        .collect();
    CandidateSet {
        user_id,
        candidates: scored,
        product_surface: product_surface as i32,
        ..Default::default()
    }
}

fn record_call(outcome: &str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(CALLS_METRIC, &[("outcome", outcome)], 1);
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for TeacherScoreSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        let params = &query.params;
        is_prod()
            && !query.has_cached_posts
            && query.scoring_sequence.is_some()
            && params.get(EnableTeacherScoreSideEffect)
            && params.get(EnablePhoenixRequestCacheSideEffect)
            && is_sampled(query.user_id, params.get(TeacherSampleBps) as f64 / 100.0)
            && !teacher_action_indices(params).is_empty()
            && teacher_cluster(params).is_some()
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let params = &query.params;
        let Some(cluster) = teacher_cluster(params) else {
            return Ok(());
        };
        let Some(prediction_request_id) = input
            .selected_candidates
            .iter()
            .find_map(|c| c.prediction_request_id)
        else {
            return Ok(());
        };
        let candidates: Vec<PostCandidate> = teacher_candidates(
            &input.selected_candidates,
            &input.non_selected_candidates,
            params.get(TeacherTopK),
        )
        .into_iter()
        .cloned()
        .collect();
        let serve_time_ms = candidates
            .iter()
            .find_map(|c| c.last_scored_at_ms)
            .map_or(query.request_time_ms, |ms| ms as i64);
        let action_indices = teacher_action_indices(params);
        let mut request =
            build_prediction_request(query, &candidates, product_surface(query), serve_time_ms);
        request.requested_action_indices = action_indices.clone();
        request.return_log_map = true;
        request.return_logprob = false;
        let timeout = Duration::from_millis(params.get(TeacherTimeoutMs));
        let predictions =
            match tokio::time::timeout(timeout, self.phoenix_client.predict(cluster, request, 0))
                .await
            {
                Ok(Ok(predictions)) => predictions,
                Ok(Err(err)) => {
                    record_call("predict_error");
                    return Err(format!("teacher {cluster:?} request failed: {err}"));
                }
                Err(_) => {
                    record_call("timeout");
                    return Err(format!("teacher {cluster:?} timed out after {timeout:?}"));
                }
            };

        let entry = teacher_entry(
            query.user_id,
            product_surface(query),
            &candidates,
            &predictions,
            &action_indices,
        );
        if entry.candidates.is_empty() {
            record_call("empty");
            return Ok(());
        }
        let scored: HashSet<u64> = entry.candidates.iter().map(|t| t.tweet_id).collect();
        let scored_candidates: Vec<PostCandidate> = candidates
            .into_iter()
            .filter(|c| scored.contains(&c.get_original_tweet_id()))
            .collect();

        let mut entries = vec![(
            teacher_cache_key(prediction_request_id),
            encode_cache_value(&entry)?,
        )];
        for mut tweet_info in build_tweet_infos(query, &scored_candidates, query.request_time_ms) {
            clear_post_engagers(&mut tweet_info);
            entries.push((
                candidate_cache_key(prediction_request_id, tweet_info.tweet_id),
                encode_cache_value(&tweet_info)?,
            ));
        }
        let written = write_to_both_dcs(
            self.teacher_cache_atla_client.as_ref(),
            self.teacher_cache_pdxa_client.as_ref(),
            entries,
            params.get(TeacherTtlSeconds),
        )
        .await;
        record_call(if written.is_ok() {
            "written"
        } else {
            "write_error"
        });
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_recsys_proto::{
        ActionName, CandidateDistributionSet, NextActionDistribution, PredictNextActionsResponse,
    };

    fn candidate(tweet_id: u64, weighted_score: Option<f64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            weighted_score,
            ..Default::default()
        }
    }

    #[test]
    fn teacher_candidates_are_the_selected_then_the_best_ranked_unselected() {
        let selected = vec![candidate(1, Some(9.0)), candidate(2, Some(8.0))];
        let mut retweet_of_selected = candidate(8, Some(5.0));
        retweet_of_selected.retweeted_tweet_id = Some(1);
        let non_selected = vec![
            retweet_of_selected,
            candidate(3, Some(0.5)),
            candidate(4, Some(2.0)),
            candidate(5, None),
            candidate(6, Some(0.0)),
            candidate(2, Some(3.0)),
            candidate(7, Some(1.0)),
        ];
        let ids: Vec<u64> = teacher_candidates(&selected, &non_selected, 5)
            .iter()
            .map(|c| c.tweet_id)
            .collect();
        assert_eq!(ids, vec![1, 2, 4, 7, 3]);
    }

    #[test]
    fn teacher_entry_keeps_the_configured_heads_of_each_scored_candidate() {
        let mut retweet = candidate(11, Some(1.0));
        retweet.retweeted_tweet_id = Some(10);
        let candidates = vec![candidate(1, Some(2.0)), retweet, candidate(2, Some(1.0))];
        let fav = ActionName::ServerTweetFav as u32;
        let reply = ActionName::ServerTweetReply as u32;
        let distribution = |tweet_id: u64, fav_prob: f32, reply_prob: f32| NextActionDistribution {
            candidate: Some(TweetInfo {
                tweet_id,
                ..Default::default()
            }),
            index_to_logits: [(fav, fav_prob.ln()), (reply, reply_prob.ln())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let predictions = PredictNextActions::new(PredictNextActionsResponse {
            distribution_sets: vec![CandidateDistributionSet {
                candidate_distributions: vec![
                    distribution(1, 0.5, 0.25),
                    distribution(10, 0.125, 0.75),
                ],
                ..Default::default()
            }],
            ..Default::default()
        });

        let entry = teacher_entry(
            7,
            ProductSurface::HomeTimelineRanking,
            &candidates,
            &predictions,
            &[fav, reply],
        );

        assert_eq!(entry.user_id, 7);
        let retweeting: Vec<u64> = entry
            .candidates
            .iter()
            .map(|t| t.retweeting_tweet_id)
            .collect();
        assert_eq!(retweeting, vec![0, 11]);
        let rows: Vec<(u64, Vec<u32>, Vec<f32>)> = entry
            .candidates
            .iter()
            .map(|t| {
                let score_info = t.score_info.clone().unwrap_or_default();
                (
                    t.tweet_id,
                    score_info.prediction_heads,
                    score_info.prediction_values,
                )
            })
            .collect();
        let expected = [(1, [0.5, 0.25]), (10, [0.125, 0.75])];
        assert_eq!(rows.len(), expected.len());
        for ((tweet_id, heads, values), (expected_id, expected_values)) in rows.iter().zip(expected)
        {
            assert_eq!(*tweet_id, expected_id);
            assert_eq!(heads, &vec![fav, reply]);
            for (value, expected_value) in values.iter().zip(expected_values) {
                assert!(
                    (value - expected_value).abs() < 1e-6,
                    "{value} vs {expected_value}"
                );
            }
        }
    }
}
