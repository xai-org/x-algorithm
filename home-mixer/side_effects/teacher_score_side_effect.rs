use crate::models::candidate::{CandidateHelpers, PhoenixScores, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixRequestCacheSideEffect, EnableTeacherScoreSideEffect, TeacherActionIndices,
    TeacherTtlSeconds,
};
use crate::scorers::phoenix_scorer::{is_prod_cluster, PhoenixScorer};
use crate::side_effects::phoenix_request_cache_side_effect::{
    encode_cache_value, product_surface, teacher_cache_key,
};
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::Params;
use xai_recsys_proto::{ActionName, CandidateSet, ProductSurface, ScoreInfo, TweetInfo};
use xai_stats_receiver::global_stats_receiver;

const CALLS_METRIC: &str = "TeacherScoreSideEffect.calls";

pub struct TeacherScoreSideEffect {
    teacher_cache_client: Arc<dyn RedisClient>,
}

impl TeacherScoreSideEffect {
    pub fn new(teacher_cache_client: Arc<dyn RedisClient>) -> Self {
        Self {
            teacher_cache_client,
        }
    }
}

fn teacher_action_indices(params: &Params) -> Vec<u32> {
    params
        .get(TeacherActionIndices)
        .into_iter()
        .filter_map(|index| u32::try_from(index).ok())
        .collect()
}

fn served_prob(scores: &PhoenixScores, action: ActionName) -> Option<f64> {
    match action {
        ActionName::ServerTweetFav => scores.favorite_score,
        ActionName::ServerTweetReply => scores.reply_score,
        ActionName::ServerTweetRetweet => scores.retweet_score,
        ActionName::ClientTweetPhotoExpand => scores.photo_expand_score,
        ActionName::ClientTweetVideoOpen => scores.video_open_score,
        ActionName::ClientTweetClick => scores.click_score,
        ActionName::ClientTweetOpenLink => scores.open_link_score,
        ActionName::ClientTweetClickProfile => scores.profile_click_score,
        ActionName::ClientTweetVideoQualityView => scores.vqv_score,
        ActionName::ClientTweetShare => scores.share_score,
        ActionName::ClientTweetClickSendViaDirectMessage => scores.share_via_dm_score,
        ActionName::ClientTweetShareViaCopyLink => scores.share_via_copy_link_score,
        ActionName::ClientTweetBookmark => scores.bookmark_score,
        ActionName::ClientTweetRecapDwelled => scores.dwell_score,
        ActionName::ServerTweetQuote => scores.quote_score,
        ActionName::ClientQuotedTweetClick => scores.quoted_click_score,
        ActionName::ClientQuotedTweetVideoQualityView => scores.quoted_vqv_score,
        ActionName::ClientTweetFollowAuthor => scores.follow_author_score,
        ActionName::ClientTweetNotInterestedIn => scores.not_interested_score,
        ActionName::ClientTweetBlockAuthor => scores.block_author_score,
        ActionName::ClientTweetMuteAuthor => scores.mute_author_score,
        ActionName::ClientTweetReport => scores.report_score,
        ActionName::ClientTweetRecapNotDwelled => scores.not_dwelled_score,
        ActionName::ClientAppExit => scores.immersive_exit_score,
        ActionName::ServerTweetPostUnexplored => scores.post_unexplored_score,
        ActionName::ServerTweetPostVideo => scores.post_video_score,
        ActionName::ClientTweetClickGrokAnalyze => scores.grok_analyze_score,
        _ => None,
    }
}

pub(crate) fn teacher_entry(
    user_id: u64,
    product_surface: ProductSurface,
    served: &[PostCandidate],
    action_indices: &[u32],
) -> CandidateSet {
    let actions: Vec<Option<ActionName>> = action_indices
        .iter()
        .map(|&index| ActionName::try_from(index as i32).ok())
        .collect();
    let mut seen = HashSet::new();
    let scored = served
        .iter()
        .filter(|c| seen.insert(c.get_original_tweet_id()))
        .filter_map(|c| {
            let prediction_values = actions
                .iter()
                .map(|action| served_prob(&c.phoenix_scores, (*action)?).map(|prob| prob as f32))
                .collect::<Option<Vec<f32>>>()?;
            Some(TweetInfo {
                tweet_id: c.get_original_tweet_id(),
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
            && params.get(EnableTeacherScoreSideEffect)
            && params.get(EnablePhoenixRequestCacheSideEffect)
            && !teacher_action_indices(params).is_empty()
            && (query.has_cached_posts
                || (query.scoring_sequence.is_some()
                    && is_prod_cluster(&PhoenixScorer::resolve_cluster(&query))))
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let params = &query.params;
        let Some(prediction_request_id) = input
            .selected_candidates
            .iter()
            .find_map(|c| c.prediction_request_id)
        else {
            return Ok(());
        };
        let entry = teacher_entry(
            query.user_id,
            product_surface(query),
            &input.selected_candidates,
            &teacher_action_indices(params),
        );
        if entry.candidates.is_empty() {
            record_call("empty");
            return Ok(());
        }
        let key = teacher_cache_key(prediction_request_id);
        let value = encode_cache_value(&entry)?;
        let ttl_seconds = params.get(TeacherTtlSeconds);
        let written = if query.has_cached_posts {
            self.teacher_cache_client
                .append_if_exists_ex(key, value, ttl_seconds)
                .await
                .map(|appended| if appended { "appended" } else { "no_entry" })
        } else {
            self.teacher_cache_client
                .set_ex(key, value, ttl_seconds)
                .await
                .map(|()| "written")
        };
        record_call(match &written {
            Ok(outcome) => outcome,
            Err(_) => "write_error",
        });
        written.map(|_| ()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::PredictNextActions;
    use xai_candidate_pipeline::component_library::clients::redis_client::MockRedisClient;
    use xai_feature_switches::{FeatureSwitches, Param, RecipientBuilder, Value};
    use xai_recsys_proto::{
        CandidateDistributionSet, NextActionDistribution, PredictNextActionsResponse,
    };

    fn candidate(tweet_id: u64, weighted_score: Option<f64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            weighted_score,
            ..Default::default()
        }
    }

    #[test]
    fn teacher_entry_keeps_the_configured_heads_of_each_served_post_once() {
        let scored = |tweet_id: u64, fav: f64, reply: f64| {
            let mut c = candidate(tweet_id, Some(1.0));
            c.phoenix_scores.favorite_score = Some(fav);
            c.phoenix_scores.reply_score = Some(reply);
            c
        };
        let mut retweet = scored(11, 0.125, 0.75);
        retweet.retweeted_tweet_id = Some(10);
        let mut retweet_of_served = scored(12, 0.9, 0.9);
        retweet_of_served.retweeted_tweet_id = Some(1);
        let candidates = vec![
            scored(1, 0.5, 0.25),
            retweet,
            candidate(2, Some(1.0)),
            retweet_of_served,
        ];
        let fav = ActionName::ServerTweetFav as u32;
        let reply = ActionName::ServerTweetReply as u32;

        let entry = teacher_entry(
            7,
            ProductSurface::HomeTimelineRanking,
            &candidates,
            &[fav, reply],
        );

        assert_eq!(entry.user_id, 7);
        let rows: Vec<(u64, u64, Vec<u32>, Vec<f32>)> = entry
            .candidates
            .iter()
            .map(|t| {
                let score_info = t.score_info.clone().unwrap_or_default();
                (
                    t.tweet_id,
                    t.retweeting_tweet_id,
                    score_info.prediction_heads,
                    score_info.prediction_values,
                )
            })
            .collect();
        assert_eq!(
            rows,
            vec![
                (1, 0, vec![fav, reply], vec![0.5, 0.25]),
                (10, 11, vec![fav, reply], vec![0.125, 0.75]),
            ]
        );
    }

    #[tokio::test]
    async fn later_pages_append_their_posts_to_the_request_entry() {
        let fs = FeatureSwitches::new(vec![]).unwrap();
        let mut results = fs.match_recipient(&RecipientBuilder::new().build());
        results.override_value(
            TeacherActionIndices.key().to_string(),
            Value::Array(vec![Value::from(ActionName::ServerTweetFav as i64)]),
        );
        let params: Params = results.into();
        let page = |has_cached_posts: bool, prediction_request_id: u64, tweet_id: u64| {
            let mut post = candidate(tweet_id, Some(1.0));
            post.phoenix_scores.favorite_score = Some(0.5);
            post.prediction_request_id = Some(prediction_request_id);
            Arc::new(SideEffectInput {
                query: Arc::new(ScoredPostsQuery {
                    params: params.clone(),
                    has_cached_posts,
                    ..Default::default()
                }),
                selected_candidates: vec![post],
                non_selected_candidates: vec![],
            })
        };
        let cache = Arc::new(MockRedisClient::default());
        let side_effect = TeacherScoreSideEffect::new(cache.clone());

        side_effect.side_effect(page(false, 7, 1)).await.unwrap();
        side_effect.side_effect(page(true, 7, 2)).await.unwrap();
        side_effect.side_effect(page(true, 8, 3)).await.unwrap();

        let stored = cache.get(teacher_cache_key(7)).await.unwrap();
        let entry =
            CandidateSet::decode(zstd::decode_all(stored.as_slice()).unwrap().as_slice()).unwrap();
        let served: Vec<u64> = entry.candidates.iter().map(|t| t.tweet_id).collect();
        assert_eq!(served, vec![1, 2]);
        assert!(cache.get_opt(teacher_cache_key(8)).await.unwrap().is_none());
    }

    #[test]
    fn served_probs_are_the_scores_the_ranker_returned() {
        let heads = [
            ActionName::ServerTweetFav,
            ActionName::ServerTweetReply,
            ActionName::ServerTweetQuote,
            ActionName::ServerTweetRetweet,
            ActionName::ClientTweetRecapDwelled,
            ActionName::ClientTweetRecapNotDwelled,
            ActionName::ClientTweetVideoQualityView,
            ActionName::ClientTweetPhotoExpand,
            ActionName::ClientTweetReport,
            ActionName::ClientTweetNotInterestedIn,
            ActionName::ClientTweetFollowAuthor,
            ActionName::ClientTweetBlockAuthor,
            ActionName::ClientTweetMuteAuthor,
            ActionName::ClientTweetClick,
            ActionName::ClientTweetClickProfile,
            ActionName::ClientTweetShareViaCopyLink,
            ActionName::ClientTweetClickSendViaDirectMessage,
            ActionName::ClientTweetShare,
            ActionName::ClientQuotedTweetClick,
            ActionName::ClientTweetOpenLink,
            ActionName::ServerTweetPostUnexplored,
        ];
        let prob = |k: usize| 0.01 * (k + 1) as f32;
        let predictions = PredictNextActions::new(PredictNextActionsResponse {
            distribution_sets: vec![CandidateDistributionSet {
                candidate_distributions: vec![NextActionDistribution {
                    candidate: Some(TweetInfo {
                        tweet_id: 1,
                        ..Default::default()
                    }),
                    index_to_logits: heads
                        .iter()
                        .enumerate()
                        .map(|(k, &head)| (head as u32, prob(k).ln()))
                        .collect(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        });
        let scores = predictions.candidate_scores(&1);

        for (k, &head) in heads.iter().enumerate() {
            let served = served_prob(&scores, head).unwrap_or_default() as f32;
            assert!(
                (served - prob(k)).abs() < 1e-6,
                "{head:?}: {served} vs {}",
                prob(k)
            );
        }
    }
}
