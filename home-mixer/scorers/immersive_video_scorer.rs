use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::VideoCarouselImmersiveClusterId;
use crate::util::egress::PredictionDispatch;
use crate::util::phoenix_request::build_immersive_prediction_request;
use crate::util::video_carousel;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::PhoenixCluster;
use xai_candidate_pipeline::scorer::Scorer;

pub struct ImmersiveVideoScorer {
    pub dispatch: PredictionDispatch,
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for ImmersiveVideoScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
            && video_carousel::ranks_by_immersive_scores(query)
            && query.immersive_columnar_scoring_sequence.is_some()
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let needs_scores = |candidate: &PostCandidate| {
            video_carousel::is_carousel_video(candidate) && candidate.immersive_scores.is_none()
        };
        let keep_existing = |candidate: &PostCandidate| PostCandidate {
            immersive_scores: candidate.immersive_scores.clone(),
            ..Default::default()
        };
        let videos: Vec<&PostCandidate> = candidates
            .iter()
            .filter(|candidate| needs_scores(candidate))
            .collect();
        if videos.is_empty() {
            return candidates.iter().map(|c| Ok(keep_existing(c))).collect();
        }

        let cluster = PhoenixCluster::parse(&query.params.get(VideoCarouselImmersiveClusterId));
        let request = build_immersive_prediction_request(query, &videos);
        let predictions = match self.dispatch.predict(query, cluster, request).await {
            Ok(predictions) => predictions,
            Err(err) => {
                return vec![Err(format!("Immersive prediction failed: {err}")); candidates.len()];
            }
        };

        candidates
            .iter()
            .map(|candidate| {
                if !needs_scores(candidate) {
                    return Ok(keep_existing(candidate));
                }
                Ok(PostCandidate {
                    immersive_scores: Some(
                        predictions.candidate_scores(&candidate.get_original_tweet_id()),
                    ),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.immersive_scores = scored.immersive_scores;
    }
}
