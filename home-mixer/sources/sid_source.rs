use std::sync::{Arc, LazyLock};

use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_recsys_sid_retrieval_proto::RetrieveRequest;
use xai_stats_receiver::global_stats_receiver;

use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::clients::sid_retrieval_client::SidRetrievalClient;
use crate::models::candidate::{PostCandidate, RetrievalSource};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableSidSource, SidSourceMaxPerSeed, SidSourceMaxResults, SidSourceMaxSeeds,
    SidSourceMinPrefixDepth,
};
use crate::sources::simclusters_source::{has_post_signals, post_signal_ids};

const METRIC_NAME: &str = "SidSource.retrieved";

static CALLER: LazyLock<String> = LazyLock::new(|| {
    format!(
        "home-mixer-{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "prod".to_string())
    )
});

pub struct SidSource {
    client: Arc<dyn SidRetrievalClient>,
    core_data_hydrator: CoreDataCandidateHydrator,
}

impl SidSource {
    pub fn new(
        client: Arc<dyn SidRetrievalClient>,
        core_data_hydrator: CoreDataCandidateHydrator,
    ) -> Self {
        Self {
            client,
            core_data_hydrator,
        }
    }
}

pub fn sid_source_enabled(query: &ScoredPostsQuery) -> bool {
    query.params.get(EnableSidSource)
}

fn record(kind: &'static str, count: usize) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(METRIC_NAME, &[("type", kind)], count as u64);
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for SidSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        sid_source_enabled(query)
            && !query.is_topic_request()
            && !query.in_network_only
            && !query.has_cached_posts
            && has_post_signals(query)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let mut seed_post_ids = post_signal_ids(query);
        seed_post_ids.truncate(query.params.get(SidSourceMaxSeeds));
        if seed_post_ids.is_empty() {
            return Ok(vec![]);
        }
        record("seeds", seed_post_ids.len());

        let response = self
            .client
            .retrieve(RetrieveRequest {
                seed_post_ids,
                max_results: query.params.get(SidSourceMaxResults),
                max_per_seed: query.params.get(SidSourceMaxPerSeed),
                min_prefix_depth: query.params.get(SidSourceMinPrefixDepth),
                max_prefix_depth: 0,
                caller: CALLER.clone(),
            })
            .await
            .map_err(|e| format!("SidSource: {e}"))?;
        record("retrieved", response.candidates.len());

        let mut candidates: Vec<PostCandidate> = response
            .candidates
            .into_iter()
            .enumerate()
            .map(|(index, c)| PostCandidate {
                tweet_id: c.post_id as u64,
                served_type: Some(pb::ServedType::ForYouPhoenixRetrievalMoe),
                retrieval_sources: vec![RetrievalSource {
                    served_type: pb::ServedType::ForYouPhoenixRetrievalMoe,
                    cluster: None,
                    score: Some(c.shared_prefix_depth as f32),
                    position: Some(index as u32 + 1),
                }],
                ..Default::default()
            })
            .collect();

        if let Ok(fetched) = self.core_data_hydrator.fetch(query, &candidates).await {
            self.core_data_hydrator
                .apply(query, &mut candidates, fetched);
        }
        candidates.retain(|c| c.author_id != 0);
        record("hydrated", candidates.len());
        Ok(candidates)
    }
}
