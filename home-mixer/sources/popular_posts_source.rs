use crate::models::candidate::{PostCandidate, RetrievalSource};
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnablePopularPostsSource, PopularPostsMaxResults};
use crate::util::popular_authors::now_ms;
use crate::util::popular_posts::PopularPostsCache;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

const METRIC: &str = "PopularPostsSource";
const SERVED_TYPE: pb::ServedType = pb::ServedType::ForYouPopularPosts;

pub struct PopularPostsSource {
    pub popular_posts: Arc<PopularPostsCache>,
}

fn record(stage: &str, value: usize) {
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.incr(METRIC, &[("stage", stage)], value as u64);
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PopularPostsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
            && !query.in_network_only
            && query.params.get(EnablePopularPostsSource)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let now = now_ms();
        self.popular_posts.maybe_spawn_refresh(now);
        let limit = query.params.get(PopularPostsMaxResults) as usize;
        let (generated_at_ms, posts) = self.popular_posts.top_posts(limit);
        record("requests", 1);
        if posts.is_empty() {
            record("empty", 1);
            return Ok(Vec::new());
        }
        if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
            receiver.gauge(
                "PopularPostsSource.ListAgeSecs",
                &[],
                (now - generated_at_ms) as f64 / 1000.0,
            );
        }
        let seen: HashSet<u64> = query.seen_ids.iter().copied().collect();
        let candidates: Vec<PostCandidate> = posts
            .into_iter()
            .filter(|p| !seen.contains(&p.post_id))
            .map(|p| PostCandidate {
                tweet_id: p.post_id,
                author_id: p.author_id,
                served_type: Some(SERVED_TYPE),
                retrieval_sources: vec![RetrievalSource::from_served_type(SERVED_TYPE)],
                ..Default::default()
            })
            .collect();
        record("fetched", candidates.len());
        Ok(candidates)
    }
}
