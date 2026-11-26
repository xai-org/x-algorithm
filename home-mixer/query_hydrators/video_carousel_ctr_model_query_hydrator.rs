use crate::clients::vm_ranker_client::VMRankerClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableVideoCarouselCtrModel, VideoCarouselCtrModelName, VideoCarouselEmbeddingVersion,
};
use crate::util::video_carousel;
use crate::util::video_carousel_ctr_model::{self, CtrModel};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

const REFRESH_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Default)]
struct CachedModel {
    model: Option<Arc<CtrModel>>,
    fetched_at: Option<Instant>,
    fetching: bool,
}

pub struct VideoCarouselCtrModelQueryHydrator {
    client: Arc<dyn VMRankerClient>,
    models: Arc<Mutex<HashMap<String, CachedModel>>>,
}

impl VideoCarouselCtrModelQueryHydrator {
    pub fn new(client: Arc<dyn VMRankerClient>) -> Self {
        Self {
            client,
            models: Arc::default(),
        }
    }

    fn cached_or_refresh(&self, name: &str) -> Option<Arc<CtrModel>> {
        let (model, refresh) = {
            let mut models = self.models.lock().ok()?;
            let cached = models.entry(name.to_string()).or_default();
            let stale = cached
                .fetched_at
                .is_none_or(|fetched_at| fetched_at.elapsed() >= REFRESH_INTERVAL);
            let refresh = stale && !cached.fetching;
            cached.fetching |= refresh;
            (cached.model.clone(), refresh)
        };
        if refresh {
            let client = Arc::clone(&self.client);
            let models = Arc::clone(&self.models);
            let name = name.to_string();
            tokio::spawn(async move {
                let fetched = video_carousel_ctr_model::fetch(client.as_ref(), &name).await;
                if let Ok(mut models) = models.lock() {
                    let cached = models.entry(name).or_default();
                    cached.fetching = false;
                    cached.fetched_at = Some(Instant::now());
                    if let Some(model) = fetched {
                        cached.model = Some(Arc::new(model));
                    }
                }
            });
        }
        model
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for VideoCarouselCtrModelQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query) && query.params.get(EnableVideoCarouselCtrModel)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let name = query.params.get(VideoCarouselCtrModelName);
        let cached = if name.is_empty() {
            None
        } else {
            self.cached_or_refresh(&name)
        };
        let (model, outcome) = match cached {
            Some(model) if model.embedding == query.params.get(VideoCarouselEmbeddingVersion) => {
                (Some(model), "model")
            }
            Some(_) => (None, "embedding_mismatch"),
            None => (None, "fallback"),
        };
        video_carousel_ctr_model::record_request(outcome);
        Ok(ScoredPostsQuery {
            video_carousel_ctr_model: Arc::new(model.map(OnceLock::from).unwrap_or_default()),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        if let Some(model) = hydrated.video_carousel_ctr_model.get() {
            let _ = query.video_carousel_ctr_model.set(Arc::clone(model));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tonic::Status;
    use xai_vm_ranker_proto::{RankRequest, RankResponse};

    use crate::clients::vm_ranker_client::VMRankerCluster;

    struct FakeClient {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl VMRankerClient for FakeClient {
        async fn rank(
            &self,
            _cluster: VMRankerCluster,
            _request: RankRequest,
        ) -> Result<RankResponse, Status> {
            Err(Status::unimplemented("rank"))
        }

        async fn get_carousel_model(&self, name: &str) -> Result<Vec<u8>, Status> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match name {
                "lr" => Ok(video_carousel_ctr_model::test_model_json().into_bytes()),
                _ => Err(Status::not_found(name.to_string())),
            }
        }
    }

    #[tokio::test]
    async fn first_request_triggers_one_fetch_and_later_requests_get_the_model() {
        let client = Arc::new(FakeClient {
            calls: AtomicUsize::new(0),
        });
        let hydrator = VideoCarouselCtrModelQueryHydrator::new(client.clone());

        assert!(hydrator.cached_or_refresh("lr").is_none());
        assert!(hydrator.cached_or_refresh("lr").is_none());
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            hydrator
                .cached_or_refresh("lr")
                .map(|model| model.embedding.clone()),
            Some("v8_search_1".to_string())
        );

        assert!(hydrator.cached_or_refresh("missing").is_none());
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(hydrator.cached_or_refresh("missing").is_none());
        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
    }

    async fn for_you_query_gets_model(
        hydrator: &VideoCarouselCtrModelQueryHydrator,
        embedding: &str,
    ) -> bool {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_video_carousel_ctr_model_name".to_string(),
            "lr",
        );
        results.override_fs(
            "rust_home_mixer_video_carousel_embedding_version".to_string(),
            embedding,
        );
        let for_you = ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        };
        let mut phoenix = for_you.clone();
        let hydrated = hydrator.hydrate(&phoenix).await.unwrap();
        hydrator.update(&mut phoenix, hydrated);
        for_you.video_carousel_ctr_model.get().is_some()
    }

    #[tokio::test]
    async fn shares_the_model_with_the_for_you_query_only_when_its_embedding_matches() {
        let hydrator = VideoCarouselCtrModelQueryHydrator::new(Arc::new(FakeClient {
            calls: AtomicUsize::new(0),
        }));

        assert!(!for_you_query_gets_model(&hydrator, "v8_search_1").await);
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(for_you_query_gets_model(&hydrator, "v8_search_1").await);
        assert!(!for_you_query_gets_model(&hydrator, "v8_search_video_1").await);
    }
}
