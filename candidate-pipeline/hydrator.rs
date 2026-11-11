use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use crate::SPAN_LEVEL;
use std::any::{type_name_of_val, Any};
use std::borrow::Cow;
use std::hash::Hash;
use tonic::async_trait;
use tracing::warn;
use xai_stats_receiver::global_stats_receiver;

#[async_trait]
pub trait Hydrator<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>>;

    async fn hydrate_for_stage(
        &self,
        query: &Q,
        candidates: &[C],
        _stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        self.hydrate(query, candidates).await
    }

    #[xai_stats_macro::receive_stats(latency=Bucket50To500, size=Bucket500To2500)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "hydrator", fields(name = self.name()))]
    async fn run(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let hydrated = self.hydrate_for_stage(query, candidates, stage).await;
        let expected_len = candidates.len();
        let results = if hydrated.len() == expected_len {
            hydrated
        } else {
            let message = format!(
                "Hydrator length_mismatch expected={} got={}",
                expected_len,
                hydrated.len()
            );
            warn!(
                "Skipped: length_mismatch expected={} got={}",
                expected_len,
                hydrated.len()
            );
            vec![Err(message); expected_len]
        };
        stats.finish_with_results(&results);
        results
    }

    fn update(&self, candidate: &mut C, hydrated: C);

    fn update_all(&self, candidates: &mut [C], hydrated: Vec<Result<C, String>>) {
        for (candidate, hydrated) in candidates.iter_mut().zip(hydrated) {
            if let Ok(hydrated) = hydrated {
                self.update(candidate, hydrated);
            }
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}

#[async_trait]
pub trait CacheStore<K, V>: Send + Sync {
    async fn get(&self, key: &K) -> Option<V>;
    async fn insert(&self, key: K, value: V);
}

#[async_trait]
pub trait CachedHydrator<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    type CacheKey: Eq + Hash + Send + Sync + 'static;
    type CacheValue: Clone + Send + Sync + 'static;

    fn enable(&self, _query: &Q) -> bool {
        true
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue>;
    fn cache_key(&self, candidate: &C) -> Self::CacheKey;
    fn cache_key_for(&self, _query: &Q, candidate: &C) -> Self::CacheKey {
        self.cache_key(candidate)
    }
    fn cache_value(&self, hydrated: &C) -> Self::CacheValue;

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> C;
    async fn hydrate_from_client(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>>;

    fn already_hydrated(&self, _candidate: &C) -> bool {
        false
    }

    fn update(&self, candidate: &mut C, hydrated: C);

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat_cache(&self, cache_hits: usize, cache_misses: usize, stage: PipelineStage) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.cache", self.name());
            if cache_hits > 0 {
                receiver.incr(
                    metric_name.as_str(),
                    &stage.stat_labels(self.name(), "cache_hit"),
                    cache_hits as u64,
                );
            }
            if cache_misses > 0 {
                receiver.incr(
                    metric_name.as_str(),
                    &stage.stat_labels(self.name(), "cache_miss"),
                    cache_misses as u64,
                );
            }
        }
    }
}

#[async_trait]
impl<Q, C, T> Hydrator<Q, C> for T
where
    Q: PipelineQuery,
    C: PipelineCandidate,
    T: CachedHydrator<Q, C> + ?Sized,
{
    fn enable(&self, query: &Q) -> bool {
        CachedHydrator::enable(self, query)
    }

    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>> {
        self.hydrate_for_stage(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn hydrate_for_stage(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        let mut results = Vec::new();
        let mut missing_keys = Vec::new();
        let mut missing_indices = Vec::new();
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;

        for (index, candidate) in candidates.iter().enumerate() {
            if self.already_hydrated(candidate) {
                if results.is_empty() {
                    results.resize_with(candidates.len(), || None);
                }
                results[index] = Some(Ok(self.hydrate_from_cache(self.cache_value(candidate))));
                continue;
            }
            let key = self.cache_key_for(query, candidate);
            match self.cache_store().get(&key).await {
                Some(value) => {
                    if results.is_empty() {
                        results.resize_with(candidates.len(), || None);
                    }
                    results[index] = Some(Ok(self.hydrate_from_cache(value)));
                    cache_hits += 1;
                }
                None => {
                    missing_keys.push(key);
                    missing_indices.push(index);
                    cache_misses += 1;
                }
            }
        }

        self.stat_cache(cache_hits, cache_misses, stage);

        if !missing_indices.is_empty() {
            let missing_candidates: Cow<'_, [C]> = if results.is_empty() {
                Cow::Borrowed(candidates)
            } else {
                Cow::Owned(
                    missing_indices
                        .iter()
                        .map(|&i| candidates[i].clone())
                        .collect(),
                )
            };
            let hydrated_missing = self.hydrate_from_client(query, &missing_candidates).await;
            if hydrated_missing.len() != missing_indices.len() {
                let message = format!(
                    "CachedHydrator length_mismatch expected={} got={}",
                    missing_indices.len(),
                    hydrated_missing.len()
                );
                return vec![Err(message); candidates.len()];
            }

            if results.is_empty() {
                for (key, hydrated) in missing_keys.into_iter().zip(&hydrated_missing) {
                    if let Ok(hydrated) = hydrated {
                        self.cache_store()
                            .insert(key, self.cache_value(hydrated))
                            .await;
                    }
                }
                return hydrated_missing;
            }

            for ((index, key), hydrated) in missing_indices
                .into_iter()
                .zip(missing_keys)
                .zip(hydrated_missing)
            {
                if let Ok(ref hydrated_candidate) = hydrated {
                    let value = self.cache_value(hydrated_candidate);
                    self.cache_store().insert(key, value).await;
                }
                results[index] = Some(hydrated);
            }
        }

        results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| Err("Missing hydration result for candidate".to_string()))
            })
            .collect()
    }

    fn update(&self, candidate: &mut C, hydrated: C) {
        CachedHydrator::update(self, candidate, hydrated);
    }
}
