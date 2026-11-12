use crate::SPAN_LEVEL;
use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use std::any::{Any, type_name_of_val};
use std::borrow::Cow;
use std::hash::Hash;
use tonic::async_trait;
use tracing::warn;
use xai_stats_receiver::global_stats_receiver;

pub struct HydratorOutput(Box<dyn Any + Send>);

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
        let results = checked_len(
            self.hydrate_for_stage(query, candidates, stage).await,
            candidates.len(),
        );
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

    async fn fetch_output(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> HydratorOutput {
        HydratorOutput(Box::new(self.run(query, candidates, stage).await))
    }

    fn apply_output(&self, _query: &Q, candidates: &mut [C], output: HydratorOutput) {
        if let Ok(hydrated) = output.0.downcast::<Vec<Result<C, String>>>() {
            self.update_all(candidates, *hydrated);
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}

fn checked_len<C: Clone>(
    hydrated: Vec<Result<C, String>>,
    expected: usize,
) -> Vec<Result<C, String>> {
    if hydrated.len() == expected {
        return hydrated;
    }
    let message = format!(
        "Hydrator length_mismatch expected={} got={}",
        expected,
        hydrated.len()
    );
    warn!(
        "Skipped: length_mismatch expected={} got={}",
        expected,
        hydrated.len()
    );
    vec![Err(message); expected]
}

#[async_trait]
pub trait TwoPhaseHydrator<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    type Fetched: Send + 'static;

    fn enable(&self, _query: &Q) -> bool {
        true
    }

    async fn fetch(&self, query: &Q, candidates: &[C]) -> Result<Self::Fetched, String>;

            fn apply(&self, query: &Q, candidates: &mut [C], fetched: Self::Fetched);

            fn merge(&self, candidate: &mut C, hydrated: C);

    #[xai_stats_macro::receive_stats(name = "run", latency = Bucket50To500)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "hydrator", fields(name = self.name()))]
    async fn run_fetch(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Result<Self::Fetched, String> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let fetched = self.fetch(query, candidates).await;
        let errors = if fetched.is_ok() { 0 } else { candidates.len() };
        stats.finish_with_errors(candidates.len(), errors);
        fetched
    }

        async fn legacy_hydrate(
        &self,
        query: &Q,
        candidates: &[C],
        _stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        match self.fetch(query, candidates).await {
            Ok(fetched) => {
                let mut hydrated = candidates.to_vec();
                self.apply(query, &mut hydrated, fetched);
                hydrated.into_iter().map(Ok).collect()
            }
            Err(err) => vec![Err(err); candidates.len()],
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}

#[async_trait]
impl<Q, C, T> Hydrator<Q, C> for T
where
    Q: PipelineQuery,
    C: PipelineCandidate,
    T: TwoPhaseHydrator<Q, C>,
{
    fn enable(&self, query: &Q) -> bool {
        TwoPhaseHydrator::enable(self, query)
    }

    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>> {
        self.legacy_hydrate(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn hydrate_for_stage(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        self.legacy_hydrate(query, candidates, stage).await
    }

    fn update(&self, candidate: &mut C, hydrated: C) {
        TwoPhaseHydrator::merge(self, candidate, hydrated);
    }

    async fn fetch_output(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> HydratorOutput {
        HydratorOutput(Box::new(self.run_fetch(query, candidates, stage).await))
    }

    fn apply_output(&self, query: &Q, candidates: &mut [C], output: HydratorOutput) {
        if let Ok(fetched) = output.0.downcast::<Result<T::Fetched, String>>()
            && let Ok(fetched) = *fetched
        {
            TwoPhaseHydrator::apply(self, query, candidates, fetched);
        }
    }

    fn name(&self) -> &'static str {
        TwoPhaseHydrator::name(self)
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

    #[xai_stats_macro::receive_stats(name = "run", latency = Bucket50To500, size = Bucket500To2500)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "hydrator", fields(name = self.name()))]
    async fn run_cached(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let results = checked_len(
            self.hydrate_cached(query, candidates, stage).await,
            candidates.len(),
        );
        stats.finish_with_results(&results);
        results
    }

    async fn hydrate_cached(
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
impl<Q, C, T> TwoPhaseHydrator<Q, C> for T
where
    Q: PipelineQuery,
    C: PipelineCandidate,
    T: CachedHydrator<Q, C>,
{
    type Fetched = Vec<Result<C, String>>;

    fn enable(&self, query: &Q) -> bool {
        CachedHydrator::enable(self, query)
    }

    async fn fetch(&self, query: &Q, candidates: &[C]) -> Result<Self::Fetched, String> {
        Ok(self
            .hydrate_cached(query, candidates, PipelineStage::Hydrator)
            .await)
    }

    fn apply(&self, _query: &Q, candidates: &mut [C], fetched: Self::Fetched) {
        for (candidate, hydrated) in candidates.iter_mut().zip(fetched) {
            if let Ok(hydrated) = hydrated {
                CachedHydrator::update(self, candidate, hydrated);
            }
        }
    }

    async fn run_fetch(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Result<Self::Fetched, String> {
        Ok(self.run_cached(query, candidates, stage).await)
    }

    async fn legacy_hydrate(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        self.hydrate_cached(query, candidates, stage).await
    }

    fn merge(&self, candidate: &mut C, hydrated: C) {
        CachedHydrator::update(self, candidate, hydrated);
    }

    fn name(&self) -> &'static str {
        CachedHydrator::name(self)
    }
}
