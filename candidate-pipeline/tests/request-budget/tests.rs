use crate::candidate_pipeline::{CandidatePipeline, PipelineQuery};
use crate::filter::{Filter, FilterResult};
use crate::hydrator::Hydrator;
use crate::query_hydrator::QueryHydrator;
use crate::scorer::Scorer;
use crate::selector::Selector;
use crate::side_effect::{SideEffect, SideEffectInput};
use crate::source::Source;
use crate::{Decider, Params};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;
use tonic::async_trait;

#[derive(Clone, Default)]
struct Query {
    params: Params,
    limit: Option<usize>,
    hydrated: usize,
}
impl PipelineQuery for Query {
    fn params(&self) -> &Params {
        &self.params
    }
    fn decider(&self) -> Option<&Decider> {
        None
    }
}

#[derive(Default)]
struct Calls {
    sources: AtomicUsize,
    scorers: AtomicUsize,
    filters: AtomicUsize,
    finalize: AtomicUsize,
    captured: Mutex<Option<SideEffectInput<Query, u64>>>,
    notified: Notify,
}

struct TagQuery {
    limit: Option<usize>,
}
#[async_trait]
impl QueryHydrator<Query> for TagQuery {
    async fn hydrate(&self, query: &Query) -> Result<Query, String> {
        let mut hydrated = query.clone();
        hydrated.hydrated += 1;
        if self.limit.is_some() {
            hydrated.limit = self.limit;
        }
        Ok(hydrated)
    }
    fn update(&self, query: &mut Query, hydrated: Query) {
        *query = hydrated;
    }
}

struct NumberSource(Arc<Calls>);
#[async_trait]
impl Source<Query, u64> for NumberSource {
    async fn source(&self, _: &Query) -> Result<Vec<u64>, String> {
        self.0.sources.fetch_add(1, Ordering::SeqCst);
        Ok(vec![1, 2, 3, 4, 5, 6])
    }
}

struct RemoveNumber {
    number: u64,
    calls: Arc<Calls>,
}
impl Filter<Query, u64> for RemoveNumber {
    fn filter(&self, _: &Query, candidates: Vec<u64>) -> FilterResult<u64> {
        self.calls.filters.fetch_add(1, Ordering::SeqCst);
        let (removed, kept) = candidates.into_iter().partition(|c| *c == self.number);
        FilterResult { kept, removed }
    }
}

struct CountScorer(Arc<Calls>);
#[async_trait]
impl Scorer<Query, u64> for CountScorer {
    async fn score(&self, _: &Query, candidates: &[u64]) -> Vec<Result<u64, String>> {
        self.0.scorers.fetch_add(1, Ordering::SeqCst);
        candidates.iter().copied().map(Ok).collect()
    }
    fn update(&self, candidate: &mut u64, scored: u64) {
        *candidate = scored;
    }
}

struct OrderedSelector;
impl Selector<Query, u64> for OrderedSelector {
    fn score(&self, candidate: &u64) -> f64 {
        -(*candidate as f64)
    }
}

struct CaptureSideEffect(Arc<Calls>);
#[async_trait]
impl SideEffect<Query, u64> for CaptureSideEffect {
    async fn side_effect(&self, input: Arc<SideEffectInput<Query, u64>>) -> Result<(), String> {
        *self.0.captured.lock().unwrap() = Some((*input).clone());
        self.0.notified.notify_one();
        Ok(())
    }
}

struct BudgetPipeline {
    calls: Arc<Calls>,
    query_hydrators: Vec<Box<dyn QueryHydrator<Query>>>,
    dependent: Vec<Box<dyn QueryHydrator<Query>>>,
    sources: Vec<Box<dyn Source<Query, u64>>>,
    filters: Vec<Box<dyn Filter<Query, u64>>>,
    post_filters: Vec<Box<dyn Filter<Query, u64>>>,
    scorers: Vec<Box<dyn Scorer<Query, u64>>>,
    effects: Arc<Vec<Box<dyn SideEffect<Query, u64>>>>,
    append_final: bool,
    size: usize,
}
impl BudgetPipeline {
    fn new(hydrated_limit: Option<usize>, append_final: bool, size: usize) -> Self {
        let calls = Arc::new(Calls::default());
        Self {
            query_hydrators: vec![Box::new(TagQuery {
                limit: hydrated_limit,
            })],
            dependent: vec![Box::new(TagQuery { limit: None })],
            sources: vec![Box::new(NumberSource(calls.clone()))],
            filters: vec![Box::new(RemoveNumber {
                number: 2,
                calls: calls.clone(),
            })],
            post_filters: vec![Box::new(RemoveNumber {
                number: 4,
                calls: calls.clone(),
            })],
            scorers: vec![Box::new(CountScorer(calls.clone()))],
            effects: Arc::new(vec![Box::new(CaptureSideEffect(calls.clone()))]),
            calls,
            append_final,
            size,
        }
    }
    async fn captured(&self) -> SideEffectInput<Query, u64> {
        self.calls.notified.notified().await;
        self.calls.captured.lock().unwrap().clone().unwrap()
    }
}

#[async_trait]
impl CandidatePipeline<Query, u64> for BudgetPipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Query>>] {
        &self.query_hydrators
    }
    fn dependent_query_hydrators(&self) -> &[Box<dyn QueryHydrator<Query>>] {
        &self.dependent
    }
    fn sources(&self) -> &[Box<dyn Source<Query, u64>>] {
        &self.sources
    }
    fn hydrators(&self) -> &[Box<dyn Hydrator<Query, u64>>] {
        &[]
    }
    fn filters(&self) -> &[Box<dyn Filter<Query, u64>>] {
        &self.filters
    }
    fn scorers(&self) -> &[Box<dyn Scorer<Query, u64>>] {
        &self.scorers
    }
    fn selector(&self) -> &dyn Selector<Query, u64> {
        &OrderedSelector
    }
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<Query, u64>>] {
        &[]
    }
    fn post_selection_filters(&self) -> &[Box<dyn Filter<Query, u64>>] {
        &self.post_filters
    }
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<Query, u64>>>> {
        self.effects.clone()
    }
    fn result_size(&self) -> usize {
        self.size
    }
    fn request_result_limit(&self, query: &Query) -> Option<usize> {
        query.limit
    }
    fn finalize(&self, _: &Query, candidates: &mut Vec<u64>) {
        self.calls.finalize.fetch_add(1, Ordering::SeqCst);
        if self.append_final {
            candidates.push(99);
        }
    }
}

#[tokio::test]
async fn no_request_limit_keeps_existing_selection_and_side_effects() {
    let pipeline = BudgetPipeline::new(None, false, 3);
    let result = pipeline.execute(Query::default()).await;
    assert_eq!(result.selected_candidates, vec![1, 3, 5]);
    assert_eq!(result.filtered_candidates, vec![2, 4]);
    assert_eq!(result.retrieved_candidates, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(result.query.hydrated, 2);
    let captured = pipeline.captured().await;
    assert_eq!(captured.selected_candidates, result.selected_candidates);
    assert_eq!(captured.non_selected_candidates, vec![6]);
}

#[tokio::test]
async fn positive_limit_preserves_order_and_reports_only_served_candidates() {
    let pipeline = BudgetPipeline::new(None, false, 3);
    let result = pipeline
        .execute(Query {
            limit: Some(1),
            ..Default::default()
        })
        .await;
    assert_eq!(result.selected_candidates, vec![1]);
    assert_eq!(result.filtered_candidates, vec![2, 4]);
    let captured = pipeline.captured().await;
    assert_eq!(captured.selected_candidates, vec![1]);
    assert_eq!(captured.non_selected_candidates, vec![6, 3, 5]);
    assert_eq!(captured.query.limit, Some(1));
}

#[tokio::test]
async fn zero_budget_skips_sources_scoring_filters_finalize_and_serving_effects() {
    let pipeline = BudgetPipeline::new(None, true, 3);
    let result = pipeline
        .execute(Query {
            limit: Some(0),
            ..Default::default()
        })
        .await;
    tokio::task::yield_now().await;
    assert!(result.selected_candidates.is_empty());
    assert!(result.retrieved_candidates.is_empty());
    assert!(result.filtered_candidates.is_empty());
    assert_eq!(result.query.hydrated, 2);
    assert_eq!(result.query.limit, Some(0));
    assert_eq!(pipeline.calls.sources.load(Ordering::SeqCst), 0);
    assert_eq!(pipeline.calls.scorers.load(Ordering::SeqCst), 0);
    assert_eq!(pipeline.calls.filters.load(Ordering::SeqCst), 0);
    assert_eq!(pipeline.calls.finalize.load(Ordering::SeqCst), 0);
    assert!(pipeline.calls.captured.lock().unwrap().is_none());
}

#[tokio::test]
async fn budget_is_resolved_after_query_and_dependent_query_hydration() {
    let pipeline = BudgetPipeline::new(Some(0), false, 3);
    let result = pipeline.execute(Query::default()).await;
    assert_eq!(result.query.limit, Some(0));
    assert_eq!(result.query.hydrated, 2);
    assert_eq!(pipeline.calls.sources.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cap_is_enforced_after_finalize_can_add_candidates() {
    let pipeline = BudgetPipeline::new(None, true, 3);
    let result = pipeline
        .execute(Query {
            limit: Some(2),
            ..Default::default()
        })
        .await;
    assert_eq!(result.selected_candidates, vec![1, 3]);
    let captured = pipeline.captured().await;
    assert_eq!(captured.non_selected_candidates, vec![6, 5, 99]);
    assert_eq!(captured.selected_candidates, vec![1, 3]);
}

#[tokio::test]
async fn oversized_request_budget_cannot_expand_configured_pipeline_size() {
    let pipeline = BudgetPipeline::new(None, true, 3);
    let result = pipeline
        .execute(Query {
            limit: Some(usize::MAX),
            ..Default::default()
        })
        .await;
    assert_eq!(result.selected_candidates, vec![1, 3, 5]);
    assert_eq!(
        pipeline.captured().await.non_selected_candidates,
        vec![6, 99]
    );
}

#[tokio::test]
async fn existing_finalize_behavior_is_unchanged_without_an_opt_in_limit() {
    let pipeline = BudgetPipeline::new(None, true, 3);
    let result = pipeline.execute(Query::default()).await;
    assert_eq!(result.selected_candidates, vec![1, 3, 5, 99]);
    assert_eq!(
        pipeline.captured().await.selected_candidates,
        vec![1, 3, 5, 99]
    );
}

#[tokio::test]
async fn existing_zero_static_size_keeps_legacy_execution_without_opt_in() {
    let pipeline = BudgetPipeline::new(None, false, 0);
    let result = pipeline.execute(Query::default()).await;
    assert!(result.selected_candidates.is_empty());
    assert_eq!(pipeline.calls.sources.load(Ordering::SeqCst), 1);
    assert_eq!(pipeline.calls.scorers.load(Ordering::SeqCst), 1);
    assert_eq!(
        pipeline.captured().await.non_selected_candidates,
        vec![1, 3, 5, 6]
    );
}
