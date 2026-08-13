use crate::filter::Filter;
use crate::hydrator::Hydrator;
use crate::pipeline_summary::{self, StageStats};
use crate::query_hydrator::QueryHydrator;
use crate::scorer::Scorer;
use crate::selector::SelectResult;
use crate::selector::Selector;
use crate::side_effect::{SideEffect, SideEffectInput};
use crate::source::Source;
use crate::util;
use crate::SPAN_LEVEL;
use futures::future::join_all;
use std::any::type_name_of_val;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tonic::async_trait;
use tracing::{field::Empty, Span};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

const FINAL_RESULT_SIZE_SCOPE: [(&str, &str); 1] = [("requests", "result_size")];
const FINAL_RESULT_EMPTY_SCOPE: [(&str, &str); 1] = [("requests", "result_empty")];

#[derive(Copy, Clone, Debug)]
pub enum PipelineStage {
    QueryHydrator,
    DependentQueryHydrator,
    Source,
    Hydrator,
    PostSelectionHydrator,
    Filter,
    PostSelectionFilter,
    Scorer,
    Selector,
    SideEffect,
}

pub struct PipelineComponents {
    pub stage: PipelineStage,
    pub components: Vec<String>,
}

pub struct PipelineResult<Q, C> {
    pub retrieved_candidates: Vec<C>,
    pub filtered_candidates: Vec<C>,
    pub selected_candidates: Vec<C>,
    pub query: Arc<Q>,
}

pub struct PostSelectionResult<C> {
    pub selected: Vec<C>,
    pub filtered: Vec<C>,
    pub non_selected: Vec<C>,
}

impl<Q: Default, C> PipelineResult<Q, C> {
    pub fn empty() -> Self {
        Self {
            retrieved_candidates: vec![],
            filtered_candidates: vec![],
            selected_candidates: vec![],
            query: Arc::new(Q::default()),
        }
    }
}
pub trait PipelineQuery: Clone + Send + Sync + 'static {
    fn params(&self) -> &xai_feature_switches::Params;
    fn decider(&self) -> Option<&xai_decider::Decider>;
}

pub trait PipelineCandidate: Clone + Send + Sync + 'static {}
impl<T> PipelineCandidate for T where T: Clone + Send + Sync + 'static {}

#[async_trait]
pub trait CandidatePipeline<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>];
    fn dependent_query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>] {
        &[]
    }
    fn sources(&self) -> &[Box<dyn Source<Q, C>>];
    fn hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn scorers(&self) -> &[Box<dyn Scorer<Q, C>>];
    fn selector(&self) -> &dyn Selector<Q, C>;
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn post_selection_filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<Q, C>>>>;
    fn result_size(&self) -> usize;
    /// Whether `Selector::non_selected` is a ranked reserve that is safe to hydrate and serve.
    ///
    /// Custom selectors may use `non_selected` for side-effect accounting or placeholders, so
    /// backfill must be explicitly enabled by pipelines whose selector guarantees real candidates.
    fn enable_post_selection_backfill(&self) -> bool {
        false
    }
    fn finalize(&self, _query: &Q, _candidates: &mut Vec<C>) {}

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn execute(&self, query: Q) -> PipelineResult<Q, C> {
        xai_stats_receiver::with_scoped_metric_labels(
            vec![("pipeline".to_string(), self.name().to_string())],
            pipeline_summary::scope(self.execute_stages(query)),
        )
        .await
    }

    async fn execute_stages(&self, query: Q) -> PipelineResult<Q, C> {
        let start = Instant::now();

        let hydrated_query = self.hydrate_query(query).await;
        let hydrated_query = self.hydrate_dependent_query(hydrated_query).await;

        let candidates = self.fetch_candidates(&hydrated_query).await;

        let hydrated_candidates = self.hydrate(&hydrated_query, candidates).await;

        let (kept_candidates, mut filtered_candidates) =
            self.filter(&hydrated_query, hydrated_candidates.clone());

        let scored_candidates = self.score(&hydrated_query, kept_candidates).await;

        let SelectResult {
            selected: selected_candidates,
            non_selected: non_selected_candidates,
        } = self.select(&hydrated_query, scored_candidates);

        let PostSelectionResult {
            selected: mut final_candidates,
            filtered: post_selection_filtered_candidates,
            non_selected: non_selected_candidates,
        } = self
            .post_selection(
                &hydrated_query,
                selected_candidates,
                non_selected_candidates,
            )
            .await;
        filtered_candidates.extend(post_selection_filtered_candidates);

        self.finalize(&hydrated_query, &mut final_candidates);

        self.stat_result_size(&final_candidates);
        pipeline_summary::emit(self.name(), start, final_candidates.len());

        let arc_hydrated_query = Arc::new(hydrated_query);
        let input = Arc::new(SideEffectInput {
            query: arc_hydrated_query.clone(),
            selected_candidates: final_candidates.clone(),
            non_selected_candidates,
        });
        self.run_side_effects(input);

        PipelineResult {
            retrieved_candidates: hydrated_candidates,
            filtered_candidates,
            selected_candidates: final_candidates,
            query: arc_hydrated_query,
        }
    }

    async fn post_selection(
        &self,
        query: &Q,
        selected_candidates: Vec<C>,
        mut non_selected_candidates: Vec<C>,
    ) -> PostSelectionResult<C> {
        if self.enable_post_selection_backfill() {
            return self
                .post_selection_with_backfill(query, selected_candidates, non_selected_candidates)
                .await;
        }

        // Preserve the original single-pass contract for selectors whose `non_selected` output
        // is not a serveable reserve (for example, synthetic side-effect placeholders).
        let hydrated = self
            .hydrate_post_selection(query, selected_candidates)
            .await;
        let (mut selected, filtered) = self.filter_post_selection(query, hydrated);
        let truncated = selected.split_off(self.result_size().min(selected.len()));
        non_selected_candidates.extend(truncated);
        PostSelectionResult {
            selected,
            filtered,
            non_selected: non_selected_candidates,
        }
    }

    async fn post_selection_with_backfill(
        &self,
        query: &Q,
        selected_candidates: Vec<C>,
        non_selected_candidates: Vec<C>,
    ) -> PostSelectionResult<C> {
        let result_size = self.result_size();
        // Preserve the selector's reserve order while removing every attempted candidate from
        // the non-selected side-effect bucket exactly once.
        let mut reserve: VecDeque<C> = non_selected_candidates.into();
        let mut filtered = Vec::new();

        let hydrated = self
            .hydrate_post_selection(query, selected_candidates)
            .await;
        let (mut selected, removed) = self.filter_post_selection(query, hydrated);
        filtered.extend(removed);

        while selected.len() < result_size && !reserve.is_empty() {
            let needed = result_size - selected.len();
            let batch_size = needed.min(reserve.len());
            let batch: Vec<C> = reserve.drain(..batch_size).collect();
            let hydrated = self.hydrate_post_selection(query, batch).await;
            selected.extend(hydrated);
            // Some late filters evaluate relationships across the whole slate (for example,
            // conversation deduplication), so a reserve batch cannot be filtered in isolation.
            let (kept, removed) = self.filter_post_selection(query, selected);
            selected = kept;
            filtered.extend(removed);
        }

        let truncated = selected.split_off(result_size.min(selected.len()));
        let mut non_selected: Vec<C> = reserve.into_iter().collect();
        non_selected.extend(truncated);

        PostSelectionResult {
            selected,
            filtered,
            non_selected,
        }
    }

    fn components(&self) -> Vec<PipelineComponents> {
        fn stage<T: ?Sized>(
            stage: PipelineStage,
            items: &[Box<T>],
            name: impl Fn(&T) -> &str,
        ) -> PipelineComponents {
            PipelineComponents {
                stage,
                components: items
                    .iter()
                    .map(|item| name(item.as_ref()).to_string())
                    .collect(),
            }
        }

        vec![
            stage(PipelineStage::QueryHydrator, self.query_hydrators(), |h| {
                h.name()
            }),
            stage(
                PipelineStage::DependentQueryHydrator,
                self.dependent_query_hydrators(),
                |h| h.name(),
            ),
            stage(PipelineStage::Source, self.sources(), |s| s.name()),
            stage(PipelineStage::Hydrator, self.hydrators(), |h| h.name()),
            stage(PipelineStage::Filter, self.filters(), |f| f.name()),
            stage(PipelineStage::Scorer, self.scorers(), |s| s.name()),
            PipelineComponents {
                stage: PipelineStage::Selector,
                components: vec![self.selector().name().to_string()],
            },
            stage(
                PipelineStage::PostSelectionHydrator,
                self.post_selection_hydrators(),
                |h| h.name(),
            ),
            stage(
                PipelineStage::PostSelectionFilter,
                self.post_selection_filters(),
                |f| f.name(),
            ),
            stage(
                PipelineStage::SideEffect,
                self.side_effects().as_ref(),
                |s| s.name(),
            ),
        ]
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "query_hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
    ))]
    async fn hydrate_query(&self, query: Q) -> Q {
        let stats = StageStats::begin(PipelineStage::QueryHydrator);
        let all = self.query_hydrators();
        let hydrators: Vec<_> = all.iter().filter(|h| h.enable(&query)).collect();
        stats.record_components(all.len(), hydrators.len());
        let hydrate_futures = hydrators.iter().map(|h| h.run(&query));
        let results = join_all(hydrate_futures).await;

        let mut hydrated_query = query;
        for (hydrator, result) in hydrators.iter().zip(results) {
            if let Ok(hydrated) = result {
                hydrator.update(&mut hydrated_query, hydrated);
            }
        }
        stats.finish();
        hydrated_query
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "dependent_query_hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
    ))]
    async fn hydrate_dependent_query(&self, query: Q) -> Q {
        let all = self.dependent_query_hydrators();
        if all.is_empty() {
            return query;
        }
        let stats = StageStats::begin(PipelineStage::DependentQueryHydrator);
        let hydrators: Vec<_> = all.iter().filter(|h| h.enable(&query)).collect();
        stats.record_components(all.len(), hydrators.len());
        let hydrate_futures = hydrators.iter().map(|h| h.run(&query));
        let results = join_all(hydrate_futures).await;

        let mut hydrated_query = query;
        for (hydrator, result) in hydrators.iter().zip(results) {
            if let Ok(hydrated) = result {
                hydrator.update(&mut hydrated_query, hydrated);
            }
        }
        stats.finish();
        hydrated_query
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "sources", fields(
        total_count = Empty,
        enabled_count = Empty,
        candidate_count = Empty,
    ))]
    async fn fetch_candidates(&self, query: &Q) -> Vec<C> {
        let stats = StageStats::begin(PipelineStage::Source);
        let all = self.sources();
        let sources: Vec<_> = all.iter().filter(|s| s.enable(query)).collect();
        stats.record_components(all.len(), sources.len());
        let source_futures = sources.iter().map(|s| s.run(query));
        let results = join_all(source_futures).await;

        let mut collected = Vec::new();
        for mut candidates in results.into_iter().flatten() {
            collected.append(&mut candidates);
        }
        Span::current().record("candidate_count", collected.len());
        stats.finish_with_size(collected.len());
        collected
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
    ))]
    async fn hydrate(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        self.run_hydrators(query, candidates, self.hydrators(), PipelineStage::Hydrator)
            .await
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "post_selection_hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
    ))]
    async fn hydrate_post_selection(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        self.run_hydrators(
            query,
            candidates,
            self.post_selection_hydrators(),
            PipelineStage::PostSelectionHydrator,
        )
        .await
    }

    async fn run_hydrators(
        &self,
        query: &Q,
        mut candidates: Vec<C>,
        hydrators: &[Box<dyn Hydrator<Q, C>>],
        stage: PipelineStage,
    ) -> Vec<C> {
        let stats = StageStats::begin(stage);
        let enabled: Vec<_> = hydrators.iter().filter(|h| h.enable(query)).collect();
        stats.record_components(hydrators.len(), enabled.len());
        let hydrate_futures = enabled.iter().map(|h| h.run(query, &candidates));
        let results = join_all(hydrate_futures).await;
        for (hydrator, result) in enabled.iter().zip(results) {
            hydrator.update_all(&mut candidates, result);
        }
        stats.finish_with_size(candidates.len());
        candidates
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "filters", fields(
        total_count = Empty,
        enabled_count = Empty,
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn filter(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
        self.run_filters(query, candidates, self.filters(), PipelineStage::Filter)
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "post_selection_filters", fields(
        total_count = Empty,
        enabled_count = Empty,
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn filter_post_selection(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
        self.run_filters(
            query,
            candidates,
            self.post_selection_filters(),
            PipelineStage::PostSelectionFilter,
        )
    }

    fn run_filters(
        &self,
        query: &Q,
        mut candidates: Vec<C>,
        filters: &[Box<dyn Filter<Q, C>>],
        stage: PipelineStage,
    ) -> (Vec<C>, Vec<C>) {
        let stats = StageStats::begin(stage);
        let enabled: Vec<_> = filters.iter().filter(|f| f.enable(query)).collect();
        stats.record_components(filters.len(), enabled.len());
        let mut all_removed = Vec::new();
        let mut removed_per_filter: Vec<(String, usize)> = Vec::new();
        for filter in enabled {
            let result = filter.run(query, candidates);
            if !result.removed.is_empty() {
                removed_per_filter.push((filter.name().to_string(), result.removed.len()));
            }
            candidates = result.kept;
            all_removed.extend(result.removed);
        }
        stats.finish_filters(candidates.len(), all_removed.len(), removed_per_filter);
        (candidates, all_removed)
    }

    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "scorers", fields(
        total_count = Empty,
        enabled_count = Empty,
    ))]
    async fn score(&self, query: &Q, mut candidates: Vec<C>) -> Vec<C> {
        let stats = StageStats::begin(PipelineStage::Scorer);
        let all = self.scorers();
        let scorers: Vec<_> = all.iter().filter(|s| s.enable(query)).collect();
        stats.record_components(all.len(), scorers.len());
        for scorer in scorers {
            let scored = scorer.run(query, &candidates).await;
            scorer.update_all(&mut candidates, scored);
        }
        stats.finish_with_size(candidates.len());
        candidates
    }

    fn select(&self, query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        if self.selector().enable(query) {
            self.selector().run(query, candidates)
        } else {
            SelectResult {
                selected: candidates,
                non_selected: vec![],
            }
        }
    }

    fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
        let side_effects = self.side_effects();
        let pipeline_label = vec![("pipeline".to_string(), self.name().to_string())];
        tokio::spawn(xai_stats_receiver::with_scoped_metric_labels(
            pipeline_label,
            async move {
                let futures = side_effects
                    .iter()
                    .filter(|se| se.enable(input.query.clone()))
                    .map(|se| se.run(input.clone()));
                let _ = join_all(futures).await;
            },
        ));
    }

    fn stat_result_size(&self, final_candidates: &[C]) {
        if let Some(receiver) = global_stats_receiver() {
            let response_size = final_candidates.len();
            let metric_name = format!("{}.execute", self.name());
            receiver.observe(
                metric_name.as_str(),
                &FINAL_RESULT_SIZE_SCOPE,
                response_size as f64,
                HistogramBuckets::Bucket0To50,
            );
            if response_size == 0 {
                receiver.incr(metric_name.as_str(), &FINAL_RESULT_EMPTY_SCOPE, 1u64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterResult;
    use std::collections::HashSet;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct TestQuery {
        params: xai_feature_switches::Params,
    }

    impl PipelineQuery for TestQuery {
        fn params(&self) -> &xai_feature_switches::Params {
            &self.params
        }

        fn decider(&self) -> Option<&xai_decider::Decider> {
            None
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestCandidate {
        id: usize,
        group: usize,
        hydrated: bool,
    }

    impl TestCandidate {
        fn new(id: usize) -> Self {
            Self {
                id,
                group: id,
                hydrated: false,
            }
        }

        fn with_group(id: usize, group: usize) -> Self {
            Self {
                id,
                group,
                hydrated: false,
            }
        }
    }

    struct RecordingHydrator {
        batches: Arc<Mutex<Vec<Vec<usize>>>>,
    }

    #[async_trait]
    impl Hydrator<TestQuery, TestCandidate> for RecordingHydrator {
        async fn hydrate(
            &self,
            _query: &TestQuery,
            candidates: &[TestCandidate],
        ) -> Vec<Result<TestCandidate, String>> {
            self.batches
                .lock()
                .unwrap()
                .push(candidates.iter().map(|candidate| candidate.id).collect());
            candidates
                .iter()
                .cloned()
                .map(|mut candidate| {
                    candidate.hydrated = true;
                    Ok(candidate)
                })
                .collect()
        }

        fn update(&self, candidate: &mut TestCandidate, hydrated: TestCandidate) {
            candidate.hydrated = hydrated.hydrated;
        }
    }

    struct DropIdsFilter {
        ids: HashSet<usize>,
    }

    impl Filter<TestQuery, TestCandidate> for DropIdsFilter {
        fn filter(
            &self,
            _query: &TestQuery,
            candidates: Vec<TestCandidate>,
        ) -> FilterResult<TestCandidate> {
            let (removed, kept): (Vec<_>, Vec<_>) = candidates
                .into_iter()
                .partition(|candidate| self.ids.contains(&candidate.id));
            assert!(kept.iter().all(|candidate| candidate.hydrated));
            FilterResult { kept, removed }
        }
    }

    struct DedupGroupFilter;

    impl Filter<TestQuery, TestCandidate> for DedupGroupFilter {
        fn filter(
            &self,
            _query: &TestQuery,
            candidates: Vec<TestCandidate>,
        ) -> FilterResult<TestCandidate> {
            let mut seen = HashSet::new();
            let mut kept = Vec::new();
            let mut removed = Vec::new();
            for candidate in candidates {
                if seen.insert(candidate.group) {
                    kept.push(candidate);
                } else {
                    removed.push(candidate);
                }
            }
            FilterResult { kept, removed }
        }
    }

    struct ScoreSelector;

    impl Selector<TestQuery, TestCandidate> for ScoreSelector {
        fn score(&self, candidate: &TestCandidate) -> f64 {
            -(candidate.id as f64)
        }
    }

    struct PlaceholderSelector;

    impl Selector<TestQuery, TestCandidate> for PlaceholderSelector {
        fn select(
            &self,
            _query: &TestQuery,
            candidates: Vec<TestCandidate>,
        ) -> SelectResult<TestCandidate> {
            SelectResult {
                selected: candidates,
                non_selected: vec![TestCandidate::new(999), TestCandidate::new(1000)],
            }
        }

        fn score(&self, candidate: &TestCandidate) -> f64 {
            -(candidate.id as f64)
        }
    }

    struct TestPipeline {
        query_hydrators: Vec<Box<dyn QueryHydrator<TestQuery>>>,
        sources: Vec<Box<dyn Source<TestQuery, TestCandidate>>>,
        hydrators: Vec<Box<dyn Hydrator<TestQuery, TestCandidate>>>,
        filters: Vec<Box<dyn Filter<TestQuery, TestCandidate>>>,
        scorers: Vec<Box<dyn Scorer<TestQuery, TestCandidate>>>,
        selector: Box<dyn Selector<TestQuery, TestCandidate>>,
        post_selection_hydrators: Vec<Box<dyn Hydrator<TestQuery, TestCandidate>>>,
        post_selection_filters: Vec<Box<dyn Filter<TestQuery, TestCandidate>>>,
        side_effects: Arc<Vec<Box<dyn SideEffect<TestQuery, TestCandidate>>>>,
        result_size: usize,
        enable_backfill: bool,
    }

    impl CandidatePipeline<TestQuery, TestCandidate> for TestPipeline {
        fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<TestQuery>>] {
            &self.query_hydrators
        }

        fn sources(&self) -> &[Box<dyn Source<TestQuery, TestCandidate>>] {
            &self.sources
        }

        fn hydrators(&self) -> &[Box<dyn Hydrator<TestQuery, TestCandidate>>] {
            &self.hydrators
        }

        fn filters(&self) -> &[Box<dyn Filter<TestQuery, TestCandidate>>] {
            &self.filters
        }

        fn scorers(&self) -> &[Box<dyn Scorer<TestQuery, TestCandidate>>] {
            &self.scorers
        }

        fn selector(&self) -> &dyn Selector<TestQuery, TestCandidate> {
            self.selector.as_ref()
        }

        fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<TestQuery, TestCandidate>>] {
            &self.post_selection_hydrators
        }

        fn post_selection_filters(&self) -> &[Box<dyn Filter<TestQuery, TestCandidate>>] {
            &self.post_selection_filters
        }

        fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<TestQuery, TestCandidate>>>> {
            Arc::clone(&self.side_effects)
        }

        fn result_size(&self) -> usize {
            self.result_size
        }

        fn enable_post_selection_backfill(&self) -> bool {
            self.enable_backfill
        }
    }

    fn pipeline(
        result_size: usize,
        dropped_ids: impl IntoIterator<Item = usize>,
    ) -> (TestPipeline, Arc<Mutex<Vec<Vec<usize>>>>) {
        pipeline_with_filters(
            result_size,
            vec![Box::new(DropIdsFilter {
                ids: dropped_ids.into_iter().collect(),
            })],
        )
    }

    fn pipeline_with_filters(
        result_size: usize,
        post_selection_filters: Vec<Box<dyn Filter<TestQuery, TestCandidate>>>,
    ) -> (TestPipeline, Arc<Mutex<Vec<Vec<usize>>>>) {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let pipeline = TestPipeline {
            query_hydrators: vec![],
            sources: vec![],
            hydrators: vec![],
            filters: vec![],
            scorers: vec![],
            selector: Box::new(ScoreSelector),
            post_selection_hydrators: vec![Box::new(RecordingHydrator {
                batches: Arc::clone(&batches),
            })],
            post_selection_filters,
            side_effects: Arc::new(vec![]),
            result_size,
            enable_backfill: true,
        };
        (pipeline, batches)
    }

    fn candidates(range: std::ops::Range<usize>) -> Vec<TestCandidate> {
        range.map(TestCandidate::new).collect()
    }

    fn ids(candidates: &[TestCandidate]) -> Vec<usize> {
        candidates.iter().map(|candidate| candidate.id).collect()
    }

    #[tokio::test]
    async fn default_pipeline_never_hydrates_or_promotes_non_selected_placeholders() {
        let (mut pipeline, batches) = pipeline(3, [0]);
        pipeline.enable_backfill = false;
        pipeline.selector = Box::new(PlaceholderSelector);
        let selector_result = pipeline.select(&TestQuery::default(), candidates(0..3));
        let placeholders = selector_result.non_selected.clone();

        let result = pipeline
            .post_selection(
                &TestQuery::default(),
                selector_result.selected,
                selector_result.non_selected,
            )
            .await;

        assert_eq!(ids(&result.selected), vec![1, 2]);
        assert_eq!(ids(&result.filtered), vec![0]);
        assert_eq!(result.non_selected, placeholders);
        assert_eq!(*batches.lock().unwrap(), vec![vec![0, 1, 2]]);
        assert!(
            result
                .non_selected
                .iter()
                .all(|candidate| !candidate.hydrated)
        );
    }

    #[tokio::test]
    async fn backfills_after_more_than_selector_oversampling_is_filtered() {
        let (pipeline, batches) = pipeline(35, 0..20);
        let result = pipeline
            .post_selection_with_backfill(
                &TestQuery::default(),
                candidates(0..50),
                candidates(50..70),
            )
            .await;

        assert_eq!(ids(&result.selected), (20..55).collect::<Vec<_>>());
        assert_eq!(ids(&result.filtered), (0..20).collect::<Vec<_>>());
        assert_eq!(ids(&result.non_selected), (55..70).collect::<Vec<_>>());
        assert_eq!(
            *batches.lock().unwrap(),
            vec![(0..50).collect::<Vec<_>>(), (50..55).collect::<Vec<_>>()]
        );

        let all_ids: Vec<_> = result
            .selected
            .iter()
            .chain(&result.filtered)
            .chain(&result.non_selected)
            .map(|candidate| candidate.id)
            .collect();
        assert_eq!(all_ids.len(), 70);
        assert_eq!(all_ids.iter().copied().collect::<HashSet<_>>().len(), 70);
    }

    #[tokio::test]
    async fn filters_reserve_incrementally_without_reordering_survivors() {
        let dropped = (0..20).chain([50, 52, 54]);
        let (pipeline, batches) = pipeline(35, dropped);
        let result = pipeline
            .post_selection_with_backfill(
                &TestQuery::default(),
                candidates(0..50),
                candidates(50..70),
            )
            .await;

        let expected_selected: Vec<_> = (20..50).chain([51, 53, 55, 56, 57]).collect();
        let expected_filtered: Vec<_> = (0..20).chain([50, 52, 54]).collect();
        assert_eq!(ids(&result.selected), expected_selected);
        assert_eq!(ids(&result.filtered), expected_filtered);
        assert_eq!(ids(&result.non_selected), (58..70).collect::<Vec<_>>());
        assert_eq!(
            *batches.lock().unwrap(),
            vec![
                (0..50).collect::<Vec<_>>(),
                (50..55).collect::<Vec<_>>(),
                (55..58).collect::<Vec<_>>()
            ]
        );
    }

    #[tokio::test]
    async fn reserve_filters_see_already_selected_candidates() {
        let (pipeline, batches) = pipeline_with_filters(
            5,
            vec![
                Box::new(DropIdsFilter {
                    ids: HashSet::from([0]),
                }),
                Box::new(DedupGroupFilter),
            ],
        );
        let result = pipeline
            .post_selection_with_backfill(
                &TestQuery::default(),
                candidates(0..5),
                vec![
                    TestCandidate::with_group(5, 1),
                    TestCandidate::new(6),
                    TestCandidate::new(7),
                ],
            )
            .await;

        assert_eq!(ids(&result.selected), vec![1, 2, 3, 4, 6]);
        assert_eq!(ids(&result.filtered), vec![0, 5]);
        assert_eq!(ids(&result.non_selected), vec![7]);
        assert_eq!(
            *batches.lock().unwrap(),
            vec![vec![0, 1, 2, 3, 4], vec![5], vec![6]]
        );
    }
}
