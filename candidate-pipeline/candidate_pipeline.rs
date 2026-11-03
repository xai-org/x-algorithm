use crate::filter::Filter;
use crate::hydrator::Hydrator;
use crate::pipeline_summary::{self, PipelineTrace, StageStats};
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
use std::sync::Arc;
use std::time::Instant;
use tonic::async_trait;
use tracing::{field::Empty, Span};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

const FINAL_RESULT_SIZE_SCOPE: [(&str, &str); 1] = [("requests", "result_size")];
const FINAL_RESULT_EMPTY_SCOPE: [(&str, &str); 1] = [("requests", "result_empty")];

#[derive(Copy, Clone, Debug, strum::IntoStaticStr)]
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

impl PipelineStage {
    pub fn stat_labels(
        self,
        component: &'static str,
        requests: &'static str,
    ) -> [(&'static str, &'static str); 3] {
        [
            ("requests", requests),
            ("stage", self.into()),
            ("component", component),
        ]
    }
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
    fn result_size_for(&self, _query: &Q) -> usize {
        self.result_size()
    }
    fn finalize(&self, _query: &Q, _candidates: &mut Vec<C>) {}

    async fn execute(&self, query: Q) -> PipelineResult<Q, C> {
        self.execute_traced(query).await.0
    }

    async fn execute_traced(&self, query: Q) -> (PipelineResult<Q, C>, Option<PipelineTrace>) {
        xai_stats_receiver::with_scoped_metric_labels(
            vec![("pipeline".to_string(), self.name().to_string())],
            pipeline_summary::scope(
                self.name(),
                type_name_of_val(self),
                self.execute_instrumented(query),
            ),
        )
        .await
    }

    #[xai_stats_macro::receive_stats(name = "execute", latency = Bucket500To2500)]
    async fn execute_instrumented(&self, query: Q) -> PipelineResult<Q, C> {
        self.execute_stages(query).await
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
            non_selected: mut non_selected_candidates,
        } = self.select(&hydrated_query, scored_candidates);

        let post_selection_hydrated_candidates = self
            .hydrate_post_selection(&hydrated_query, selected_candidates)
            .await;

        let (mut final_candidates, post_selection_filtered_candidates) =
            self.filter_post_selection(&hydrated_query, post_selection_hydrated_candidates);
        filtered_candidates.extend(post_selection_filtered_candidates);

        let truncated_candidates = final_candidates.split_off(
            self.result_size_for(&hydrated_query)
                .min(final_candidates.len()),
        );
        non_selected_candidates.extend(truncated_candidates);

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
        let hydrate_futures = hydrators
            .iter()
            .map(|h| h.run(&query, PipelineStage::QueryHydrator));
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
        let hydrate_futures = hydrators
            .iter()
            .map(|h| h.run(&query, PipelineStage::DependentQueryHydrator));
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
        let source_futures = sources.iter().map(|s| s.run(query, PipelineStage::Source));
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
        let hydrate_futures = enabled.iter().map(|h| h.run(query, &candidates, stage));
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
        for filter in enabled {
            let result = filter.run(query, candidates, stage);
            candidates = result.kept;
            all_removed.extend(result.removed);
        }
        stats.finish_filters(candidates.len(), all_removed.len());
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
            let scored = scorer.run(query, &candidates, PipelineStage::Scorer).await;
            scorer.update_all(&mut candidates, scored);
        }
        stats.finish_with_size(candidates.len());
        candidates
    }

    fn select(&self, query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        let stats = StageStats::begin(PipelineStage::Selector);
        let enabled = self.selector().enable(query);
        stats.record_components(1, enabled as usize);
        let result = if enabled {
            self.selector()
                .run(query, candidates, PipelineStage::Selector)
        } else {
            SelectResult {
                selected: candidates,
                non_selected: vec![],
            }
        };
        stats.finish_with_size(result.selected.len());
        result
    }

    fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
        let side_effects = self.side_effects();
        tokio::spawn(xai_stats_receiver::with_scoped_metric_labels(
            vec![("pipeline".to_string(), self.name().to_string())],
            async move {
                let futures = side_effects
                    .iter()
                    .filter(|se| se.enable(input.query.clone()))
                    .map(|se| se.run(input.clone(), PipelineStage::SideEffect));
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
