use std::cell::RefCell;
use std::fmt;
use std::future::Future;
use std::time::Instant;
use tracing::{info, Span};

use crate::candidate_pipeline::PipelineStage;

impl PipelineStage {
    fn summary_name(&self) -> &'static str {
        match self {
            PipelineStage::QueryHydrator => "query_hydrators",
            PipelineStage::DependentQueryHydrator => "dependent_query_hydrators",
            PipelineStage::Source => "sources",
            PipelineStage::Hydrator => "hydrators",
            PipelineStage::PostSelectionHydrator => "post_selection_hydrators",
            PipelineStage::Filter => "filters",
            PipelineStage::PostSelectionFilter => "post_selection_filters",
            PipelineStage::Scorer => "scorers",
            PipelineStage::Selector => "selector",
            PipelineStage::SideEffect => "side_effects",
        }
    }
}

tokio::task_local! {
    static ACTIVE: RefCell<Option<PipelineSummary>>;
}

pub async fn scope<F: Future>(
    pipeline: &'static str,
    type_name: &'static str,
    fut: F,
) -> (F::Output, Option<PipelineTrace>) {
    let nested = ACTIVE.try_with(|_| ()).is_ok();
    let (output, trace) = ACTIVE
        .scope(
            RefCell::new(Some(PipelineSummary::new(pipeline, type_name))),
            async {
                let output = fut.await;
                let summary = ACTIVE.with(|summary| summary.borrow_mut().take());
                (output, summary.map(PipelineSummary::finish))
            },
        )
        .await;
    if nested {
        with_active(|parent| {
            if let Some(stage) = parent.trace.stages.last_mut() {
                stage.nested.extend(trace.clone());
            }
        });
    }
    (output, trace)
}

pub fn emit(pipeline: &str, start: Instant, result_size: usize) {
    with_active(|summary| {
        info!(
            latency_ms = start.elapsed().as_millis() as u64,
            result_size, "{} Summary:{}", pipeline, summary.trace
        );
    });
}

pub struct StageStats {
    stage: PipelineStage,
    start: Instant,
}

impl StageStats {
    pub fn begin(stage: PipelineStage) -> Self {
        with_active(|summary| {
            summary.stage_mut(stage);
        });
        Self {
            stage,
            start: Instant::now(),
        }
    }

    pub fn record_components(&self, total: usize, enabled: usize) {
        let span = Span::current();
        span.record("total_count", total);
        span.record("enabled_count", enabled);
        with_active(|summary| {
            let stage = summary.stage_mut(self.stage);
            stage.total = total;
            stage.enabled = enabled;
        });
    }

    pub fn finish(self) {
        let latency_us = self.latency_us();
        let recorded = with_active(|summary| {
            summary.stage_mut(self.stage).latency_us = Some(latency_us);
        });
        if !recorded {
            info!("latency_ms={}", latency_us / 1000);
        }
    }

    pub fn finish_with_size(self, size: usize) {
        let latency_us = self.latency_us();
        let recorded = with_active(|summary| {
            let stage = summary.stage_mut(self.stage);
            stage.latency_us = Some(latency_us);
            stage.size = Some(size);
        });
        if !recorded {
            info!("latency_ms={} size={}", latency_us / 1000, size);
        }
    }

    pub fn finish_filters(self, kept: usize, removed: usize) {
        let total = kept + removed;
        let rate = if total > 0 {
            removed as f64 / total as f64
        } else {
            0.0
        };
        let span = Span::current();
        span.record("kept_count", kept);
        span.record("removed_count", removed);
        span.record("filter_rate", format!("{:.3}", rate).as_str());
        let latency_us = self.latency_us();
        let recorded = with_active(|summary| {
            let stage = summary.stage_mut(self.stage);
            stage.latency_us = Some(latency_us);
            stage.kept = Some(kept);
            stage.removed = Some(removed);
        });
        if !recorded {
            info!("kept {}, removed {}", kept, removed);
        }
    }

    fn latency_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
}

pub struct ComponentStats {
    stage: PipelineStage,
    component: ComponentTrace,
    start: Instant,
}

impl ComponentStats {
    pub fn begin(stage: PipelineStage, name: &'static str, type_name: &'static str) -> Self {
        Self {
            stage,
            component: ComponentTrace {
                name,
                type_name,
                ..Default::default()
            },
            start: Instant::now(),
        }
    }

    pub fn finish(self) {
        self.record();
    }

    pub fn finish_with_input(mut self, input: usize) {
        self.component.input_count = Some(input);
        self.record();
    }

    pub fn finish_source(mut self, fetched: usize) {
        self.component.fetched = Some(fetched);
        self.record();
    }

    pub fn finish_failed(mut self) {
        self.component.errors = Some(1);
        self.record();
    }

    pub fn finish_with_results<C>(mut self, results: &[Result<C, String>]) {
        self.component.input_count = Some(results.len());
        self.component.errors = Some(results.iter().filter(|result| result.is_err()).count());
        self.record();
    }

    pub fn finish_filter(mut self, kept: usize, removed: usize) {
        self.component.input_count = Some(kept + removed);
        self.component.kept = Some(kept);
        self.component.removed = Some(removed);
        self.record();
    }

    fn record(mut self) {
        self.component.latency_us = self.start.elapsed().as_micros() as u64;
        with_active(|summary| {
            self.component.offset_us = self.start.duration_since(summary.start).as_micros() as u64;
            summary
                .stage_mut(self.stage)
                .components
                .push(self.component);
        });
    }
}

struct PipelineSummary {
    start: Instant,
    trace: PipelineTrace,
}

impl PipelineSummary {
    fn new(pipeline: &'static str, type_name: &'static str) -> Self {
        Self {
            start: Instant::now(),
            trace: PipelineTrace {
                pipeline,
                type_name,
                ..Default::default()
            },
        }
    }

    fn finish(mut self) -> PipelineTrace {
        self.trace.latency_us = self.start.elapsed().as_micros() as u64;
        self.trace
    }

    fn stage_mut(&mut self, stage: PipelineStage) -> &mut StageTrace {
        let name = stage.summary_name();
        if let Some(idx) = self.trace.stages.iter().position(|s| s.name == name) {
            &mut self.trace.stages[idx]
        } else {
            self.trace.stages.push(StageTrace {
                name,
                offset_us: self.start.elapsed().as_micros() as u64,
                ..Default::default()
            });
            self.trace.stages.last_mut().unwrap()
        }
    }
}

#[derive(Clone, Default)]
pub struct PipelineTrace {
    pub pipeline: &'static str,
    pub type_name: &'static str,
    pub latency_us: u64,
    pub stages: Vec<StageTrace>,
}

#[derive(Clone, Default)]
pub struct StageTrace {
    pub name: &'static str,
    pub total: usize,
    pub enabled: usize,
    pub offset_us: u64,
    pub latency_us: Option<u64>,
    pub size: Option<usize>,
    pub kept: Option<usize>,
    pub removed: Option<usize>,
    pub components: Vec<ComponentTrace>,
    pub nested: Vec<PipelineTrace>,
}

#[derive(Clone, Default)]
pub struct ComponentTrace {
    pub name: &'static str,
    pub type_name: &'static str,
    pub offset_us: u64,
    pub latency_us: u64,
    pub input_count: Option<usize>,
    pub fetched: Option<usize>,
    pub kept: Option<usize>,
    pub removed: Option<usize>,
    pub errors: Option<usize>,
}

pub struct ComponentFailure {
    pub name: &'static str,
    pub errors: usize,
    pub total: usize,
}

impl PipelineTrace {
    pub fn failures(&self) -> Vec<ComponentFailure> {
        self.stages
            .iter()
            .flat_map(|stage| {
                let own = stage.components.iter().filter_map(ComponentTrace::failure);
                own.chain(stage.nested.iter().flat_map(PipelineTrace::failures))
            })
            .collect()
    }
}

impl ComponentTrace {
    fn failure(&self) -> Option<ComponentFailure> {
        let errors = self.errors.filter(|errors| *errors > 0)?;
        Some(ComponentFailure {
            name: self.name,
            errors,
            total: self.input_count.unwrap_or(1),
        })
    }
}

impl fmt::Display for PipelineTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for stage in &self.stages {
            write!(
                f,
                " {}{{total={} enabled={}",
                stage.name, stage.total, stage.enabled
            )?;
            if let Some(latency_us) = stage.latency_us {
                write!(f, " latency_ms={}", latency_us / 1000)?;
            }
            let fetched: Vec<_> = stage
                .components
                .iter()
                .filter_map(|c| c.fetched.map(|n| (c.name, n)))
                .collect();
            if !fetched.is_empty() {
                write!(f, " fetched=[{}]", Counts(&fetched))?;
            }
            if let Some(size) = stage.size {
                write!(f, " size={}", size)?;
            }
            if let (Some(kept), Some(removed)) = (stage.kept, stage.removed) {
                write!(f, " kept={} removed={}", kept, removed)?;
                let removed_per_filter: Vec<_> = stage
                    .components
                    .iter()
                    .filter_map(|c| c.removed.filter(|n| *n > 0).map(|n| (c.name, n)))
                    .collect();
                if !removed_per_filter.is_empty() {
                    write!(f, " removed_per_filter=[{}]", Counts(&removed_per_filter))?;
                }
            }
            write!(f, "}}")?;
        }
        Ok(())
    }
}

struct Counts<'a>(&'a [(&'a str, usize)]);

impl fmt::Display for Counts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (name, count)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}={}", name, count)?;
        }
        Ok(())
    }
}

fn with_active(f: impl FnOnce(&mut PipelineSummary)) -> bool {
    ACTIVE
        .try_with(|summary| {
            if let Some(summary) = summary.borrow_mut().as_mut() {
                f(summary);
            }
        })
        .is_ok()
}
