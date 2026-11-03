use xai_candidate_pipeline::pipeline_summary::{ComponentTrace, PipelineTrace, StageTrace};
use xai_home_mixer_proto as pb;

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableRanking;

const X_ALGORITHM_HOME_MIXER_URL: &str =
    "https://github.com/xai-org/x-algorithm/blob/main/home-mixer";

pub fn code_url(type_name: &str) -> String {
    let path = type_name.split('<').next().unwrap_or(type_name);
    let mut segments: Vec<&str> = path.split("::").collect();
    if segments.len() < 3 || segments.remove(0) != "xai_home_mixer" {
        return String::new();
    }
    segments.pop();
    format!("{X_ALGORITHM_HOME_MIXER_URL}/{}.rs", segments.join("/"))
}

pub fn pipeline_trace(trace: &PipelineTrace) -> pb::PipelineTrace {
    pb::PipelineTrace {
        pipeline: trace.pipeline.to_string(),
        latency_us: trace.latency_us,
        stages: trace.stages.iter().map(stage_trace).collect(),
        code_url: code_url(trace.type_name),
    }
}

fn stage_trace(stage: &StageTrace) -> pb::PipelineStageTrace {
    pb::PipelineStageTrace {
        name: stage.name.to_string(),
        total_count: stage.total as u32,
        enabled_count: stage.enabled as u32,
        offset_us: stage.offset_us,
        latency_us: stage.latency_us.unwrap_or(0),
        input_count: stage.kept.zip(stage.removed).map(|(k, r)| (k + r) as u32),
        candidate_count: stage.size.map(|n| n as u32),
        kept_count: stage.kept.map(|n| n as u32),
        removed_count: stage.removed.map(|n| n as u32),
        components: stage.components.iter().map(component_trace).collect(),
        nested: stage.nested.iter().map(pipeline_trace).collect(),
    }
}

fn component_trace(component: &ComponentTrace) -> pb::PipelineComponentTrace {
    pb::PipelineComponentTrace {
        name: component.name.to_string(),
        offset_us: component.offset_us,
        latency_us: component.latency_us,
        input_count: component.input_count.map(|n| n as u32),
        candidate_count: component.fetched.map(|n| n as u32),
        kept_count: component.kept.map(|n| n as u32),
        removed_count: component.removed.map(|n| n as u32),
        filter_rate: component
            .kept
            .zip(component.removed)
            .map(|(kept, removed)| {
                let total = kept + removed;
                if total > 0 {
                    removed as f32 / total as f32
                } else {
                    0.0
                }
            }),
        code_url: code_url(component.type_name),
    }
}

pub fn post_scores(query: &ScoredPostsQuery, candidate: &PostCandidate) -> pb::UnderTheHoodScores {
    let value_model = if query.params.get(EnableRanking) {
        "vm_ranker"
    } else {
        "weighted"
    };
    pb::UnderTheHoodScores {
        weighted_score: candidate.weighted_score,
        value_model: value_model.to_string(),
        top_heads: candidate
            .top_heads
            .iter()
            .map(|&(ref head, contribution, share)| pb::HeadContribution {
                head: head.clone(),
                contribution,
                share,
            })
            .collect(),
    }
}
