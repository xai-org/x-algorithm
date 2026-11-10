use xai_candidate_pipeline::component_library::clients::{JetfuelClient, JetfuelView};
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

pub const TRACE_FRAME_ROUTE: &str = "/cards/under_the_hood/trace";

pub async fn trace_frame(
    jetfuel_client: &dyn JetfuelClient,
    query: &ScoredPostsQuery,
    trace: &pb::PipelineTrace,
) -> Option<pb::Frame> {
    let view = JetfuelView {
        client_app_id: i64::from(query.client_app_id),
        client_version: query.client_version.clone(),
        locale: query.language_code.clone(),
        country_code: query.country_code.clone(),
        user_id: (query.user_id != 0).then_some(query.user_id),
        ..Default::default()
    };
    let body = trace_json(trace).to_string();
    let payload = match jetfuel_client
        .post_frame(TRACE_FRAME_ROUTE, &body, &view)
        .await
    {
        Ok(Some(payload)) => payload,
        Ok(None) => return None,
        Err(e) => {
            tracing::warn!(
                request_id = query.request_id,
                error = %e,
                "under-the-hood trace frame fetch failed"
            );
            return None;
        }
    };
    Some(pb::Frame {
        payload,
        height: 0,
        is_dismissible: false,
        placement_id: 0,
        route: TRACE_FRAME_ROUTE.to_string(),
        occurrence: 0,
        fetch_route: TRACE_FRAME_ROUTE.to_string(),
    })
}

fn trace_json(trace: &pb::PipelineTrace) -> serde_json::Value {
    serde_json::json!({
        "pipeline": trace.pipeline,
        "latency_us": trace.latency_us,
        "code_url": trace.code_url,
        "stages": trace.stages.iter().map(|stage| serde_json::json!({
            "name": stage.name,
            "total_count": stage.total_count,
            "enabled_count": stage.enabled_count,
            "offset_us": stage.offset_us,
            "latency_us": stage.latency_us,
            "input_count": stage.input_count,
            "candidate_count": stage.candidate_count,
            "kept_count": stage.kept_count,
            "removed_count": stage.removed_count,
            "components": stage.components.iter().map(|component| serde_json::json!({
                "name": component.name,
                "offset_us": component.offset_us,
                "latency_us": component.latency_us,
                "input_count": component.input_count,
                "candidate_count": component.candidate_count,
                "kept_count": component.kept_count,
                "removed_count": component.removed_count,
                "filter_rate": component.filter_rate,
                "code_url": component.code_url,
            })).collect::<Vec<_>>(),
            "nested": stage.nested.iter().map(trace_json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}
