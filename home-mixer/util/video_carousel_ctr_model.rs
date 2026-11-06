use crate::clients::vm_ranker_client::VMRankerClient;
use serde::Deserialize;
use tracing::warn;
use xai_stats_receiver::global_stats_receiver;

const EMBEDDING_DIM: usize = 1024;
const FETCH_METRIC: &str = "VideoCarouselCtrModel.fetch";
const REQUEST_METRIC: &str = "VideoCarouselCtrModel.request";

#[derive(Debug, Deserialize)]
pub struct CtrModel {
    pub embedding: String,
    intercept: f64,
    video_open_weight: f64,
    video_open_logit_min: f64,
    embedding_weights: Vec<f32>,
}

impl CtrModel {
    pub fn score(&self, video_open: f64, embedding: Option<&[f32]>) -> f64 {
        let video_open = video_open.clamp(1e-6, 1.0 - 1e-6);
        let video_open_logit = (video_open / (1.0 - video_open))
            .ln()
            .max(self.video_open_logit_min);
        let embedding_term: f64 = embedding.map_or(0.0, |embedding| {
            embedding
                .iter()
                .zip(&self.embedding_weights)
                .map(|(x, w)| f64::from(x * w))
                .sum()
        });
        self.intercept + self.video_open_weight * video_open_logit + embedding_term
    }
}

pub async fn fetch(client: &dyn VMRankerClient, name: &str) -> Option<CtrModel> {
    let fetched = match client.get_carousel_model(name).await {
        Ok(bytes) => parse(&bytes).map_err(|e| ("invalid", e)),
        Err(status) => Err(("rpc_error", status.message().to_string())),
    };
    match fetched {
        Ok(model) => {
            record(FETCH_METRIC, "ok");
            Some(model)
        }
        Err((outcome, e)) => {
            record(FETCH_METRIC, outcome);
            warn!("video carousel CTR model {name}: {e}");
            None
        }
    }
}

pub fn record_request(outcome: &str) {
    record(REQUEST_METRIC, outcome);
}

fn record(metric: &str, outcome: &str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(metric, &[("outcome", outcome)], 1);
    }
}

fn parse(bytes: &[u8]) -> Result<CtrModel, String> {
    let model: CtrModel = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if model.embedding_weights.len() != EMBEDDING_DIM {
        return Err(format!(
            "expected {EMBEDDING_DIM} embedding weights, got {}",
            model.embedding_weights.len()
        ));
    }
    Ok(model)
}

#[cfg(test)]
pub(crate) fn test_model_json() -> String {
    serde_json::json!({
        "embedding": "v8_search_1",
        "intercept": -1.0,
        "video_open_weight": 0.5,
        "video_open_logit_min": -8.0,
        "embedding_weights": vec![0.5f32; EMBEDDING_DIM],
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_model_file_and_rejects_bad_ones() {
        let parsed = parse(test_model_json().as_bytes()).unwrap();
        let embedding = vec![0.0f32; EMBEDDING_DIM];
        assert!((parsed.score(0.5, Some(&embedding)) + 1.0).abs() < 1e-9);
        assert!(parse(
            test_model_json()
                .replace("\"embedding\":\"v8_search_1\",", "")
                .as_bytes()
        )
        .is_err());
        assert!(parse(b"{").is_err());
    }
}
