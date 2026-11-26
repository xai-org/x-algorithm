use prost::Message;
use std::collections::{BTreeSet, HashMap};

const KEY_PREFIX: &str = "phoenix-shadow-scores";
const ZSTD_COMPRESSION_LEVEL: i32 = 3;

pub type ShadowScoresByPost = HashMap<u64, HashMap<String, f64>>;

#[derive(Clone, PartialEq, prost::Message)]
struct StoredShadowScores {
    #[prost(string, repeated, tag = "1")]
    heads: Vec<String>,
    #[prost(uint64, repeated, tag = "2")]
    post_ids: Vec<u64>,
    #[prost(float, repeated, tag = "3")]
    values: Vec<f32>,
}

pub fn shadow_scores_key(prediction_request_id: u64) -> String {
    format!("{KEY_PREFIX}_{prediction_request_id}")
}

pub fn encode_shadow_scores(scores: &ShadowScoresByPost) -> Result<Vec<u8>, String> {
    let heads: Vec<String> = scores
        .values()
        .flat_map(|post_scores| post_scores.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let width = heads.len();
    let mut post_ids = Vec::with_capacity(scores.len());
    let mut values = vec![f32::NAN; scores.len() * width];
    {
        let head_index: HashMap<&str, usize> = heads
            .iter()
            .enumerate()
            .map(|(i, head)| (head.as_str(), i))
            .collect();
        for (row, (post_id, post_scores)) in scores.iter().enumerate() {
            post_ids.push(*post_id);
            for (head, value) in post_scores {
                values[row * width + head_index[head.as_str()]] = *value as f32;
            }
        }
    }
    let stored = StoredShadowScores {
        heads,
        post_ids,
        values,
    };
    zstd::encode_all(stored.encode_to_vec().as_slice(), ZSTD_COMPRESSION_LEVEL)
        .map_err(|err| err.to_string())
}

pub fn decode_shadow_scores(bytes: &[u8]) -> Result<ShadowScoresByPost, String> {
    let raw = zstd::decode_all(bytes).map_err(|err| err.to_string())?;
    let stored = StoredShadowScores::decode(raw.as_slice()).map_err(|err| err.to_string())?;
    let width = stored.heads.len();
    if width == 0 {
        return Ok(HashMap::new());
    }
    if stored.values.len() != stored.post_ids.len() * width {
        return Err(format!(
            "shadow scores shape mismatch: {} values for {} posts x {} heads",
            stored.values.len(),
            stored.post_ids.len(),
            width
        ));
    }
    Ok(stored
        .post_ids
        .iter()
        .zip(stored.values.chunks(width))
        .map(|(post_id, row)| {
            let post_scores = stored
                .heads
                .iter()
                .zip(row)
                .filter(|(_, value)| !value.is_nan())
                .map(|(head, value)| (head.clone(), f64::from(*value)))
                .collect();
            (*post_id, post_scores)
        })
        .collect())
}
