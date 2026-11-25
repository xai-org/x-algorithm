// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::error::Error;
use crate::features::{FEATURE_DIM, FEATURE_VERSION, RequestContext};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"UDB1";
const HEADER_LEN: usize = 8;
pub const N_SURFACE: usize = 3;
pub const N_HOUR: usize = 24;
pub const N_DOW: usize = 7;
pub const N_APP: usize = 32;
const SURFACES: [&str; N_SURFACE] = ["other", "home", "immersive"];

#[derive(Debug, Deserialize, Serialize)]
struct Metadata {
    feature_version: String,
    feature_dim: usize,
    head_names: Vec<String>,
    n_layers: usize,
    emb_dim: usize,
    #[serde(default)]
    surfaces: Vec<String>,
    #[serde(default)]
    trained_at_ms: i64,
    #[serde(default)]
    train_records: i64,
    tensors: IndexMap<String, Vec<usize>>,
}

struct Tensors {
    by_name: IndexMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tensors {
    fn take(&mut self, name: &str, expected: &[usize]) -> Result<Vec<f32>, Error> {
        let (shape, data) = self
            .by_name
            .shift_remove(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if shape != expected {
            return Err(Error::TensorShape {
                name: name.to_string(),
                expected: expected.to_vec(),
                actual: shape,
            });
        }
        Ok(data)
    }
}

struct Layer {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Layer {
    fn apply(&self, x: &[f32], relu: bool) -> Vec<f32> {
        (0..self.out_dim)
            .map(|j| {
                let mut acc = self.b[j] as f64;
                for (i, &xi) in x.iter().enumerate() {
                    acc += xi as f64 * self.w[i * self.out_dim + j] as f64;
                }
                let y = acc as f32;
                if relu { y.max(0.0) } else { y }
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub user_logits: Vec<f32>,
    pub user_log_dwell: f32,
    pub global_logits: Vec<f32>,
    pub global_log_dwell: f32,
}

pub struct UserDebiasModel {
    head_names: Vec<String>,
    feature_dim: usize,
    emb_dim: usize,
    trained_at_ms: i64,
    train_records: i64,
    feat_mean: Vec<f32>,
    feat_std: Vec<f32>,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    emb_surface: Vec<f32>,
    emb_hour: Vec<f32>,
    emb_dow: Vec<f32>,
    emb_app: Vec<f32>,
    hidden: Vec<Layer>,
    output: Layer,
    global_logit_by_surface: Vec<f32>,
    global_log_dwell_by_surface: Vec<f32>,
}

impl UserDebiasModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Self::load(bytes, FEATURE_DIM)
    }

    fn load(bytes: &[u8], expected_feature_dim: usize) -> Result<Self, Error> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::Truncated);
        }
        if &bytes[..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        let json_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let json_end = HEADER_LEN + json_len;
        if bytes.len() < json_end {
            return Err(Error::Truncated);
        }
        let meta: Metadata = serde_json::from_slice(&bytes[HEADER_LEN..json_end])?;
        if meta.feature_version != FEATURE_VERSION {
            return Err(Error::FeatureVersion(meta.feature_version));
        }
        if meta.feature_dim != expected_feature_dim {
            return Err(Error::FeatureDim(meta.feature_dim));
        }

        let payload = &bytes[json_end..];
        let total_floats: usize = meta
            .tensors
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        if payload.len() != total_floats * 4 {
            return Err(Error::PayloadSize {
                expected: total_floats * 4,
                actual: payload.len(),
            });
        }
        let mut offset = 0;
        let mut by_name = IndexMap::with_capacity(meta.tensors.len());
        for (name, shape) in &meta.tensors {
            let count = shape.iter().product::<usize>();
            let data: Vec<f32> = payload[offset..offset + count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            offset += count * 4;
            by_name.insert(name.clone(), (shape.clone(), data));
        }
        let mut tensors = Tensors { by_name };

        let d = meta.feature_dim;
        let e = meta.emb_dim;
        let n_heads = meta.head_names.len();
        if n_heads == 0 {
            return Err(Error::Invalid("head_names is empty".to_string()));
        }
        let feat_mean = tensors.take("feat_mean", &[d])?;
        let feat_std = tensors.take("feat_std", &[d])?;
        if !feat_std.iter().all(|&s| s > 0.0 && s.is_finite()) {
            return Err(Error::Invalid(
                "feat_std must be finite and positive".to_string(),
            ));
        }
        let ln_g = tensors.take("ln_g", &[d])?;
        let ln_b = tensors.take("ln_b", &[d])?;
        let emb_surface = tensors.take("emb_surface", &[N_SURFACE, e])?;
        let emb_hour = tensors.take("emb_hour", &[N_HOUR, e])?;
        let emb_dow = tensors.take("emb_dow", &[N_DOW, e])?;
        let emb_app = tensors.take("emb_app", &[N_APP, e])?;

        let mut in_dim = d + 4 * e;
        let mut hidden = Vec::with_capacity(meta.n_layers);
        for li in 1..=meta.n_layers {
            let w_name = format!("w{li}");
            let out_dim = tensors
                .by_name
                .get(&w_name)
                .and_then(|(shape, _)| shape.get(1).copied())
                .ok_or_else(|| Error::MissingTensor(w_name.clone()))?;
            let w = tensors.take(&w_name, &[in_dim, out_dim])?;
            let b = tensors.take(&format!("b{li}"), &[out_dim])?;
            hidden.push(Layer {
                w,
                b,
                in_dim,
                out_dim,
            });
            in_dim = out_dim;
        }
        let extra_layer = format!("w{}", meta.n_layers + 1);
        if tensors.by_name.contains_key(&extra_layer) {
            return Err(Error::Invalid(format!(
                "tensor {extra_layer} present but n_layers = {}",
                meta.n_layers
            )));
        }
        let output = Layer {
            w: tensors.take("w_out", &[in_dim, n_heads + 1])?,
            b: tensors.take("b_out", &[n_heads + 1])?,
            in_dim,
            out_dim: n_heads + 1,
        };
        let global_logit_by_surface =
            tensors.take("global_logit_by_surface", &[N_SURFACE, n_heads])?;
        let global_log_dwell_by_surface =
            tensors.take("global_log_dwell_by_surface", &[N_SURFACE])?;

        Ok(Self {
            head_names: meta.head_names,
            feature_dim: d,
            emb_dim: e,
            trained_at_ms: meta.trained_at_ms,
            train_records: meta.train_records,
            feat_mean,
            feat_std,
            ln_g,
            ln_b,
            emb_surface,
            emb_hour,
            emb_dow,
            emb_app,
            hidden,
            output,
            global_logit_by_surface,
            global_log_dwell_by_surface,
        })
    }

    pub fn head_names(&self) -> &[String] {
        &self.head_names
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn trained_at_ms(&self) -> i64 {
        self.trained_at_ms
    }

    pub fn train_records(&self) -> i64 {
        self.train_records
    }

    pub fn predict(&self, features: &[f32], ctx: &RequestContext) -> Prediction {
        assert_eq!(
            features.len(),
            self.feature_dim,
            "feature vector length does not match model feature_dim"
        );
        let surface = ctx.surface_id.min(N_SURFACE - 1);
        let hour = ctx.hour.min(N_HOUR - 1);
        let dow = ctx.dow.min(N_DOW - 1);
        let app = ctx.app.min(N_APP - 1);
        let e = self.emb_dim;

        let mut x: Vec<f32> = Vec::with_capacity(self.feature_dim + 4 * e);
        for (i, &u) in features.iter().enumerate() {
            let z = (u - self.feat_mean[i]) / self.feat_std[i];
            x.push(z * self.ln_g[i] + self.ln_b[i]);
        }
        x.extend_from_slice(&self.emb_surface[surface * e..(surface + 1) * e]);
        x.extend_from_slice(&self.emb_hour[hour * e..(hour + 1) * e]);
        x.extend_from_slice(&self.emb_dow[dow * e..(dow + 1) * e]);
        x.extend_from_slice(&self.emb_app[app * e..(app + 1) * e]);
        debug_assert_eq!(
            x.len(),
            self.hidden.first().map_or(self.output.in_dim, |l| l.in_dim)
        );

        for layer in &self.hidden {
            x = layer.apply(&x, true);
        }
        let mut out = self.output.apply(&x, false);
        let user_log_dwell = out.pop().unwrap_or(0.0);
        let n_heads = self.head_names.len();
        Prediction {
            user_logits: out,
            user_log_dwell,
            global_logits: self.global_logit_by_surface[surface * n_heads..(surface + 1) * n_heads]
                .to_vec(),
            global_log_dwell: self.global_log_dwell_by_surface[surface],
        }
    }
}

pub struct ArtifactTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn write_artifact(
    head_names: &[String],
    trained_at_ms: i64,
    train_records: i64,
    tensors: &[ArtifactTensor],
) -> Result<Vec<u8>, Error> {
    write_artifact_checked(
        head_names,
        trained_at_ms,
        train_records,
        tensors,
        FEATURE_DIM,
    )
}

fn write_artifact_checked(
    head_names: &[String],
    trained_at_ms: i64,
    train_records: i64,
    tensors: &[ArtifactTensor],
    expected_feature_dim: usize,
) -> Result<Vec<u8>, Error> {
    let shape_of = |name: &str| {
        tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    };
    let feature_dim = shape_of("feat_mean")?.first().copied().unwrap_or(0);
    let emb_dim = shape_of("emb_surface")?.get(1).copied().unwrap_or(0);
    let n_layers = (1..)
        .take_while(|li| tensors.iter().any(|t| t.name == format!("w{li}")))
        .count();
    let mut shapes = IndexMap::with_capacity(tensors.len());
    for t in tensors {
        if t.data.len() != t.shape.iter().product::<usize>() {
            return Err(Error::Invalid(format!(
                "tensor {} has {} values for shape {:?}",
                t.name,
                t.data.len(),
                t.shape
            )));
        }
        if shapes.insert(t.name.clone(), t.shape.clone()).is_some() {
            return Err(Error::Invalid(format!("duplicate tensor {}", t.name)));
        }
    }
    let meta = Metadata {
        feature_version: FEATURE_VERSION.to_string(),
        feature_dim,
        head_names: head_names.to_vec(),
        n_layers,
        emb_dim,
        surfaces: SURFACES.map(String::from).to_vec(),
        trained_at_ms,
        train_records,
        tensors: shapes,
    };
    let json = serde_json::to_vec(&meta)?;
    let json_len = u32::try_from(json.len())
        .map_err(|_| Error::Invalid(format!("metadata is {} bytes", json.len())))?;
    let payload_floats: usize = tensors.iter().map(|t| t.data.len()).sum();
    let mut out = Vec::with_capacity(HEADER_LEN + json.len() + 4 * payload_floats);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&json_len.to_le_bytes());
    out.extend_from_slice(&json);
    for t in tensors {
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    UserDebiasModel::load(&out, expected_feature_dim)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(meta_json: &str, tensors: &[&[f32]]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        out.extend_from_slice(meta_json.as_bytes());
        for t in tensors {
            for v in t.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn toy_artifact_round_trip_matches_hand_computed_forward() {
        let emb_hour: Vec<f32> = (0..N_HOUR).map(|h| h as f32 * 0.01).collect();
        let emb_dow: Vec<f32> = (0..N_DOW).map(|d| d as f32 * 0.1).collect();
        let emb_app: Vec<f32> = (0..N_APP).map(|a| a as f32 * 0.001).collect();
        let meta = concat!(
            r#"{"feature_version": "serving_v1", "feature_dim": 2, "head_names": ["fav", "reply"], "#,
            r#""n_layers": 1, "emb_dim": 1, "surfaces": ["other", "home", "immersive"], "#,
            r#""trained_at_ms": 1700000000000, "train_records": 42, "tensors": {"#,
            r#""feat_mean": [2], "feat_std": [2], "ln_g": [2], "ln_b": [2], "#,
            r#""emb_surface": [3, 1], "emb_hour": [24, 1], "emb_dow": [7, 1], "emb_app": [32, 1], "#,
            r#""w_out": [2, 3], "b_out": [3], "global_logit_by_surface": [3, 2], "#,
            r#""global_log_dwell_by_surface": [3], "w1": [6, 2], "b1": [2]}}"#
        );
        let data: [&[f32]; 14] = [
            &[1.0, 2.0],
            &[2.0, 4.0],
            &[1.0, 2.0],
            &[0.5, -0.5],
            &[0.1, 0.2, 0.3],
            &emb_hour,
            &emb_dow,
            &emb_app,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.1, 0.2, 0.3],
            &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            &[1.0, 2.0, 3.0],
            &[
                1.0, -1.0, 0.5, 0.5, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 10.0, -10.0,
            ],
            &[0.1, 0.2],
        ];
        let bytes = artifact(meta, &data);
        let model = UserDebiasModel::load(&bytes, 2).unwrap();
        assert_eq!(model.head_names(), ["fav", "reply"]);
        assert_eq!(model.trained_at_ms(), 1_700_000_000_000);
        assert_eq!(model.train_records(), 42);

        let ctx = RequestContext {
            surface_id: 1,
            hour: 5,
            dow: 3,
            app: 7,
        };
        let pred = model.predict(&[3.0, 6.0], &ctx);
        let tol = 1e-5f32;
        assert!((pred.user_logits[0] - 3.22).abs() < tol, "{pred:?}");
        assert!((pred.user_logits[1] - 6.44).abs() < tol, "{pred:?}");
        assert!((pred.user_log_dwell - 9.66).abs() < tol, "{pred:?}");
        assert_eq!(pred.global_logits, vec![-3.0, -4.0]);
        assert_eq!(pred.global_log_dwell, 2.0);

        let parsed: Metadata = serde_json::from_str(meta).unwrap();
        let tensors: Vec<ArtifactTensor> = parsed
            .tensors
            .iter()
            .zip(data)
            .map(|((name, shape), values)| ArtifactTensor {
                name: name.clone(),
                shape: shape.clone(),
                data: values.to_vec(),
            })
            .collect();
        let written = write_artifact_checked(
            &parsed.head_names,
            parsed.trained_at_ms,
            parsed.train_records,
            &tensors,
            2,
        )
        .unwrap();
        let written_json_end =
            HEADER_LEN + u32::from_le_bytes(written[4..8].try_into().unwrap()) as usize;
        assert_eq!(
            written[written_json_end..],
            bytes[HEADER_LEN + meta.len()..]
        );
        let reloaded = UserDebiasModel::load(&written, 2).unwrap();
        assert_eq!(reloaded.head_names(), model.head_names());
        assert_eq!(reloaded.trained_at_ms(), model.trained_at_ms());
        assert_eq!(reloaded.train_records(), model.train_records());
        assert_eq!(reloaded.predict(&[3.0, 6.0], &ctx), pred);
        assert!(matches!(
            write_artifact(&parsed.head_names, 0, 0, &tensors),
            Err(Error::FeatureDim(2))
        ));

        assert!(matches!(
            UserDebiasModel::from_bytes(&bytes),
            Err(Error::FeatureDim(2))
        ));
        assert!(matches!(
            UserDebiasModel::load(b"NOPE", 2),
            Err(Error::Truncated)
        ));
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            UserDebiasModel::load(&bad_magic, 2),
            Err(Error::BadMagic)
        ));
        let truncated = &bytes[..bytes.len() - 4];
        assert!(matches!(
            UserDebiasModel::load(truncated, 2),
            Err(Error::PayloadSize { .. })
        ));
    }
}
