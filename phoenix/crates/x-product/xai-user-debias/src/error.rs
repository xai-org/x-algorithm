// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::features::{FEATURE_DIM, FEATURE_VERSION};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("missing column {0}")]
    MissingColumn(&'static str),
    #[error("column {name} has unsupported type {data_type}")]
    UnsupportedColumnType {
        name: &'static str,
        data_type: String,
    },
    #[error("bad artifact magic")]
    BadMagic,
    #[error("artifact truncated")]
    Truncated,
    #[error("artifact metadata: {0}")]
    Metadata(#[from] serde_json::Error),
    #[error("feature version {0:?} != {FEATURE_VERSION:?}")]
    FeatureVersion(String),
    #[error("feature dim {0} != {FEATURE_DIM}")]
    FeatureDim(usize),
    #[error("missing tensor {0}")]
    MissingTensor(String),
    #[error("tensor {name} shape {actual:?} != {expected:?}")]
    TensorShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor payload {actual} bytes != {expected}")]
    PayloadSize { expected: usize, actual: usize },
    #[error("{0}")]
    Invalid(String),
}
