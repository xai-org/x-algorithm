use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tonic::async_trait;
use xai_cache::{KVCacheError, Key, Value};
use xai_manhattan::ManhattanError;
use xai_safety_label_store::types::encode_lkey;
use xai_x_thrift::servo_repo::{CachedValue, CachedValueStatus};
use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};

use crate::decode::RawSafetyLabel;
use crate::metrics::LabelMetrics;
use crate::mh_client::{FetchResult, ManhattanLabelFetcher};
use crate::remote::{cache_key, CacheRead, RemoteSource};
use crate::source::SafetyLabelSource;

#[derive(Default)]
pub struct FakeLabels {
    labels: HashMap<u64, Vec<(SafetyLabelType, SafetyLabel)>>,
    not_found: HashSet<u64>,
    failing: HashSet<u64>,
}

impl FakeLabels {
    #[must_use]
    pub fn label(mut self, tweet_id: u64, label_type: SafetyLabelType, label: SafetyLabel) -> Self {
        self.labels
            .entry(tweet_id)
            .or_default()
            .push((label_type, label));
        self
    }

    #[must_use]
    pub fn not_found(mut self, tweet_id: u64) -> Self {
        self.not_found.insert(tweet_id);
        self
    }

    #[must_use]
    pub fn failing(mut self, tweet_id: u64) -> Self {
        self.failing.insert(tweet_id);
        self
    }

    pub fn into_source(self) -> SafetyLabelSource {
        let backends = Arc::new(self);
        let remote = RemoteSource::new(
            Arc::<Self>::clone(&backends),
            backends,
            LabelMetrics::default(),
        );
        SafetyLabelSource::new(Arc::new(remote), None, LabelMetrics::default())
    }
}

#[async_trait]
impl CacheRead for FakeLabels {
    async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, Result<Option<Value>, KVCacheError>> {
        let not_found: Vec<Key> = self
            .not_found
            .iter()
            .filter_map(|&id| cache_key(id).ok())
            .collect();
        let answer = |key: &Key| {
            if !not_found.contains(key) {
                return Ok(None);
            }
            cached_value_not_found()
                .map(Some)
                .map_err(|e| KVCacheError::Client(e.to_string()))
        };
        keys.iter().map(|key| (key.clone(), answer(key))).collect()
    }
}

#[async_trait]
impl ManhattanLabelFetcher for FakeLabels {
    async fn fetch_labels(&self, tweet_ids: &[i64]) -> Result<Vec<FetchResult>, ManhattanError> {
        let fetch = |id: u64| {
            if self.failing.contains(&id) || self.not_found.contains(&id) {
                return Err(ManhattanError::NativeProtocol("fake failure".to_owned()));
            }
            let labels = self.labels.get(&id).map_or(&[][..], Vec::as_slice);
            labels
                .iter()
                .map(|(label_type, label)| raw_label(*label_type, label))
                .collect::<thrift::Result<_>>()
                .map_err(|e| ManhattanError::NativeProtocol(e.to_string()))
        };
        Ok(tweet_ids
            .iter()
            .map(|&id| fetch(id.cast_unsigned()))
            .collect())
    }
}

pub(crate) fn mval(label: &SafetyLabel) -> thrift::Result<Vec<u8>> {
    let mut buf = vec![0x0C, 0x00, 0x04];
    buf.extend(xai_x_thrift::serialize_binary(label)?);
    buf.push(0x00);
    Ok(buf)
}

fn raw_label(label_type: SafetyLabelType, label: &SafetyLabel) -> thrift::Result<RawSafetyLabel> {
    Ok(RawSafetyLabel {
        lkey: encode_lkey(label_type),
        mval: mval(label)?,
    })
}

pub(crate) fn cached_value_not_found() -> thrift::Result<Vec<u8>> {
    xai_x_thrift::serialize_binary(&CachedValue {
        status: Some(CachedValueStatus::NOT_FOUND),
        ..Default::default()
    })
}
