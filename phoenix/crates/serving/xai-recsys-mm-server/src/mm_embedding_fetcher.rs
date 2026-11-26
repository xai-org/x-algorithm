// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use lazy_static::lazy_static;
use object_store::ObjectMeta;
use object_store::path::Path as ObjectPath;
use prometheus::{IntCounterVec, register_int_counter_vec};
use tokio::time::sleep;

use crate::o2::O2Client;

lazy_static! {
    static ref FETCHER_ERRORS: IntCounterVec = register_int_counter_vec!(
        "mm_embedding_fetcher_errors_total",
        "MM embedding fetcher errors by method and fatality",
        &["method", "fatal"]
    )
    .unwrap();
    static ref FETCHER_SOURCE: IntCounterVec = register_int_counter_vec!(
        "mm_embedding_fetcher_source_total",
        "Successful MM embedding fetches from O2, by method",
        &["source", "method"]
    )
    .unwrap();
}

const RETRY_TIMEOUT: Duration = Duration::from_secs(1);
const MAX_RETRIES: usize = 9;

#[derive(Clone)]
pub struct MmEmbeddingFetcher {
    primary: O2Client,
}

impl MmEmbeddingFetcher {
    pub fn from_env(bucket: &str, prefix: &str) -> Result<Self> {
        Ok(Self {
            primary: O2Client::from_env(bucket, prefix)?,
        })
    }

    pub fn new(primary: O2Client) -> Self {
        Self { primary }
    }

    pub fn primary(&self) -> &O2Client {
        &self.primary
    }

    pub fn bucket(&self) -> &str {
        self.primary.bucket()
    }

    pub fn prefix(&self) -> &str {
        self.primary.prefix()
    }

    pub async fn list(&self, key: &str) -> Result<Vec<ObjectMeta>> {
        let mut attempt = 0;

        loop {
            match self.primary.try_list(key).await {
                Ok(objects) => {
                    FETCHER_SOURCE.with_label_values(&["o2", "list"]).inc();
                    return Ok(objects);
                }
                Err(err) => {
                    FETCHER_ERRORS.with_label_values(&["list", "false"]).inc();
                    log::warn!("O2 list error: {}; attempt={}", err, attempt + 1);

                    if attempt >= MAX_RETRIES {
                        FETCHER_ERRORS.with_label_values(&["list", "true"]).inc();
                        return Err(err).context("Failed to list objects");
                    }

                    sleep(RETRY_TIMEOUT * 2u32.pow(attempt as u32)).await;
                    attempt += 1;
                }
            }
        }
    }

    pub async fn get_raw(&self, path: &ObjectPath) -> Result<Bytes> {
        let mut attempt = 0;

        loop {
            match self.primary.try_get_raw(path).await {
                Ok(bytes) => {
                    FETCHER_SOURCE.with_label_values(&["o2", "get_raw"]).inc();
                    return Ok(bytes);
                }
                Err(err) => {
                    FETCHER_ERRORS
                        .with_label_values(&["get_raw", "false"])
                        .inc();
                    log::warn!(
                        "O2 get error for {}: {}; attempt={}",
                        path,
                        err,
                        attempt + 1
                    );

                    if attempt >= MAX_RETRIES {
                        FETCHER_ERRORS.with_label_values(&["get_raw", "true"]).inc();
                        return Err(err).with_context(|| format!("Failed to get object: {}", path));
                    }

                    sleep(RETRY_TIMEOUT * 2u32.pow(attempt as u32)).await;
                    attempt += 1;
                }
            }
        }
    }
}
