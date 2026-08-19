// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::env;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use half::f16;
use lazy_static::lazy_static;
use prometheus::{
    Gauge, Histogram, IntCounterVec, exponential_buckets, register_gauge, register_histogram,
    register_int_counter_vec,
};

use crate::embedding_cache::{EmbeddingCache, LmdbEmbeddingCache};
use crate::mm_embedding_fetcher::MmEmbeddingFetcher;
use crate::snapshot::{
    EmbeddingRecord, fetch_recent_embeddings_from_o2, read_embeddings_from_parquet,
};

const MAX_EMBEDDING_CACHE_SIZE: usize = 150_000_000;

const NUM_SHARDS: usize = 256;

const BASE_PATH: &str = "/dev/shm/mm_embeddings";

const EMBEDDING_TTL: Duration = Duration::from_secs(2 * 24 * 3600);

lazy_static! {
    static ref MM_EMBEDDING_SUCCESS_RATIO: Histogram = register_histogram!(
        "mm_embedding_success_ratio",
        "Ratio of posts for which we successfully got embeddings",
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0
        ]
    )
    .unwrap();
    static ref FETCH_MM_EMBEDDINGS_DURATION: Histogram = register_histogram!(
        "fetch_mm_embeddings_duration_seconds",
        "Duration of fetch_mm_embeddings method in seconds",
        exponential_buckets(0.001, 1.25, 33).unwrap()
    )
    .unwrap();
    static ref MM_EMBEDDING_CACHE_SIZE: Gauge = register_gauge!(
        "mm_embedding_cache_size",
        "Number of embeddings currently in the cache"
    )
    .unwrap();
    static ref MM_EMBEDDING_TWEET_LOOKUP: IntCounterVec = register_int_counter_vec!(
        "mm_embedding_tweet_lookup_total",
        "Total number of tweet embedding lookups by status",
        &["status"]
    )
    .unwrap();
}

#[derive(Clone)]
enum CacheShard {
    InProcess(Arc<EmbeddingCache>),
    Lmdb(Arc<LmdbEmbeddingCache>),
}

impl CacheShard {
    fn len(&self) -> usize {
        match self {
            Self::InProcess(c) => c.len(),
            Self::Lmdb(c) => c.len(),
        }
    }

    fn copy_to(&self, post_id: u64, dst: &mut [f16]) -> bool {
        match self {
            Self::InProcess(c) => c.copy_to(post_id, dst),
            Self::Lmdb(c) => c.copy_to(post_id, dst),
        }
    }

    fn get(&self, post_id: u64) -> Option<Arc<Vec<f16>>> {
        match self {
            Self::InProcess(c) => c.get(post_id),
            Self::Lmdb(c) => {
                let emb_dim = c.emb_dim();
                let mut dst = vec![f16::ZERO; emb_dim];
                if c.copy_to(post_id, &mut dst) {
                    Some(Arc::new(dst))
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct MmEmbeddingsClient {
    shards: Vec<CacheShard>,
    num_shards: usize,
}

pub type O2PreloadFuture = Pin<Box<dyn Future<Output = Result<()>> + Send>>;

impl MmEmbeddingsClient {
    fn shard(&self, key: u64) -> &CacheShard {
        &self.shards[(key % self.num_shards as u64) as usize]
    }

    pub fn total_len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    pub async fn preload_embeddings(&self, embeddings: Vec<EmbeddingRecord>, parallel: bool) {
        let num_shards = self.num_shards;
        let mut buckets: Vec<Vec<(u64, Vec<f16>)>> = (0..num_shards).map(|_| Vec::new()).collect();
        for record in embeddings {
            let idx = (record.post_id % num_shards as u64) as usize;
            buckets[idx].push((record.post_id, record.embedding));
        }

        if parallel {
            let mut handles = Vec::with_capacity(num_shards);
            for (shard_idx, bucket) in buckets.into_iter().enumerate() {
                let shard = self.shards[shard_idx].clone();
                handles.push(tokio::task::spawn_blocking(move || match shard {
                    CacheShard::InProcess(c) => {
                        c.bulk_insert(bucket.into_iter().map(|(id, emb)| (id, Arc::new(emb))));
                    }
                    CacheShard::Lmdb(c) => {
                        c.insert_many(bucket);
                    }
                }));
            }
            for handle in handles {
                if let Err(e) = handle.await {
                    log::error!("Shard insertion task panicked: {:#}", e);
                }
            }
        } else {
            let shards = self.shards.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || {
                for (shard_idx, bucket) in buckets.into_iter().enumerate() {
                    match &shards[shard_idx] {
                        CacheShard::InProcess(c) => {
                            c.bulk_insert(bucket.into_iter().map(|(id, emb)| (id, Arc::new(emb))));
                        }
                        CacheShard::Lmdb(c) => {
                            c.insert_many(bucket);
                        }
                    }
                }
            })
            .await
            {
                log::error!("Shard insertion task panicked: {:#}", e);
            }
        }

        MM_EMBEDDING_CACHE_SIZE.set(self.total_len() as f64);
    }

    pub fn get(&self, post_id: u64) -> Option<Arc<Vec<f16>>> {
        self.shard(post_id).get(post_id)
    }

    pub async fn fetch_mm_embeddings(
        &self,
        post_ids: Vec<u64>,
        emb_dim: usize,
        candidate_seq_len: usize,
    ) -> Result<Vec<f16>> {
        let start_time = std::time::Instant::now();
        assert!(emb_dim > 0);
        let total_posts = post_ids.len();
        let processed = candidate_seq_len.min(total_posts);
        let mut results = vec![f16::ZERO; emb_dim * processed];
        let mut missing = 0usize;

        for (idx, &post_id) in post_ids.iter().take(processed).enumerate() {
            let start = idx * emb_dim;
            if !self
                .shard(post_id)
                .copy_to(post_id, &mut results[start..start + emb_dim])
            {
                missing += 1;
            }
        }

        let found = processed - missing;
        MM_EMBEDDING_TWEET_LOOKUP
            .with_label_values(&["found"])
            .inc_by(found as u64);
        MM_EMBEDDING_TWEET_LOOKUP
            .with_label_values(&["not_found"])
            .inc_by(missing as u64);

        if total_posts > 0 {
            MM_EMBEDDING_SUCCESS_RATIO.observe((total_posts - missing) as f64 / total_posts as f64);
        }

        FETCH_MM_EMBEDDINGS_DURATION.observe(start_time.elapsed().as_secs_f64());
        Ok(results)
    }
}

fn o2_env_vars() -> (String, String) {
    let bucket = env::var("MM_O2_BUCKET").unwrap_or_else(|_| "recsys-mm-embeddings".to_string());
    let prefix = env::var("MM_O2_PREFIX").unwrap_or_else(|_| "prod".to_string());
    (bucket, prefix)
}

pub fn get_mm_client(
    worker_id: usize,
    emb_dim: usize,
    shared: bool,
) -> Result<(MmEmbeddingsClient, O2PreloadFuture)> {
    let is_writer = worker_id == 0;
    let shard_capacity = MAX_EMBEDDING_CACHE_SIZE / NUM_SHARDS;

    let shards: Vec<CacheShard> = if shared {
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..NUM_SHARDS)
                .map(|shard_idx| {
                    s.spawn(move || {
                        let lmdb_path =
                            PathBuf::from(format!("{}_shard_{}.lmdb", BASE_PATH, shard_idx));
                        let data_path =
                            PathBuf::from(format!("{}_shard_{}.dat", BASE_PATH, shard_idx));
                        let cache = if is_writer {
                            LmdbEmbeddingCache::create(
                                &lmdb_path,
                                &data_path,
                                shard_capacity,
                                emb_dim,
                                EMBEDDING_TTL,
                            )
                        } else {
                            LmdbEmbeddingCache::open(&lmdb_path, &data_path, EMBEDDING_TTL)
                        };
                        CacheShard::Lmdb(Arc::new(cache.unwrap_or_else(|e| {
                            panic!("LmdbEmbeddingCache shard {} failed: {}", shard_idx, e)
                        })))
                    })
                })
                .collect();
            for handle in handles {
                shards.push(handle.join().expect("shard creation thread panicked"));
            }
        });
        shards
    } else {
        log::info!(
            "Creating in-process MM cache ({} shards, capacity={})",
            NUM_SHARDS,
            MAX_EMBEDDING_CACHE_SIZE
        );
        (0..NUM_SHARDS)
            .map(|_| {
                CacheShard::InProcess(Arc::new(EmbeddingCache::new(shard_capacity, EMBEDDING_TTL)))
            })
            .collect()
    };

    let client = MmEmbeddingsClient {
        shards,
        num_shards: NUM_SHARDS,
    };

    if !is_writer {
        log::info!(
            "Worker {}: opened MM cache (read-only, {} shards)",
            worker_id,
            NUM_SHARDS
        );
        return Ok((client, Box::pin(async { Ok(()) })));
    }

    if let Ok(path) = env::var("MM_LOCAL_SNAPSHOT")
        && !path.trim().is_empty()
    {
        log::warn!(
            "MM_LOCAL_SNAPSHOT set ({path}): loading MM embeddings from a LOCAL parquet \
             and SKIPPING O2 (offline/dev path, NOT production behavior)."
        );
        let data = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("MM_LOCAL_SNAPSHOT: failed to read {path}: {e}"))?;
        let records = read_embeddings_from_parquet(bytes::Bytes::from(data), &path)?;
        let n = records.len();
        let client_for_load = client.clone();
        let load_future = async move {
            client_for_load.preload_embeddings(records, true).await;
            log::warn!("MM_LOCAL_SNAPSHOT: loaded {n} MM embeddings into the cache (NOT O2).");
            Ok(())
        };
        return Ok((client, Box::pin(load_future)));
    }

    let mm_client_for_bg = client.clone();
    let (o2_bucket, o2_prefix) = o2_env_vars();

    match MmEmbeddingFetcher::from_env(&o2_bucket, &o2_prefix) {
        Ok(o2_client) => {
            log::info!("Successfully created O2 client, fetching embeddings for last 2 days...");
            let o2_future = async move {
                let fetch_start = std::time::Instant::now();
                match fetch_recent_embeddings_from_o2(o2_client, Arc::new(mm_client_for_bg)).await {
                    Ok(()) => {
                        log::info!(
                            "Successfully fetched and cached initial embeddings from O2 in {:.2}s",
                            fetch_start.elapsed().as_secs_f64()
                        );
                        Ok(())
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to fetch embeddings from O2 after {:.2}s: {}",
                            fetch_start.elapsed().as_secs_f64(),
                            e
                        );
                        Err(e)
                    }
                }
            };
            Ok((client, Box::pin(o2_future)))
        }
        Err(e) => {
            log::error!("Failed to create O2 client: {}", e);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::EmbeddingRecord;
    use std::time::Instant;

    #[tokio::test]
    #[ignore]
    async fn bench_serving() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        let emb_dim = 1024;
        let num_entries = 150_000_000;
        let num_lookups = 2816;
        let num_rounds = 100;
        let batch_size = 2_000_000;

        const TWEPOCH_MS: u64 = 1288834974657;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let base_id = ((now_ms - TWEPOCH_MS) << 22) | 1;

        let data_size_gb = (num_entries as f64 * (8 + emb_dim * 2) as f64) / 1e9;
        eprintln!(
            "\n=== Hybrid Sharded Benchmark: {} entries, {} shards, emb_dim={}, data={:.0} GB ===",
            num_entries, NUM_SHARDS, emb_dim, data_size_gb
        );

        let (client, _) = get_mm_client(0, emb_dim, true).unwrap();

        let start = Instant::now();
        for batch_start in (0..num_entries).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_entries);
            let embeddings: Vec<EmbeddingRecord> = (batch_start..batch_end)
                .map(|i| EmbeddingRecord {
                    post_id: base_id + i as u64,
                    embedding: vec![f16::from_f32(0.1); emb_dim],
                })
                .collect();
            client.preload_embeddings(embeddings, true).await;
            if batch_start > 0 && (batch_start / batch_size) % 5 == 0 {
                let rate = batch_start as f64 / start.elapsed().as_secs_f64();
                eprintln!(
                    "  insert: {}/{} ({:.0} entries/sec)",
                    batch_start, num_entries, rate
                );
            }
        }
        let insert_elapsed = start.elapsed();
        eprintln!(
            "  insert:         {:>8.2} sec ({:.0} entries/sec)",
            insert_elapsed.as_secs_f64(),
            num_entries as f64 / insert_elapsed.as_secs_f64()
        );
        let effective_capacity = (num_entries / NUM_SHARDS) * NUM_SHARDS;
        assert_eq!(client.total_len(), effective_capacity);

        let all_ids: Vec<u64> = (0..num_entries as u64).map(|i| base_id + i).collect();
        let random_batches: Vec<Vec<u64>> = (0..num_rounds)
            .map(|round| {
                (0..num_lookups)
                    .map(|i| {
                        let mut h = DefaultHasher::new();
                        (round * num_lookups + i).hash(&mut h);
                        all_ids[(h.finish() as usize) % num_entries]
                    })
                    .collect()
            })
            .collect();

        let start = Instant::now();
        for batch in &random_batches {
            let _ = client
                .fetch_mm_embeddings(batch.clone(), emb_dim, num_lookups)
                .await
                .unwrap();
        }
        let lookup_elapsed = start.elapsed();
        let per_request_ms = lookup_elapsed.as_secs_f64() * 1000.0 / num_rounds as f64;
        eprintln!(
            "  fetch_mm_embeddings: {:>8.2} ms/request ({} random candidates)",
            per_request_ms, num_lookups
        );
        eprintln!(
            "                       {:>8.0} ns/lookup",
            lookup_elapsed.as_nanos() as f64 / (num_rounds * num_lookups) as f64
        );
        eprintln!();
    }
}
