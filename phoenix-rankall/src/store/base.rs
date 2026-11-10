use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use log::{info, warn};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::snowflake::timestamp_secs_to_snowflake;
use super::writer::{atomic_write_parquet, find_latest_versioned_file, read_parquet_file};
use super::{SnapshotStore, StoreConfig, StoreError, WindowRouter};
use crate::config::WindowConfig;
use crate::metrics::PipelineMetrics;
use crate::processor::record::{EngagementCounts, IndexRecord, PostId};

#[derive(Debug, Clone)]
enum StoredValue {
        Core(i64),
            Full(Arc<IndexRecord>),
        Sid { author_id: i64, post_sid: Vec<i32> },
}

impl StoredValue {
    fn from_record(record: IndexRecord) -> (i64, Self) {
        let post_id = record.post_id();
        match record {
            IndexRecord::Core { author_id, .. } | IndexRecord::Ads { author_id, .. } => {
                (post_id, StoredValue::Core(author_id))
            }
            IndexRecord::Sid {
                author_id,
                post_sid,
                ..
            } => (
                post_id,
                StoredValue::Sid {
                    author_id,
                    post_sid,
                },
            ),
            other => (post_id, StoredValue::Full(Arc::new(other))),
        }
    }

    fn merge(existing: Self, incoming: Self) -> Self {
        match (existing, incoming) {
            (StoredValue::Full(old), StoredValue::Full(new)) => StoredValue::Full(Arc::new(
                old.as_ref().clone().merge_with(new.as_ref().clone()),
            )),
            (_, incoming) => incoming,
        }
    }

    fn author_id(&self) -> i64 {
        match self {
            StoredValue::Core(author_id) => *author_id,
            StoredValue::Sid { author_id, .. } => *author_id,
            StoredValue::Full(record) => record.author_id(),
        }
    }

            fn post_sid(&self) -> Option<&[i32]> {
        match self {
            StoredValue::Sid { post_sid, .. } => Some(post_sid.as_slice()),
            _ => None,
        }
    }

        fn sid_is_missing(&self) -> bool {
        match self.post_sid() {
            Some(codes) => codes.is_empty() || codes.first() == Some(&-1),
            None => false,
        }
    }
}

type WindowData = HashMap<PostId, StoredValue>;

fn upsert_stored(map: &mut WindowData, post_id: PostId, incoming: StoredValue) {
    let merged = match map.remove(&post_id) {
        Some(existing) => StoredValue::merge(existing, incoming),
        None => incoming,
    };
    map.insert(post_id, merged);
}

type BufferMap = HashMap<String, WindowData>;

fn posts_in_index_windows(
    main_data: &BufferMap,
    pending: &BufferMap,
    window_name: &str,
    windows: &[WindowConfig],
    index_name: &str,
) -> Vec<PostId> {
    let Some(target) = main_data.get(window_name) else {
        return Vec::new();
    };
    let index_windows: Vec<String> = windows
        .iter()
        .filter(|w| w.name == index_name)
        .map(WindowConfig::window_name)
        .collect();
    target
        .keys()
        .filter(|post_id| {
            index_windows.iter().any(|w| {
                main_data.get(w).is_some_and(|m| m.contains_key(*post_id))
                    || pending.get(w).is_some_and(|m| m.contains_key(*post_id))
            })
        })
        .copied()
        .collect()
}

fn drop_posts_in_excluded_index(
    main_data: &mut BufferMap,
    pending: &BufferMap,
    wc: &WindowConfig,
    windows: &[WindowConfig],
) -> usize {
    let Some(index_name) = &wc.exclude_posts_in else {
        return 0;
    };
    let window_name = wc.window_name();
    let excluded = posts_in_index_windows(main_data, pending, &window_name, windows, index_name);
    if let Some(main) = main_data.get_mut(&window_name) {
        for post_id in &excluded {
            main.remove(post_id);
        }
    }
    if !excluded.is_empty() {
        info!(
            "dropped {} records from '{window_name}' that are in '{index_name}' windows",
            excluded.len()
        );
    }
    excluded.len()
}

struct SharedState {
        main_data: BufferMap,
        pending: BufferMap,
}

pub struct BaseSnapshotStore {
    config: StoreConfig,
    router: WindowRouter,
    state: Arc<Mutex<SharedState>>,
    metrics: Arc<PipelineMetrics>,
    cancel: CancellationToken,
    dump_handle: Mutex<Option<JoinHandle<()>>>,
    dump_completed: Arc<AtomicBool>,
}

const PREFLOOR_SLACK_SECS: f64 = 6.0 * 3600.0;

impl BaseSnapshotStore {
        pub fn new(config: StoreConfig, router: WindowRouter, metrics: Arc<PipelineMetrics>) -> Self {
        std::fs::create_dir_all(&config.output_dir).ok();

        Self {
            config,
            router,
            state: Arc::new(Mutex::new(SharedState {
                main_data: HashMap::new(),
                pending: HashMap::new(),
            })),
            metrics,
            cancel: CancellationToken::new(),
            dump_handle: Mutex::new(None),
            dump_completed: Arc::new(AtomicBool::new(false)),
        }
    }

        async fn load_existing_data(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock().await;
        let mut total_loaded: usize = 0;

        for wc in &self.config.windows {
            let window_name = wc.window_name();
            let symlink_path = self
                .config
                .output_dir
                .join(format!("{window_name}.parquet"));

            let public_path = if symlink_path.exists() {
                Some(symlink_path.clone())
            } else {
                find_latest_versioned_file(&self.config.output_dir, &window_name)
            };

            let mut window_data = HashMap::new();
            if let Some(path) = &public_path {
                match read_parquet_file(path) {
                    Ok(batches) => {
                        for batch in &batches {
                            load_post_ids_from_batch(
                                batch,
                                &mut window_data,
                                &self.config.pipeline,
                            );
                        }
                    }
                    Err(e) => warn!("failed to load {}: {e}", path.display()),
                }
            }
            if wc.min_age.is_some() {
                let pf_name = format!("prefloor_{window_name}");
                let pf_link = self.config.output_dir.join(format!("{pf_name}.parquet"));
                let pf_path = if pf_link.exists() {
                    Some(pf_link)
                } else {
                    find_latest_versioned_file(&self.config.output_dir, &pf_name)
                };
                if let Some(pf_path) = pf_path {
                    match read_parquet_file(&pf_path) {
                        Ok(pf_batches) => {
                            for batch in &pf_batches {
                                load_post_ids_from_batch(
                                    batch,
                                    &mut window_data,
                                    &self.config.pipeline,
                                );
                            }
                        }
                        Err(e) => warn!("failed to load {}: {e}", pf_path.display()),
                    }
                }
            }

            if public_path.is_none() {
                let now = chrono::Utc::now().timestamp() as f64;
                let retention_cutoff =
                    timestamp_secs_to_snowflake(now - wc.retention.as_secs() as f64);
                let min_age_cutoff = wc
                    .min_age
                    .map(|d| timestamp_secs_to_snowflake(now - d.as_secs() as f64));
                let mut entries: Vec<(i64, &StoredValue)> = window_data
                    .iter()
                    .map(|(&pid, val)| (pid, val))
                    .filter(|&(pid, _)| {
                        pid >= retention_cutoff && min_age_cutoff.is_none_or(|c| pid < c)
                    })
                    .collect();
                entries.sort_by_key(|(pid, _)| *pid);
                info!(
                    "public file missing for window '{window_name}', regenerating with {} records",
                    entries.len()
                );
                let regen = if entries.is_empty() {
                    Ok(empty_batch_for_window(&window_name, &self.config.pipeline))
                } else {
                    stored_to_batch(
                        &entries,
                        &window_name,
                        &self.config.pipeline,
                        self.config.sid_num_levels,
                    )
                };
                match regen {
                    Ok(batch) => {
                        if let Err(e) = atomic_write_parquet(
                            &self.config.output_dir,
                            &window_name,
                            now as i64,
                            &batch,
                            self.config.versions_to_keep,
                        ) {
                            warn!("failed to regenerate public parquet for {window_name}: {e}");
                        }
                    }
                    Err(e) => warn!("failed to build regen batch for {window_name}: {e}"),
                }
            }
            let count = window_data.len();
            total_loaded += count;
            info!("loaded {count} records into window '{window_name}'");
            state.main_data.insert(window_name.clone(), window_data);
        }

        if total_loaded > 0 {
            info!(
                "loaded {total_loaded} total records across {} windows",
                self.config.windows.len()
            );
        }
        Ok(())
    }

            pub async fn get_missing_sid_post_ids(&self, max_age_hours: f64) -> Vec<PostId> {
        let cutoff = if max_age_hours > 0.0 {
            let cutoff_secs = chrono::Utc::now().timestamp() as f64 - max_age_hours * 3600.0;
            super::snowflake::timestamp_secs_to_snowflake(cutoff_secs)
        } else {
            i64::MIN
        };

        let state = self.state.lock().await;
        let mut missing: Vec<PostId> = Vec::new();

        for window in state.main_data.values() {
            missing.extend(window.iter().filter_map(|(&pid, val)| {
                if pid < cutoff {
                    return None;
                }
                if val.sid_is_missing() {
                    Some(pid)
                } else {
                    None
                }
            }));
        }

        for window in state.pending.values() {
            missing.extend(window.iter().filter_map(|(&pid, val)| {
                if pid < cutoff {
                    return None;
                }
                if val.sid_is_missing() {
                    Some(pid)
                } else {
                    None
                }
            }));
        }

        missing.sort_unstable();
        missing.dedup();
        missing
    }

                                    pub async fn update_sids(
        &self,
        sid_map: HashMap<PostId, Vec<i32>>,
    ) -> Result<usize, StoreError> {
        if sid_map.is_empty() {
            return Ok(0);
        }
        let mut state = self.state.lock().await;
        let mut updated = 0usize;

        for window in state.main_data.values_mut() {
            for (pid, codes) in &sid_map {
                if let Some(slot) = window.get_mut(pid)
                    && let StoredValue::Sid { post_sid, .. } = slot
                {
                    *post_sid = codes.clone();
                    updated += 1;
                }
            }
        }

        for window in state.pending.values_mut() {
            for (pid, codes) in &sid_map {
                if let Some(slot) = window.get_mut(pid)
                    && let StoredValue::Sid { post_sid, .. } = slot
                {
                    *post_sid = codes.clone();
                    updated += 1;
                }
            }
        }

        Ok(updated)
    }

        async fn dump_all_windows(&self) -> anyhow::Result<()> {
        let now = chrono::Utc::now().timestamp() as f64;
        let epoch_secs = now as i64;

        self.metrics.store_dump_started.inc();

        let pending_snapshot = {
            let mut state = self.state.lock().await;
            let mut snapshot = HashMap::new();
            std::mem::swap(&mut state.pending, &mut snapshot);
            snapshot
        };

        for wc in &self.config.windows {
            let window_name = wc.window_name();
            let retention_secs = wc.retention.as_secs() as f64;
            let min_age_cutoff = wc
                .min_age
                .map(|d| timestamp_secs_to_snowflake(now - d.as_secs() as f64));
            let prefloor_cutoff = wc.min_age.map(|d| {
                timestamp_secs_to_snowflake(now - d.as_secs() as f64 - PREFLOOR_SLACK_SECS)
            });

            if let Some(pending) = pending_snapshot.get(&window_name)
                && !pending.is_empty()
            {
                let mut state = self.state.lock().await;
                let main = state.main_data.entry(window_name.clone()).or_default();
                for (post_id, record) in pending {
                    upsert_stored(main, *post_id, record.clone());
                }
            }

            let cutoff = timestamp_secs_to_snowflake(now - retention_secs);
            let mut state = self.state.lock().await;
            let dropped = drop_posts_in_excluded_index(
                &mut state.main_data,
                &pending_snapshot,
                wc,
                &self.config.windows,
            );
            let main = state.main_data.entry(window_name.clone()).or_default();

            let before_count = main.len();
            main.retain(|post_id, _| *post_id >= cutoff);
            let removed = before_count - main.len() + dropped;
            if removed > 0 {
                self.metrics
                    .store_compaction_removed_total
                    .inc_by(removed as u64);
                main.shrink_to_fit();
            }

            let pipeline = &self.config.pipeline;
            let sid_num_levels = self.config.sid_num_levels;
            let in_memory = main.len();
            let mut entries: Vec<(i64, &StoredValue)> = main
                .iter()
                .map(|(&pid, val)| (pid, val))
                .filter(|&(pid, _)| min_age_cutoff.is_none_or(|c| pid < c))
                .collect();
            entries.sort_by_key(|(pid, _)| *pid);
            let batch = if entries.is_empty() {
                empty_batch_for_window(&window_name, pipeline)
            } else {
                stored_to_batch(&entries, &window_name, pipeline, sid_num_levels)?
            };
            let prefloor_batch = match prefloor_cutoff {
                Some(c) => {
                    let mut young: Vec<(i64, &StoredValue)> = main
                        .iter()
                        .map(|(&pid, val)| (pid, val))
                        .filter(|&(pid, _)| pid >= c)
                        .collect();
                    young.sort_by_key(|(pid, _)| *pid);
                    Some(if young.is_empty() {
                        empty_batch_for_window(&window_name, pipeline)
                    } else {
                        stored_to_batch(&young, &window_name, pipeline, sid_num_levels)?
                    })
                }
                None => None,
            };

            let output_dir = self.config.output_dir.clone();
            let wn = window_name.clone();
            let keep = self.config.versions_to_keep;
            let written = tokio::task::spawn_blocking(move || {
                if let Some(pb) = prefloor_batch {
                    atomic_write_parquet(
                        &output_dir,
                        &format!("prefloor_{wn}"),
                        epoch_secs,
                        &pb,
                        keep,
                    )?;
                }
                atomic_write_parquet(&output_dir, &wn, epoch_secs, &batch, keep)
            })
            .await??;

            let parts: Vec<&str> = window_name.rsplitn(2, '_').collect();
            let (window_type, index_name) = if parts.len() == 2 {
                (parts[0], parts[1])
            } else {
                ("unknown", window_name.as_str())
            };
            self.metrics
                .store_window_records_written
                .with_label_values(&[index_name, window_type])
                .inc_by(written as f64);
            self.metrics
                .store_window_last_dump_records
                .with_label_values(&[index_name, window_type])
                .set(written as f64);
            self.metrics
                .store_buffer_size
                .with_label_values(&[index_name, window_type])
                .set(in_memory as f64);
        }

        self.metrics.store_dump_completed.inc();
        self.metrics.store_dump_last_success_ts.set(epoch_secs);
        self.dump_completed
            .store(true, std::sync::atomic::Ordering::Release);

        Ok(())
    }
}

#[async_trait]
impl SnapshotStore for BaseSnapshotStore {
    async fn add_batch(&self, records: Vec<IndexRecord>) -> Result<(), StoreError> {
        let mut state = self.state.lock().await;
        for record in records {
            let window_names = (self.router)(&record, &self.config.windows);
            let (post_id, stored) = StoredValue::from_record(record);
            for wn in window_names {
                upsert_stored(
                    state.pending.entry(wn).or_default(),
                    post_id,
                    stored.clone(),
                );
            }
        }
        Ok(())
    }

    async fn start(&self) -> Result<(), StoreError> {
        self.load_existing_data()
            .await
            .map_err(StoreError::ParquetWrite)?;

        let interval = Duration::from_secs(self.config.compaction_interval_secs);

        let config = self.config.clone();
        let metrics = Arc::clone(&self.metrics);


        let store_state = Arc::clone(&self.state);
        let store_cancel = self.cancel.clone();
        let store_dump_completed: Arc<AtomicBool> = Arc::clone(&self.dump_completed);

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {},
                    _ = store_cancel.cancelled() => break,
                }

                info!("running periodic dump...");
                let now = chrono::Utc::now().timestamp() as f64;
                let epoch_secs = now as i64;

                metrics.store_dump_started.inc();

                let pending_snapshot = {
                    let mut st = store_state.lock().await;
                    let mut snapshot = HashMap::new();
                    std::mem::swap(&mut st.pending, &mut snapshot);
                    snapshot
                };

                let mut all_ok = true;
                for wc in &config.windows {
                    let window_name = wc.window_name();
                    let retention_secs = wc.retention.as_secs() as f64;
                    let min_age_cutoff = wc
                        .min_age
                        .map(|d| timestamp_secs_to_snowflake(now - d.as_secs() as f64));
                    let prefloor_cutoff = wc.min_age.map(|d| {
                        timestamp_secs_to_snowflake(now - d.as_secs() as f64 - PREFLOOR_SLACK_SECS)
                    });

                    if let Some(pending) = pending_snapshot.get(&window_name)
                        && !pending.is_empty()
                    {
                        let mut st = store_state.lock().await;
                        let main = st.main_data.entry(window_name.clone()).or_default();
                        for (post_id, record) in pending {
                            upsert_stored(main, *post_id, record.clone());
                        }
                    }

                    let cutoff = timestamp_secs_to_snowflake(now - retention_secs);
                    let mut st = store_state.lock().await;
                    let dropped = drop_posts_in_excluded_index(
                        &mut st.main_data,
                        &pending_snapshot,
                        wc,
                        &config.windows,
                    );
                    let main = st.main_data.entry(window_name.clone()).or_default();

                    let before = main.len();
                    main.retain(|pid, _| *pid >= cutoff);
                    let removed = before - main.len() + dropped;
                    if removed > 0 {
                        metrics
                            .store_compaction_removed_total
                            .inc_by(removed as u64);
                        main.shrink_to_fit();
                    }

                    let in_memory = main.len();
                    let mut entries: Vec<(i64, &StoredValue)> = main
                        .iter()
                        .map(|(&pid, val)| (pid, val))
                        .filter(|&(pid, _)| min_age_cutoff.is_none_or(|c| pid < c))
                        .collect();
                    entries.sort_by_key(|(pid, _)| *pid);
                    let batch_result = if entries.is_empty() {
                        Ok(empty_batch_for_window(&window_name, &config.pipeline))
                    } else {
                        stored_to_batch(
                            &entries,
                            &window_name,
                            &config.pipeline,
                            config.sid_num_levels,
                        )
                    };
                    let prefloor_result = match prefloor_cutoff {
                        Some(c) => {
                            let mut young: Vec<(i64, &StoredValue)> = main
                                .iter()
                                .map(|(&pid, val)| (pid, val))
                                .filter(|&(pid, _)| pid >= c)
                                .collect();
                            young.sort_by_key(|(pid, _)| *pid);
                            if young.is_empty() {
                                Some(Ok(empty_batch_for_window(&window_name, &config.pipeline)))
                            } else {
                                Some(stored_to_batch(
                                    &young,
                                    &window_name,
                                    &config.pipeline,
                                    config.sid_num_levels,
                                ))
                            }
                        }
                        None => None,
                    };
                    match batch_result {
                        Ok(batch) => {
                            let dir = config.output_dir.clone();
                            let wn = window_name.clone();
                            let keep = config.versions_to_keep;
                            let prefloor_batch = match prefloor_result {
                                Some(Ok(pb)) => Some(pb),
                                Some(Err(e)) => {
                                    warn!(
                                        "prefloor batch conversion failed for {window_name}: {e}"
                                    );
                                    all_ok = false;
                                    continue;
                                }
                                None => None,
                            };
                            match tokio::task::spawn_blocking(move || {
                                if let Some(pb) = prefloor_batch {
                                    atomic_write_parquet(
                                        &dir,
                                        &format!("prefloor_{wn}"),
                                        epoch_secs,
                                        &pb,
                                        keep,
                                    )?;
                                }
                                atomic_write_parquet(&dir, &wn, epoch_secs, &batch, keep)
                            })
                            .await
                            {
                                Ok(Ok(written)) => {
                                    let parts: Vec<&str> = window_name.rsplitn(2, '_').collect();
                                    let (wt, idx) = if parts.len() == 2 {
                                        (parts[0], parts[1])
                                    } else {
                                        ("unknown", window_name.as_str())
                                    };
                                    metrics
                                        .store_window_records_written
                                        .with_label_values(&[idx, wt])
                                        .inc_by(written as f64);
                                    metrics
                                        .store_window_last_dump_records
                                        .with_label_values(&[idx, wt])
                                        .set(written as f64);
                                    metrics
                                        .store_buffer_size
                                        .with_label_values(&[idx, wt])
                                        .set(in_memory as f64);
                                }
                                Ok(Err(e)) => {
                                    warn!("dump failed for {window_name}: {e}");
                                    all_ok = false;
                                }
                                Err(e) => {
                                    warn!("dump task panicked for {window_name}: {e}");
                                    all_ok = false;
                                }
                            }
                        }
                        Err(e) => {
                            warn!("batch conversion failed for {window_name}: {e}");
                            all_ok = false;
                        }
                    }
                }

                if all_ok {
                    metrics.store_dump_completed.inc();
                    metrics.store_dump_last_success_ts.set(epoch_secs);
                    store_dump_completed.store(true, std::sync::atomic::Ordering::Release);
                    info!("periodic dump complete");
                } else {
                    warn!("periodic dump had failures; not signaling completion (offsets held)");
                }
            }
        });

        *self.dump_handle.lock().await = Some(handle);
        info!(
            "snapshot store started with {} windows",
            self.config.windows.len()
        );
        Ok(())
    }

    async fn stop(&self) -> Result<(), StoreError> {
        self.cancel.cancel();

        if let Some(handle) = self.dump_handle.lock().await.take() {
            let _ = handle.await;
        }

        info!("running final dump before shutdown...");
        self.dump_all_windows()
            .await
            .map_err(StoreError::ParquetWrite)?;

        info!("snapshot store stopped");
        Ok(())
    }

    fn dump_completed_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.dump_completed)
    }
}


fn is_sid_pipeline(pipeline: &str) -> bool {
    matches!(pipeline, "sid" | "sid-tail")
}

fn stored_to_batch(
    entries: &[(i64, &StoredValue)],
    window_name: &str,
    pipeline: &str,
    sid_num_levels: usize,
) -> anyhow::Result<RecordBatch> {
    if entries.is_empty() {
        anyhow::bail!("cannot create batch from empty entries");
    }

    match pipeline {
        "metadata" => stored_to_metadata_batch(entries),
        "topic" if window_name.contains("topic") => stored_to_topic_batch(entries),
        p if is_sid_pipeline(p) => stored_to_sid_batch(entries, sid_num_levels),
        _ => stored_to_core_batch(entries),
    }
}

fn stored_to_core_batch(entries: &[(i64, &StoredValue)]) -> anyhow::Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
    ]));

    let post_ids: Vec<i64> = entries.iter().map(|(pid, _)| *pid).collect();
    let author_ids: Vec<i64> = entries.iter().map(|(_, val)| val.author_id()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(post_ids)) as ArrayRef,
            Arc::new(Int64Array::from(author_ids)) as ArrayRef,
        ],
    )?;
    Ok(batch)
}

fn stored_to_sid_batch(
    entries: &[(i64, &StoredValue)],
    sid_num_levels: usize,
) -> anyhow::Result<RecordBatch> {
    use arrow::array::{Int64Builder, ListBuilder};

    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
        Field::new(
            "post_sid",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]));

    let mut post_ids = Vec::with_capacity(entries.len());
    let mut author_ids = Vec::with_capacity(entries.len());
    let mut sid_builder = ListBuilder::new(Int64Builder::new());

    let append_sentinel = |b: &mut ListBuilder<Int64Builder>| {
        for _ in 0..sid_num_levels {
            b.values().append_value(-1);
        }
        b.append(true);
    };

    for (pid, val) in entries {
        post_ids.push(*pid);
        author_ids.push(val.author_id());
        match val {
            StoredValue::Sid { post_sid, .. } => {
                if post_sid.len() == sid_num_levels {
                    for code in post_sid {
                        sid_builder.values().append_value(*code as i64);
                    }
                    sid_builder.append(true);
                } else {
                    append_sentinel(&mut sid_builder);
                }
            }
            _ => append_sentinel(&mut sid_builder),
        }
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(post_ids)) as ArrayRef,
            Arc::new(Int64Array::from(author_ids)) as ArrayRef,
            Arc::new(sid_builder.finish()) as ArrayRef,
        ],
    )?;
    Ok(batch)
}

fn stored_to_topic_batch(entries: &[(i64, &StoredValue)]) -> anyhow::Result<RecordBatch> {
    use crate::processor::topic_names::topic_ids_to_names;
    use arrow::array::{Int64Builder, ListBuilder, StringBuilder};

    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
        Field::new(
            "topic_entity_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        Field::new(
            "topic_names",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));

    let mut post_ids = Vec::with_capacity(entries.len());
    let mut author_ids = Vec::with_capacity(entries.len());
    let mut topic_ids_builder = ListBuilder::new(Int64Builder::new());
    let mut topic_names_builder = ListBuilder::new(StringBuilder::new());

    for (pid, val) in entries {
        post_ids.push(*pid);
        author_ids.push(val.author_id());

        match val {
            StoredValue::Full(record) => {
                if let IndexRecord::Topic {
                    topic_entity_ids, ..
                } = record.as_ref()
                {
                    for tid in topic_entity_ids {
                        topic_ids_builder.values().append_value(*tid);
                    }
                    topic_ids_builder.append(true);

                    let names = topic_ids_to_names(topic_entity_ids);
                    for name in &names {
                        topic_names_builder.values().append_value(name);
                    }
                    topic_names_builder.append(true);
                } else {
                    topic_ids_builder.append(true);
                    topic_names_builder.append(true);
                }
            }
            StoredValue::Core(_) | StoredValue::Sid { .. } => {
                topic_ids_builder.append(true);
                topic_names_builder.append(true);
            }
        }
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(post_ids)) as ArrayRef,
            Arc::new(Int64Array::from(author_ids)) as ArrayRef,
            Arc::new(topic_ids_builder.finish()) as ArrayRef,
            Arc::new(topic_names_builder.finish()) as ArrayRef,
        ],
    )?;
    Ok(batch)
}

fn stored_to_metadata_batch(entries: &[(i64, &StoredValue)]) -> anyhow::Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
        Field::new("has_video", DataType::Boolean, true),
        Field::new("video_duration_ms", DataType::Int64, true),
        Field::new("has_image", DataType::Boolean, true),
        Field::new("author_followers_count", DataType::Int64, true),
        Field::new("retweet_count", DataType::Int64, true),
        Field::new("reply_count", DataType::Int64, true),
        Field::new("fav_count", DataType::Int64, true),
        Field::new("quote_count", DataType::Int64, true),
        Field::new("bookmark_count", DataType::Int64, true),
        Field::new("view_count", DataType::Int64, true),
        Field::new("not_interested_in_count", DataType::Int64, true),
        Field::new("report_count", DataType::Int64, true),
        Field::new("block_count", DataType::Int64, true),
    ]));

    let mut post_ids = Vec::with_capacity(entries.len());
    let mut author_ids = Vec::with_capacity(entries.len());
    let mut has_video = Vec::with_capacity(entries.len());
    let mut video_duration_ms = Vec::with_capacity(entries.len());
    let mut has_image = Vec::with_capacity(entries.len());
    let mut author_followers = Vec::with_capacity(entries.len());
    let mut retweet = Vec::with_capacity(entries.len());
    let mut reply = Vec::with_capacity(entries.len());
    let mut fav = Vec::with_capacity(entries.len());
    let mut quote = Vec::with_capacity(entries.len());
    let mut bookmark = Vec::with_capacity(entries.len());
    let mut view = Vec::with_capacity(entries.len());
    let mut not_interested = Vec::with_capacity(entries.len());
    let mut report = Vec::with_capacity(entries.len());
    let mut block = Vec::with_capacity(entries.len());

    for (pid, val) in entries {
        post_ids.push(*pid);
        author_ids.push(val.author_id());

        match val {
            StoredValue::Full(record) => match record.as_ref() {
                IndexRecord::Metadata {
                    has_video: hv,
                    has_image: hi,
                    video_duration_ms: vd,
                    author_followers_count: af,
                    engagement: e,
                    ..
                } => {
                    has_video.push(*hv);
                    video_duration_ms.push(*vd);
                    has_image.push(*hi);
                    author_followers.push(*af);
                    retweet.push(e.retweet_count);
                    reply.push(e.reply_count);
                    fav.push(e.fav_count);
                    quote.push(e.quote_count);
                    bookmark.push(e.bookmark_count);
                    view.push(e.view_count);
                    not_interested.push(e.not_interested_in_count);
                    report.push(e.report_count);
                    block.push(e.block_count);
                }
                _ => {
                    has_video.push(false);
                    video_duration_ms.push(0);
                    has_image.push(false);
                    author_followers.push(0);
                    retweet.push(0);
                    reply.push(0);
                    fav.push(0);
                    quote.push(0);
                    bookmark.push(0);
                    view.push(0);
                    not_interested.push(0);
                    report.push(0);
                    block.push(0);
                }
            },
            StoredValue::Core(_) | StoredValue::Sid { .. } => {
                has_video.push(false);
                video_duration_ms.push(0);
                has_image.push(false);
                author_followers.push(0);
                retweet.push(0);
                reply.push(0);
                fav.push(0);
                quote.push(0);
                bookmark.push(0);
                view.push(0);
                not_interested.push(0);
                report.push(0);
                block.push(0);
            }
        }
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(post_ids)) as ArrayRef,
            Arc::new(Int64Array::from(author_ids)) as ArrayRef,
            Arc::new(BooleanArray::from(has_video)) as ArrayRef,
            Arc::new(Int64Array::from(video_duration_ms)) as ArrayRef,
            Arc::new(BooleanArray::from(has_image)) as ArrayRef,
            Arc::new(Int64Array::from(author_followers)) as ArrayRef,
            Arc::new(Int64Array::from(retweet)) as ArrayRef,
            Arc::new(Int64Array::from(reply)) as ArrayRef,
            Arc::new(Int64Array::from(fav)) as ArrayRef,
            Arc::new(Int64Array::from(quote)) as ArrayRef,
            Arc::new(Int64Array::from(bookmark)) as ArrayRef,
            Arc::new(Int64Array::from(view)) as ArrayRef,
            Arc::new(Int64Array::from(not_interested)) as ArrayRef,
            Arc::new(Int64Array::from(report)) as ArrayRef,
            Arc::new(Int64Array::from(block)) as ArrayRef,
        ],
    )?;
    Ok(batch)
}

fn load_post_ids_from_batch(batch: &RecordBatch, target: &mut WindowData, pipeline: &str) {
    use arrow::array::ListArray;

    let post_col = batch
        .column_by_name("post_id")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
    let author_col = batch
        .column_by_name("author_id")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

    let (Some(post_ids), Some(author_ids)) = (post_col, author_col) else {
        warn!("parquet file missing post_id or author_id column");
        return;
    };

    if is_sid_pipeline(pipeline) {
        let sid_col = batch
            .column_by_name("post_sid")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        for i in 0..batch.num_rows() {
            let post_id = post_ids.value(i);
            let author_id = author_ids.value(i);
            let post_sid = sid_col
                .map(|col| {
                    col.value(i)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .map(|a| a.values().iter().map(|&v| v as i32).collect::<Vec<_>>())
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            target.insert(
                post_id,
                StoredValue::Sid {
                    author_id,
                    post_sid,
                },
            );
        }
    } else if pipeline == "metadata" {
        let i64_col = |name: &str| {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        };
        let bool_col = |name: &str| {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        };
        let i64_at = |col: Option<&Int64Array>, i: usize| {
            col.filter(|c| c.is_valid(i))
                .map(|c| c.value(i))
                .unwrap_or(0)
        };
        let bool_at = |col: Option<&BooleanArray>, i: usize| {
            col.filter(|c| c.is_valid(i))
                .map(|c| c.value(i))
                .unwrap_or(false)
        };

        let has_video = bool_col("has_video");
        let has_image = bool_col("has_image");
        let video_duration_ms = i64_col("video_duration_ms");
        let author_followers_count = i64_col("author_followers_count");
        let retweet_count = i64_col("retweet_count");
        let reply_count = i64_col("reply_count");
        let fav_count = i64_col("fav_count");
        let quote_count = i64_col("quote_count");
        let bookmark_count = i64_col("bookmark_count");
        let view_count = i64_col("view_count");
        let not_interested_in_count = i64_col("not_interested_in_count");
        let report_count = i64_col("report_count");
        let block_count = i64_col("block_count");

        for i in 0..batch.num_rows() {
            let post_id = post_ids.value(i);
            let author_id = author_ids.value(i);
            target.insert(
                post_id,
                StoredValue::Full(Arc::new(IndexRecord::Metadata {
                    post_id,
                    author_id,
                    has_video: bool_at(has_video, i),
                    has_image: bool_at(has_image, i),
                    video_duration_ms: i64_at(video_duration_ms, i),
                    author_followers_count: i64_at(author_followers_count, i),
                    engagement: EngagementCounts {
                        retweet_count: i64_at(retweet_count, i),
                        reply_count: i64_at(reply_count, i),
                        fav_count: i64_at(fav_count, i),
                        quote_count: i64_at(quote_count, i),
                        bookmark_count: i64_at(bookmark_count, i),
                        view_count: i64_at(view_count, i),
                        not_interested_in_count: i64_at(not_interested_in_count, i),
                        report_count: i64_at(report_count, i),
                        block_count: i64_at(block_count, i),
                    },
                })),
            );
        }
    } else {
        for i in 0..batch.num_rows() {
            let post_id = post_ids.value(i);
            let author_id = author_ids.value(i);
            target.insert(post_id, StoredValue::Core(author_id));
        }
    }
}

fn empty_batch_for_window(window_name: &str, pipeline: &str) -> RecordBatch {
    if pipeline == "metadata" {
        empty_metadata_batch()
    } else if is_sid_pipeline(pipeline) {
        empty_sid_batch()
    } else if window_name.contains("topic") {
        empty_topic_batch()
    } else {
        empty_core_batch()
    }
}

fn empty_sid_batch() -> RecordBatch {
    use arrow::array::{Int64Builder, ListBuilder};
    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
        Field::new(
            "post_sid",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(ListBuilder::new(Int64Builder::new()).finish()) as ArrayRef,
        ],
    )
    .expect("empty sid batch creation cannot fail")
}

fn empty_core_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
        ],
    )
    .expect("empty batch creation cannot fail")
}

fn empty_topic_batch() -> RecordBatch {
    use arrow::array::{Int64Builder, ListBuilder, StringBuilder};
    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
        Field::new(
            "topic_entity_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        Field::new(
            "topic_names",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(ListBuilder::new(Int64Builder::new()).finish()) as ArrayRef,
            Arc::new(ListBuilder::new(StringBuilder::new()).finish()) as ArrayRef,
        ],
    )
    .expect("empty topic batch creation cannot fail")
}

fn empty_metadata_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("post_id", DataType::Int64, true),
        Field::new("author_id", DataType::Int64, true),
        Field::new("has_video", DataType::Boolean, true),
        Field::new("video_duration_ms", DataType::Int64, true),
        Field::new("has_image", DataType::Boolean, true),
        Field::new("author_followers_count", DataType::Int64, true),
        Field::new("retweet_count", DataType::Int64, true),
        Field::new("reply_count", DataType::Int64, true),
        Field::new("fav_count", DataType::Int64, true),
        Field::new("quote_count", DataType::Int64, true),
        Field::new("bookmark_count", DataType::Int64, true),
        Field::new("view_count", DataType::Int64, true),
        Field::new("not_interested_in_count", DataType::Int64, true),
        Field::new("report_count", DataType::Int64, true),
        Field::new("block_count", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(BooleanArray::from(Vec::<bool>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(BooleanArray::from(Vec::<bool>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
        ],
    )
    .expect("empty metadata batch creation cannot fail")
}


pub fn prefix_window_router() -> WindowRouter {
    Box::new(|record: &IndexRecord, windows: &[WindowConfig]| {
        let index_name = record.index_name();
        windows
            .iter()
            .filter(|w| w.name == index_name)
            .map(|w| w.window_name())
            .collect()
    })
}

pub fn metadata_window_router() -> WindowRouter {
    Box::new(|_record: &IndexRecord, windows: &[WindowConfig]| {
        windows.iter().map(|w| w.window_name()).collect()
    })
}

pub fn sid_window_router() -> WindowRouter {
    Box::new(|record: &IndexRecord, windows: &[WindowConfig]| {
        let index_name = record.index_name();
        windows
            .iter()
            .filter(|w| w.name == index_name)
            .map(|w| w.window_name())
            .collect()
    })
}

pub fn topic_window_router() -> WindowRouter {
    Box::new(|record: &IndexRecord, _windows: &[WindowConfig]| {
        let index_name = record.index_name();
        match index_name {
            "1fav" => vec!["1fav_1day".to_string()],
            "1fav_topic" => vec!["1fav_topic_1day".to_string()],
            "1fav_topic_option_1" => vec!["1fav_topic_option_1_1day".to_string()],
            "1fav_topic_option_2" => vec!["1fav_topic_option_2_1day".to_string()],
            "1fav_topic_option_3" => vec!["1fav_topic_option_3_1day".to_string()],
            "1fav_topic_option_4" => vec!["1fav_topic_option_4_1day".to_string()],
            "1fav_topic_option_5" => vec!["1fav_topic_option_5_1day".to_string()],
            _ => vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WindowConfig;
    use crate::processor::record::IndexRecord;
    use crate::store::snowflake::timestamp_secs_to_snowflake;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn make_metrics() -> Arc<PipelineMetrics> {
        Arc::new(PipelineMetrics::new(prometheus::Registry::new()).unwrap())
    }

    fn make_core_record(post_id: i64, author_id: i64, index_name: &str) -> IndexRecord {
        IndexRecord::Core {
            post_id,
            author_id,
            index_name: index_name.to_string(),
        }
    }

    fn make_metadata_record(post_id: i64, author_id: i64, fav_count: i64) -> IndexRecord {
        IndexRecord::Metadata {
            post_id,
            author_id,
            has_video: false,
            has_image: false,
            video_duration_ms: 0,
            author_followers_count: 0,
            engagement: crate::processor::record::EngagementCounts {
                fav_count,
                ..Default::default()
            },
        }
    }

    #[test]
    fn prefix_router_routes_to_matching_windows() {
        let windows = vec![
            WindowConfig::new("1fav", 24),
            WindowConfig::new("1fav", 48),
            WindowConfig::new("video", 48),
        ];
        let router = prefix_window_router();
        let record = make_core_record(1, 1, "1fav");
        let result = router(&record, &windows);
        assert_eq!(result, vec!["1fav_1day", "1fav_2day"]);
    }

    #[test]
    fn prefix_router_returns_empty_for_unknown_index() {
        let windows = vec![WindowConfig::new("1fav", 24)];
        let router = prefix_window_router();
        let record = make_core_record(1, 1, "unknown");
        let result = router(&record, &windows);
        assert!(result.is_empty());
    }

    #[test]
    fn metadata_router_routes_to_all_windows() {
        let windows = vec![
            WindowConfig::new("metadata", 24),
            WindowConfig::new("metadata", 48),
        ];
        let router = metadata_window_router();
        let record = IndexRecord::Metadata {
            post_id: 1,
            author_id: 1,
            has_video: false,
            has_image: false,
            video_duration_ms: 0,
            author_followers_count: 0,
            engagement: Default::default(),
        };
        let result = router(&record, &windows);
        assert_eq!(result, vec!["metadata_1day", "metadata_2day"]);
    }

    #[test]
    fn topic_router_explicit_mapping() {
        let windows = vec![];
        let router = topic_window_router();

        let r1 = make_core_record(1, 1, "1fav");
        assert_eq!(router(&r1, &windows), vec!["1fav_1day"]);

        let r2 = make_core_record(1, 1, "1fav_topic");
        assert_eq!(router(&r2, &windows), vec!["1fav_topic_1day"]);

        let r3 = make_core_record(1, 1, "unknown");
        assert!(router(&r3, &windows).is_empty());
    }

    #[test]
    fn stored_to_core_batch_roundtrip() {
        let entries: Vec<(i64, StoredValue)> =
            vec![(100, StoredValue::Core(10)), (200, StoredValue::Core(20))];
        let refs: Vec<(i64, &StoredValue)> = entries.iter().map(|(p, v)| (*p, v)).collect();
        let batch = stored_to_core_batch(&refs).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        let pids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(pids.values(), &[100, 200]);
        let aids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(aids.values(), &[10, 20]);
    }

    #[tokio::test]
    async fn store_add_and_dump() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("1fav", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        };

        let store = BaseSnapshotStore::new(config, prefix_window_router(), metrics);

        let now_secs = chrono::Utc::now().timestamp() as f64;
        let recent_post_id = timestamp_secs_to_snowflake(now_secs - 100.0);

        let records = vec![
            make_core_record(recent_post_id, 10, "1fav"),
            make_core_record(recent_post_id + 1, 20, "1fav"),
        ];

        store.add_batch(records).await.unwrap();

        store.dump_all_windows().await.unwrap();

        let symlink = dir.path().join("1fav_1day.parquet");
        assert!(symlink.exists(), "symlink should exist after dump");

        let batches = read_parquet_file(&symlink).unwrap();
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[tokio::test]
    async fn metadata_keeps_fav_when_later_event_has_zero_engagement() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("metadata", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "metadata".to_string(),
            sid_num_levels: 6,
        };
        let store = BaseSnapshotStore::new(config, metadata_window_router(), metrics);

        let now_secs = chrono::Utc::now().timestamp() as f64;
        let post_id = timestamp_secs_to_snowflake(now_secs - 100.0);

        store
            .add_batch(vec![make_metadata_record(post_id, 10, 50)])
            .await
            .unwrap();
        store.dump_all_windows().await.unwrap();
        store
            .add_batch(vec![make_metadata_record(post_id, 10, 0)])
            .await
            .unwrap();
        store.dump_all_windows().await.unwrap();

        let symlink = dir.path().join("metadata_1day.parquet");
        let batches = read_parquet_file(&symlink).unwrap();
        assert_eq!(batches[0].num_rows(), 1);
        let favs = batches[0]
            .column_by_name("fav_count")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(favs.value(0), 50);
    }

    #[tokio::test]
    async fn metadata_reload_preserves_fav() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("metadata", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "metadata".to_string(),
            sid_num_levels: 6,
        };
        let store1 = BaseSnapshotStore::new(config, metadata_window_router(), Arc::clone(&metrics));

        let now_secs = chrono::Utc::now().timestamp() as f64;
        let post_id = timestamp_secs_to_snowflake(now_secs - 100.0);
        store1
            .add_batch(vec![IndexRecord::Metadata {
                post_id,
                author_id: 10,
                has_video: false,
                has_image: true,
                video_duration_ms: 0,
                author_followers_count: 99,
                engagement: crate::processor::record::EngagementCounts {
                    fav_count: 50,
                    view_count: 200,
                    ..Default::default()
                },
            }])
            .await
            .unwrap();
        store1.dump_all_windows().await.unwrap();

        let config2 = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("metadata", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "metadata".to_string(),
            sid_num_levels: 6,
        };
        let store2 = BaseSnapshotStore::new(config2, metadata_window_router(), metrics);
        store2.load_existing_data().await.unwrap();
        store2
            .add_batch(vec![make_metadata_record(post_id, 10, 0)])
            .await
            .unwrap();
        store2.dump_all_windows().await.unwrap();

        let batches = read_parquet_file(&dir.path().join("metadata_1day.parquet")).unwrap();
        let col = |name: &str| {
            batches[0]
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        };
        assert_eq!(col("fav_count"), 50);
        assert_eq!(col("view_count"), 200);
        assert_eq!(col("author_followers_count"), 99);
    }

    #[tokio::test]
    async fn store_retention_filters_old_records() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("test", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        };

        let store = BaseSnapshotStore::new(config, prefix_window_router(), metrics);

        let now_secs = chrono::Utc::now().timestamp() as f64;
        let recent_id = timestamp_secs_to_snowflake(now_secs - 100.0);
        let old_id = timestamp_secs_to_snowflake(now_secs - 2.0 * 86400.0);

        let records = vec![
            make_core_record(recent_id, 10, "test"),
            make_core_record(old_id, 20, "test"),
        ];

        store.add_batch(records).await.unwrap();
        store.dump_all_windows().await.unwrap();

        let symlink = dir.path().join("test_1day.parquet");
        let batches = read_parquet_file(&symlink).unwrap();
        assert_eq!(
            batches[0].num_rows(),
            1,
            "old record should be filtered out"
        );
    }

    #[tokio::test]
    async fn store_loads_existing_data() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();

        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("1fav", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        };
        let store1 = BaseSnapshotStore::new(config, prefix_window_router(), Arc::clone(&metrics));

        let now_secs = chrono::Utc::now().timestamp() as f64;
        let pid = timestamp_secs_to_snowflake(now_secs - 100.0);
        store1
            .add_batch(vec![make_core_record(pid, 10, "1fav")])
            .await
            .unwrap();
        store1.dump_all_windows().await.unwrap();

        let config2 = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("1fav", 24)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        };
        let store2 = BaseSnapshotStore::new(config2, prefix_window_router(), metrics);
        store2.load_existing_data().await.unwrap();

        let state = store2.state.lock().await;
        let main = state.main_data.get("1fav_1day").unwrap();
        assert_eq!(main.len(), 1, "should have loaded 1 record from disk");
        assert!(main.contains_key(&pid));
    }

                #[tokio::test]
    async fn rare_event_arrives_to_previously_empty_window() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![
                WindowConfig::new("1fav", 24),
                WindowConfig::new("evergreen_video", 24 * 365 * 5),
            ],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        };

        let store = BaseSnapshotStore::new(config, prefix_window_router(), metrics);

        let now_secs = chrono::Utc::now().timestamp() as f64;
        let recent_id = timestamp_secs_to_snowflake(now_secs - 100.0);

        store
            .add_batch(vec![make_core_record(recent_id, 10, "1fav")])
            .await
            .unwrap();

        store.dump_all_windows().await.unwrap();

        let fav_path = dir.path().join("1fav_1day.parquet");
        let evergreen_path = dir.path().join("evergreen_video_1825day.parquet");
        assert!(fav_path.exists(), "1fav should exist");
        assert!(evergreen_path.exists(), "evergreen should exist (empty)");

        let evergreen_batches = read_parquet_file(&evergreen_path).unwrap();
        let evergreen_rows: usize = evergreen_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(evergreen_rows, 0, "evergreen should be empty");

        store.dump_all_windows().await.unwrap();

        let evergreen_id = timestamp_secs_to_snowflake(now_secs - 50.0);
        store
            .add_batch(vec![make_core_record(evergreen_id, 99, "evergreen_video")])
            .await
            .unwrap();

        store.dump_all_windows().await.unwrap();

        let evergreen_batches = read_parquet_file(&evergreen_path).unwrap();
        let total_rows: usize = evergreen_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 1,
            "evergreen should have 1 record after rare event"
        );

        let batch = &evergreen_batches[0];
        let pids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let aids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(pids.value(0), evergreen_id);
        assert_eq!(aids.value(0), 99);
    }

            #[tokio::test]
    async fn rare_events_accumulate_across_dumps() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let config = StoreConfig {
            output_dir: dir.path().to_path_buf(),
            windows: vec![WindowConfig::new("evergreen_video", 24 * 365 * 5)],
            compaction_interval_secs: 999,
            versions_to_keep: 3,
            pipeline: "main".to_string(),
            sid_num_levels: 6,
        };

        let store = BaseSnapshotStore::new(config, prefix_window_router(), metrics);
        let now_secs = chrono::Utc::now().timestamp() as f64;

        let id1 = timestamp_secs_to_snowflake(now_secs - 200.0);
        store
            .add_batch(vec![make_core_record(id1, 10, "evergreen_video")])
            .await
            .unwrap();
        store.dump_all_windows().await.unwrap();

        let id2 = timestamp_secs_to_snowflake(now_secs - 150.0);
        let id3 = timestamp_secs_to_snowflake(now_secs - 100.0);
        store
            .add_batch(vec![
                make_core_record(id2, 20, "evergreen_video"),
                make_core_record(id3, 30, "evergreen_video"),
            ])
            .await
            .unwrap();
        store.dump_all_windows().await.unwrap();

        let path = dir.path().join("evergreen_video_1825day.parquet");
        let batches = read_parquet_file(&path).unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3, "all 3 evergreen records should persist");

        let pids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(pids.value(0) < pids.value(1), "should be sorted by post_id");
        assert!(pids.value(1) < pids.value(2), "should be sorted by post_id");
    }


    fn make_sid_record(post_id: i64, author_id: i64, post_sid: Vec<i32>) -> IndexRecord {
        IndexRecord::Sid {
            post_id,
            author_id,
            index_name: "1fav".to_string(),
            post_sid,
        }
    }

    fn make_sid_store(dir: &TempDir) -> BaseSnapshotStore {
        BaseSnapshotStore::new(
            StoreConfig {
                output_dir: dir.path().to_path_buf(),
                windows: vec![WindowConfig::new("1fav", 24)],
                compaction_interval_secs: 999,
                versions_to_keep: 3,
                pipeline: "sid".to_string(),
                sid_num_levels: 6,
            },
            sid_window_router(),
            make_metrics(),
        )
    }

    #[test]
    fn sid_router_routes_only_matching_index_name() {
        let windows = vec![WindowConfig::new("1fav", 24)];
        let router = sid_window_router();
        let r1 = make_core_record(1, 1, "1fav");
        let r2 = make_core_record(2, 2, "video");
        let r3 = make_core_record(3, 3, "32fav");
        assert_eq!(router(&r1, &windows), vec!["1fav_1day"]);
        assert!(router(&r2, &windows).is_empty());
        assert!(router(&r3, &windows).is_empty());
    }

    #[tokio::test]
    async fn sid_get_missing_returns_only_empty_post_sid_rows() {
        let dir = TempDir::new().unwrap();
        let store = make_sid_store(&dir);
        store
            .add_batch(vec![
                make_sid_record(100, 10, vec![]),
                make_sid_record(200, 20, vec![1, 2, 3]),
                make_sid_record(300, 30, vec![]),
            ])
            .await
            .unwrap();
        let mut missing = store.get_missing_sid_post_ids(0.0).await;
        missing.sort();
        assert_eq!(missing, vec![100, 300]);
    }

    #[tokio::test]
    async fn sid_get_missing_filters_by_age() {
        let dir = TempDir::new().unwrap();
        let store = make_sid_store(&dir);
        let now_secs = chrono::Utc::now().timestamp() as f64;
        let young_pid = timestamp_secs_to_snowflake(now_secs - 60.0);
        let old_pid = timestamp_secs_to_snowflake(now_secs - 3600.0 * 5.0);
        store
            .add_batch(vec![
                make_sid_record(young_pid, 1, vec![]),
                make_sid_record(old_pid, 2, vec![]),
            ])
            .await
            .unwrap();
        let missing = store.get_missing_sid_post_ids(1.0).await;
        assert_eq!(missing, vec![young_pid]);
    }

    #[tokio::test]
    async fn sid_update_sids_patches_pending_and_main() {
        let dir = TempDir::new().unwrap();
        let store = make_sid_store(&dir);
        store
            .add_batch(vec![
                make_sid_record(100, 10, vec![]),
                make_sid_record(200, 20, vec![]),
            ])
            .await
            .unwrap();
        let mut patch = HashMap::new();
        patch.insert(100i64, vec![7, 8, 9]);
        let updated = store.update_sids(patch).await.unwrap();
        assert_eq!(updated, 1, "exactly one row patched");

        let mut missing = store.get_missing_sid_post_ids(0.0).await;
        missing.sort();
        assert_eq!(missing, vec![200]);
    }

    #[tokio::test]
    async fn sid_update_sids_skips_unknown_post_ids_silently() {
        let dir = TempDir::new().unwrap();
        let store = make_sid_store(&dir);
        store
            .add_batch(vec![make_sid_record(100, 10, vec![])])
            .await
            .unwrap();
        let mut patch = HashMap::new();
        patch.insert(999_i64, vec![1, 2, 3]);
        let updated = store.update_sids(patch).await.unwrap();
        assert_eq!(updated, 0, "unknown post_id silently skipped");
    }

    #[test]
    fn sid_writer_emits_sentinel_for_missing_rows() {
        use arrow::array::ListArray;

        let entries: Vec<(i64, StoredValue)> = vec![
            (
                100,
                StoredValue::Sid {
                    author_id: 10,
                    post_sid: vec![1, 2, 3, 4, 5, 6],
                },
            ),
            (
                200,
                StoredValue::Sid {
                    author_id: 20,
                    post_sid: vec![],
                },
            ),
            (
                300,
                StoredValue::Sid {
                    author_id: 30,
                    post_sid: vec![1, 2],
                },
            ),
        ];
        let refs: Vec<(i64, &StoredValue)> = entries.iter().map(|(p, v)| (*p, v)).collect();
        let batch = stored_to_sid_batch(&refs, 6).unwrap();

        assert_eq!(batch.num_rows(), 3);
        let sid_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        let row0 = sid_col.value(0);
        let row0 = row0.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(row0.values(), &[1, 2, 3, 4, 5, 6]);

        let row1 = sid_col.value(1);
        let row1 = row1.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(row1.values(), &[-1, -1, -1, -1, -1, -1]);

        let row2 = sid_col.value(2);
        let row2 = row2.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(row2.values(), &[-1, -1, -1, -1, -1, -1]);
    }

    #[tokio::test]
    async fn sid_get_missing_recognizes_sentinel_rows() {
        let dir = TempDir::new().unwrap();
        let store = make_sid_store(&dir);
        store
            .add_batch(vec![
                make_sid_record(100, 10, vec![]),
                make_sid_record(200, 20, vec![-1, -1, -1, -1, -1, -1]),
                make_sid_record(300, 30, vec![1, 2, 3, 4, 5, 6]),
            ])
            .await
            .unwrap();
        let mut missing = store.get_missing_sid_post_ids(0.0).await;
        missing.sort();
        assert_eq!(missing, vec![100, 200]);
    }

    #[tokio::test]
    async fn sid_parquet_roundtrip_preserves_sentinel_and_codes() {
        let dir = TempDir::new().unwrap();
        let metrics = make_metrics();
        let store = BaseSnapshotStore::new(
            StoreConfig {
                output_dir: dir.path().to_path_buf(),
                windows: vec![WindowConfig::new("1fav", 24)],
                compaction_interval_secs: 999,
                versions_to_keep: 3,
                pipeline: "sid".to_string(),
                sid_num_levels: 6,
            },
            sid_window_router(),
            Arc::clone(&metrics),
        );
        let now_secs = chrono::Utc::now().timestamp() as f64;
        let pid_a = timestamp_secs_to_snowflake(now_secs - 100.0);
        let pid_b = timestamp_secs_to_snowflake(now_secs - 50.0);
        store
            .add_batch(vec![
                make_sid_record(pid_a, 10, vec![]),
                make_sid_record(pid_b, 20, vec![1, 2, 3, 4, 5, 6]),
            ])
            .await
            .unwrap();
        store.dump_all_windows().await.unwrap();

        let store2 = BaseSnapshotStore::new(
            StoreConfig {
                output_dir: dir.path().to_path_buf(),
                windows: vec![WindowConfig::new("1fav", 24)],
                compaction_interval_secs: 999,
                versions_to_keep: 3,
                pipeline: "sid".to_string(),
                sid_num_levels: 6,
            },
            sid_window_router(),
            metrics,
        );
        store2.load_existing_data().await.unwrap();

        let mut missing = store2.get_missing_sid_post_ids(0.0).await;
        missing.sort();
        assert_eq!(missing, vec![pid_a]);

        let state = store2.state.lock().await;
        let main = state.main_data.get("1fav_1day").unwrap();
        let StoredValue::Sid { post_sid, .. } = main.get(&pid_b).unwrap() else {
            panic!("expected Sid variant");
        };
        assert_eq!(post_sid, &vec![1, 2, 3, 4, 5, 6]);
        let StoredValue::Sid { post_sid, .. } = main.get(&pid_a).unwrap() else {
            panic!("expected Sid variant");
        };
        assert_eq!(post_sid, &vec![-1, -1, -1, -1, -1, -1]);
    }
}
