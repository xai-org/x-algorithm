// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::stream::{self, StreamExt};
use half::f16;

use crate::mm_embedding_client::MmEmbeddingsClient;
use crate::mm_embedding_fetcher::MmEmbeddingFetcher;
use crate::o2::O2Client;
use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayRef, AsArray, Float32Builder, LargeListArray, LargeListBuilder, StringArray,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use lazy_static::lazy_static;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use prometheus::{IntCounter, IntGauge, register_int_counter, register_int_gauge};

static O2_DOWNLOAD_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);

lazy_static! {
    pub static ref SNAPSHOT_UPLOAD_SUCCESS: IntCounter = register_int_counter!(
        "mm_embedding_snapshot_upload_success_total",
        "Total number of successful snapshot uploads to O2"
    )
    .unwrap();
    pub static ref SNAPSHOT_UPLOAD_FAILURE: IntCounter = register_int_counter!(
        "mm_embedding_snapshot_upload_failure_total",
        "Total number of failed snapshot uploads to O2"
    )
    .unwrap();
    pub static ref SNAPSHOT_EMBEDDINGS_COUNT: IntGauge = register_int_gauge!(
        "mm_embedding_snapshot_embeddings_count",
        "Number of embeddings in the current snapshot"
    )
    .unwrap();
    pub static ref SNAPSHOT_READ_SUCCESS: IntCounter = register_int_counter!(
        "mm_embedding_snapshot_read_success_total",
        "Total number of successful snapshot reads from O2"
    )
    .unwrap();
    pub static ref SNAPSHOT_READ_FAILURE: IntCounter = register_int_counter!(
        "mm_embedding_snapshot_read_failure_total",
        "Total number of failed snapshot reads from O2"
    )
    .unwrap();
    pub static ref SNAPSHOT_READ_EMBEDDINGS_TOTAL: IntCounter = register_int_counter!(
        "mm_embedding_snapshot_read_embeddings_total",
        "Total number of embeddings read from O2 snapshots"
    )
    .unwrap();
}

pub struct EmbeddingRecord {
    pub post_id: u64,
    pub embedding: Vec<f16>,
}

#[derive(Clone)]
pub struct EmbeddingStore {
    inner: Arc<RwLock<Vec<EmbeddingRecord>>>,
    version: String,
}

impl EmbeddingStore {
    pub fn new(version: String) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Vec::new())),
            version,
        }
    }

    pub fn insert(&self, post_id: u64, embedding: Vec<f32>) {
        let embedding_f16: Vec<f16> = embedding.iter().map(|&v| f16::from_f32(v)).collect();
        let mut store = self.inner.write().unwrap();
        store.push(EmbeddingRecord {
            post_id,
            embedding: embedding_f16,
        });
        SNAPSHOT_EMBEDDINGS_COUNT.set(store.len() as i64);
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn take_snapshot(&self) -> Vec<EmbeddingRecord> {
        let mut store = self.inner.write().unwrap();
        let records = store.drain(..).collect();
        SNAPSHOT_EMBEDDINGS_COUNT.set(0);
        records
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().is_empty()
    }
}

pub const SNAPSHOT_ROWS_PER_GROUP: usize = 4_096;

pub fn write_embeddings_to_parquet(records: &[EmbeddingRecord], version: &str) -> Result<Bytes> {
    let (bytes, _) = write_embedding_rows_to_parquet(
        records.iter().map(|r| (r.post_id, r.embedding.as_slice())),
        version,
        SNAPSHOT_ROWS_PER_GROUP,
    )?;
    Ok(bytes)
}

pub fn write_embedding_rows_to_parquet<'a>(
    rows: impl IntoIterator<Item = (u64, &'a [f16])>,
    version: &str,
    rows_per_group: usize,
) -> Result<(Bytes, usize)> {
    let mut rows = rows.into_iter().peekable();
    let Some(&(_, first)) = rows.peek() else {
        log::error!("No embeddings to write");
        return Ok((Bytes::new(), 0));
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("postID", DataType::UInt64, false),
        Field::new("version", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
    ]));
    let max_rows = rows.size_hint().1.unwrap_or(rows_per_group);
    let buffer = Vec::with_capacity(max_rows.saturating_mul(first.len() * 4) + (1 << 20));
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .set_max_row_group_row_count(Some(rows_per_group.max(1)))
        .build();
    let mut writer = ArrowWriter::try_new(buffer, schema.clone(), Some(props))
        .context("Failed to create parquet writer")?;

    let mut embedding_builder = LargeListBuilder::new(Float32Builder::new());
    let mut written = 0usize;
    loop {
        let mut post_ids = Vec::with_capacity(rows_per_group);
        for (post_id, embedding) in rows.by_ref().take(rows_per_group) {
            post_ids.push(post_id);
            let values_builder = embedding_builder.values();
            for &val in embedding {
                values_builder.append_value(val.to_f32());
            }
            embedding_builder.append(true);
        }
        if post_ids.is_empty() {
            break;
        }
        let n = post_ids.len();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(post_ids)) as ArrayRef,
                Arc::new(StringArray::from(vec![version; n])) as ArrayRef,
                Arc::new(embedding_builder.finish()) as ArrayRef,
            ],
        )
        .context("Failed to create record batch")?;
        writer
            .write(&batch)
            .context("Failed to write record batch")?;
        written += n;
    }

    let buffer = writer
        .into_inner()
        .context("Failed to close parquet writer")?;
    Ok((Bytes::from(buffer), written))
}

pub async fn dump_snapshot(store: &EmbeddingStore, o2_client: &O2Client, dst_dir: &str) {
    let records = store.take_snapshot();
    if records.is_empty() {
        log::warn!("No embeddings to snapshot, skipping upload");
        return;
    }

    let snapshot_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let version = store.version().to_string();
    let record_count = records.len();

    log::info!(
        "Creating snapshot {} with {} embeddings",
        snapshot_id,
        record_count
    );

    let parquet_bytes =
        match tokio::task::spawn_blocking(move || write_embeddings_to_parquet(&records, &version))
            .await
        {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(e)) => {
                log::error!(
                    "Failed to write snapshot {} to parquet: {:#}",
                    snapshot_id,
                    e
                );
                SNAPSHOT_UPLOAD_FAILURE.inc();
                return;
            }
            Err(e) => {
                log::error!(
                    "Parquet writing task panicked for snapshot {}: {}",
                    snapshot_id,
                    e
                );
                SNAPSHOT_UPLOAD_FAILURE.inc();
                return;
            }
        };

    let parquet_size_mb = parquet_bytes.len() as f64 / (1024.0 * 1024.0);
    log::info!(
        "Snapshot {} parquet size: {:.2} MB",
        snapshot_id,
        parquet_size_mb
    );

    let key = format!("{}/snapshot_{}", dst_dir, snapshot_id);
    match o2_client.put(&key, parquet_bytes).await {
        Ok(()) => {
            log::info!(
                "Successfully uploaded snapshot {} with {} embeddings to bucket {} with prefix {}/{}",
                snapshot_id,
                record_count,
                o2_client.bucket(),
                o2_client.prefix(),
                dst_dir,
            );
            SNAPSHOT_UPLOAD_SUCCESS.inc();
        }
        Err(e) => {
            log::error!("Failed to upload snapshot {} to O2: {:#}", snapshot_id, e);
            SNAPSHOT_UPLOAD_FAILURE.inc();
        }
    }
}

pub fn read_embeddings_from_parquet(data: Bytes, file_name: &str) -> Result<Vec<EmbeddingRecord>> {
    if data.is_empty() {
        return Ok(vec![]);
    }

    let parse_start = Instant::now();

    let reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .context("Failed to create parquet reader")?
        .build()
        .context("Failed to build parquet reader")?;

    let mut embeddings = Vec::new();

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

        let post_id_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("Failed to get postID column as UInt64Array")?;

        let embedding_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<LargeListArray>()
            .context("Failed to get embedding column as LargeListArray")?;

        for i in 0..batch.num_rows() {
            let post_id = post_id_array.value(i);

            let embedding_list = embedding_col.value(i);
            let embedding: Vec<f16> = embedding_list
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect();

            embeddings.push(EmbeddingRecord { post_id, embedding });
        }
    }

    log::debug!(
        "Parsed parquet file: {} ({} embeddings) in {:?}",
        file_name,
        embeddings.len(),
        parse_start.elapsed()
    );

    Ok(embeddings)
}

pub async fn read_embeddings_from_o2(
    mm_fetcher: &MmEmbeddingFetcher,
    path: &object_store::path::Path,
) -> Result<Vec<EmbeddingRecord>> {
    let download_start = Instant::now();
    let data = mm_fetcher
        .get_raw(path)
        .await
        .with_context(|| format!("Failed to read file from O2: {}", path))?;

    let download_count = O2_DOWNLOAD_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    if download_count.is_multiple_of(10) {
        log::info!(
            "Downloaded file from O2: {} ({:.1} MB) in {:?}",
            path,
            data.len() as f64 / (1024.0 * 1024.0),
            download_start.elapsed()
        );
    }

    let path_str = path.to_string();
    let path_str_clone = path_str.clone();
    tokio::task::spawn_blocking(move || read_embeddings_from_parquet(data, &path_str_clone))
        .await
        .with_context(|| format!("Parquet parsing task panicked for O2 file: {}", path_str))?
        .with_context(|| format!("Failed to parse parquet from O2 file: {}", path_str))
}

pub fn start_o2_reader_task(mm_fetcher: MmEmbeddingFetcher, mm_client: Arc<MmEmbeddingsClient>) {
    tokio::spawn(async move {
        log::info!(
            "O2 reader task started: watching bucket '{}' with prefix '{}'",
            mm_fetcher.bucket(),
            mm_fetcher.prefix(),
        );

        if let Err(e) = fetch_recent_embeddings_from_o2(mm_fetcher, mm_client).await {
            log::error!("Failed to fetch embeddings from O2: {:#}", e);
        }
    });
}

pub fn snapshot_time(obj: &object_store::ObjectMeta) -> DateTime<Utc> {
    let path: &str = obj.location.as_ref();
    let ts = path
        .rsplit('/')
        .next()
        .and_then(|name| name.strip_prefix("snapshot_"))
        .and_then(|s| s.parse::<i64>().ok())
        .and_then(|ts| DateTime::from_timestamp(ts, 0));
    ts.unwrap_or(obj.last_modified)
}

pub async fn list_recent_embedding_objects(
    mm_fetcher: &MmEmbeddingFetcher,
    cutoff_time: DateTime<Utc>,
) -> Result<Vec<object_store::ObjectMeta>> {
    let (objects_minutes, objects_hourly) =
        tokio::join!(mm_fetcher.list("embeddings"), mm_fetcher.list("hourly"));
    let objects_minutes = objects_minutes.context("Failed to list O2 objects")?;
    let objects_hourly = objects_hourly.context("Failed to list O2 objects")?;

    let mut recent_hourly_objects: Vec<_> = objects_hourly
        .into_iter()
        .filter(|obj| snapshot_time(obj) >= cutoff_time)
        .collect();

    let newest_hourly_data_time = recent_hourly_objects
        .iter()
        .map(snapshot_time)
        .max()
        .unwrap_or(cutoff_time);

    let mut recent_minute_objects: Vec<_> = objects_minutes
        .into_iter()
        .filter(|obj| snapshot_time(obj) > newest_hourly_data_time)
        .collect();

    recent_hourly_objects.sort_by_key(|o| std::cmp::Reverse(snapshot_time(o)));
    recent_minute_objects.sort_by_key(|o| std::cmp::Reverse(snapshot_time(o)));
    let mut recent_objects = recent_minute_objects;
    recent_objects.extend(recent_hourly_objects);

    log::info!(
        "Found {} recent snapshot files (hourly cutoff: {:?})",
        recent_objects.len(),
        newest_hourly_data_time,
    );

    Ok(recent_objects)
}

const DEFAULT_O2_FETCH_CONCURRENCY: usize = 24;

fn o2_fetch_concurrency() -> usize {
    match std::env::var("MM_O2_FETCH_CONCURRENCY") {
        Err(_) => DEFAULT_O2_FETCH_CONCURRENCY,
        Ok(raw) => parse_o2_fetch_concurrency(&raw),
    }
}

fn parse_o2_fetch_concurrency(raw: &str) -> usize {
    match raw.trim().parse::<usize>() {
        Ok(n) if n >= 1 => n,
        _ => {
            log::warn!(
                "Ignoring invalid MM_O2_FETCH_CONCURRENCY={:?} (must be an integer >= 1); \
                 using default {}",
                raw,
                DEFAULT_O2_FETCH_CONCURRENCY
            );
            DEFAULT_O2_FETCH_CONCURRENCY
        }
    }
}

pub async fn fetch_recent_embeddings_from_o2(
    mm_fetcher: MmEmbeddingFetcher,
    mm_client: Arc<MmEmbeddingsClient>,
) -> Result<()> {
    let method_start = Instant::now();
    let two_days_ago: DateTime<Utc> = Utc::now() - chrono::Duration::days(2);

    log::info!(
        "Fetching embeddings from O2 bucket '{}' with prefix '{}', modified after {}",
        mm_fetcher.bucket(),
        mm_fetcher.prefix(),
        two_days_ago
    );

    let recent_objects = list_recent_embedding_objects(&mm_fetcher, two_days_ago).await?;

    let mut last_processed_time: DateTime<Utc> = two_days_ago;

    let total_files = recent_objects.len();
    let concurrency = o2_fetch_concurrency();
    log::info!(
        "Preloading {} O2 embedding file(s) with fetch concurrency {} (MM_O2_FETCH_CONCURRENCY)",
        total_files,
        concurrency
    );
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));

    let mm_fetcher_dl = mm_fetcher.clone();

    let results = stream::iter(recent_objects.into_iter().rev().map(move |obj| {
        let snap_time = snapshot_time(&obj);
        let path_str = obj.location.to_string();
        let location = obj.location;
        let size = obj.size;
        let mm_fetcher = mm_fetcher_dl.clone();
        let semaphore_clone = semaphore.clone();

        async move {
            let _permit = semaphore_clone
                .acquire_owned()
                .await
                .expect("semaphore closed");
            log::debug!(
                "Reading file: {} (size: {:.1} MB, snapshot_time: {:?})",
                path_str,
                size as f64 / (1024.0 * 1024.0),
                snap_time
            );

            let result = read_embeddings_from_o2(&mm_fetcher, &location).await;
            (path_str, snap_time, result)
        }
    }))
    .buffered(concurrency);

    let (tx, mut rx) = tokio::sync::mpsc::channel(concurrency);

    let download_task = tokio::spawn(async move {
        futures::pin_mut!(results);
        while let Some(item) = results.next().await {
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });

    let mut processed_count = 0usize;
    while let Some((path_str, snap_time, result)) = rx.recv().await {
        match result {
            Ok(embeddings) => {
                SNAPSHOT_READ_SUCCESS.inc();
                SNAPSHOT_READ_EMBEDDINGS_TOTAL.inc_by(embeddings.len() as u64);
                let embeddings_num = embeddings.len();
                mm_client.preload_embeddings(embeddings, true).await;
                processed_count += 1;
                log::info!(
                    "{}/{} Loaded {} embeddings from {}",
                    processed_count,
                    total_files,
                    embeddings_num,
                    path_str
                );
            }
            Err(e) => {
                log::error!("Failed to read embeddings from {}: {:#}", path_str, e);
                SNAPSHOT_READ_FAILURE.inc();
            }
        }
        last_processed_time = last_processed_time.max(snap_time);
    }

    if let Err(e) = download_task.await {
        log::error!("Download task panicked: {:#}", e);
    }

    let mm_fetcher = mm_fetcher.clone();
    let mm_client = mm_client.clone();

    tokio::spawn(async move {
        let poll_interval = Duration::from_secs(30);
        log::info!("O2 watcher task started: polling every {:?}", poll_interval);

        loop {
            tokio::time::sleep(poll_interval).await;

            let mut new_last_processed_time = last_processed_time;

            match mm_fetcher.list("embeddings").await {
                Ok(objects) => {
                    for obj in objects.into_iter().rev() {
                        let snap_time = snapshot_time(&obj);
                        let path_str = obj.location.to_string();

                        if snap_time <= last_processed_time {
                            continue;
                        }

                        log::debug!(
                            "Found new file: {} (size: {} bytes, snapshot_time: {:?})",
                            path_str,
                            obj.size,
                            snap_time
                        );

                        match mm_fetcher.get_raw(&obj.location).await {
                            Ok(data) => {
                                let client = mm_client.clone();
                                let parse_path = path_str.clone();
                                match tokio::task::spawn_blocking(move || {
                                    client.ingest_parquet_bytes_sync(data, &parse_path)
                                })
                                .await
                                {
                                    Ok(Ok(count)) => {
                                        log::info!(
                                            "Loaded {} new embeddings from {}",
                                            count,
                                            path_str
                                        );
                                        SNAPSHOT_READ_SUCCESS.inc();
                                        SNAPSHOT_READ_EMBEDDINGS_TOTAL.inc_by(count as u64);
                                    }
                                    Ok(Err(e)) => {
                                        log::error!(
                                            "Failed to parse/insert embeddings from {}: {e:#}",
                                            path_str
                                        );
                                        SNAPSHOT_READ_FAILURE.inc();
                                    }
                                    Err(e) => {
                                        log::error!(
                                            "Blocking ingest panicked for {}: {e:#}",
                                            path_str
                                        );
                                        SNAPSHOT_READ_FAILURE.inc();
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to read embeddings from {}: {e:#}", path_str);
                                SNAPSHOT_READ_FAILURE.inc();
                            }
                        }

                        new_last_processed_time = new_last_processed_time.max(snap_time);
                    }
                }
                Err(e) => {
                    log::error!("Failed to list O2 objects: {e:#}");
                }
            }

            last_processed_time = new_last_processed_time;
        }
    });

    log::info!(
        "fetch_recent_embeddings_from_o2 finished initial load ({} files) in {:?}",
        total_files,
        method_start.elapsed()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::ObjectMeta;
    use object_store::path::Path as ObjectPath;

    fn make_obj(path: &str, last_modified: DateTime<Utc>) -> ObjectMeta {
        ObjectMeta {
            location: ObjectPath::from(path),
            last_modified,
            size: 0,
            e_tag: None,
            version: None,
        }
    }

    #[test]
    fn snapshot_time_parses_filename() {
        let ts = 1_719_500_000i64;
        let obj = make_obj(&format!("hourly/snapshot_{ts}"), Utc::now());
        assert_eq!(
            snapshot_time(&obj),
            DateTime::from_timestamp(ts, 0).unwrap()
        );
    }

    #[test]
    fn snapshot_time_falls_back_to_last_modified() {
        let now = Utc::now();
        let obj = make_obj("some/other_file.parquet", now);
        assert_eq!(snapshot_time(&obj), now);
    }

    #[test]
    fn snapshot_time_handles_nested_path() {
        let ts = 1_700_000_000i64;
        let obj = make_obj(&format!("prefix/embeddings/snapshot_{ts}"), Utc::now());
        assert_eq!(
            snapshot_time(&obj),
            DateTime::from_timestamp(ts, 0).unwrap()
        );
    }

    #[test]
    fn o2_fetch_concurrency_parses_valid_values() {
        assert_eq!(parse_o2_fetch_concurrency("8"), 8);
        assert_eq!(parse_o2_fetch_concurrency("  16 "), 16);
        assert_eq!(parse_o2_fetch_concurrency("1"), 1);
    }

    #[test]
    fn o2_fetch_concurrency_falls_back_on_invalid_values() {
        assert_eq!(
            parse_o2_fetch_concurrency("0"),
            DEFAULT_O2_FETCH_CONCURRENCY
        );
        assert_eq!(
            parse_o2_fetch_concurrency("-1"),
            DEFAULT_O2_FETCH_CONCURRENCY
        );
        assert_eq!(
            parse_o2_fetch_concurrency("abc"),
            DEFAULT_O2_FETCH_CONCURRENCY
        );
        assert_eq!(parse_o2_fetch_concurrency(""), DEFAULT_O2_FETCH_CONCURRENCY);
    }

    fn record(post_id: u64, values: &[f32]) -> EmbeddingRecord {
        EmbeddingRecord {
            post_id,
            embedding: values.iter().map(|&v| f16::from_f32(v)).collect(),
        }
    }

    #[test]
    fn chunked_parquet_round_trips_through_the_engine_reader() {
        let records: Vec<_> = (1..=5u64)
            .map(|id| record(id, &[id as f32, -0.5, 0.25]))
            .collect();
        let (bytes, written) = write_embedding_rows_to_parquet(
            records.iter().map(|r| (r.post_id, r.embedding.as_slice())),
            "v8_2",
            2,
        )
        .unwrap();
        assert_eq!(written, 5);

        let meta = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        assert_eq!(meta.num_row_groups(), 3);

        let read = read_embeddings_from_parquet(bytes, "test").unwrap();
        let got: Vec<(u64, Vec<f16>)> =
            read.into_iter().map(|r| (r.post_id, r.embedding)).collect();
        let want: Vec<(u64, Vec<f16>)> = records
            .into_iter()
            .map(|r| (r.post_id, r.embedding))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn chunked_parquet_matches_the_single_batch_schema() {
        let records = vec![record(7, &[1.0, 2.0])];
        let bytes = write_embeddings_to_parquet(&records, "v5_1").unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        let schema = reader.schema().clone();
        let fields: Vec<(&str, &DataType, bool)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type(), f.is_nullable()))
            .collect();
        let list = DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true)));
        assert_eq!(
            fields,
            vec![
                ("postID", &DataType::UInt64, false),
                ("version", &DataType::Utf8, false),
                ("embedding", &list, false),
            ]
        );
        let batch = reader.build().unwrap().next().unwrap().unwrap();
        assert_eq!(batch.column(1).as_string::<i32>().value(0), "v5_1");
    }

    #[test]
    fn chunked_parquet_writes_nothing_for_no_rows() {
        let (bytes, written) =
            write_embedding_rows_to_parquet(std::iter::empty(), "v8_2", 2).unwrap();
        assert_eq!((bytes.len(), written), (0, 0));
    }
}
