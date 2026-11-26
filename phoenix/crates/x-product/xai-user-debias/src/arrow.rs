// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::error::Error;
use crate::features::{SequenceRows, VOCAB};
use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Int32Array, Int64Array, UInt64Array,
};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use std::io::Cursor;

pub const COL_IMPRESSED_TIME_MS: &str = "impressedTimeMs";
pub const COL_ACTION_NAME_MULTI_HOT: &str = "actionNameMultiHot";
pub const COL_PRODUCT_SURFACE: &str = "productSurface";
pub const COL_TIMEZONE_ID: &str = "timezoneId";
pub const COL_DWELL_TIME_MS: &str = "dwellTimeMs";
pub const COL_HAS_MEDIA: &str = "hasMedia";
pub const COL_MEDIA_CONTENT_TYPE: &str = "mediaContentType";
pub const COL_FIRST_VIDEO_DURATION_MS: &str = "firstVideoDurationMs";
pub const COL_IS_AUTHOR_FOLLOWED_BY_VIEWER: &str = "isAuthorFollowedByViewer";
pub const COL_TWEET_POSITION: &str = "tweetPosition";

pub fn parse_sequence(ipc_bytes: &[u8]) -> Result<SequenceRows, Error> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)?;
    let mut rows = SequenceRows::default();
    for batch in reader {
        append_batch(&mut rows, &batch?)?;
    }
    Ok(rows)
}

pub fn append_batch(rows: &mut SequenceRows, batch: &RecordBatch) -> Result<(), Error> {
    rows.ts_ms.extend(i64_values(batch, COL_IMPRESSED_TIME_MS)?);
    rows.acts.extend(action_masks(batch)?);
    rows.surface.extend(i32_values(batch, COL_PRODUCT_SURFACE)?);
    rows.tz.extend(i32_values(batch, COL_TIMEZONE_ID)?);
    rows.dwell_sec.extend(
        i64_values(batch, COL_DWELL_TIME_MS)?
            .into_iter()
            .map(|ms| ms as f32 / 1000.0),
    );
    rows.has_media.extend(bool_values(batch, COL_HAS_MEDIA)?);
    rows.media_type
        .extend(i32_values(batch, COL_MEDIA_CONTENT_TYPE)?);
    rows.video_dur_ms
        .extend(i64_values(batch, COL_FIRST_VIDEO_DURATION_MS)?);
    rows.in_network
        .extend(bool_values(batch, COL_IS_AUTHOR_FOLLOWED_BY_VIEWER)?);
    rows.position.extend(i32_values(batch, COL_TWEET_POSITION)?);
    Ok(())
}

fn column<'a>(batch: &'a RecordBatch, name: &'static str) -> Result<&'a ArrayRef, Error> {
    batch.column_by_name(name).ok_or(Error::MissingColumn(name))
}

fn cast_column(
    batch: &RecordBatch,
    name: &'static str,
    data_type: &DataType,
) -> Result<ArrayRef, Error> {
    let col = column(batch, name)?;
    if col.data_type() == data_type {
        return Ok(col.clone());
    }
    cast(col.as_ref(), data_type).map_err(|_| Error::UnsupportedColumnType {
        name,
        data_type: col.data_type().to_string(),
    })
}

fn i64_values(batch: &RecordBatch, name: &'static str) -> Result<Vec<i64>, Error> {
    let col = cast_column(batch, name, &DataType::Int64)?;
    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
    Ok((0..arr.len())
        .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
        .collect())
}

fn i32_values(batch: &RecordBatch, name: &'static str) -> Result<Vec<i32>, Error> {
    let col = cast_column(batch, name, &DataType::Int32)?;
    let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
    Ok((0..arr.len())
        .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
        .collect())
}

fn bool_values(batch: &RecordBatch, name: &'static str) -> Result<Vec<bool>, Error> {
    let col = cast_column(batch, name, &DataType::Boolean)?;
    let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
    Ok((0..arr.len())
        .map(|i| arr.is_valid(i) && arr.value(i))
        .collect())
}

fn action_masks(batch: &RecordBatch) -> Result<Vec<u64>, Error> {
    let col = column(batch, COL_ACTION_NAME_MULTI_HOT)?;
    let unsupported = || Error::UnsupportedColumnType {
        name: COL_ACTION_NAME_MULTI_HOT,
        data_type: col.data_type().to_string(),
    };
    match col.data_type() {
        DataType::FixedSizeList(field, _) if *field.data_type() == DataType::Boolean => {
            let list = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let hot = list
                .values()
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(unsupported)?;
            let vocab = list.value_length() as usize;
            let tracked = vocab.min(VOCAB);
            Ok((0..list.len())
                .map(|row| {
                    if !list.is_valid(row) {
                        return 0;
                    }
                    let base = row * vocab;
                    (0..tracked)
                        .filter(|&i| hot.is_valid(base + i) && hot.value(base + i))
                        .fold(0u64, |mask, i| mask | (1u64 << i))
                })
                .collect())
        }
        DataType::UInt64 => {
            let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            Ok((0..arr.len())
                .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                .collect())
        }
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_valid(i) {
                        arr.value(i) as u64
                    } else {
                        0
                    }
                })
                .collect())
        }
        _ => Err(unsupported()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int16Array, Int32Array, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use std::sync::Arc;

    #[test]
    fn parses_served_aggregated_schema_columns() {
        let list_field = Arc::new(Field::new("item", DataType::Boolean, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_IMPRESSED_TIME_MS, DataType::Int64, false),
            Field::new(COL_DWELL_TIME_MS, DataType::Int64, true),
            Field::new(COL_FIRST_VIDEO_DURATION_MS, DataType::Int64, true),
            Field::new(COL_TIMEZONE_ID, DataType::Int32, true),
            Field::new(COL_PRODUCT_SURFACE, DataType::Int32, true),
            Field::new(COL_HAS_MEDIA, DataType::Boolean, true),
            Field::new(COL_IS_AUTHOR_FOLLOWED_BY_VIEWER, DataType::Boolean, true),
            Field::new(COL_TWEET_POSITION, DataType::Int32, true),
            Field::new(COL_MEDIA_CONTENT_TYPE, DataType::Int16, true),
            Field::new(
                COL_ACTION_NAME_MULTI_HOT,
                DataType::FixedSizeList(list_field.clone(), VOCAB as i32),
                true,
            ),
        ]));
        let mut hot = vec![false; 3 * VOCAB];
        hot[1] = true;
        hot[13] = true;
        hot[VOCAB + 29] = true;
        hot[2 * VOCAB + 63] = true;
        let multi_hot = FixedSizeListArray::new(
            list_field,
            VOCAB as i32,
            Arc::new(BooleanArray::from(hot)),
            None,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1_000i64, 2_000, 3_000])),
                Arc::new(Int64Array::from(vec![Some(1500i64), None, Some(250)])),
                Arc::new(Int64Array::from(vec![Some(0i64), Some(30_000), None])),
                Arc::new(Int32Array::from(vec![Some(21), None, Some(3)])),
                Arc::new(Int32Array::from(vec![Some(1), Some(10), None])),
                Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
                Arc::new(BooleanArray::from(vec![Some(false), Some(true), None])),
                Arc::new(Int32Array::from(vec![Some(4), Some(40), None])),
                Arc::new(Int16Array::from(vec![Some(2i16), None, Some(1)])),
                Arc::new(multi_hot),
            ],
        )
        .unwrap();
        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let rows = parse_sequence(&buf).unwrap();
        assert_eq!(
            rows,
            SequenceRows {
                acts: vec![(1 << 1) | (1 << 13), 1 << 29, 1 << 63],
                ts_ms: vec![1_000, 2_000, 3_000],
                surface: vec![1, 10, 0],
                tz: vec![21, 0, 3],
                dwell_sec: vec![1.5, 0.0, 0.25],
                has_media: vec![true, false, false],
                media_type: vec![2, 0, 1],
                video_dur_ms: vec![0, 30_000, 0],
                in_network: vec![false, true, false],
                position: vec![4, 40, 0],
            }
        );
    }
}
