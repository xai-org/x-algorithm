// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use arrow::array::{Array, AsArray, BooleanArray, Float32Array, Int32Array, Int64Array};
use arrow::ipc::reader::StreamReader;
use bytes::Bytes;
use std::io::Cursor;
use std::time::Instant;
use tokio::sync::oneshot;
use tonic::Status;
use xai_recsys::{
    model_config::ModelConfig,
    util::{
        InputBuffer, PRODUCT_SURFACE_CATEGORICAL_IDX, compute_user_ip_hashes, parse_ipv4_to_u32,
        record_sid_coverage, stamp_i32_as_categorical, stamp_local_time_features,
        stamp_semantic_ids,
    },
};
use xai_recsys_proto as pb;

pub struct RetrievalInputBuffer {
    pub user_hashes: Vec<i32>,
    pub user_ip_hashes: Vec<i32>,
    pub history_post_hashes: Vec<i32>,
    pub history_auth_hashes: Vec<i32>,
    pub history_actions: Vec<bool>,
    pub history_continuous_actions: Vec<f32>,
    pub history_product_surfaces: Vec<i32>,
    pub user_categorical_features: Vec<i16>,
    pub user_bool_features: Vec<bool>,
    pub user_float_features: Vec<f32>,
    pub user_int64_features: Vec<i64>,
    pub user_installed_apps: Vec<bool>,
    pub history_categorical_features: Vec<i16>,
    pub history_bool_features: Vec<bool>,
    pub history_float_features: Vec<f32>,
    pub history_int64_features: Vec<i64>,
    pub history_post_ids: Vec<i64>,
    pub history_semantic_ids: Vec<u16>,
}

impl RetrievalInputBuffer {
    pub fn compute_for_item(
        model_config: &ModelConfig,
        sequence: &Option<pb::UserActionSequence>,
        ip_address: &str,
        client_context: Option<&pb::ClientContext>,
        user_context: Option<&pb::UserContext>,
    ) -> Self {
        use xai_recsys_proto::user_action_sequence_data_container::Data;

        let num_user_hashes = model_config.hash_table.num_user_hashes();
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let history_seq_len = model_config.history_seq_len;
        let output_vocab_size = model_config.hash_table.output_vocab_size;
        let num_continuous_actions = model_config.hash_table.num_continuous_actions;

        let mut user_hashes = vec![0i32; num_user_hashes];
        let mut history_post_hashes = vec![0i32; history_seq_len * num_item_hashes];
        let mut history_auth_hashes = vec![0i32; history_seq_len * num_author_hashes];
        let mut history_actions = vec![false; history_seq_len * output_vocab_size];
        let mut history_continuous_actions = vec![0.0f32; history_seq_len * num_continuous_actions];
        let mut history_product_surfaces = vec![0i32; history_seq_len];
        let mut history_impr_ts = vec![0i32; history_seq_len];
        let mut history_tz_enums = vec![0i16; history_seq_len];
        let mut history_post_ids = vec![0i64; history_seq_len];
        let sid_num_levels = model_config.sid_num_levels;
        let mut history_semantic_ids = vec![0u16; history_seq_len * sid_num_levels];

        let user_id = match sequence {
            Some(sequence) => sequence.user_id as i64,
            None => 0,
        };
        for (k, slot) in user_hashes.iter_mut().enumerate().take(num_user_hashes) {
            *slot = model_config.hash_table.hash_user_id(user_id, k);
        }

        let request_ip = if ip_address.is_empty() {
            0i64
        } else {
            parse_ipv4_to_u32(ip_address) as i64
        };
        let user_ip_hashes = compute_user_ip_hashes(model_config, request_ip);

        let _empty = Vec::new();
        let agg_user_actions = match sequence {
            Some(sequence) => match &sequence.user_actions_data {
                Some(container) => match &container.data {
                    Some(Data::OrderedAggregatedUserActionsList(list)) => {
                        &list.aggregated_user_actions
                    }
                    _ => &_empty,
                },
                None => &_empty,
            },
            None => &_empty,
        };

        let start = agg_user_actions.len().saturating_sub(history_seq_len);
        let user_actions = &agg_user_actions[start..];

        let mut valid_entry_count = 0;
        for aggregated_user_action in user_actions.iter() {
            if let Some(tweet_info) = &aggregated_user_action.tweet_info {
                if tweet_info.author_id == 0 || tweet_info.tweet_id == 0 {
                    continue;
                }
                let author_id = tweet_info.author_id as i64;
                let tweet_id = tweet_info.tweet_id as i64;
                let base_author_idx = valid_entry_count * num_author_hashes;
                let base_tweet_idx = valid_entry_count * num_item_hashes;

                for k in 0..num_author_hashes {
                    history_auth_hashes[base_author_idx + k] =
                        model_config.hash_table.hash_author_id(author_id, k);
                }

                for k in 0..num_item_hashes {
                    history_post_hashes[base_tweet_idx + k] =
                        model_config.hash_table.hash_item_id(tweet_id, k);
                }

                history_post_ids[valid_entry_count] = tweet_id;
                stamp_semantic_ids(
                    &mut history_semantic_ids,
                    valid_entry_count,
                    sid_num_levels,
                    &tweet_info.semantic_ids,
                );

                let base_idx = valid_entry_count * output_vocab_size;
                let continuous_base_idx = valid_entry_count * num_continuous_actions;
                let mut total_dwell_time: f64 = 0.0;
                for action_info in &aggregated_user_action.actions {
                    let action_idx = action_info.action_name as usize;
                    if action_idx < output_vocab_size {
                        history_actions[base_idx + action_idx] = true;
                    }
                    if let Some(ref meta) = action_info.user_action_meta {
                        total_dwell_time += meta.dwell_time;
                    }
                }

                let dwell_time_seconds = total_dwell_time / 1000.0;
                let dwell_time_idx = pb::ContinuousActionName::DwellTime as usize;
                if dwell_time_idx < num_continuous_actions {
                    history_continuous_actions[continuous_base_idx + dwell_time_idx] =
                        dwell_time_seconds as f32;
                }

                let product_surface_val = aggregated_user_action
                    .actions
                    .iter()
                    .filter_map(|action| {
                        action
                            .user_action_meta
                            .as_ref()
                            .map(|meta| meta.product_surface)
                    })
                    .next()
                    .unwrap_or(pb::ProductSurface::Unknown as i32);
                history_product_surfaces[valid_entry_count] = product_surface_val;

                history_impr_ts[valid_entry_count] =
                    (aggregated_user_action.impressed_time_ms / 1000) as i32;

                let tz_enum_val = aggregated_user_action
                    .actions
                    .iter()
                    .filter_map(|action| {
                        action
                            .user_action_meta
                            .as_ref()
                            .filter(|meta| !meta.time_zone.is_empty())
                            .map(|meta| pb::timezone_string_to_enum(&meta.time_zone) as i16)
                    })
                    .next()
                    .unwrap_or(0);
                history_tz_enums[valid_entry_count] = tz_enum_val;

                valid_entry_count += 1;
            }
        }

        if sid_num_levels > 0 {
            let present = (0..valid_entry_count)
                .filter(|&i| history_semantic_ids[i * sid_num_levels] != 0)
                .count() as u64;
            record_sid_coverage("history", present, valid_entry_count as u64);
        }

        let n_post_cat = model_config.hash_table.num_post_categorical_features;
        let n_post_bool = model_config.hash_table.num_post_bool_features;
        let n_post_float = model_config.hash_table.num_post_float_features;
        let n_post_int64 = model_config.hash_table.num_post_int64_features;

        let mut history_categorical_features = vec![0i16; history_seq_len * n_post_cat];
        if n_post_cat > PRODUCT_SURFACE_CATEGORICAL_IDX {
            for (i, &ps) in history_product_surfaces.iter().enumerate() {
                history_categorical_features[i * n_post_cat + PRODUCT_SURFACE_CATEGORICAL_IDX] =
                    ps as i16;
            }
        }
        stamp_local_time_features(
            &history_impr_ts,
            &history_tz_enums,
            &mut history_categorical_features,
            n_post_cat,
        );

        let user_features = InputBuffer::extract_user_features(
            &model_config.hash_table,
            client_context,
            user_context,
        );

        Self {
            user_hashes,
            user_ip_hashes,
            history_post_hashes,
            history_auth_hashes,
            history_actions,
            history_continuous_actions,
            history_product_surfaces,
            user_categorical_features: user_features.categorical_features,
            user_bool_features: user_features.bool_features,
            user_float_features: user_features.float_features,
            user_int64_features: user_features.int64_features,
            user_installed_apps: user_features.installed_apps,
            history_categorical_features,
            history_bool_features: vec![false; history_seq_len * n_post_bool],
            history_float_features: vec![0.0f32; history_seq_len * n_post_float],
            history_int64_features: vec![0i64; history_seq_len * n_post_int64],
            history_post_ids,
            history_semantic_ids,
        }
    }

    pub fn compute_from_columnar_bytes(
        model_config: &ModelConfig,
        columnar_bytes: &[u8],
        user_id: u64,
        ip_address: &str,
        client_context: Option<&pb::ClientContext>,
        user_context: Option<&pb::UserContext>,
    ) -> Result<Self, arrow::error::ArrowError> {
        let num_user_hashes = model_config.hash_table.num_user_hashes();
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let history_seq_len = model_config.history_seq_len;
        let output_vocab_size = model_config.hash_table.output_vocab_size;
        let num_continuous_actions = model_config.hash_table.num_continuous_actions;

        let mut user_hashes = vec![0i32; num_user_hashes];
        let user_id_i64 = user_id as i64;
        for (k, slot) in user_hashes.iter_mut().enumerate().take(num_user_hashes) {
            *slot = model_config.hash_table.hash_user_id(user_id_i64, k);
        }

        let request_ip = if ip_address.is_empty() {
            0i64
        } else {
            parse_ipv4_to_u32(ip_address) as i64
        };
        let user_ip_hashes = compute_user_ip_hashes(model_config, request_ip);

        let mut history_post_hashes = vec![0i32; history_seq_len * num_item_hashes];
        let mut history_auth_hashes = vec![0i32; history_seq_len * num_author_hashes];
        let mut history_actions = vec![false; history_seq_len * output_vocab_size];
        let mut history_continuous_actions = vec![0.0f32; history_seq_len * num_continuous_actions];
        let mut history_product_surfaces = vec![0i32; history_seq_len];
        let mut history_impr_ts = vec![0i32; history_seq_len];
        let mut history_tz_enums = vec![0i16; history_seq_len];
        let mut history_post_ids = vec![0i64; history_seq_len];
        let sid_num_levels = model_config.sid_num_levels;
        let mut history_semantic_ids = vec![0u16; history_seq_len * sid_num_levels];

        let cursor = Cursor::new(columnar_bytes);
        let mut reader = StreamReader::try_new(cursor, None)?;

        let batch = match reader.next() {
            Some(Ok(batch)) if batch.num_rows() > 0 => batch,
            Some(Err(e)) => return Err(e),
            _ => {
                let user_features = InputBuffer::extract_user_features(
                    &model_config.hash_table,
                    client_context,
                    user_context,
                );
                let n_post_cat = model_config.hash_table.num_post_categorical_features;
                let n_post_bool = model_config.hash_table.num_post_bool_features;
                let n_post_float = model_config.hash_table.num_post_float_features;
                let n_post_int64 = model_config.hash_table.num_post_int64_features;
                return Ok(Self {
                    user_hashes,
                    user_ip_hashes,
                    history_post_hashes,
                    history_auth_hashes,
                    history_actions,
                    history_continuous_actions,
                    history_product_surfaces,
                    user_categorical_features: user_features.categorical_features,
                    user_bool_features: user_features.bool_features,
                    user_float_features: user_features.float_features,
                    user_int64_features: user_features.int64_features,
                    user_installed_apps: user_features.installed_apps,
                    history_categorical_features: vec![0i16; history_seq_len * n_post_cat],
                    history_bool_features: vec![false; history_seq_len * n_post_bool],
                    history_float_features: vec![0.0f32; history_seq_len * n_post_float],
                    history_int64_features: vec![0i64; history_seq_len * n_post_int64],
                    history_post_ids: vec![0i64; history_seq_len],
                    history_semantic_ids,
                });
            }
        };

        let num_rows = batch.num_rows();

        let tweet_ids = batch
            .column_by_name("tweetId")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let author_ids = batch
            .column_by_name("authorId")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let product_surfaces = batch
            .column_by_name("productSurface")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let action_multi_hot = batch.column_by_name("actionNameMultiHot");
        let continuous_actions = batch.column_by_name("continuousActionValuesSeq");
        let col_impressed_time_ms = batch
            .column_by_name("impressedTimeMs")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_timezone_id = batch
            .column_by_name("timezoneId")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let col_semantic_id = batch.column_by_name("semanticId");

        let start_row = num_rows.saturating_sub(history_seq_len);
        let mut valid_entry_count = 0;

        for row_idx in start_row..num_rows {
            let tweet_id = tweet_ids.map_or(0i64, |arr| arr.value(row_idx));
            let author_id = author_ids.map_or(0i64, |arr| arr.value(row_idx));

            if tweet_id == 0 || author_id == 0 {
                continue;
            }

            let base_tweet_idx = valid_entry_count * num_item_hashes;
            let base_author_idx = valid_entry_count * num_author_hashes;

            for k in 0..num_item_hashes {
                history_post_hashes[base_tweet_idx + k] =
                    model_config.hash_table.hash_item_id(tweet_id, k);
            }

            for k in 0..num_author_hashes {
                history_auth_hashes[base_author_idx + k] =
                    model_config.hash_table.hash_author_id(author_id, k);
            }

            history_post_ids[valid_entry_count] = tweet_id;
            if let Some(sid_col) = col_semantic_id {
                let fsl = sid_col.as_fixed_size_list();
                if !fsl.is_null(row_idx) {
                    let inner = fsl.value(row_idx);
                    if let Some(codes) = inner.as_any().downcast_ref::<Int32Array>() {
                        stamp_semantic_ids(
                            &mut history_semantic_ids,
                            valid_entry_count,
                            sid_num_levels,
                            codes.values(),
                        );
                    }
                }
            }

            if let Some(ps) = product_surfaces {
                history_product_surfaces[valid_entry_count] = ps.value(row_idx);
            }

            if let Some(action_col) = action_multi_hot {
                let fsl = action_col.as_fixed_size_list();
                let inner_bools = fsl.value(row_idx);
                if let Some(bools) = inner_bools.as_any().downcast_ref::<BooleanArray>() {
                    let base_idx = valid_entry_count * output_vocab_size;
                    let copy_len = output_vocab_size.min(bools.len());
                    for j in 0..copy_len {
                        history_actions[base_idx + j] = bools.value(j);
                    }
                }
            }

            if let Some(cont_col) = continuous_actions {
                let fsl = cont_col.as_fixed_size_list();
                let inner_floats = fsl.value(row_idx);
                if let Some(floats) = inner_floats.as_any().downcast_ref::<Float32Array>() {
                    let continuous_base_idx = valid_entry_count * num_continuous_actions;
                    let copy_len = num_continuous_actions.min(floats.len());
                    for j in 0..copy_len {
                        history_continuous_actions[continuous_base_idx + j] = floats.value(j);
                    }
                }
            }

            if let Some(ts) = col_impressed_time_ms {
                history_impr_ts[valid_entry_count] = (ts.value(row_idx) / 1000) as i32;
            }

            if let Some(tz) = col_timezone_id {
                history_tz_enums[valid_entry_count] = tz.value(row_idx) as i16;
            }

            valid_entry_count += 1;
        }

        if sid_num_levels > 0 {
            let present = (0..valid_entry_count)
                .filter(|&i| history_semantic_ids[i * sid_num_levels] != 0)
                .count() as u64;
            record_sid_coverage("history", present, valid_entry_count as u64);
        }

        let n_post_cat = model_config.hash_table.num_post_categorical_features;
        let n_post_bool = model_config.hash_table.num_post_bool_features;
        let n_post_float = model_config.hash_table.num_post_float_features;
        let n_post_int64 = model_config.hash_table.num_post_int64_features;

        let mut history_categorical_features = vec![0i16; history_seq_len * n_post_cat];
        stamp_i32_as_categorical(
            &history_product_surfaces,
            &mut history_categorical_features,
            n_post_cat,
            PRODUCT_SURFACE_CATEGORICAL_IDX,
        );
        stamp_local_time_features(
            &history_impr_ts,
            &history_tz_enums,
            &mut history_categorical_features,
            n_post_cat,
        );

        let user_features = InputBuffer::extract_user_features(
            &model_config.hash_table,
            client_context,
            user_context,
        );

        Ok(Self {
            user_hashes,
            user_ip_hashes,
            history_post_hashes,
            history_auth_hashes,
            history_actions,
            history_continuous_actions,
            history_product_surfaces,
            user_categorical_features: user_features.categorical_features,
            user_bool_features: user_features.bool_features,
            user_float_features: user_features.float_features,
            user_int64_features: user_features.int64_features,
            user_installed_apps: user_features.installed_apps,
            history_categorical_features,
            history_bool_features: vec![false; history_seq_len * n_post_bool],
            history_float_features: vec![0.0f32; history_seq_len * n_post_float],
            history_int64_features: vec![0i64; history_seq_len * n_post_int64],
            history_post_ids,
            history_semantic_ids,
        })
    }
}

pub struct PredictRequestItem {
    pub sequence: Option<pb::UserActionSequence>,
    pub return_logprob: bool,
    pub top_logprobs_num: u32,
    pub return_log_map: bool,
    pub requested_action_indices: Vec<u32>,
    pub requested_continuous_action_indices: Vec<u32>,
    pub return_candidate_tweet_id_only: bool,
    pub candidate_set: pb::CandidateSet,
    pub sender: Option<oneshot::Sender<Result<pb::CandidateDistributionSet, Status>>>,
    pub input_buffer: Option<InputBuffer>,
    pub enqueue_time: Instant,
    pub deadline: Option<Instant>,
    pub columnar_sequence_bytes: Option<Bytes>,
}

impl Default for PredictRequestItem {
    fn default() -> Self {
        Self {
            sequence: None,
            return_logprob: false,
            top_logprobs_num: 0,
            return_log_map: false,
            requested_action_indices: Vec::new(),
            requested_continuous_action_indices: Vec::new(),
            return_candidate_tweet_id_only: false,
            candidate_set: pb::CandidateSet::default(),
            sender: None,
            input_buffer: None,
            enqueue_time: Instant::now(),
            deadline: None,
            columnar_sequence_bytes: None,
        }
    }
}

pub struct RetrieveRequestItem {
    pub user_id: u64,
    pub sequence: Option<pb::UserActionSequence>,
    pub top_k: u32,
    pub topic_entity_ids: Vec<u64>,
    pub topic_filter_mode: i32,
    pub sender: Option<oneshot::Sender<Result<pb::ScoredCandidates, Status>>>,
    pub input_buffer: Option<RetrievalInputBuffer>,
    pub enqueue_time: Instant,
    pub deadline: Option<Instant>,
    pub columnar_sequence_bytes: Option<Bytes>,
    pub eligible_posts_bloom_filter: Vec<u8>,
    pub mm_query_embeddings: Option<Vec<half::f16>>,
    pub mm_query_seed_post_ids: Option<Vec<u64>>,
}

pub fn fill_candidate_search_query_embeddings(
    sq_slice: &mut [f32],
    items: &[PredictRequestItem],
    length_of_input: usize,
    candidate_seq_len: usize,
    search_query_embedding_dim: usize,
    num_item_hashes: usize,
) {
    use rayon::prelude::*;

    sq_slice.fill(0.0);
    sq_slice
        .par_chunks_exact_mut(candidate_seq_len * search_query_embedding_dim)
        .take(length_of_input)
        .zip(items.par_iter())
        .for_each(|(search_query_emb_row, item)| {
            if let Some(ref input_buffer) = item.input_buffer {
                let src = &input_buffer.candidate_search_query_embeddings;
                if src.len() == search_query_embedding_dim {
                    let n_rep =
                        input_buffer.num_real_candidates(num_item_hashes, candidate_seq_len);
                    xai_recsys::util::repeat_query_into(search_query_emb_row, src, n_rep);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid_test_config() -> ModelConfig {
        let mut mc = ModelConfig::from_params(
            8,
            vec![1],
            vec![0],
            1009,
            8,
            0,
            vec![1],
            vec![0],
            1009,
            8,
            vec![1],
            vec![0],
            1009,
            4,
            2,
            4,
            1,
            0,
            0,
            0,
            vec![],
            vec![],
            1009,
            0,
            7,
            0,
            4,
            0,
            0,
            1,
            0,
            0,
            0,
            false,
        );
        mc.sid_num_levels = 3;
        mc
    }

    #[test]
    fn proto_history_semantic_ids_are_read_from_tweet_info() {
        let sequence = Some(pb::UserActionSequence {
            user_actions_data: Some(pb::UserActionSequenceDataContainer {
                data: Some(
                    pb::user_action_sequence_data_container::Data::OrderedAggregatedUserActionsList(
                        pb::AggregatedUserActionList {
                            aggregated_user_actions: vec![pb::AggregatedUserAction {
                                tweet_info: Some(pb::TweetInfo {
                                    tweet_id: 11,
                                    author_id: 21,
                                    semantic_ids: vec![0, 7, -1],
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ),
                ),
            }),
            ..Default::default()
        });
        let buf =
            RetrievalInputBuffer::compute_for_item(&sid_test_config(), &sequence, "", None, None);
        assert_eq!(
            buf.history_semantic_ids,
            vec![1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    const SQ_DIM: usize = 2;
    const SQ_SEQ_LEN: usize = 2;
    const SQ_ROW: usize = SQ_SEQ_LEN * SQ_DIM;
    const SQ_ITEM_HASHES: usize = 1;

    fn sq_item(query: Option<Vec<f32>>, real_candidates: usize) -> PredictRequestItem {
        let input_buffer = query.map(|candidate_search_query_embeddings| InputBuffer {
            candidate_post_hashes: (0..real_candidates as i32).map(|i| i + 1).collect(),
            candidate_search_query_embeddings,
            ..Default::default()
        });
        PredictRequestItem {
            input_buffer,
            ..Default::default()
        }
    }

    fn sq_item_without_input_buffer() -> PredictRequestItem {
        PredictRequestItem::default()
    }

    fn fill_sq(sq_slice: &mut [f32], items: &[PredictRequestItem], length_of_input: usize) {
        fill_candidate_search_query_embeddings(
            sq_slice,
            items,
            length_of_input,
            SQ_SEQ_LEN,
            SQ_DIM,
            SQ_ITEM_HASHES,
        );
    }

    #[test]
    fn search_query_row_for_item_without_input_buffer_is_zeroed() {
        let items = vec![
            sq_item(Some(vec![1.0, 2.0]), SQ_SEQ_LEN),
            sq_item_without_input_buffer(),
        ];
        let mut sq_slice = vec![9.0f32; 2 * SQ_ROW];

        fill_sq(&mut sq_slice, &items, items.len());

        assert_eq!(sq_slice[..SQ_ROW], [1.0, 2.0, 1.0, 2.0]);
        assert_eq!(sq_slice[SQ_ROW..], [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn search_query_rows_beyond_length_of_input_are_zeroed() {
        let items = vec![sq_item(Some(vec![1.0, 2.0]), SQ_SEQ_LEN)];
        let mut sq_slice = vec![9.0f32; 3 * SQ_ROW];

        fill_sq(&mut sq_slice, &items, items.len());

        assert_eq!(sq_slice[..SQ_ROW], [1.0, 2.0, 1.0, 2.0]);
        assert_eq!(sq_slice[SQ_ROW..], [0.0; 2 * SQ_ROW]);
    }

    #[test]
    fn queryless_request_does_not_inherit_reused_search_query_rows() {
        let mut sq_slice = vec![0.0f32; 2 * SQ_ROW];
        let with_query = vec![
            sq_item(Some(vec![1.0, 2.0]), SQ_SEQ_LEN),
            sq_item(Some(vec![3.0, 4.0]), SQ_SEQ_LEN),
        ];
        fill_sq(&mut sq_slice, &with_query, with_query.len());
        assert_eq!(sq_slice, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);

        let queryless = vec![sq_item(Some(Vec::new()), SQ_SEQ_LEN), sq_item(None, 0)];
        fill_sq(&mut sq_slice, &queryless, queryless.len());

        assert_eq!(sq_slice, vec![0.0; 2 * SQ_ROW]);
    }

    #[test]
    fn malformed_width_search_query_leaves_no_reused_values() {
        let mut sq_slice = vec![9.0f32; SQ_ROW];
        let items = vec![sq_item(Some(vec![1.0, 2.0, 3.0]), SQ_SEQ_LEN)];

        fill_sq(&mut sq_slice, &items, items.len());

        assert_eq!(sq_slice, vec![0.0; SQ_ROW]);
    }

    #[test]
    fn search_query_is_repeated_only_across_real_candidates() {
        let items = vec![sq_item(Some(vec![1.0, 2.0]), 1)];
        let mut sq_slice = vec![9.0f32; SQ_ROW];

        fill_sq(&mut sq_slice, &items, items.len());

        assert_eq!(sq_slice, vec![1.0, 2.0, 0.0, 0.0]);
    }
}
