// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
#[derive(Debug, Clone)]
pub struct HashTableConfig {
    pub user_id_table_size: usize,
    pub user_hash_scales: Vec<i64>,
    pub user_biases: Vec<i64>,
    pub user_modulus: i64,

    pub item_id_table_size: usize,
    pub item_hash_vocab_size: usize,
    pub item_hash_scales: Vec<i64>,
    pub item_biases: Vec<i64>,
    pub item_modulus: i64,

    pub author_id_table_size: usize,
    pub author_hash_scales: Vec<i64>,
    pub author_biases: Vec<i64>,
    pub author_modulus: i64,

    pub ip_id_table_size: usize,
    pub ip_hash_scales: Vec<i64>,
    pub ip_biases: Vec<i64>,
    pub ip_modulus: i64,

    pub output_vocab_size: usize,
    pub num_continuous_actions: usize,
    pub search_query_embedding_dim: usize,

    pub num_user_categorical_features: usize,
    pub num_user_bool_features: usize,
    pub num_user_float_features: usize,
    pub num_user_int64_features: usize,
    pub num_user_installed_apps: usize,

    pub num_post_categorical_features: usize,
    pub num_post_bool_features: usize,
    pub num_post_float_features: usize,
    pub num_post_int64_features: usize,

    pub enable_stale_post: bool,
}

impl HashTableConfig {
    pub fn offset_user(&self) -> usize {
        1 + 64
    }

    pub fn offset_item(&self) -> usize {
        self.offset_user() + self.user_id_table_size
    }

    pub fn offset_author(&self) -> usize {
        self.offset_item() + self.item_id_table_size
    }

    pub fn offset_ip(&self) -> usize {
        self.offset_author() + self.author_id_table_size
    }

    pub fn num_user_hashes(&self) -> usize {
        self.user_hash_scales.len()
    }

    pub fn num_item_hashes(&self) -> usize {
        self.item_hash_scales.len()
    }

    pub fn num_author_hashes(&self) -> usize {
        self.author_hash_scales.len()
    }

    pub fn num_ip_hashes(&self) -> usize {
        self.ip_hash_scales.len()
    }

    pub fn effective_item_hash_buckets(&self) -> usize {
        if self.item_hash_vocab_size > 0 {
            self.item_hash_vocab_size
        } else {
            self.item_id_table_size
        }
    }

    pub fn hash_user_id(&self, user_id: i64, index: usize) -> i32 {
        if user_id == 0 {
            return 0;
        }
        if self.user_id_table_size == 0 {
            return i32::from(index == 0);
        }

        let mut hash_result =
            (user_id * self.user_hash_scales[index] + self.user_biases[index]) % self.user_modulus;
        if hash_result < 0 {
            hash_result += self.user_modulus;
        }
        let hash_bucket = (hash_result % (self.user_id_table_size as i64 - 1) + 1) as usize;
        (hash_bucket + self.offset_user()) as i32
    }

    pub fn hash_item_id(&self, item_id: i64, index: usize) -> i32 {
        if item_id == 0 {
            return 0;
        }

        let mut hash_result =
            (item_id * self.item_hash_scales[index] + self.item_biases[index]) % self.item_modulus;
        if hash_result < 0 {
            hash_result += self.item_modulus;
        }
        let hash_bucket =
            (hash_result % (self.effective_item_hash_buckets() as i64 - 1) + 1) as usize;
        (hash_bucket + self.offset_item()) as i32
    }

    pub fn hash_author_id(&self, author_id: i64, index: usize) -> i32 {
        if author_id == 0 {
            return 0;
        }

        let mut hash_result = (author_id * self.author_hash_scales[index]
            + self.author_biases[index])
            % self.author_modulus;
        if hash_result < 0 {
            hash_result += self.author_modulus;
        }
        let hash_bucket = (hash_result % (self.author_id_table_size as i64 - 1) + 1) as usize;
        (hash_bucket + self.offset_author()) as i32
    }

    pub fn hash_user_ids(&self, user_ids: &[i64]) -> Vec<i32> {
        let n = user_ids.len();
        let m = self.num_user_hashes();

        let mut hashes = Vec::with_capacity(n * m);

        for user_id in user_ids.iter() {
            for j in 0..m {
                hashes.push(self.hash_user_id(*user_id, j));
            }
        }

        hashes
    }

    pub fn hash_item_ids(&self, item_ids: &[i64]) -> Vec<i32> {
        let n = item_ids.len();
        let m = self.num_item_hashes();

        let mut hashes = Vec::with_capacity(n * m);

        for item_id in item_ids.iter() {
            for j in 0..m {
                hashes.push(self.hash_item_id(*item_id, j));
            }
        }

        hashes
    }

    pub fn hash_author_ids(&self, author_ids: &[i64]) -> Vec<i32> {
        let n = author_ids.len();
        let m = self.num_author_hashes();

        let mut hashes = Vec::with_capacity(n * m);

        for author_id in author_ids.iter() {
            for j in 0..m {
                hashes.push(self.hash_author_id(*author_id, j));
            }
        }

        hashes
    }

    pub fn hash_ip_id(&self, ip_id: i64, index: usize) -> i32 {
        if ip_id == 0 || self.ip_id_table_size == 0 {
            return 0;
        }

        let mut hash_result =
            (ip_id * self.ip_hash_scales[index] + self.ip_biases[index]) % self.ip_modulus;
        if hash_result < 0 {
            hash_result += self.ip_modulus;
        }
        let hash_bucket = (hash_result % (self.ip_id_table_size as i64 - 1) + 1) as usize;
        (hash_bucket + self.offset_ip()) as i32
    }

    pub fn hash_ip_ids(&self, ip_ids: &[i64]) -> Vec<i32> {
        let n = ip_ids.len();
        let m = self.num_ip_hashes();

        let mut hashes = Vec::with_capacity(n * m);

        for ip_id in ip_ids.iter() {
            for j in 0..m {
                hashes.push(self.hash_ip_id(*ip_id, j));
            }
        }

        hashes
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hash_table: HashTableConfig,
    pub history_seq_len: usize,
    pub candidate_seq_len: usize,
    pub multimodal_embedding_dim: usize,
    pub search_query_embedding_dim: usize,
    pub num_categorical_features: usize,
    pub sid_num_levels: usize,
}

impl ModelConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn from_params(
        user_id_table_size: usize,
        user_hash_scales: Vec<i64>,
        user_biases: Vec<i64>,
        user_modulus: i64,
        item_id_table_size: usize,
        item_hash_vocab_size: usize,
        item_hash_scales: Vec<i64>,
        item_biases: Vec<i64>,
        item_modulus: i64,
        author_id_table_size: usize,
        author_hash_scales: Vec<i64>,
        author_biases: Vec<i64>,
        author_modulus: i64,
        output_vocab_size: usize,
        num_continuous_actions: usize,
        history_seq_len: usize,
        candidate_seq_len: usize,
        multimodal_embedding_dim: usize,
        search_query_embedding_dim: usize,
        ip_id_table_size: usize,
        ip_hash_scales: Vec<i64>,
        ip_biases: Vec<i64>,
        ip_modulus: i64,
        num_categorical_features: usize,
        num_user_categorical_features: usize,
        num_user_bool_features: usize,
        num_user_float_features: usize,
        num_user_int64_features: usize,
        num_user_installed_apps: usize,
        num_post_categorical_features: usize,
        num_post_bool_features: usize,
        num_post_float_features: usize,
        num_post_int64_features: usize,
        enable_stale_post: bool,
    ) -> Self {
        ModelConfig {
            hash_table: HashTableConfig {
                user_id_table_size,
                user_hash_scales,
                user_biases,
                user_modulus,
                item_id_table_size,
                item_hash_vocab_size,
                item_hash_scales,
                item_biases,
                item_modulus,
                author_id_table_size,
                author_hash_scales,
                author_biases,
                author_modulus,
                ip_id_table_size,
                ip_hash_scales,
                ip_biases,
                ip_modulus,
                output_vocab_size,
                num_continuous_actions,
                search_query_embedding_dim,
                num_user_categorical_features,
                num_user_bool_features,
                num_user_float_features,
                num_user_int64_features,
                num_user_installed_apps,
                num_post_categorical_features,
                num_post_bool_features,
                num_post_float_features,
                num_post_int64_features,
                enable_stale_post,
            },
            history_seq_len,
            candidate_seq_len,
            multimodal_embedding_dim,
            search_query_embedding_dim,
            num_categorical_features,
            sid_num_levels: 0,
        }
    }
}
