pub mod analysis_processor;
pub mod blacklist;
pub mod r#gen;
pub mod main_processor;
pub mod metadata_processor;
pub mod record;
pub mod sid_popular_video_processor;
pub mod sid_processor;
pub mod sid_tail_processor;
pub mod thrift_types;
pub mod topic_names;
pub mod topic_processor;

use record::IndexRecord;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    DeserializationError,
    InvalidData,
    Filtered,
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeserializationError => write!(f, "deserialization_error"),
            Self::InvalidData => write!(f, "invalid_data"),
            Self::Filtered => write!(f, "filtered"),
        }
    }
}

#[derive(Debug, Default)]
pub struct ProcessorStats {
    pub total_processed: u64,
    pub total_success: u64,
    pub total_filtered: u64,
    pub total_invalid: u64,
    pub total_deser_error: u64,
}

impl ProcessorStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            return 0.0;
        }
        self.total_success as f64 / self.total_processed as f64 * 100.0
    }
}

pub trait RecordProcessor: Send + Sync {
    fn process_batch(&mut self, raw: &[Vec<u8>]) -> Vec<IndexRecord>;

    fn stats(&self) -> &ProcessorStats;
}
