use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(name = "xai-recsys-rankall")]
pub struct Cli {
    #[arg(long, value_enum, env = "PIPELINE")]
    pub pipeline: PipelineKind,

    #[arg(long, env = "KAFKA_TOPIC")]
    pub topic: Option<String>,

    #[arg(long, env = "KAFKA_GROUP")]
    pub group: String,

    #[arg(
        long,
        default_value = "/s/kafka/main-2:kafka-tls",
        env = "KAFKA_BOOTSTRAP"
    )]
    pub bootstrap: String,

    #[arg(long, default_value_t = 12, env = "SEEK_HOURS")]
    pub seek_hours: u64,

    #[arg(long, default_value_t = false)]
    pub force_seek: bool,

    #[arg(long, env = "OUTPUT_DIR")]
    pub output_dir: PathBuf,

    #[arg(long, default_value_t = 9090, env = "METRICS_PORT")]
    pub metrics_port: u16,

    #[arg(long, default_value_t = 180, env = "DUMP_INTERVAL_SECS")]
    pub dump_interval_secs: u64,

    #[arg(long, default_value_t = 3, env = "VERSIONS_TO_KEEP")]
    pub versions_to_keep: usize,

    #[arg(long, default_value_t = false)]
    pub compact_protocol: bool,

    #[arg(
        long,
        default_value = "phoenix.HomeExperiment1Lap7.",
        env = "SCORE_PREFIX"
    )]
    pub score_prefix: String,

    #[arg(long, default_value_t = true, env = "COMMIT_AFTER_DUMP")]
    pub commit_after_dump: bool,

    #[arg(long, default_value_t = 0, env = "COMMIT_INTERVAL_SECS")]
    pub commit_interval_secs: u64,

    #[arg(long, default_value_t = 500, env = "MAX_POLL_RECORDS")]
    pub max_poll_records: usize,

    #[arg(long, env = "SID_ENDPOINT")]
    pub sid_endpoint: Option<String>,

    #[arg(long, default_value_t = 180.0, env = "SID_MIN_POST_AGE_SECONDS")]
    pub sid_min_post_age_seconds: f64,

    #[arg(long, default_value_t = 10.0, env = "SID_REQUEST_TIMEOUT_SECONDS")]
    pub sid_request_timeout_seconds: f64,

    #[arg(long, default_value_t = 5000, env = "SID_BACKFILL_BATCH_SIZE")]
    pub sid_backfill_batch_size: usize,

    #[arg(long, default_value_t = 800_000, env = "SID_BACKFILL_QPS_CAP")]
    pub sid_backfill_qps_cap: usize,

    #[arg(long, default_value_t = 30, env = "SID_BACKFILL_INTERVAL_SECS")]
    pub sid_backfill_interval_secs: u64,

    #[arg(long, default_value_t = 1.0, env = "SID_BACKFILL_MAX_AGE_HOURS")]
    pub sid_backfill_max_age_hours: f64,

    #[arg(long, default_value_t = 6, env = "SID_NUM_LEVELS")]
    pub sid_num_levels: usize,

    #[arg(long, default_value_t = 1000, env = "TAIL_MAX_AUTHOR_FOLLOWERS")]
    pub tail_max_author_followers: i64,

    #[arg(long, default_value_t = 0, env = "TAIL_MIN_FAV_COUNT")]
    pub tail_min_fav_count: i64,
}

impl Cli {
    pub fn effective_topic(&self) -> &str {
        if let Some(ref t) = self.topic {
            return t;
        }
        self.pipeline.default_topic()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineKind {
    Main,
    Topic,
    Metadata,
    Sid,
    #[value(name = "sid-tail")]
    SidTail,
    MmMetadata,
    Ads,
    Analysis,
}

impl PipelineKind {
    pub fn default_topic(&self) -> &'static str {
        match self {
            Self::Main | Self::Sid => "phoenix_rank_all_indexing_event",
            Self::Topic => "phoenix_rank_all_indexing_event_backup",
            Self::Metadata | Self::MmMetadata | Self::SidTail => "phoenix_rankall_metadata_event",
            Self::Analysis => "home_mixer_phoenix_scored_candidates",
            Self::Ads => "",
        }
    }

    pub fn is_implemented(&self) -> bool {
        matches!(
            self,
            Self::Main | Self::Topic | Self::Metadata | Self::Analysis | Self::Sid | Self::SidTail
        )
    }

    pub fn needs_sid_endpoint(&self) -> bool {
        matches!(self, Self::Sid | Self::SidTail)
    }

    pub fn window_configs(&self) -> Vec<WindowConfig> {
        match self {
            Self::Main => vec![
                WindowConfig::new("post_creation", 24),
                WindowConfig::new("1fav", 24),
                WindowConfig::new("1fav", 48),
                WindowConfig::new("32fav", 24),
                WindowConfig::new("video", 48),
                WindowConfig::new("video", 96),
                WindowConfig::new("video", 168),
                WindowConfig::new("video", 336),
                WindowConfig::new("video", 720),
                WindowConfig::new("nsfw_video", 48),
                WindowConfig::new("nsfw_video", 168),
                WindowConfig::new("evergreen_video", 24 * 365 * 5),
                WindowConfig::new("evergreen_nsfw_video", 24 * 365 * 5),
                WindowConfig::new("evergreen_video_grok", 24 * 30),
            ],
            Self::Topic => vec![
                WindowConfig::new("1fav", 24),
                WindowConfig::new("1fav_topic", 24),
                WindowConfig::new("1fav_topic_option_1", 24),
                WindowConfig::new("1fav_topic_option_2", 24),
                WindowConfig::new("1fav_topic_option_3", 24),
                WindowConfig::new("1fav_topic_option_4", 24),
                WindowConfig::new("1fav_topic_option_5", 24),
            ],
            Self::Metadata => vec![
                WindowConfig::new("metadata", 24),
                WindowConfig::new("metadata", 48),
                WindowConfig::new("metadata", 72),
            ],
            Self::MmMetadata => vec![
                WindowConfig::new("mm_emb_metadata", 24),
                WindowConfig::new("mm_emb_metadata_video", 48),
                WindowConfig::new("mm_emb_metadata_video", 96),
            ],
            Self::Sid => vec![
                WindowConfig::new("1fav", 24),
                WindowConfig::new("video", 48),
                WindowConfig::new("video", 96),
                WindowConfig::new("video", 24 * 14),
                WindowConfig::new("nsfw_video", 48),
                WindowConfig::new("nsfw_video", 168),
                WindowConfig::new("nsfw_video", 24 * 14),
                WindowConfig::new("evergreen_video", 24 * 365 * 5),
                WindowConfig::new("imagine", 96),
                WindowConfig::new("evergreen_video_grok", 24 * 30),
            ],
            Self::SidTail => vec![WindowConfig::new("tail", 24)],
            Self::Analysis | Self::Ads => vec![],
        }
    }
}

impl fmt::Display for PipelineKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Main => write!(f, "main"),
            Self::Topic => write!(f, "topic"),
            Self::Metadata => write!(f, "metadata"),
            Self::MmMetadata => write!(f, "mm_metadata"),
            Self::Ads => write!(f, "ads"),
            Self::Analysis => write!(f, "analysis"),
            Self::Sid => write!(f, "sid"),
            Self::SidTail => write!(f, "sid-tail"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub name: String,
    pub retention: Duration,
}

impl WindowConfig {
    pub fn new(name: impl Into<String>, retention_hours: u64) -> Self {
        Self {
            name: name.into(),
            retention: Duration::from_secs(retention_hours * 3600),
        }
    }

    pub fn window_name(&self) -> String {
        let days = self.retention.as_secs() / 86400;
        format!("{}_{days}day", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_name_formatting() {
        assert_eq!(WindowConfig::new("1fav", 24).window_name(), "1fav_1day");
        assert_eq!(WindowConfig::new("video", 48).window_name(), "video_2day");
        assert_eq!(WindowConfig::new("video", 168).window_name(), "video_7day");
        assert_eq!(
            WindowConfig::new("evergreen_video", 24 * 365 * 5).window_name(),
            "evergreen_video_1825day"
        );
        assert_eq!(
            WindowConfig::new("evergreen_video_grok", 24 * 30).window_name(),
            "evergreen_video_grok_30day"
        );
    }

    #[test]
    fn main_pipeline_windows() {
        let configs = PipelineKind::Main.window_configs();
        let names: Vec<String> = configs.iter().map(|w| w.window_name()).collect();
        assert!(names.contains(&"1fav_1day".to_string()));
        assert!(names.contains(&"1fav_2day".to_string()));
        assert!(names.contains(&"video_2day".to_string()));
        assert!(names.contains(&"post_creation_1day".to_string()));
        assert!(names.contains(&"evergreen_video_1825day".to_string()));
        assert!(names.contains(&"evergreen_video_grok_30day".to_string()));
        assert_eq!(configs.len(), 14);
    }

    #[test]
    fn topic_pipeline_windows() {
        let configs = PipelineKind::Topic.window_configs();
        let names: Vec<String> = configs.iter().map(|w| w.window_name()).collect();
        assert!(names.contains(&"1fav_1day".to_string()));
        assert!(names.contains(&"1fav_topic_1day".to_string()));
        assert!(names.contains(&"1fav_topic_option_1_1day".to_string()));
        assert_eq!(configs.len(), 7);
    }

    #[test]
    fn metadata_pipeline_windows() {
        let configs = PipelineKind::Metadata.window_configs();
        let names: Vec<String> = configs.iter().map(|w| w.window_name()).collect();
        assert_eq!(
            names,
            vec!["metadata_1day", "metadata_2day", "metadata_3day"]
        );
    }

    #[test]
    fn default_topics() {
        assert_eq!(
            PipelineKind::Main.default_topic(),
            "phoenix_rank_all_indexing_event"
        );
        assert_eq!(
            PipelineKind::Topic.default_topic(),
            "phoenix_rank_all_indexing_event_backup"
        );
        assert_eq!(
            PipelineKind::Metadata.default_topic(),
            "phoenix_rankall_metadata_event"
        );
        assert_eq!(
            PipelineKind::SidTail.default_topic(),
            "phoenix_rankall_metadata_event"
        );
        assert_eq!(
            PipelineKind::Analysis.default_topic(),
            "home_mixer_phoenix_scored_candidates"
        );
    }

    #[test]
    fn sid_tail_windows() {
        let names: Vec<String> = PipelineKind::SidTail
            .window_configs()
            .iter()
            .map(|w| w.window_name())
            .collect();
        assert_eq!(names, vec!["tail_1day"]);
    }

    #[test]
    fn implemented_variants() {
        assert!(PipelineKind::Main.is_implemented());
        assert!(PipelineKind::Topic.is_implemented());
        assert!(PipelineKind::Metadata.is_implemented());
        assert!(PipelineKind::Analysis.is_implemented());
        assert!(PipelineKind::Sid.is_implemented());
        assert!(PipelineKind::SidTail.is_implemented());
        assert!(PipelineKind::SidTail.needs_sid_endpoint());
        assert!(!PipelineKind::MmMetadata.is_implemented());
        assert!(!PipelineKind::Ads.is_implemented());
    }

    #[test]
    fn sid_default_topic_matches_main() {
        assert_eq!(
            PipelineKind::Sid.default_topic(),
            PipelineKind::Main.default_topic(),
        );
    }

    #[test]
    fn sid_pipeline_includes_evergreen_video_grok_window() {
        let names: Vec<String> = PipelineKind::Sid
            .window_configs()
            .iter()
            .map(|w| w.window_name())
            .collect();
        assert!(
            names.contains(&"evergreen_video_grok_30day".to_string()),
            "Sid window list missing evergreen_video_grok_30day: {names:?}",
        );
    }

    #[test]
    fn sid_pipeline_includes_nsfw_video_7day_window() {
        let names: Vec<String> = PipelineKind::Sid
            .window_configs()
            .iter()
            .map(|w| w.window_name())
            .collect();
        for w in ["nsfw_video_7day", "nsfw_video_14day", "video_14day"] {
            assert!(
                names.contains(&w.to_string()),
                "Sid window list missing {w}: {names:?}",
            );
        }
    }

    #[test]
    fn sid_pipeline_includes_imagine_4day_window() {
        let names: Vec<String> = PipelineKind::Sid
            .window_configs()
            .iter()
            .map(|w| w.window_name())
            .collect();
        assert!(
            names.contains(&"imagine_4day".to_string()),
            "Sid window list missing imagine_4day: {names:?}",
        );
        for unexpected in [
            "imagine_1day",
            "imagine_2day",
            "imagine_7day",
            "imagine_30day",
        ] {
            assert!(
                !names.contains(&unexpected.to_string()),
                "unexpected imagine window {unexpected} in Sid list: {names:?}",
            );
        }
    }
}
