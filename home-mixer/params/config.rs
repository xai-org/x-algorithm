pub const FS_PATH: &str = "/usr/local/config/features/home-mixer/main/rust_home_mixer.yml";

pub const STRINGCENTER_BUNDLE_PATH: &str = "/config/stringcenter/timelinemixer.json";

pub fn decider_path() -> String {
    let dc = std::env::var("DATACENTER").expect("DATACENTER env var must be set");
    format!(
        "/usr/local/config/overlays/home-mixer/home-mixer/prod/{}/decider_overlay.yml",
        dc
    )
}

pub const TEST_USER_IDS: &[u64] = &[123, 234];
pub const TRACE_USER_IDS: &[u64] = &[];

pub const MAX_GRPC_MESSAGE_SIZE: usize = 128 * 1024 * 1024;
pub const TOP_K_CANDIDATES_TO_SELECT: usize = 50;
pub const RESULT_SIZE: usize = 35;
pub const FEED_MODULE_SLOTS: usize = 4;
pub const MAX_JETFUEL_FRAMES_PER_RESPONSE: usize = 8;
pub const FOR_YOU_MAX_RESULT_SIZE: usize =
    RESULT_SIZE + FEED_MODULE_SLOTS + MAX_JETFUEL_FRAMES_PER_RESPONSE;
pub const RANKED_FOLLOWING_MAX_RESULT_SIZE: usize = 38;
pub const FOLLOWING_MAX_RESULT_SIZE: usize = 110;
pub const FOLLOWING_ADS_TOP_K: usize = 9;
pub const FOLLOWING_POST_FETCH_SIZE: usize = FOLLOWING_MAX_RESULT_SIZE - FOLLOWING_ADS_TOP_K;
pub const FOLLOWING_PIPELINE_RESULT_SIZE: usize = FOLLOWING_MAX_RESULT_SIZE + 2;

pub const WHO_TO_FOLLOW_POSITION: usize = 6;
pub const FOLLOWING_WHO_TO_FOLLOW_MIN_POSTS: usize = 10;
pub const PROMPTS_POSITION: i32 = 0;
pub const FEED_SURVEY_POSITION: usize = 12;

pub const UAS_WINDOW_TIME_MS: i64 = 300_000;

pub const MAX_POST_AGE: u64 = 48 * 60 * 60;

pub const NEW_USER_OON_WEIGHT_FACTOR: f64 = 0.00001;
pub const NEW_USER_MIN_FOLLOWING: usize = 5;
pub const NEGATIVE_SCORES_OFFSET: f64 = 0.001;
