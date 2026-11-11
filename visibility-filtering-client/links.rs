macro_rules! rules {
    ($path:literal) => {
        concat!("https://help.x.com/rules-and-policies/", $path)
    };
}
macro_rules! using {
    ($path:literal) => {
        concat!("https://help.x.com/using-x/", $path)
    };
}

pub const NOTICES_ON_TWITTER: &str = rules!("notices-on-x");
pub const SENSITIVE_MEDIA_APPEAL_PREFIX: &str =
    "https://x.com/i/safety/report_story_start?source=appealtweet&reported_tweet_id=";
pub const ENFORCEMENT_OPTIONS: &str = rules!("enforcement-options");
pub const PUBLIC_INTEREST: &str = rules!("public-interest");
pub const TWITTER_RULES: &str = rules!("x-rules");
pub const ELECTION_INTEGRITY: &str = rules!("election-integrity-policy");
pub const CRISIS_MISINFORMATION: &str = rules!("crisis-misinformation");
pub const MANIPULATED_MEDIA: &str = rules!("manipulated-media");
pub const HATEFUL_CONDUCT_POLICY: &str = rules!("hateful-conduct-policy");
pub const ABUSIVE_BEHAVIOR: &str = rules!("abusive-behavior");
pub const AGE_ASSURANCE: &str = rules!("age-assurance");

pub const COMMUNITIES: &str = using!("communities");
pub const PREMIUM_CONTENT: &str = "https://help.x.com/en/using-x/x-premium";

pub const APPLE_APP_STORE_UPDATE: &str = "https://apps.apple.com/us/app/x/id333903271%20https://apps.apple.com/app/apple-store/id333903271?pt=9551&ct=age-assurance-update&mt=8";
pub const GOOGLE_PLAY_STORE_UPDATE: &str = "https://play.google.com/store/apps/details?id=com.twitter.android&referrer=utm_source%3D9551%26utm_medium%3D8%26utm_campaign%3Dage-assurance-update";

pub const PROFILE_SETTINGS: &str = "https://x.com/settings/profile";
pub const X_DOT_COM: &str = "https://x.com";

pub const BLURRED_MEDIA_TOMBSTONE_IMAGE: &str =
    "https://pbs.twimg.com/media/GxJIrSUagAAK-ZP?format=jpg&name=240x240";
