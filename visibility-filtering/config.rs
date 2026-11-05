use crate::quote::QuotePassMode;

pub const ENV_AUTHOR_CACHE_CAPACITY: &str = "VF_AUTHOR_CACHE_CAPACITY";
pub const ENV_TWEET_CACHE_CAPACITY: &str = "VF_TWEET_CACHE_CAPACITY";
pub const ENV_SAFETY_LABEL_CACHE_CAPACITY: &str = "VF_SAFETY_LABEL_CACHE_CAPACITY";
pub const ENV_CACHE_WARM_ENABLED: &str = "VF_CACHE_WARM_ENABLED";
pub const ENV_REFERENCE: &str = "VF_REFERENCE";
pub const ENV_DARK_TRAFFIC_ENABLED: &str = "DARK_TRAFFIC_ENABLED";
pub const ENV_APP_ENV: &str = "APP_ENV";
pub const ENV_FS_PATH: &str = "VF_FS_PATH";
pub const ENV_GIZMODUCK_CLIENT_ID: &str = "VF_GIZMODUCK_CLIENT_ID";
pub const ENV_TWEMCACHE_CLIENT_NAME: &str = "VF_TWEMCACHE_CLIENT_NAME";
pub const ENV_LABEL_CACHE_EDS: &str = "VF_LABEL_CACHE_EDS";
pub const ENV_QUOTE_PASS: &str = "VF_QUOTE_PASS";

const DEFAULT_FS_PATH: &str = "/usr/local/config/features/visibility/main/rust_vf.yml";

pub fn fs_path() -> String {
    std::env::var(ENV_FS_PATH).unwrap_or_else(|_| DEFAULT_FS_PATH.to_string())
}

pub(crate) fn gizmoduck_client_id() -> String {
    resolve_gizmoduck_client_id(
        std::env::var(ENV_GIZMODUCK_CLIENT_ID).ok().as_deref(),
        std::env::var(ENV_APP_ENV).ok().as_deref(),
    )
}

pub(crate) fn twemcache_client_name() -> String {
    resolve_twemcache_client_name(std::env::var(ENV_TWEMCACHE_CLIENT_NAME).ok().as_deref())
}

pub(crate) fn label_cache_eds() -> Option<String> {
    std::env::var(ENV_LABEL_CACHE_EDS)
        .ok()
        .filter(|eds| !eds.trim().is_empty())
}

pub fn resolve_gizmoduck_client_id(configured: Option<&str>, app_env: Option<&str>) -> String {
    match configured {
        Some(id) => id.to_string(),
        None => format!("visibility-filtering-service.{}", app_env.unwrap_or("prod")),
    }
}

pub fn resolve_twemcache_client_name(configured: Option<&str>) -> String {
    configured
        .unwrap_or("visibility-filtering-service")
        .to_string()
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
pub(crate) fn refuse_reference() {
    if let Ok(reference) = std::env::var(ENV_REFERENCE)
        && reference != "none"
    {
        panic!("{ENV_REFERENCE}={reference} needs the staging binary's serve");
    }
}

pub(crate) fn refuse_test_users_in_prod() {
    refuse_test_users(std::env::var(ENV_APP_ENV).ok().as_deref());
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
fn refuse_test_users(app_env: Option<&str>) {
    if app_env == Some("prod") {
        panic!("--test-users is staging-only, but {ENV_APP_ENV}=prod");
    }
}

const DEFAULT_CACHE_CAPACITY: usize = 1_000_000;

pub(crate) fn author_cache_capacity() -> Option<usize> {
    cache_capacity(ENV_AUTHOR_CACHE_CAPACITY)
}

pub(crate) fn tweet_cache_capacity() -> Option<usize> {
    cache_capacity(ENV_TWEET_CACHE_CAPACITY)
}

pub(crate) fn safety_label_cache_capacity() -> Option<usize> {
    cache_capacity(ENV_SAFETY_LABEL_CACHE_CAPACITY)
}

fn cache_capacity(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok();
    let capacity = parse_capacity(value.as_deref());
    if let Some(value) = value
        && value.trim().parse::<usize>().is_err()
    {
        tracing::warn!(
            name,
            value,
            ?capacity,
            "unparsable cache capacity; keeping the default"
        );
    }
    capacity
}

fn parse_capacity(value: Option<&str>) -> Option<usize> {
    let capacity = value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_CACHE_CAPACITY);
    (capacity > 0).then_some(capacity)
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
pub(crate) fn quote_pass() -> QuotePassMode {
    let value = std::env::var(ENV_QUOTE_PASS).ok();
    parse_quote_pass(value.as_deref())
        .unwrap_or_else(|| panic!("{ENV_QUOTE_PASS}={value:?} is neither off nor shadow"))
}

fn parse_quote_pass(value: Option<&str>) -> Option<QuotePassMode> {
    match value {
        None | Some("off") => Some(QuotePassMode::Off),
        Some("shadow") => Some(QuotePassMode::Shadow),
        Some(_) => None,
    }
}

pub(crate) fn cache_warm_enabled() -> bool {
    parse_env_flag(std::env::var(ENV_CACHE_WARM_ENABLED).ok().as_deref())
}

pub(crate) fn parse_env_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CACHE_CAPACITY, QuotePassMode, parse_capacity, parse_env_flag, parse_quote_pass,
        refuse_test_users, resolve_gizmoduck_client_id, resolve_twemcache_client_name,
    };

    #[test]
    fn only_prod_refuses_test_users() {
        refuse_test_users(Some("staging"));
        refuse_test_users(None);
        assert!(std::panic::catch_unwind(|| refuse_test_users(Some("prod"))).is_err());
    }

    #[test]
    fn the_quote_pass_is_off_unless_set_to_shadow_and_refuses_any_other_value() {
        for (value, expected) in [
            (None, Some(QuotePassMode::Off)),
            (Some("off"), Some(QuotePassMode::Off)),
            (Some("shadow"), Some(QuotePassMode::Shadow)),
            (Some(""), None),
            (Some("on"), None),
            (Some("Shadow"), None),
        ] {
            assert_eq!(parse_quote_pass(value), expected, "{value:?}");
        }
    }

    #[test]
    fn zero_turns_a_cache_off_and_an_unparsable_capacity_keeps_the_default() {
        for (value, expected) in [
            (None, Some(DEFAULT_CACHE_CAPACITY)),
            (Some(""), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("-5"), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("-1"), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("2M"), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("0"), None),
            (Some(" 0 "), None),
            (Some(" 2000000 "), Some(2_000_000)),
        ] {
            assert_eq!(parse_capacity(value), expected, "{value:?}");
        }
    }

    #[test]
    fn client_ids_default_to_historical_values_and_overrides_win() {
        assert_eq!(
            resolve_gizmoduck_client_id(None, Some("prod")),
            "visibility-filtering-service.prod"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(None, Some("staging")),
            "visibility-filtering-service.staging"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(None, None),
            "visibility-filtering-service.prod"
        );
        assert_eq!(
            resolve_twemcache_client_name(None),
            "visibility-filtering-service"
        );
    }

    #[test]
    fn environment_flags_accept_only_enabled_values() {
        for (value, expected) in [
            (None, false),
            (Some(""), false),
            (Some("0"), false),
            (Some("false"), false),
            (Some("FALSE"), false),
            (Some("no"), false),
            (Some("off"), false),
            (Some("other"), false),
            (Some("1"), true),
            (Some("true"), true),
            (Some("TRUE"), true),
            (Some("yes"), true),
            (Some("on"), true),
        ] {
            assert_eq!(parse_env_flag(value), expected, "{value:?}");
        }
    }
}
