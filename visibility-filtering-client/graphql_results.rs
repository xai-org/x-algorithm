use include_dir::{include_dir, Dir};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;
use xai_x_thrift::action::{self, Action, AnyInterstitial, MediaInterstitial};
use xai_x_thrift::safety_result::{FilteredReason, SafetyResult};

static TRANSLATIONS: Dir = include_dir!("$CARGO_MANIFEST_DIR/resources/translations");
const DEFAULT_LANG: &str = "en";

static COPY: LazyLock<HashMap<String, HashMap<String, String>>> = LazyLock::new(|| {
    #[derive(serde::Deserialize)]
    struct Catalog {
        data: Vec<Entry>,
    }
    #[derive(serde::Deserialize)]
    struct Entry {
        resname: Option<String>,
        source: Option<String>,
    }
    let mut by_lang = HashMap::new();
    for file in TRANSLATIONS.files() {
        let Some(stem) = file.path().file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem == "languages" || stem == "localization_clff" {
            continue;
        }
        let Some(raw) = file.contents_utf8() else {
            continue;
        };
        let Ok(catalog) = serde_yaml::from_str::<Catalog>(raw) else {
            continue;
        };
        let map: HashMap<String, String> = catalog
            .data
            .into_iter()
            .filter_map(|e| Some((e.resname?, e.source?)))
            .collect();
        by_lang.insert(stem.to_string(), map);
    }
    by_lang
});

fn ascii_lowercase(lang: &str) -> Cow<'_, str> {
    if lang.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(lang.to_ascii_lowercase())
    } else {
        Cow::Borrowed(lang)
    }
}

pub fn translate(resname: &str, lang: &str) -> Option<&'static str> {
    let lang = ascii_lowercase(lang);
    let primary = lang.split(['-', '_']).next().unwrap_or(&lang);
    [&*lang, primary, DEFAULT_LANG]
        .into_iter()
        .find_map(|l| COPY.get(l).and_then(|m| m.get(resname)))
        .map(String::as_str)
}

static RTL_LANGS: LazyLock<HashMap<String, bool>> = LazyLock::new(|| {
    #[derive(serde::Deserialize)]
    struct Lang {
        #[serde(default)]
        rtl: bool,
    }
    let Some(raw) = TRANSLATIONS
        .get_file("languages.yml")
        .and_then(|f| f.contents_utf8())
    else {
        return HashMap::new();
    };
    serde_yaml::from_str::<HashMap<String, Lang>>(raw)
        .map(|m| m.into_iter().map(|(k, v)| (k, v.rtl)).collect())
        .unwrap_or_default()
});

pub fn is_rtl(lang: &str) -> bool {
    let lang = ascii_lowercase(lang);
    let primary = lang.split(['-', '_']).next().unwrap_or(&lang);
    [&*lang, primary]
        .into_iter()
        .find_map(|l| RTL_LANGS.get(l).copied())
        .unwrap_or(false)
        || icu_rtl(primary)
}

fn icu_rtl(primary: &str) -> bool {
    matches!(
        primary,
        "ar" | "ckb" | "dv" | "fa" | "he" | "iw" | "ji" | "ps" | "sd" | "ug" | "ur" | "yi"
    )
}

pub fn get_safety_result(reason: &FilteredReason) -> Option<&SafetyResult> {
    match reason {
        FilteredReason::SafetyResult(sr) => Some(sr),
        _ => None,
    }
}

pub fn get_appealable(sr: &SafetyResult) -> Option<&action::Appealable> {
    match &sr.action {
        Action::Appealable(a) => Some(a),
        Action::TweetInterstitial(ti) => ti.appealable.as_ref(),
        _ => None,
    }
}

pub fn get_soft_intervention(sr: &SafetyResult) -> Option<&action::SoftIntervention> {
    match &sr.action {
        Action::SoftIntervention(si) => Some(si),
        Action::TweetInterstitial(ti) => ti.soft_intervention.as_ref(),
        _ => None,
    }
}

pub fn get_tweet_visibility_nudge(sr: &SafetyResult) -> Option<&action::TweetVisibilityNudge> {
    match &sr.action {
        Action::TweetVisibilityNudge(n) => Some(n),
        Action::TweetInterstitial(ti) => ti.tweet_visibility_nudge.as_deref(),
        _ => None,
    }
}

pub fn get_limited_actions_policy(sr: &SafetyResult) -> Option<&action::LimitedActionsPolicy> {
    match &sr.action {
        Action::LimitedEngagements(le) => le.limited_actions_policy.as_ref(),
        Action::TweetInterstitial(ti) => match (&ti.limited_engagements, &ti.interstitial) {
            (Some(le), _) => le.limited_actions_policy.as_ref(),
            (None, Some(AnyInterstitial::InterstitialLimitedEngagements(ile))) => {
                ile.limited_actions_policy.as_ref()
            }
            _ => None,
        },
        Action::InterstitialLimitedEngagements(ile) => ile.limited_actions_policy.as_ref(),
        _ => None,
    }
}

pub fn get_tombstone(reason: &FilteredReason) -> Option<&action::Tombstone> {
    match get_safety_result(reason).map(|sr| &sr.action) {
        Some(Action::Tombstone(t)) => Some(t),
        _ => None,
    }
}

pub fn get_media_interstitial(sr: &SafetyResult) -> Option<&MediaInterstitial> {
    match &sr.action {
        Action::ComposedMediaVisibilityResults(m) => m.media_interstitial.as_deref(),
        Action::TweetInterstitial(ti) => ti
            .all_media_visibility_results
            .as_ref()
            .and_then(|m| m.media_interstitial.as_deref()),
        _ => None,
    }
}

fn has_media_blur_reason(ti: &action::TweetInterstitial) -> bool {
    let candidates = [
        ti.interstitial.as_ref().and_then(|any| match any {
            AnyInterstitial::Interstitial(i) => i.reason.as_ref(),
            _ => None,
        }),
        ti.media_action.as_ref().and_then(|ma| match ma {
            action::MediaAction::Interstitial(i) => i.reason.as_ref(),
            _ => None,
        }),
    ];
    candidates
        .into_iter()
        .flatten()
        .any(|reason| blurred_image_interstitial_resnames(reason).is_some())
}

pub fn resolve_blurred_image_interstitial(
    sr: &SafetyResult,
) -> Option<action::BlurredImageInterstitial> {
    match get_media_interstitial(sr) {
        Some(MediaInterstitial::BlurredImageInterstitial(b)) => Some(b.clone()),
        _ => None,
    }
}

pub fn is_dropped_media_blur(reason: &FilteredReason) -> bool {
    let Some(sr) = get_safety_result(reason) else {
        return false;
    };
    get_media_interstitial(sr).is_none()
        && matches!(
            &sr.action,
            Action::TweetInterstitial(ti) if has_media_blur_reason(ti)
        )
}

pub fn get_fosnr_limited_engagement_appealable_reason(
    reason: &FilteredReason,
) -> Option<&action::AppealableReason> {
    let sr = get_safety_result(reason)?;
    let Action::TweetInterstitial(ti) = &sr.action else {
        return None;
    };
    match ti.limited_engagements.as_ref()?.reason.as_ref()? {
        action::LimitedEngagementReason::FosnrReason(ar) => Some(ar),
        _ => None,
    }
}

fn drop_reason(reason: &FilteredReason) -> Option<&action::DropReason> {
    match &get_safety_result(reason)?.action {
        Action::Drop(d) => d.reason.as_ref(),
        _ => None,
    }
}

pub fn is_tombstone(reason: &FilteredReason) -> bool {
    get_tombstone(reason).and_then(|t| t.reason).is_some()
}

pub fn is_drop_exclusive_tweet(reason: &FilteredReason) -> bool {
    matches!(
        drop_reason(reason),
        Some(action::DropReason::ExclusiveTweet(true))
    )
}

pub fn is_nsfw_viewer_is_underage(reason: &FilteredReason) -> bool {
    matches!(
        drop_reason(reason),
        Some(action::DropReason::NsfwViewerIsUnderage(true))
    )
}

pub fn is_nsfw_viewer_has_no_stated_age(reason: &FilteredReason) -> bool {
    matches!(
        drop_reason(reason),
        Some(action::DropReason::NsfwViewerHasNoStatedAge(true))
    )
}

pub fn is_nsfw_logged_out(reason: &FilteredReason) -> bool {
    matches!(
        drop_reason(reason),
        Some(action::DropReason::NsfwLoggedOut(true))
    )
}

pub fn is_drop_premium_tweet(reason: &FilteredReason) -> bool {
    matches!(
        drop_reason(reason),
        Some(action::DropReason::PremiumTweet(true))
    )
}

pub fn blurred_image_interstitial_resnames(
    reason: &action::InterstitialReason,
) -> Option<(&'static str, &'static str)> {
    use action::InterstitialReason as R;
    macro_rules! copy {
        ($k:literal) => {
            Some((
                concat!("media_interstitials.blurred_image_interstitial_title_", $k),
                concat!("media_interstitials.blurred_image_interstitial_text_", $k),
            ))
        };
    }
    match reason {
        R::ContainsNsfwMedia(_) | R::Sensitive(_) => copy!("nsfw"),
        R::Nudity(_) => copy!("nudity"),
        R::Violence(_) => copy!("violence"),
        R::DisturbingOrControversial(_) => copy!("disturbing_or_controversial"),
        R::SensitiveUser(_) => copy!("sensitive_user"),
        _ => None,
    }
}

pub fn tombstone_resname_and_link(
    reason: action::TombstoneReason,
) -> Option<(&'static str, &'static str)> {
    use crate::links;
    use action::TombstoneReason as R;
    let mapped = match reason {
        R::DELETED => ("tombstone.deleted", links::NOTICES_ON_TWITTER),
        R::BOUNCED | R::BOUNCE_DELETED => ("tombstone.violated", links::ENFORCEMENT_OPTIONS),
        R::PROTECTED_AUTHOR | R::AUTHOR_BLOCKS_VIEWER => {
            ("tombstone.limited", links::NOTICES_ON_TWITTER)
        }
        R::SUSPENDED_AUTHOR => ("tombstone.suspended", links::NOTICES_ON_TWITTER),
        R::DEACTIVATED_AUTHOR => ("tombstone.deactivated", links::NOTICES_ON_TWITTER),
        R::NSFW_VIEWER_IS_UNDERAGE => ("tombstone.underage", links::NOTICES_ON_TWITTER),
        R::GRAPHIC_VIEWER_IS_UNDERAGE => ("tombstone.graphic_underage", links::NOTICES_ON_TWITTER),
        R::NSFW_VIEWER_HAS_NO_STATED_AGE => ("tombstone.no_stated_age", links::NOTICES_ON_TWITTER),
        R::GRAPHIC_VIEWER_HAS_NO_STATED_AGE => {
            ("tombstone.graphic_no_stated_age", links::NOTICES_ON_TWITTER)
        }
        R::NSFW_LOGGED_OUT => ("tombstone.logged_out_age", links::NOTICES_ON_TWITTER),
        R::COMMUNITY_TWEET_HIDDEN => ("tombstone.community_tweet_hidden", links::COMMUNITIES),
        R::PREMIUM_TWEET => ("tombstone.premium_content", links::PREMIUM_CONTENT),
        R::SENSITIVE_VIEWER_AGE_VERIFICATION => {
            ("tombstone.age_verification", links::AGE_ASSURANCE)
        }
        R::UPDATE_APP_IOS => ("tombstone.update_ios", links::AGE_ASSURANCE),
        R::UPDATE_APP_ANDROID => ("tombstone.update_android", links::AGE_ASSURANCE),
        R::LOCAL_REGULATIONS => ("tombstone.local_regulations", links::NOTICES_ON_TWITTER),
        R::DEVELOPMENT_ONLY => ("tombstone.development_only", links::NOTICES_ON_TWITTER),
        R::UNSPECIFIED => ("tombstone.unavailable", links::NOTICES_ON_TWITTER),
        _ => return None,
    };
    Some(mapped)
}

pub fn contextual_thrift_interstitial_resname_link(
    reason: &action::InterstitialReason,
) -> Option<(&'static str, Option<&'static str>)> {
    use crate::links;
    use action::InterstitialReason as R;
    let mapped = match reason {
        R::MatchesMutedKeyword(_) => ("interstitial.muted_keyword", None),
        R::ViewerReportedTweet(_) | R::ViewerReportedAuthor(_) => ("interstitial.reported", None),
        R::ViewerBlocksAuthor(_) => ("interstitial.blocked", None),
        R::ViewerMutesAuthor(_) => ("interstitial.muted", None),
        R::DevelopmentOnly(_) => ("interstitial.development_only", None),
        R::HatefulConduct(_) => (
            "interstitial.hateful_conduct",
            Some(links::HATEFUL_CONDUCT_POLICY),
        ),
        R::AbusiveBehavior(_) => (
            "interstitial.abusive_behavior",
            Some(links::ABUSIVE_BEHAVIOR),
        ),
        _ => return None,
    };
    Some(mapped)
}

pub fn appealable_policy_slug_link(
    policy: i32,
    level: i32,
) -> Option<(&'static str, &'static str)> {
    use crate::links;
    let mapped = match (policy, level) {
        (2, 1) | (2, 3) => ("hateful_conduct", links::ENFORCEMENT_OPTIONS),
        (3, 1) | (3, 3) => ("abuse", links::ENFORCEMENT_OPTIONS),
        (4, 3) => ("violent_speech", links::ENFORCEMENT_OPTIONS),
        (5, 3) => ("civic_integrity", links::ELECTION_INTEGRITY),
        (6, 1) => ("samm", links::MANIPULATED_MEDIA),
        _ => return None,
    };
    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_x_thrift::action::{
        self, AnyInterstitial, Interstitial, InterstitialReason, TweetInterstitial,
    };
    use xai_x_thrift::safety_result::SafetyResult;

    fn nsfw_tweet_interstitial_without_all_media() -> SafetyResult {
        SafetyResult::new(
            None,
            Action::TweetInterstitial(TweetInterstitial::new(
                Some(AnyInterstitial::Interstitial(Interstitial::new(
                    Some(InterstitialReason::ContainsNsfwMedia(true)),
                    None,
                ))),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )),
        )
    }

    #[test]
    fn language_tags_match_case_insensitively_with_primary_and_english_fallback() {
        for lang in ["pt", "PT", "pt-BR", "PT_br"] {
            assert_eq!(
                translate("learn_more_link", lang),
                Some("Saiba mais"),
                "{lang}"
            );
        }
        assert_eq!(translate("learn_more_link", "zh-TW"), Some("了解更多"));
        for lang in ["", "xx", "XX-YY"] {
            assert_eq!(
                translate("learn_more_link", lang),
                Some("Learn more"),
                "{lang}"
            );
        }
        assert_eq!(translate("no_such_resname", "pt"), None);

        for lang in ["ar", "AR", "ar-EG", "He", "ur", "UR-pk", "ckb"] {
            assert!(is_rtl(lang), "{lang}");
        }
        for lang in ["en", "EN-gb", "", "xx"] {
            assert!(!is_rtl(lang), "{lang}");
        }
    }

    #[test]
    fn media_blur_without_wire_all_media_is_dropped() {
        let sr = nsfw_tweet_interstitial_without_all_media();
        assert!(get_media_interstitial(&sr).is_none());
        assert!(resolve_blurred_image_interstitial(&sr).is_none());
        assert!(is_dropped_media_blur(&FilteredReason::SafetyResult(sr)));
    }

    #[test]
    fn prefers_wire_all_media_visibility_results_when_present() {
        let wire = action::BlurredImageInterstitial::new(
            Some(InterstitialReason::Nudity(true)),
            None,
            None,
            None,
        );
        let sr = SafetyResult::new(
            None,
            Action::TweetInterstitial(TweetInterstitial::new(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(action::ComposedMediaVisibilityActions::new(Some(Box::new(
                    MediaInterstitial::BlurredImageInterstitial(wire),
                )))),
            )),
        );

        let blurred = resolve_blurred_image_interstitial(&sr).expect("wire payload");
        assert!(matches!(
            blurred.reason,
            Some(InterstitialReason::Nudity(true))
        ));
    }
}
