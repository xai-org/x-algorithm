use crate::models::{NotificationFilters, ViewerAge, ViewerProfile};
use std::time::Duration;
use xai_core_entities::entities::{
    AccessPolicy, GizmoduckUser, Labels, Safety, SubscriptionLevel, UniversalQualityFiltering,
    VerifiedType,
};
use xai_core_entities::gizmoduck_client::{QueryFields, ViewerData};
use xai_x_thrift::user_labels::LabelValue;

fn has_verified_badge(data: &ViewerData) -> bool {
    matches!(
        data.verified_type,
        Some(VerifiedType::Business | VerifiedType::Government)
    ) || data.is_blue_verified
}

fn has_idv_premium(data: &ViewerData) -> bool {
    matches!(
        data.subscription_level,
        Some(SubscriptionLevel::Premium | SubscriptionLevel::PremiumPlus)
    ) || matches!(
        data.verified_type,
        Some(VerifiedType::Business | VerifiedType::Government)
    )
}

fn has_age_verified_18_label(labels: &Labels) -> bool {
    labels
        .labels
        .iter()
        .any(|label| LabelValue(label.label_value) == LabelValue::AGE_VERIFIED_18)
}

fn has_blink_label(labels: &Labels) -> bool {
    labels.labels.iter().any(|label| {
        [
            LabelValue::BLINK_WORST,
            LabelValue::BLINK_BAD,
            LabelValue::BLINK_QUESTIONABLE,
        ]
        .contains(&LabelValue(label.label_value))
    })
}

fn notification_filters(user: &GizmoduckUser, fields: &[QueryFields]) -> NotificationFilters {
    let Some(settings) = user
        .mute_settings
        .as_ref()
        .filter(|_| fields.contains(&QueryFields::MUTE_SETTINGS))
    else {
        return NotificationFilters::default();
    };
    let filters = &settings.advanced_notification_filters;
    NotificationFilters {
        no_confirmed_email: filters.filter_no_confirmed_email,
        no_confirmed_phone: filters.filter_no_confirmed_phone,
        default_profile_image: filters.filter_default_profile_image,
        new_users: filters.filter_new_users,
        not_followed_by: filters.filter_not_followed_by,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum ViewerState {
    Offboarded,
    Erased,
    Deactivated,
    Suspended,
    Active,
}

impl ViewerState {
    fn of(safety: &Safety) -> Self {
        if safety.offboarded {
            Self::Offboarded
        } else if safety.erased {
            Self::Erased
        } else if safety.deactivated {
            Self::Deactivated
        } else if safety.suspended {
            Self::Suspended
        } else {
            Self::Active
        }
    }
}

#[derive(Default)]
pub(crate) struct DecodedViewer {
    pub(crate) profile: ViewerProfile,
    pub(crate) has_age_verified_18_label: bool,
    pub(crate) state: Option<ViewerState>,
}

pub(crate) fn decode_viewer(user: Option<&GizmoduckUser>, fields: &[QueryFields]) -> DecodedViewer {
    match user {
        Some(user) => DecodedViewer {
            profile: ViewerProfile {
                is_new_account: is_younger_than(user.created_at_msec, NEW_VIEWER_AGE),
                has_uqf_enabled: fields.contains(&QueryFields::SAFETY)
                    && user.safety.universal_quality_filtering
                        == UniversalQualityFiltering::Enabled,
                has_blink_label: fields.contains(&QueryFields::LABELS)
                    && has_blink_label(&user.labels),
                notification_filters: notification_filters(user, fields),
                ..profile(ViewerData::from_user(user, fields))
            },
            has_age_verified_18_label: has_age_verified_18_label(&user.labels),
            state: Some(ViewerState::of(&user.safety)),
        },
        None => DecodedViewer::default(),
    }
}

const NEW_VIEWER_AGE: Duration = Duration::from_secs(24 * 60 * 60);

pub(super) fn is_younger_than(created_at_msec: Option<i64>, age: Duration) -> bool {
    let Some(created_at) = created_at_msec
        .and_then(|ms| u64::try_from(ms).ok())
        .map(|ms| std::time::UNIX_EPOCH + Duration::from_millis(ms))
    else {
        return false;
    };
    std::time::SystemTime::now()
        .duration_since(created_at)
        .unwrap_or_default()
        < age
}

fn profile(data: ViewerData) -> ViewerProfile {
    let viewer_age = data
        .age_in_years
        .map_or(ViewerAge::NotStated, ViewerAge::Known);
    ViewerProfile {
        allows_sensitive_media: data.nsfw_view.unwrap_or(false),
        viewer_age,
        has_verified_badge: has_verified_badge(&data),
        has_idv_premium: has_idv_premium(&data),
        is_read_only: data.access_policy == AccessPolicy::BounceAllPublicWrites,
        account_country_code: data.account_country_code.map(|c| c.to_ascii_lowercase()),
        is_new_account: false,
        is_suspended: data.is_suspended,
        is_deactivated: data.is_deactivated,
        has_uqf_enabled: false,
        mention_filter_is_following: data.opted_out_oon,
        has_blink_label: false,
        notification_filters: NotificationFilters::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::Label;

    #[test]
    fn a_found_viewer_without_an_age_has_none_stated_whatever_its_state() {
        for (data, expected_age, expected_sensitive_media) in [
            (
                ViewerData {
                    user_exists: true,
                    nsfw_view: Some(false),
                    age_in_years: None,
                    ..Default::default()
                },
                ViewerAge::NotStated,
                false,
            ),
            (
                ViewerData {
                    user_exists: true,
                    nsfw_view: Some(true),
                    age_in_years: Some(30),
                    ..Default::default()
                },
                ViewerAge::Known(30),
                true,
            ),
            (
                ViewerData {
                    user_exists: false,
                    age_in_years: None,
                    ..Default::default()
                },
                ViewerAge::NotStated,
                false,
            ),
        ] {
            let viewer = profile(data);
            assert_eq!(viewer.viewer_age, expected_age);
            assert_eq!(viewer.allows_sensitive_media, expected_sensitive_media);
        }
    }

    #[test]
    fn verified_badge_requires_org_type_or_blue_check() {
        let cases = [
            (Some(VerifiedType::Business), false, true),
            (Some(VerifiedType::Government), false, true),
            (None, true, true),
            (Some(VerifiedType::User), false, false),
            (Some(VerifiedType::Notable), false, false),
            (None, false, false),
        ];
        for (verified_type, is_blue_verified, expected) in cases {
            let data = ViewerData {
                verified_type,
                is_blue_verified,
                ..Default::default()
            };
            assert_eq!(
                profile(data).has_verified_badge,
                expected,
                "{verified_type:?} blue={is_blue_verified}"
            );
        }
    }

    #[test]
    fn only_bounce_all_public_writes_makes_the_viewer_read_only() {
        for (access_policy, expected) in [
            (AccessPolicy::Normal, false),
            (AccessPolicy::BounceAll, false),
            (AccessPolicy::BounceAllWritesAndNpci, false),
            (AccessPolicy::BounceAllPublicWrites, true),
            (AccessPolicy::BounceOnUnsuspension, false),
        ] {
            let viewer = profile(ViewerData {
                access_policy,
                ..Default::default()
            });
            assert_eq!(viewer.is_read_only, expected, "{access_policy:?}");
        }
    }

    #[test]
    fn idv_premium_is_a_premium_tier_or_an_org_verified_type() {
        use SubscriptionLevel::{Basic, Premium, PremiumPlus};
        use VerifiedType::{Business, Government, Notable, User};
        let cases = [
            (Some(Premium), None, true),
            (Some(PremiumPlus), None, true),
            (Some(Basic), Some(Business), true),
            (None, Some(Government), true),
            (Some(Basic), Some(User), false),
            (None, Some(Notable), false),
            (None, None, false),
        ];
        for (subscription_level, verified_type, expected) in cases {
            let data = ViewerData {
                subscription_level,
                verified_type,
                ..Default::default()
            };
            assert_eq!(
                profile(data).has_idv_premium,
                expected,
                "{subscription_level:?} {verified_type:?}"
            );
        }
    }

    #[test]
    fn only_the_age_verified_18_label_marks_the_viewer() {
        let labels = |values: &[LabelValue]| Labels {
            labels: values
                .iter()
                .map(|value| Label {
                    label_value: value.0,
                    created_at_msec: 0,
                })
                .collect(),
        };
        for (values, expected) in [
            (
                &[LabelValue::AGE_VERIFIED_16, LabelValue::AGE_VERIFIED_18][..],
                true,
            ),
            (&[LabelValue::AGE_VERIFIED_16], false),
            (&[], false),
        ] {
            let user = GizmoduckUser {
                labels: labels(values),
                ..Default::default()
            };
            let fields = [QueryFields::SAFETY, QueryFields::LABELS];
            assert_eq!(
                decode_viewer(Some(&user), &fields).has_age_verified_18_label,
                expected,
                "{values:?}"
            );
        }
    }

    #[test]
    fn viewer_state_takes_the_first_set_flag_and_a_missing_viewer_has_none() {
        let all = Safety {
            offboarded: true,
            erased: true,
            deactivated: true,
            suspended: true,
            ..Default::default()
        };
        for (safety, expected) in [
            (all.clone(), ViewerState::Offboarded),
            (
                Safety {
                    offboarded: false,
                    ..all.clone()
                },
                ViewerState::Erased,
            ),
            (
                Safety {
                    suspended: true,
                    deactivated: true,
                    ..Default::default()
                },
                ViewerState::Deactivated,
            ),
            (
                Safety {
                    suspended: true,
                    ..Default::default()
                },
                ViewerState::Suspended,
            ),
            (Safety::default(), ViewerState::Active),
        ] {
            let user = GizmoduckUser {
                safety,
                ..Default::default()
            };
            assert_eq!(
                decode_viewer(Some(&user), &[QueryFields::SAFETY]).state,
                Some(expected)
            );
        }
        assert_eq!(decode_viewer(None, &[QueryFields::SAFETY]).state, None);
    }

    #[test]
    fn a_creation_time_ahead_of_the_clock_reads_as_new() {
        let ahead = std::time::SystemTime::now() + Duration::from_secs(60);
        let since = ahead.duration_since(std::time::UNIX_EPOCH).unwrap();
        let ms = i64::try_from(since.as_millis()).unwrap();
        assert!(is_younger_than(Some(ms), NEW_VIEWER_AGE));
    }
}
