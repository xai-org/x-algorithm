use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{NotificationFilters, PostCandidate, Viewer, ViewerProfile};
use crate::params::CountryList;
use crate::rules::context::At;
use crate::rules::rule_spec::engine::{Candidate, Engine};
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::{account_or_request_country_in, Predicate};

pub(in crate::rules) enum ViewerAccount {}

impl Feature for ViewerAccount {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ViewerProfile);
    type Value<'a> = Option<&'a ViewerProfile>;
}

impl<C: Candidate> Reads<C> for ViewerAccount {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> Option<&'a ViewerProfile> {
        match &at.viewer().viewer {
            Viewer::LoggedIn { profile, .. } => Some(profile),
            Viewer::LoggedOut => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "each variant is a leaf name the rule files and policy catalog spell"
)]
pub enum ViewerAccountLeaf {
    ViewerUnderage,
    ViewerNoStatedAge,
    ViewerAllowsSensitiveMedia,
    ViewerAccountOrRequestCountryIn(CountryList),
    ViewerHasVerifiedBadge,
    ViewerReadOnly,
    ViewerIsNewAccount,
    ViewerIsSuspended,
    ViewerIsDeactivated,
    ViewerHasUqfEnabled,
    ViewerMentionFilterIsFollowing,
}

impl Leaf for ViewerAccountLeaf {
    type Feature = ViewerAccount;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ViewerAccount>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerUnderage => {
                m.run(|profile, _| profile.is_some_and(ViewerProfile::is_underage))
            }
            Self::ViewerNoStatedAge => {
                m.run(|profile, _| profile.is_some_and(ViewerProfile::has_no_stated_age))
            }
            Self::ViewerAllowsSensitiveMedia => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.allows_sensitive_media))
            }
            Self::ViewerAccountOrRequestCountryIn(list) => m.run_with(list, |profile, facts| {
                account_or_request_country_in(profile, facts.request_country(), |country| {
                    facts.in_country_list(list, country)
                })
            }),
            Self::ViewerHasVerifiedBadge => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.has_verified_badge))
            }
            Self::ViewerReadOnly => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.is_read_only))
            }
            Self::ViewerIsNewAccount => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.is_new_account))
            }
            Self::ViewerIsSuspended => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.is_suspended))
            }
            Self::ViewerIsDeactivated => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.is_deactivated))
            }
            Self::ViewerHasUqfEnabled => {
                m.run(|profile, _| profile.is_some_and(|profile| profile.has_uqf_enabled))
            }
            Self::ViewerMentionFilterIsFollowing => m.run(|profile, _| {
                profile.is_some_and(|profile| profile.mention_filter_is_following)
            }),
        }
    }
}

impl ViewerAccountLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::ViewerAccount(self))
    }
}

pub(in crate::rules) enum ViewerNotificationFilters {}

impl Feature for ViewerNotificationFilters {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ViewerNotificationFilters);
    type Value<'a> = NotificationFilters;
}

impl<C: Candidate> Reads<C> for ViewerNotificationFilters {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> NotificationFilters {
        match &at.viewer().viewer {
            Viewer::LoggedIn { profile, .. } => profile.notification_filters,
            Viewer::LoggedOut => NotificationFilters::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "each variant is the condition the policy catalog prints, named as Scala names it"
)]
pub enum NotificationFilterLeaf {
    ViewerFiltersNoConfirmedEmail,
    ViewerFiltersNoConfirmedPhone,
    ViewerFiltersDefaultProfileImage,
    ViewerFiltersNewUsers,
    ViewerFiltersNotFollowedBy,
}

impl Leaf for NotificationFilterLeaf {
    type Feature = ViewerNotificationFilters;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ViewerNotificationFilters>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerFiltersNoConfirmedEmail => m.run(|filters, _| filters.no_confirmed_email),
            Self::ViewerFiltersNoConfirmedPhone => m.run(|filters, _| filters.no_confirmed_phone),
            Self::ViewerFiltersDefaultProfileImage => {
                m.run(|filters, _| filters.default_profile_image)
            }
            Self::ViewerFiltersNewUsers => m.run(|filters, _| filters.new_users),
            Self::ViewerFiltersNotFollowedBy => m.run(|filters, _| filters.not_followed_by),
        }
    }
}

impl NotificationFilterLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::NotificationFilter(self))
    }
}

pub(in crate::rules) enum ViewerBlinkLabel {}

impl Feature for ViewerBlinkLabel {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ViewerLabels);
    type Value<'a> = bool;
}

impl<C: Candidate> Reads<C> for ViewerBlinkLabel {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> bool {
        match &at.viewer().viewer {
            Viewer::LoggedIn { profile, .. } => profile.has_blink_label,
            Viewer::LoggedOut => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ViewerLabelLeaf {
    ViewerHasBlinkLabel,
}

impl Leaf for ViewerLabelLeaf {
    type Feature = ViewerBlinkLabel;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ViewerBlinkLabel>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerHasBlinkLabel => m.run(|has_label, _| has_label),
        }
    }
}

impl ViewerLabelLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::ViewerLabel(self))
    }
}

pub(in crate::rules) enum ViewerLabels {}

impl Feature for ViewerLabels {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ViewerLabels);
    type Value<'a> = bool;
}

impl<C: Candidate> Reads<C> for ViewerLabels {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> bool {
        match &at.viewer().viewer {
            Viewer::LoggedIn {
                has_age_verified_18_label,
                ..
            } => *has_age_verified_18_label,
            Viewer::LoggedOut => false,
        }
    }
}

pub(in crate::rules) enum ViewerCountry {}

impl Feature for ViewerCountry {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ViewerCountry);
    type Value<'a> = Option<&'a str>;
}

impl Reads<PostCandidate> for ViewerCountry {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> Option<&'a str> {
        let features = at.post().conversation_control.as_ref();
        features.and_then(|features| features.viewer_country.as_deref())
    }
}
