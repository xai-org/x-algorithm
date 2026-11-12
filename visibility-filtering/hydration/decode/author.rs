use super::viewer::is_younger_than;
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::fallback_cache::{Column, FallbackCache};
use crate::models::{AuthorFeatures, AuthorLabel, AuthorLabelSet, AuthorProfile};
use std::time::Duration;
use strum::VariantArray;
use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, UserResponseState};
use xai_x_thrift::user_labels::LabelValue;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodedAuthor {
    pub(crate) features: AuthorFeatures,
    pub(crate) labels: AuthorLabelSet,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodedActor {
    pub(crate) author: DecodedAuthor,
    pub(crate) profile: AuthorProfile,
}

pub(crate) type AuthorFallbackCache = FallbackCache<DecodedAuthor>;

pub(crate) fn fallback_cache(capacity: usize) -> AuthorFallbackCache {
    FallbackCache::new("author", capacity)
}

pub(crate) struct AuthorColumn;

impl Column for AuthorColumn {
    type Entry = DecodedAuthor;
    type Value = DecodedAuthor;
    type Stored = DecodedAuthor;
    const NAME: &'static str = "author";

    fn store(value: &DecodedAuthor) -> DecodedAuthor {
        *value
    }

    fn new_entry(stored: DecodedAuthor) -> DecodedAuthor {
        stored
    }

    fn replace(entry: &mut DecodedAuthor, stored: DecodedAuthor) -> Option<DecodedAuthor> {
        Some(std::mem::replace(entry, stored))
    }

    fn get(entry: &DecodedAuthor) -> Option<DecodedAuthor> {
        Some(*entry)
    }

    fn holds(_: &DecodedAuthor) -> bool {
        true
    }

    fn others_hold(_: &DecodedAuthor) -> bool {
        false
    }

    fn clear(_: &mut DecodedAuthor) -> Option<DecodedAuthor> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AuthorNotFound {
    #[default]
    Unresolves,
    ReadsErased,
}

impl AuthorNotFound {
    pub(crate) fn read(
        self,
        authors: RawHydrationBatch<DecodedAuthor>,
    ) -> RawHydrationBatch<DecodedAuthor> {
        match self {
            AuthorNotFound::Unresolves => authors,
            AuthorNotFound::ReadsErased => {
                let mut authors = authors.into_hydrated();
                for decoded in authors.values_mut() {
                    if let Hydrated::NotFound = decoded {
                        *decoded = Hydrated::Found(erased_author());
                    }
                }
                HydrationBatch::from_hydrated(authors)
            }
        }
    }
}

fn erased_author() -> DecodedAuthor {
    DecodedAuthor {
        features: AuthorFeatures {
            is_erased: true,
            ..AuthorFeatures::default()
        },
        labels: AuthorLabelSet::default(),
    }
}

pub(crate) fn decoded_author(user: Hydrated<GizmoduckUserResult>) -> Hydrated<DecodedAuthor> {
    match user {
        Hydrated::Found(result) | Hydrated::Partial(result) => evaluable_author_features(result),
        Hydrated::NotFound => Hydrated::NotFound,
        Hydrated::Failed => Hydrated::Failed,
        Hydrated::Unanswered => Hydrated::Unanswered,
    }
}

pub(crate) fn decoded_actor(user: Hydrated<GizmoduckUserResult>) -> Hydrated<DecodedActor> {
    match user {
        Hydrated::Found(result) | Hydrated::Partial(result) => {
            let profile = result.user.as_ref().map(author_profile).unwrap_or_default();
            evaluable_author_features(result).map(|author| DecodedActor { author, profile })
        }
        Hydrated::NotFound => Hydrated::NotFound,
        Hydrated::Failed => Hydrated::Failed,
        Hydrated::Unanswered => Hydrated::Unanswered,
    }
}

const NEW_AUTHOR_AGE: Duration = Duration::from_secs(72 * 60 * 60);

fn author_profile(user: &GizmoduckUser) -> AuthorProfile {
    AuthorProfile {
        has_default_profile_image: !user.profile.has_profile_image,
        is_new_account: is_younger_than(user.created_at_msec, NEW_AUTHOR_AGE),
    }
}

fn evaluable_author_features(result: GizmoduckUserResult) -> Hydrated<DecodedAuthor> {
    match result.response_state {
        Some(UserResponseState::Failed | UserResponseState::Filtered) => Hydrated::Failed,
        Some(
            UserResponseState::NotFound
            | UserResponseState::SoftUser
            | UserResponseState::PeriscopeUser
            | UserResponseState::NoScreenNameUser,
        ) => Hydrated::NotFound,
        None | Some(UserResponseState::Found | UserResponseState::Partial)
            if result.user.is_none() =>
        {
            Hydrated::NotFound
        }
        None | Some(UserResponseState::Partial) => Hydrated::Partial(author_features(result)),
        Some(
            UserResponseState::Found
            | UserResponseState::DeactivatedUser
            | UserResponseState::SuspendedUser
            | UserResponseState::ProtectedUser
            | UserResponseState::ErasedUser
            | UserResponseState::UnsafeUser
            | UserResponseState::OffboardedUser,
        ) => Hydrated::Found(author_features(result)),
    }
}

fn author_features(user_result: GizmoduckUserResult) -> DecodedAuthor {
    user_result
        .user
        .map(|user| {
            let mut labels = AuthorLabelSet::default();
            for label in &user.labels.labels {
                if let Some(modeled) = author_label(LabelValue(label.label_value)) {
                    labels.insert(modeled);
                }
            }
            let features = AuthorFeatures {
                is_suspended: user.safety.suspended,
                is_deactivated: user.safety.deactivated,
                is_protected: user.safety.is_protected,
                is_nsfw_user: user.safety.nsfw_user,
                is_nsfw_admin: user.safety.nsfw_admin,
                is_erased: user.safety.erased,
                is_offboarded: user.safety.offboarded,
            };
            DecodedAuthor { features, labels }
        })
        .unwrap_or_default()
}

fn author_label(value: LabelValue) -> Option<AuthorLabel> {
    AuthorLabel::VARIANTS
        .iter()
        .copied()
        .find(|&label| gizmoduck_label(label) == value)
}

const fn gizmoduck_label(label: AuthorLabel) -> LabelValue {
    match label {
        AuthorLabel::NsfwHighRecall => LabelValue::NSFW_HIGH_RECALL,
        AuthorLabel::NsfwHighPrecision => LabelValue::NSFW_HIGH_PRECISION,
        AuthorLabel::NsfwNearPerfect => LabelValue::NSFW_NEAR_PERFECT,
        AuthorLabel::NsfwAvatarImage => LabelValue::NSFW_AVATAR_IMAGE,
        AuthorLabel::NsfwBannerImage => LabelValue::NSFW_BANNER_IMAGE,
        AuthorLabel::SpamHighRecall => LabelValue::SPAM_HIGH_RECALL,
        AuthorLabel::AbusiveHighRecall => LabelValue::ABUSIVE_HIGH_RECALL,
        AuthorLabel::Compromised => LabelValue::COMPROMISED,
        AuthorLabel::ReadOnly => LabelValue::READ_ONLY,
        AuthorLabel::ImpersonationHighPrecision => LabelValue::IMPERSONATION_HIGH_PRECISION,
        AuthorLabel::DoNotAmplify => LabelValue::DO_NOT_AMPLIFY,
        AuthorLabel::NotGraduated => LabelValue::NOT_GRADUATED,
        AuthorLabel::AgathaSpamTopUser => LabelValue::AGATHA_SPAM_TOP_USER,
        AuthorLabel::EngagementSpammer => LabelValue::ENGAGEMENT_SPAMMER,
        AuthorLabel::BlinkWorst => LabelValue::BLINK_WORST,
        AuthorLabel::DownrankSpamReply => LabelValue::DOWNRANK_SPAM_REPLY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::{GizmoduckUser, Label, Labels, Safety};

    #[test]
    fn response_states_decode_as_tweetypie_reads_them() {
        let suspended = GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety: Safety {
                    suspended: true,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        let found = Hydrated::Found(());
        for (state, expected) in [
            (Some(UserResponseState::Found), found.clone()),
            (Some(UserResponseState::DeactivatedUser), found.clone()),
            (Some(UserResponseState::SuspendedUser), found.clone()),
            (Some(UserResponseState::ProtectedUser), found.clone()),
            (Some(UserResponseState::ErasedUser), found.clone()),
            (Some(UserResponseState::OffboardedUser), found.clone()),
            (Some(UserResponseState::UnsafeUser), found),
            (Some(UserResponseState::Partial), Hydrated::Partial(())),
            (None, Hydrated::Partial(())),
            (Some(UserResponseState::NotFound), Hydrated::NotFound),
            (Some(UserResponseState::SoftUser), Hydrated::NotFound),
            (Some(UserResponseState::PeriscopeUser), Hydrated::NotFound),
            (
                Some(UserResponseState::NoScreenNameUser),
                Hydrated::NotFound,
            ),
            (Some(UserResponseState::Failed), Hydrated::Failed),
            (Some(UserResponseState::Filtered), Hydrated::Failed),
        ] {
            let author = evaluable_author_features(GizmoduckUserResult {
                response_state: state,
                ..suspended.clone()
            });
            assert_eq!(shape(&author), expected, "{state:?}");
            if let Some(author) = author.value() {
                assert!(author.features.is_suspended, "{state:?}");
            }
        }
    }

    #[test]
    fn a_success_state_without_a_user_is_not_found() {
        for state in [
            None,
            Some(UserResponseState::Found),
            Some(UserResponseState::Partial),
        ] {
            let author = evaluable_author_features(GizmoduckUserResult {
                response_state: state,
                user: None,
            });
            assert_eq!(shape(&author), Hydrated::NotFound, "{state:?}");
        }
    }

    fn shape(author: &Hydrated<DecodedAuthor>) -> Hydrated<()> {
        match author {
            Hydrated::Found(_) => Hydrated::Found(()),
            Hydrated::Partial(_) => Hydrated::Partial(()),
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed => Hydrated::Failed,
            Hydrated::Unanswered => Hydrated::Unanswered,
        }
    }

    fn user_with_labels(label_values: &[i32]) -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                user_id: 1,
                labels: Labels {
                    labels: label_values
                        .iter()
                        .map(|&label_value| Label {
                            label_value,
                            created_at_msec: 0,
                        })
                        .collect(),
                },
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn every_author_label_variant_round_trips_from_its_thrift_constant() {
        for (thrift, variant) in [
            (LabelValue::NSFW_HIGH_RECALL, AuthorLabel::NsfwHighRecall),
            (
                LabelValue::NSFW_HIGH_PRECISION,
                AuthorLabel::NsfwHighPrecision,
            ),
            (LabelValue::NSFW_NEAR_PERFECT, AuthorLabel::NsfwNearPerfect),
            (LabelValue::NSFW_AVATAR_IMAGE, AuthorLabel::NsfwAvatarImage),
            (LabelValue::NSFW_BANNER_IMAGE, AuthorLabel::NsfwBannerImage),
            (LabelValue::SPAM_HIGH_RECALL, AuthorLabel::SpamHighRecall),
            (
                LabelValue::ABUSIVE_HIGH_RECALL,
                AuthorLabel::AbusiveHighRecall,
            ),
            (LabelValue::COMPROMISED, AuthorLabel::Compromised),
            (LabelValue::READ_ONLY, AuthorLabel::ReadOnly),
            (
                LabelValue::IMPERSONATION_HIGH_PRECISION,
                AuthorLabel::ImpersonationHighPrecision,
            ),
            (LabelValue::DO_NOT_AMPLIFY, AuthorLabel::DoNotAmplify),
        ] {
            let labels = author_features(user_with_labels(&[thrift.0])).labels;
            assert!(labels.has_label(variant), "{thrift:?}");
        }

        for unmodelled in [
            LabelValue::EGREGIOUS_NSFW,
            LabelValue::RECOMMENDATIONS_BLACKLIST,
        ] {
            let labels = author_features(user_with_labels(&[
                unmodelled.0,
                LabelValue::SPAM_HIGH_RECALL.0,
            ]))
            .labels;
            let mut expected = AuthorLabelSet::default();
            expected.insert(AuthorLabel::SpamHighRecall);
            assert_eq!(labels, expected, "{unmodelled:?}");
        }
    }
}
