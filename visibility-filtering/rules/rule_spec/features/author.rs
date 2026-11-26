use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    AuthorFeatures, AuthorLabel, AuthorLabelSet, AuthorProfile as Profile, QuoteLink,
};
use crate::rules::context::At;
use crate::rules::rule_spec::engine::{Candidate, Engine};
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

pub(in crate::rules) enum AuthorFacts {}

impl Feature for AuthorFacts {
    const REQUIRED: Hydrators = Hydrators::empty();
    type Value<'a> = (u64, Option<QuoteLink>);
}

impl<C: Candidate> Reads<C> for AuthorFacts {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> (u64, Option<QuoteLink>) {
        (at.author().id, at.quote().map(|quote| quote.link))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum AuthorFactsLeaf {
    ViewerIsAuthor,
    IsSelfQuote,
}

impl Leaf for AuthorFactsLeaf {
    type Feature = AuthorFacts;

    #[inline]
    fn meaning<'a, M: Meaning<'a, AuthorFacts>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerIsAuthor => m.run(|(author, _), facts| facts.viewer_id() == Some(author)),
            Self::IsSelfQuote => {
                m.run(|(author, link), _| link.is_some_and(|link| link.quoting_author == author))
            }
        }
    }
}

impl AuthorFactsLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::AuthorFacts(self))
    }
}

pub(in crate::rules) enum AuthorSafety {}

impl Feature for AuthorSafety {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::AuthorSafety);
    type Value<'a> = &'a AuthorFeatures;
}

impl<C: Candidate> Reads<C> for AuthorSafety {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> &'a AuthorFeatures {
        &at.author().features
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "each variant is a leaf name the rule files and policy catalog spell"
)]
pub enum AuthorSafetyLeaf {
    AuthorIsSuspended,
    AuthorIsDeactivated,
    AuthorIsErased,
    AuthorIsOffboarded,
    AuthorIsProtected,
    AuthorIsNsfwUser,
    AuthorIsNsfwAdmin,
}

impl Leaf for AuthorSafetyLeaf {
    type Feature = AuthorSafety;

    #[inline]
    fn meaning<'a, M: Meaning<'a, AuthorSafety>>(self, m: M) -> M::Out {
        match self {
            Self::AuthorIsSuspended => m.run(|author, _| author.is_suspended),
            Self::AuthorIsDeactivated => m.run(|author, _| author.is_deactivated),
            Self::AuthorIsErased => m.run(|author, _| author.is_erased),
            Self::AuthorIsOffboarded => m.run(|author, _| author.is_offboarded),
            Self::AuthorIsProtected => m.run(|author, _| author.is_protected),
            Self::AuthorIsNsfwUser => m.run(|author, _| author.is_nsfw_user),
            Self::AuthorIsNsfwAdmin => m.run(|author, _| author.is_nsfw_admin),
        }
    }
}

impl AuthorSafetyLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::AuthorSafety(self))
    }
}

pub(in crate::rules) enum AuthorProfile {}

impl Feature for AuthorProfile {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::AuthorProfile);
    type Value<'a> = Profile;
}

impl<C: Candidate> Reads<C> for AuthorProfile {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> Profile {
        at.author().profile
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum AuthorProfileLeaf {
    AuthorHasDefaultProfileImage,
    AuthorIsNewAccount,
}

impl Leaf for AuthorProfileLeaf {
    type Feature = AuthorProfile;

    #[inline]
    fn meaning<'a, M: Meaning<'a, AuthorProfile>>(self, m: M) -> M::Out {
        match self {
            Self::AuthorHasDefaultProfileImage => {
                m.run(|profile, _| profile.has_default_profile_image)
            }
            Self::AuthorIsNewAccount => m.run(|profile, _| profile.is_new_account),
        }
    }
}

impl AuthorProfileLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::AuthorProfile(self))
    }
}

pub(in crate::rules) enum AuthorEmail {}

impl Feature for AuthorEmail {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::AuthorConfirmedEmail);
    type Value<'a> = bool;
}

impl<C: Candidate> Reads<C> for AuthorEmail {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> bool {
        at.author().lacks_confirmed_email
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum AuthorEmailLeaf {
    AuthorHasConfirmedEmail,
}

impl Leaf for AuthorEmailLeaf {
    type Feature = AuthorEmail;

    #[inline]
    fn meaning<'a, M: Meaning<'a, AuthorEmail>>(self, m: M) -> M::Out {
        match self {
            Self::AuthorHasConfirmedEmail => m.run(|lacks, _| !lacks),
        }
    }
}

impl AuthorEmailLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::AuthorEmail(self))
    }
}

pub(in crate::rules) enum AuthorPhone {}

impl Feature for AuthorPhone {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::AuthorVerifiedPhone);
    type Value<'a> = bool;
}

impl<C: Candidate> Reads<C> for AuthorPhone {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> bool {
        at.author().lacks_verified_phone
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum AuthorPhoneLeaf {
    AuthorHasVerifiedPhone,
}

impl Leaf for AuthorPhoneLeaf {
    type Feature = AuthorPhone;

    #[inline]
    fn meaning<'a, M: Meaning<'a, AuthorPhone>>(self, m: M) -> M::Out {
        match self {
            Self::AuthorHasVerifiedPhone => m.run(|lacks, _| !lacks),
        }
    }
}

impl AuthorPhoneLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::AuthorPhone(self))
    }
}

pub(in crate::rules) enum AuthorLabels {}

impl Feature for AuthorLabels {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::AuthorLabels);
    type Value<'a> = AuthorLabelSet;
}

impl<C: Candidate> Reads<C> for AuthorLabels {
    #[inline]
    fn read<'a>(at: At<'_, 'a, C>) -> AuthorLabelSet {
        at.author().labels
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum AuthorLabelsLeaf {
    AuthorHasLabel(AuthorLabel),
}

impl Leaf for AuthorLabelsLeaf {
    type Feature = AuthorLabels;

    #[inline]
    fn meaning<'a, M: Meaning<'a, AuthorLabels>>(self, m: M) -> M::Out {
        match self {
            Self::AuthorHasLabel(label) => m.run_with(label, |labels, _| labels.has_label(label)),
        }
    }
}

impl AuthorLabelsLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::AuthorLabels(self))
    }
}
