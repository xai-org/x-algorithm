use crate::models::{HydratedTweetCandidate, SafetyLabelType, ViewerFeatures};
use crate::params::NsfwGatingCountries;
use xai_core_entities::entities::TakedownReason;
use xai_x_thrift::user_labels::LabelValue;

pub struct RuleContext<'a> {
    viewer: &'a ViewerFeatures,
    candidate: &'a HydratedTweetCandidate,
    nsfw_gating_countries: &'a NsfwGatingCountries,
}

impl<'a> RuleContext<'a> {
    pub(super) fn new(
        viewer: &'a ViewerFeatures,
        candidate: &'a HydratedTweetCandidate,
        nsfw_gating_countries: &'a NsfwGatingCountries,
    ) -> Self {
        Self {
            viewer,
            candidate,
            nsfw_gating_countries,
        }
    }

    #[inline]
    pub fn viewer(&self) -> ViewerPredicates<'_> {
        ViewerPredicates { ctx: self }
    }

    #[inline]
    pub fn tweet(&self) -> TweetPredicates<'_> {
        TweetPredicates { ctx: self }
    }

    #[inline]
    pub fn author(&self) -> AuthorPredicates<'_> {
        AuthorPredicates { ctx: self }
    }

    #[inline]
    pub fn takedown(&self) -> TakedownPredicates<'_> {
        TakedownPredicates { ctx: self }
    }

    #[inline]
    pub fn nsfw_gating_country(&self, country_code: &str) -> bool {
        self.nsfw_gating_countries.contains(country_code)
    }
}

#[derive(Clone, Copy)]
pub struct ViewerPredicates<'a> {
    ctx: &'a RuleContext<'a>,
}

impl ViewerPredicates<'_> {
    #[inline]
    pub fn is_logged_out(&self) -> bool {
        self.ctx.viewer.viewer_is_logged_out()
    }

    #[inline]
    pub fn is_underage(&self) -> bool {
        self.ctx.viewer.viewer_is_underage()
    }

    #[inline]
    pub fn has_no_stated_age(&self) -> bool {
        self.ctx.viewer.viewer_has_no_stated_age()
    }

    #[inline]
    pub fn allows_sensitive_media(&self) -> bool {
        self.ctx.viewer.allows_sensitive_media
    }

    #[inline]
    pub fn country(&self) -> Option<&str> {
        self.ctx
            .viewer
            .account_country_code
            .as_deref()
            .or(self.ctx.viewer.country_code.as_deref())
    }

    #[inline]
    pub fn is_author(&self) -> bool {
        self.ctx.candidate.is_author_viewer(self.ctx.viewer.viewer)
    }

    #[inline]
    pub fn follows_author(&self) -> bool {
        self.ctx.candidate.viewer_follows_author()
    }

    #[inline]
    pub fn blocks_author(&self) -> bool {
        self.ctx.candidate.relationship.viewer_blocks_author
    }

    #[inline]
    pub fn mutes_author(&self) -> bool {
        self.ctx.candidate.relationship.viewer_mutes_author
    }

    #[inline]
    pub fn mutes_retweets_from_author(&self) -> bool {
        self.ctx
            .candidate
            .relationship
            .viewer_mutes_retweets_from_author
    }

    #[inline]
    pub fn is_conversation_author(&self) -> bool {
        match (
            &self.ctx.candidate.exclusive_content,
            self.ctx.viewer.viewer_id(),
        ) {
            (Some(exclusive), Some(viewer_id)) => viewer_id == exclusive.conversation_author_id,
            _ => false,
        }
    }

    #[inline]
    pub fn super_follows_author(&self) -> bool {
        self.ctx
            .candidate
            .exclusive_content
            .as_ref()
            .is_some_and(|exclusive| exclusive.viewer_super_follows_author)
    }
}

#[derive(Clone, Copy)]
pub struct TweetPredicates<'a> {
    ctx: &'a RuleContext<'a>,
}

impl TweetPredicates<'_> {
    #[inline]
    pub fn has_safety_label(&self, label: SafetyLabelType) -> bool {
        self.ctx.candidate.has_safety_label(label)
    }

    #[inline]
    pub fn is_retweet(&self) -> bool {
        self.ctx.candidate.is_retweet()
    }

    #[inline]
    pub fn is_stale(&self) -> bool {
        self.ctx.candidate.is_stale_tweet()
    }

    #[inline]
    pub fn is_nullcast(&self) -> bool {
        self.ctx.candidate.is_nullcast()
    }

    #[inline]
    pub fn is_community_tweet(&self) -> bool {
        self.ctx.candidate.is_community_tweet()
    }

    #[inline]
    pub fn has_media(&self) -> bool {
        self.ctx.candidate.has_media()
    }

    #[inline]
    pub fn has_dmca_media(&self) -> bool {
        self.ctx.candidate.has_dmca_media()
    }

    #[inline]
    pub fn is_nsfw_flagged(&self) -> bool {
        self.ctx.candidate.is_nsfw_flagged()
    }

    #[inline]
    pub fn has_nsfw_user_flag(&self) -> bool {
        self.ctx.candidate.tweet_features.nsfw.user
    }

    #[inline]
    pub fn has_nsfw_admin_flag(&self) -> bool {
        self.ctx.candidate.tweet_features.nsfw.admin
    }

    #[inline]
    pub fn is_exclusive(&self) -> bool {
        self.ctx.candidate.exclusive_content.is_some()
    }
}

#[derive(Clone, Copy)]
pub struct AuthorPredicates<'a> {
    ctx: &'a RuleContext<'a>,
}

impl AuthorPredicates<'_> {
    #[inline]
    pub fn is_suspended(&self) -> bool {
        self.ctx.candidate.author_features.is_suspended
    }

    #[inline]
    pub fn is_deactivated(&self) -> bool {
        self.ctx.candidate.author_features.is_deactivated
    }

    #[inline]
    pub fn is_erased(&self) -> bool {
        self.ctx.candidate.author_features.is_erased
    }

    #[inline]
    pub fn is_offboarded(&self) -> bool {
        self.ctx.candidate.author_features.is_offboarded
    }

    #[inline]
    pub fn is_protected(&self) -> bool {
        self.ctx.candidate.author_features.is_protected
    }

    #[inline]
    pub fn is_nsfw_user(&self) -> bool {
        self.ctx.candidate.author_features.is_nsfw_user
    }

    #[inline]
    pub fn is_nsfw_admin(&self) -> bool {
        self.ctx.candidate.author_features.is_nsfw_admin
    }

    #[inline]
    pub fn has_user_label(&self, label: LabelValue) -> bool {
        self.ctx.candidate.author_has_user_label(label)
    }
}

#[derive(Clone, Copy)]
pub struct TakedownPredicates<'a> {
    ctx: &'a RuleContext<'a>,
}

impl TakedownPredicates<'_> {
    #[inline]
    pub fn legal_in_viewer_country(&self) -> bool {
        self.in_viewer_country(legal_takedown_country)
    }

    #[inline]
    pub fn local_laws_in_viewer_country(&self) -> bool {
        self.in_viewer_country(local_laws_takedown_country)
    }

    #[inline]
    fn in_viewer_country(&self, extractor: fn(&TakedownReason) -> Option<&str>) -> bool {
        let viewer_country = self
            .ctx
            .viewer
            .account_country_code
            .as_deref()
            .or(self.ctx.viewer.country_code.as_deref());
        self.ctx
            .candidate
            .tweet_features
            .takedown_reasons
            .iter()
            .filter_map(extractor)
            .any(|c| {
                c.eq_ignore_ascii_case(WORLDWIDE_COUNTRY_CODE)
                    || c.eq_ignore_ascii_case(WORLDWIDE_COPYRIGHT_COUNTRY_CODE)
                    || viewer_country.is_some_and(|v| c.eq_ignore_ascii_case(v))
            })
    }

    #[inline]
    pub fn media_restricted_in_viewer_country(&self) -> bool {
        let country = self
            .ctx
            .viewer
            .account_country_code
            .as_deref()
            .or(self.ctx.viewer.country_code.as_deref())
            .unwrap_or(WORLDWIDE_COUNTRY_CODE);
        let allow = &self.ctx.candidate.tweet_features.media.geo_allow_list;
        let deny = &self.ctx.candidate.tweet_features.media.geo_deny_list;
        (!allow.is_empty() && !allow.iter().any(|c| c.eq_ignore_ascii_case(country)))
            || deny.iter().any(|c| c.eq_ignore_ascii_case(country))
    }
}

const WORLDWIDE_COUNTRY_CODE: &str = "xx";
const WORLDWIDE_COPYRIGHT_COUNTRY_CODE: &str = "xy";

fn legal_takedown_country(reason: &TakedownReason) -> Option<&str> {
    match reason {
        TakedownReason::LegalRequest { country_code }
        | TakedownReason::UnspecifiedReason { country_code } => Some(country_code),
        TakedownReason::Dmca => Some(WORLDWIDE_COPYRIGHT_COUNTRY_CODE),
        _ => None,
    }
}

fn local_laws_takedown_country(reason: &TakedownReason) -> Option<&str> {
    match reason {
        TakedownReason::BystanderReport { country_code } => Some(country_code),
        _ => None,
    }
}
