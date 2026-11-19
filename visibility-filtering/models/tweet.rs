use std::borrow::Cow;
use std::num::NonZeroU64;
use xai_core_entities::entities::TakedownReason;
use xai_strato::Bytes;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaFeature {
    pub has_media: bool,
    pub has_uploaded_media: bool,
    pub has_dmca_media: bool,
    pub geo_allow_list: Vec<String>,
    pub geo_deny_list: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NsfwFeature {
    pub user: bool,
    pub admin: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArticleLifecycle {
    Published,
    SoftDeleted,
    Draft,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TweetFeatures {
    pub text: PostText,
    pub media: MediaFeature,
    pub takedown_reasons: Vec<TakedownReason>,
    pub nsfw: NsfwFeature,
    pub is_nullcast: bool,
    pub community_id: Option<NonZeroU64>,
    pub trusted_friends_list_id: Option<u64>,
    pub latest_edit_tweet_id: Option<u64>,
    pub exclusive_conversation_author_id: Option<u64>,
    pub article_id: Option<NonZeroU64>,
    pub narrowcast_place_id: Option<u64>,
    pub reaction_context: Option<ReactionContext>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PostText(Bytes);

impl PostText {
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }

    pub fn lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.0)
    }
}

#[cfg(test)]
impl From<&str> for PostText {
    fn from(text: &str) -> Self {
        Self(Bytes::copy_from_slice(text.as_bytes()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReactionContext {
    pub reacted_to_post_id: u64,
    pub reacted_to_user_id: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommunityModeration {
    pub is_hidden: bool,
    pub is_author_removed: bool,
}

impl CommunityModeration {
    pub fn is_moderated(self) -> bool {
        self.is_hidden || self.is_author_removed
    }
}

impl TweetFeatures {
    pub fn has_media(&self) -> bool {
        self.media.has_media
    }

    pub fn has_dmca_media(&self) -> bool {
        self.media.has_dmca_media
    }

    pub fn is_superseded_edit(&self, tweet_id: u64) -> bool {
        self.latest_edit_tweet_id
            .is_some_and(|latest| latest > tweet_id)
    }

    pub fn legal_takedown_in(&self, request_country: Option<&str>) -> bool {
        self.takedown_in(request_country, legal_takedown_country)
    }

    pub fn local_laws_takedown_in(&self, request_country: Option<&str>) -> bool {
        self.takedown_in(request_country, local_laws_takedown_country)
    }

    fn takedown_in(
        &self,
        request_country: Option<&str>,
        extractor: fn(&TakedownReason) -> Option<&str>,
    ) -> bool {
        self.takedown_reasons.iter().filter_map(extractor).any(|c| {
            is_worldwide_code(c) || request_country.is_some_and(|v| c.eq_ignore_ascii_case(v))
        })
    }

    pub fn media_restricted_in(&self, request_country: Option<&str>) -> bool {
        let country = request_country.unwrap_or(WORLDWIDE_COUNTRY_CODE);
        let allow = &self.media.geo_allow_list;
        let deny = &self.media.geo_deny_list;
        (!allow.is_empty() && !allow.iter().any(|c| c.eq_ignore_ascii_case(country)))
            || deny.iter().any(|c| c.eq_ignore_ascii_case(country))
    }
}

const WORLDWIDE_COUNTRY_CODE: &str = "xx";
const WORLDWIDE_COPYRIGHT_COUNTRY_CODE: &str = "xy";

fn is_worldwide_code(country_code: &str) -> bool {
    country_code.eq_ignore_ascii_case(WORLDWIDE_COUNTRY_CODE)
        || country_code.eq_ignore_ascii_case(WORLDWIDE_COPYRIGHT_COUNTRY_CODE)
}

fn legal_takedown_country(reason: &TakedownReason) -> Option<&str> {
    match reason {
        TakedownReason::LegalRequest { country_code }
            if country_code.eq_ignore_ascii_case(WORLDWIDE_COPYRIGHT_COUNTRY_CODE) =>
        {
            None
        }
        TakedownReason::LegalRequest { country_code }
        | TakedownReason::UnspecifiedReason { country_code } => Some(country_code),
        TakedownReason::Dmca => Some(WORLDWIDE_COPYRIGHT_COUNTRY_CODE),
        TakedownReason::BystanderReport { .. }
        | TakedownReason::HatefulImagery
        | TakedownReason::SensitiveImagery
        | TakedownReason::Unknown => None,
    }
}

fn local_laws_takedown_country(reason: &TakedownReason) -> Option<&str> {
    match reason {
        TakedownReason::BystanderReport { country_code } if !is_worldwide_code(country_code) => {
            Some(country_code)
        }
        TakedownReason::BystanderReport { .. }
        | TakedownReason::LegalRequest { .. }
        | TakedownReason::UnspecifiedReason { .. }
        | TakedownReason::Dmca
        | TakedownReason::HatefulImagery
        | TakedownReason::SensitiveImagery
        | TakedownReason::Unknown => None,
    }
}
