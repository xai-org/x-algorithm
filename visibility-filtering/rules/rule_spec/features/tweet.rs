use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{PostCandidate, TweetFeatures};
use crate::rules::context::At;
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

pub(in crate::rules) enum Tweet {}

impl Feature for Tweet {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::Tweet);
    type Value<'a> = &'a TweetFeatures;
}

impl Reads<PostCandidate> for Tweet {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> &'a TweetFeatures {
        &at.post().tweet_features
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TweetLeaf {
    TweetNsfwUserFlag,
    TweetNsfwAdminFlag,
    TweetHasDmcaMedia,
    TweetLegalTakedownInRequestCountry,
    TweetLocalLawsTakedownInRequestCountry,
    TweetMediaGeoRestrictedInRequestCountry,
    TweetIsNullcast,
    TweetIsCommunityTweet,
    TweetHasExclusiveContent,
    TweetIsTrustedFriendsTweet,
    TweetHasArticle,
    TweetHasNarrowcastPlace,
    ViewerIsConversationAuthor,
}

impl Leaf for TweetLeaf {
    type Feature = Tweet;

    #[inline]
    fn meaning<'a, M: Meaning<'a, Tweet>>(self, m: M) -> M::Out {
        match self {
            Self::TweetNsfwUserFlag => m.run(|tweet, _| tweet.nsfw.user),
            Self::TweetNsfwAdminFlag => m.run(|tweet, _| tweet.nsfw.admin),
            Self::TweetHasDmcaMedia => m.run(|tweet, _| tweet.has_dmca_media()),
            Self::TweetLegalTakedownInRequestCountry => {
                m.run(|tweet, facts| tweet.legal_takedown_in(facts.request_country()))
            }
            Self::TweetLocalLawsTakedownInRequestCountry => {
                m.run(|tweet, facts| tweet.local_laws_takedown_in(facts.request_country()))
            }
            Self::TweetMediaGeoRestrictedInRequestCountry => {
                m.run(|tweet, facts| tweet.media_restricted_in(facts.request_country()))
            }
            Self::TweetIsNullcast => m.run(|tweet, _| tweet.is_nullcast),
            Self::TweetIsCommunityTweet => m.run(|tweet, _| tweet.community_id.is_some()),
            Self::TweetHasExclusiveContent => {
                m.run(|tweet, _| tweet.exclusive_conversation_author_id.is_some())
            }
            Self::TweetIsTrustedFriendsTweet => {
                m.run(|tweet, _| tweet.trusted_friends_list_id.is_some())
            }
            Self::TweetHasArticle => m.run(|tweet, _| tweet.article_id.is_some()),
            Self::TweetHasNarrowcastPlace => m.run(|tweet, _| tweet.narrowcast_place_id.is_some()),
            Self::ViewerIsConversationAuthor => m.run(|tweet, facts| {
                tweet
                    .exclusive_conversation_author_id
                    .is_some_and(|author| facts.viewer_id() == Some(author))
            }),
        }
    }
}

impl TweetLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::Tweet(self))
    }
}
