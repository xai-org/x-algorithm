use super::article::{ArticleCoverMedia, ArticleMediaEntities};
use super::post::{PostFacts, ReplyControl};
use super::tweet::Tweet;
use super::viewer::{ViewerAccount, ViewerCountry, ViewerLabels};
use super::{EdgeFeature, Feature, In, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::region::allows_country;
use crate::models::PostCandidate;
use crate::rules::context::At;
use crate::rules::rule_spec::inputs::Interpreter;
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

type MuteRetweets = EdgeFeature<{ Hydrator::MuteRetweets as u8 }>;
type SuperFollowsExclusive = EdgeFeature<{ Hydrator::SuperFollowsExclusive as u8 }>;
type RootFollowsViewer = EdgeFeature<{ Hydrator::RootFollowsViewer as u8 }>;
type RootFollowsViewerSecondDegree = EdgeFeature<{ Hydrator::RootFollowsViewerSecondDegree as u8 }>;
type ViewerFollowsRoot = EdgeFeature<{ Hydrator::ViewerFollowsRoot as u8 }>;
type SuperFollowsRoot = EdgeFeature<{ Hydrator::SuperFollowsRoot as u8 }>;
type TrustedFriends = EdgeFeature<{ Hydrator::TrustedFriends as u8 }>;
type OutsideNarrowcastPlace = EdgeFeature<{ Hydrator::OutsideNarrowcastPlace as u8 }>;

enum ConversationRootAuthorBlocksViewer {}

impl Feature for ConversationRootAuthorBlocksViewer {
    const REQUIRED: Hydrators = Hydrators::BLOCKED_BY_CONVERSATION_ROOT_AUTHOR;
    type Value<'a> = bool;
}

impl Reads<PostCandidate> for ConversationRootAuthorBlocksViewer {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> bool {
        at.post().conversation_root_author_blocks_viewer
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CompoundLeaf {
    TweetHasMedia,
    TweetIsSupersededEdit,
    ViewerAgeVerified,
    ViewerSuperFollowsAuthor,
    ViewerIsInConversationRootAuthorNetwork,
    ViewerIsBlockedByConversationRootAuthor,
    ViewerIsInAllowedCountry,
    ViewerMutesRetweetsFromAuthor,
    ViewerIsFollowedByConversationRootAuthor,
    ViewerFollowsConversationRootAuthor,
    ViewerSuperFollowsConversationRootAuthor,
    ViewerIsTrustedFriendsListMemberOrOwner,
    ViewerIsOutsideNarrowcastPlace,
}

impl CompoundLeaf {
    #[inline]
    pub(in crate::rules) fn interpret<'a, I: Interpreter<'a>>(self, i: &I) -> I::Output<bool> {
        match self {
            Self::TweetHasMedia => i.run(
                |(
                    In::<Tweet>(tweet),
                    In::<ArticleMediaEntities>(body),
                    In::<ArticleCoverMedia>(cover),
                )| { tweet.has_media() || body || cover },
            ),
            Self::TweetIsSupersededEdit => i.run(|(In::<PostFacts>(post), In::<Tweet>(tweet))| {
                tweet.is_superseded_edit(post.tweet_id())
            }),
            Self::ViewerAgeVerified => i.run(
                |(In::<ViewerAccount>(profile), In::<ViewerLabels>(age_verified_18))| {
                    age_verified_18 || profile.is_some_and(|profile| profile.has_idv_premium)
                },
            ),
            Self::ViewerSuperFollowsAuthor => i.run(
                |(In::<Tweet>(tweet), In::<SuperFollowsExclusive>(super_follows))| {
                    tweet.exclusive_conversation_author_id.is_some() && super_follows
                },
            ),
            Self::ViewerIsInConversationRootAuthorNetwork => i.run(
                |(In::<RootFollowsViewer>(first), In::<RootFollowsViewerSecondDegree>(second))| {
                    first || second
                },
            ),
            Self::ViewerIsBlockedByConversationRootAuthor => {
                i.run(|In::<ConversationRootAuthorBlocksViewer>(blocked)| blocked)
            }
            Self::ViewerIsInAllowedCountry => i.run(
                |(In::<ReplyControl>(control), In::<ViewerCountry>(country))| {
                    control.zip(country).is_some_and(|(control, country)| {
                        allows_country(&control.allowed_country_codes, country)
                    })
                },
            ),
            Self::ViewerMutesRetweetsFromAuthor => i.run(|In::<MuteRetweets>(mutes)| mutes),
            Self::ViewerIsFollowedByConversationRootAuthor => {
                i.run(|In::<RootFollowsViewer>(follows)| follows)
            }
            Self::ViewerFollowsConversationRootAuthor => {
                i.run(|In::<ViewerFollowsRoot>(follows)| follows)
            }
            Self::ViewerSuperFollowsConversationRootAuthor => {
                i.run(|In::<SuperFollowsRoot>(super_follows)| super_follows)
            }
            Self::ViewerIsTrustedFriendsListMemberOrOwner => {
                i.run(|In::<TrustedFriends>(member)| member)
            }
            Self::ViewerIsOutsideNarrowcastPlace => {
                i.run(|In::<OutsideNarrowcastPlace>(outside)| outside)
            }
        }
    }

    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::Compound(self))
    }
}
