use crate::hydration::fallback_cache::{Column, FallbackCache};
use crate::models::{
    AuthorId, MediaFeature, NsfwFeature, PostText, PureCore, ReactionContext, ReplyTo,
    TweetFeatures, TweetId,
};
use std::num::NonZeroU64;
use xai_core_entities::entities::{PureCoreData, TakedownReason};

#[derive(Clone)]
pub(crate) struct CachedTweet {
    pure_core: Option<PureCore>,
    row: Option<CachedTweetRow>,
}

pub(crate) type TweetFallbackCache = FallbackCache<CachedTweet>;

pub(crate) fn tweet_fallback_cache(capacity: usize) -> TweetFallbackCache {
    FallbackCache::new("tweet", capacity)
}

pub(crate) struct PureCoreColumn;

impl Column for PureCoreColumn {
    type Entry = CachedTweet;
    type Value = PureCore;
    type Stored = PureCore;
    const NAME: &'static str = "pure_core";

    fn store(value: &PureCore) -> PureCore {
        *value
    }

    fn new_entry(stored: PureCore) -> CachedTweet {
        CachedTweet {
            pure_core: Some(stored),
            row: None,
        }
    }

    fn replace(entry: &mut CachedTweet, stored: PureCore) -> Option<PureCore> {
        entry.pure_core.replace(stored)
    }

    fn get(entry: &CachedTweet) -> Option<PureCore> {
        entry.pure_core
    }

    fn holds(entry: &CachedTweet) -> bool {
        entry.pure_core.is_some()
    }

    fn others_hold(entry: &CachedTweet) -> bool {
        entry.row.is_some()
    }

    fn clear(entry: &mut CachedTweet) -> Option<PureCore> {
        entry.pure_core.take()
    }
}

pub(crate) struct TweetRowColumn;

impl Column for TweetRowColumn {
    type Entry = CachedTweet;
    type Value = TweetFeatures;
    type Stored = CachedTweetRow;
    const NAME: &'static str = "tweet_row";

    fn store(value: &TweetFeatures) -> CachedTweetRow {
        CachedTweetRow::new(value)
    }

    fn new_entry(stored: CachedTweetRow) -> CachedTweet {
        CachedTweet {
            pure_core: None,
            row: Some(stored),
        }
    }

    fn replace(entry: &mut CachedTweet, stored: CachedTweetRow) -> Option<CachedTweetRow> {
        entry.row.replace(stored)
    }

    fn get(entry: &CachedTweet) -> Option<TweetFeatures> {
        entry.row.as_ref().map(CachedTweetRow::features)
    }

    fn holds(entry: &CachedTweet) -> bool {
        entry.row.is_some()
    }

    fn others_hold(entry: &CachedTweet) -> bool {
        entry.pure_core.is_some()
    }

    fn clear(entry: &mut CachedTweet) -> Option<CachedTweetRow> {
        entry.row.take()
    }
}

#[derive(Clone)]
pub(crate) struct CachedTweetRow {
    has_media: bool,
    has_uploaded_media: bool,
    has_dmca_media: bool,
    nsfw_user: bool,
    nsfw_admin: bool,
    is_nullcast: bool,
    is_trusted_friends_tweet: bool,
    trusted_friends_list_id: u64,
    community_id: Option<NonZeroU64>,
    latest_edit_tweet_id: Option<NonZeroU64>,
    exclusive_conversation_author_id: Option<NonZeroU64>,
    article_id: Option<NonZeroU64>,
    sparse: Option<Box<Sparse>>,
}

#[derive(Clone, Default)]
struct Sparse {
    geo_allow_list: Vec<String>,
    geo_deny_list: Vec<String>,
    takedown_reasons: Vec<TakedownReason>,
    narrowcast_place_id: Option<u64>,
    reaction_context: Option<ReactionContext>,
}

impl CachedTweetRow {
    fn new(tweet: &TweetFeatures) -> Self {
        let TweetFeatures {
            text: _,
            media:
                MediaFeature {
                    has_media,
                    has_uploaded_media,
                    has_dmca_media,
                    geo_allow_list,
                    geo_deny_list,
                },
            takedown_reasons,
            nsfw: NsfwFeature { user, admin },
            is_nullcast,
            community_id,
            trusted_friends_list_id,
            latest_edit_tweet_id,
            exclusive_conversation_author_id,
            article_id,
            narrowcast_place_id,
            reaction_context,
        } = tweet;
        let is_sparse = !(geo_allow_list.is_empty()
            && geo_deny_list.is_empty()
            && takedown_reasons.is_empty()
            && narrowcast_place_id.is_none()
            && reaction_context.is_none());
        Self {
            has_media: *has_media,
            has_uploaded_media: *has_uploaded_media,
            has_dmca_media: *has_dmca_media,
            nsfw_user: *user,
            nsfw_admin: *admin,
            is_nullcast: *is_nullcast,
            is_trusted_friends_tweet: trusted_friends_list_id.is_some(),
            trusted_friends_list_id: trusted_friends_list_id.unwrap_or_default(),
            community_id: *community_id,
            latest_edit_tweet_id: latest_edit_tweet_id.map(stored_id),
            exclusive_conversation_author_id: exclusive_conversation_author_id.map(stored_id),
            article_id: *article_id,
            sparse: is_sparse.then(|| {
                Box::new(Sparse {
                    geo_allow_list: geo_allow_list.clone(),
                    geo_deny_list: geo_deny_list.clone(),
                    takedown_reasons: takedown_reasons.clone(),
                    narrowcast_place_id: *narrowcast_place_id,
                    reaction_context: *reaction_context,
                })
            }),
        }
    }

    fn features(&self) -> TweetFeatures {
        let Sparse {
            geo_allow_list,
            geo_deny_list,
            takedown_reasons,
            narrowcast_place_id,
            reaction_context,
        } = self.sparse.as_deref().cloned().unwrap_or_default();
        TweetFeatures {
            text: PostText::default(),
            media: MediaFeature {
                has_media: self.has_media,
                has_uploaded_media: self.has_uploaded_media,
                has_dmca_media: self.has_dmca_media,
                geo_allow_list,
                geo_deny_list,
            },
            takedown_reasons,
            nsfw: NsfwFeature {
                user: self.nsfw_user,
                admin: self.nsfw_admin,
            },
            is_nullcast: self.is_nullcast,
            community_id: self.community_id,
            trusted_friends_list_id: self
                .is_trusted_friends_tweet
                .then_some(self.trusted_friends_list_id),
            latest_edit_tweet_id: self.latest_edit_tweet_id.map(read_id),
            exclusive_conversation_author_id: self.exclusive_conversation_author_id.map(read_id),
            article_id: self.article_id,
            narrowcast_place_id,
            reaction_context,
        }
    }
}

fn stored_id(id: u64) -> NonZeroU64 {
    NonZeroU64::new(id).unwrap_or(NonZeroU64::MAX)
}

fn read_id(id: NonZeroU64) -> u64 {
    if id == NonZeroU64::MAX {
        0
    } else {
        id.get()
    }
}

pub(crate) fn pure_core(core: &PureCoreData) -> PureCore {
    PureCore {
        author_id: AuthorId(core.author_id),
        source_tweet_id: core.source_tweet_id.map(TweetId),
        source_author_id: core.source_user_id.filter(|&id| id != 0).map(AuthorId),
        reply_to: reply_to(core),
    }
}

fn reply_to(core: &PureCoreData) -> Option<ReplyTo> {
    let conversation_id = TweetId(core.conversation_id?);
    let parent_id = TweetId(core.in_reply_to_tweet_id?);
    if parent_id != conversation_id {
        return Some(ReplyTo::Nested {
            conversation_id,
            parent_id,
        });
    }
    Some(ReplyTo::Root {
        conversation_id,
        root_author_id: AuthorId(core.in_reply_to_user_id?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Placement;
    use crate::rules::fixtures::{candidate, dropped, logged_out_viewer};
    use crate::rules::{RuleEngine, SafetyLevel};
    use xai_visibility_filtering::models::FilteredReason;

    #[test]
    fn a_share_without_its_user_id_leaves_the_source_author_unknown() {
        for source_user_id in [Some(0), None] {
            let core = PureCoreData {
                author_id: 10,
                source_tweet_id: Some(5),
                source_user_id,
                ..Default::default()
            };
            assert_eq!(pure_core(&core).source_author_id, None);
        }
    }

    #[test]
    fn a_direct_reply_names_its_root_author_and_a_nested_one_its_parent() {
        let root = Some(ReplyTo::Root {
            conversation_id: TweetId(100),
            root_author_id: AuthorId(30),
        });
        let nested = Some(ReplyTo::Nested {
            conversation_id: TweetId(100),
            parent_id: TweetId(101),
        });
        for (ids, expected) in [
            ((Some(100), Some(100), Some(30)), root),
            ((Some(100), Some(100), None), None),
            ((Some(100), Some(101), Some(30)), nested),
            ((Some(100), Some(101), None), nested),
            ((None, Some(101), Some(30)), None),
            ((Some(100), None, Some(30)), None),
        ] {
            let (conversation_id, in_reply_to_tweet_id, in_reply_to_user_id) = ids;
            let core = PureCoreData {
                author_id: 10,
                conversation_id,
                in_reply_to_tweet_id,
                in_reply_to_user_id,
                ..Default::default()
            };
            assert_eq!(pure_core(&core).reply_to, expected, "{ids:?}");
        }
    }

    #[test]
    fn a_cached_row_fits_in_56_bytes_and_keeps_a_circle_list_id_of_zero() {
        let circle = TweetFeatures {
            trusted_friends_list_id: Some(0),
            ..Default::default()
        };
        assert_eq!(CachedTweetRow::new(&circle).features(), circle);
        assert_eq!(size_of::<CachedTweetRow>(), 56);
    }

    #[test]
    fn a_cached_row_keeps_a_place_or_a_reaction_context_set_alone() {
        let local = TweetFeatures {
            narrowcast_place_id: Some(0xa000_0000_0000_0001),
            ..Default::default()
        };
        let reaction = crate::scenario::reaction_to(9, 10);
        assert_eq!(CachedTweetRow::new(&local).features(), local);
        assert_eq!(CachedTweetRow::new(&reaction).features(), reaction);
    }

    #[test]
    fn a_cached_row_drops_a_zero_id_tweet_as_its_fresh_read_does() {
        let exclusive = TweetFeatures {
            exclusive_conversation_author_id: Some(0),
            ..Default::default()
        };
        let cached = CachedTweetRow::new(&exclusive).features();
        assert_eq!(cached, exclusive);
        let post = candidate().with_tweet_features(cached).build();
        assert_eq!(
            RuleEngine::for_tests()
                .evaluate(
                    SafetyLevel::TimelineHome,
                    &logged_out_viewer(),
                    &post,
                    Placement::Standalone,
                )
                .into_verdict(),
            dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop")
        );
    }

    #[test]
    fn a_cached_row_reads_as_its_tweet_without_its_text() {
        let tweet = TweetFeatures {
            text: PostText::from("post text"),
            media: MediaFeature {
                has_media: true,
                has_uploaded_media: true,
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string()],
                geo_deny_list: vec!["de".to_string()],
            },
            takedown_reasons: vec![TakedownReason::Dmca],
            nsfw: NsfwFeature {
                user: true,
                admin: true,
            },
            is_nullcast: true,
            community_id: NonZeroU64::new(500),
            trusted_friends_list_id: Some(8),
            latest_edit_tweet_id: Some(30),
            exclusive_conversation_author_id: Some(7),
            article_id: NonZeroU64::new(8),
            narrowcast_place_id: Some(0xa000_0000_0000_0001),
            reaction_context: Some(ReactionContext {
                reacted_to_post_id: 9,
                reacted_to_user_id: 10,
            }),
        };

        assert_eq!(
            CachedTweetRow::new(&tweet).features(),
            TweetFeatures {
                text: PostText::default(),
                ..tweet.clone()
            }
        );
        assert_eq!(
            CachedTweetRow::new(&TweetFeatures::default()).features(),
            TweetFeatures::default()
        );
    }
}
