use super::key_origin::{KeyOrigin, Names};
use super::{RequestRow, Store};
use crate::hydration::decode::author::{DecodedActor, DecodedAuthor};
use crate::hydration::fetcher::Fetcher;
use crate::hydration::inventory::Stake;
use crate::hydration::relation::Relation;
use crate::hydration::sources::source;
use crate::hydration::{
    Cause, HydratedTweet, HydratedUser, Hydrator, Hydrators, Lookup, RequestRows, Unresolved,
};
use crate::models::{
    AuthorId, ConversationControlFeatures, EdgeSet, PostCandidate, PostEntity, SafetyLabelMap,
    TweetFeatures, TweetId, UserEntity,
};
use rustc_hash::FxHashMap;
use xai_core_entities::entities::ConversationControlArm;

impl Store {
    pub(super) fn hydrated_tweet(
        &self,
        row: &RequestRow,
        position: usize,
        id: TweetId,
        tweet_misses: &FxHashMap<TweetId, Option<Cause>>,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> HydratedTweet {
        let safety_labels = self
            .request_fetcher::<source::SafetyLabels>()
            .get(id.0)
            .cloned();
        match self.resolve(row, id, tweet_misses) {
            Ok(author_id) => HydratedTweet::Resolved {
                candidate: self.candidate(row, position, id, author_id, unclaimed),
                safety_labels,
            },
            Err(reason) => HydratedTweet::Unresolved {
                reason,
                deletion: self
                    .request_fetcher::<source::TesTweetState>()
                    .get(id.0)
                    .filter(|_| reason.lookup == Lookup::Tweet && reason.cause == Cause::NotFound)
                    .and_then(|state| state.deletion()),
                safety_labels,
            },
        }
    }

    pub(super) fn hydrated_user(
        &self,
        row: &RequestRow,
        position: usize,
        user: AuthorId,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> HydratedUser {
        match self.author_miss(user) {
            Some(cause) => HydratedUser::Unresolved(cause),
            None => HydratedUser::Resolved {
                user: self.actor(row, user),
                edges: self.edges(row, position),
                failed: self.failed(row, position, unclaimed),
            },
        }
    }

    fn resolve(
        &self,
        row: &RequestRow,
        id: TweetId,
        tweet_misses: &FxHashMap<TweetId, Option<Cause>>,
    ) -> Result<AuthorId, Unresolved> {
        let tweet = |cause| Unresolved {
            lookup: Lookup::Tweet,
            cause,
        };
        match tweet_misses.get(&id) {
            Some(&Some(Cause::NotFound)) => {
                self.quotes
                    .quoted_author(self.request_fetcher::<source::TesQuotedTweet>(), id)
                    .map_err(tweet)?;
            }
            Some(&Some(Cause::Failed)) => return Err(tweet(Cause::Failed)),
            Some(None) | None => {}
        }
        let author_id = row.author.ok_or_else(|| tweet(Cause::Failed))?;
        let (shared_tweet, shared_author) =
            self.shared_misses(id, tweet_misses).unwrap_or_default();
        [
            (Lookup::SharedTweet, shared_tweet),
            (Lookup::Author, self.author_miss(author_id)),
            (Lookup::SharedAuthor, shared_author),
        ]
        .into_iter()
        .filter_map(|(lookup, miss)| {
            Some(Unresolved {
                lookup,
                cause: miss?,
            })
        })
        .min_by_key(|unresolved| unresolved.cause)
        .map_or(Ok(author_id), Err)
    }

    fn shared_misses(
        &self,
        tweet_id: TweetId,
        tweet_misses: &FxHashMap<TweetId, Option<Cause>>,
    ) -> Option<(Option<Cause>, Option<Cause>)> {
        let core = self
            .request_fetcher::<source::TesPureCoreData>()
            .get(tweet_id.0)
            .filter(|_| self.is_on(Relation::RetweetSource))?;
        let source_id = core.source_tweet_id?;
        let shared_tweet = *tweet_misses.get(&source_id)?;
        let shared_author = core.source_author_id.or_else(|| {
            self.rows
                .iter()
                .find(|row| row.tweet_id() == Some(source_id))?
                .author
        });
        Some((
            shared_tweet,
            shared_author.and_then(|author_id| self.author_miss(author_id)),
        ))
    }

    fn unresolving(&self, names: Names) -> impl Iterator<Item = Hydrator> + '_ {
        self.callable.iter().filter(move |&node| {
            self.stake(node) == Stake::Exists && self.origin(node).names() == names
        })
    }

    pub(super) fn tweet_miss(
        &self,
        tweet_id: TweetId,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> Option<Cause> {
        let key = tweet_id.0;
        let has_joined_retweet_sources = self.has_joined_retweet_sources();
        self.unresolving(Names::Post)
            .filter_map(|node| {
                if has_joined_retweet_sources && !self.fetcher(node)?.is_claimed(key) {
                    unclaimed.push((node, key));
                }
                self.existence_miss(node, key)
            })
            .min()
    }

    fn author_miss(&self, author_id: AuthorId) -> Option<Cause> {
        self.unresolving(Names::Author)
            .filter_map(|node| self.existence_miss(node, author_id.get()))
            .min()
    }

    fn author(&self, author_id: AuthorId) -> UserEntity {
        let DecodedAuthor { features, labels } = self
            .request_fetcher::<source::GizmoduckAuthor>()
            .get(author_id.get())
            .copied()
            .unwrap_or_default();
        UserEntity {
            id: author_id.get(),
            features,
            labels,
            ..UserEntity::default()
        }
    }

    fn actor(&self, row: &RequestRow, user: AuthorId) -> UserEntity {
        let DecodedActor {
            author: DecodedAuthor { features, labels },
            profile,
        } = self
            .request_fetcher::<source::GizmoduckActor>()
            .get(user.get())
            .copied()
            .unwrap_or_default();
        let lacks = |fetcher: &Fetcher<bool>, origin| {
            self.key(origin, row)
                .and_then(|key| fetcher.get(key))
                .is_some_and(|held| !held)
        };
        UserEntity {
            id: user.get(),
            features,
            labels,
            profile,
            lacks_confirmed_email: lacks(
                self.request_fetcher::<source::ActorEmails>(),
                self.origin(Hydrator::AuthorConfirmedEmail),
            ),
            lacks_verified_phone: lacks(
                self.request_fetcher::<source::ActorPhones>(),
                self.origin(Hydrator::AuthorVerifiedPhone),
            ),
        }
    }

    fn edges(&self, row: &RequestRow, position: usize) -> EdgeSet {
        let unread = self.root_author_block(row).unread();
        self.callable
            .difference(unread)
            .iter()
            .filter(|&node| {
                node.is_viewer_edge()
                    && self
                        .parent_key(node, row, position)
                        .and_then(|key| self.edge_holds(node, key))
                        .unwrap_or(false)
            })
            .fold(EdgeSet::default(), EdgeSet::with)
    }

    pub(super) fn candidate(
        &self,
        row: &RequestRow,
        position: usize,
        tweet_id: TweetId,
        author_id: AuthorId,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> PostCandidate {
        let id = tweet_id.0;
        let edges = self.edges(row, position);
        PostCandidate {
            author: match self.request_rows {
                RequestRows::ActorPosts => self.actor(row, author_id),
                RequestRows::Posts | RequestRows::Users => self.author(author_id),
            },
            edges,
            post: PostEntity {
                tweet_id: id,
                source_tweet_id: self.source_tweet_id(tweet_id).map(|source| source.0),
                tweet_features: TweetFeatures::default(),
                safety_labels: self
                    .request_fetcher::<source::SafetyLabels>()
                    .get(id)
                    .map(|labels| SafetyLabelMap::from_proto_label_types(labels))
                    .unwrap_or_default(),
                conversation_control: self.control(row).cloned().map(|control| {
                    ConversationControlFeatures {
                        viewer_country: self
                            .facts
                            .viewer_id()
                            .filter(|_| control.arm == ConversationControlArm::Co)
                            .and_then(|viewer_id| {
                                self.request_fetcher::<source::ViewerCountry>()
                                    .get(viewer_id)
                            })
                            .cloned(),
                        control,
                    }
                }),
                community_moderation: self
                    .request_fetcher::<source::CommunityModeration>()
                    .get(id)
                    .copied()
                    .unwrap_or_default(),
                viewer_is_community_moderator: self
                    .key(KeyOrigin::ModeratedCommunity, row)
                    .and_then(|community_id| {
                        self.request_fetcher::<source::CommunityModerator>()
                            .get(community_id)
                    })
                    .copied(),
                viewer_is_removed_from_community: self
                    .key(KeyOrigin::TweetCommunity, row)
                    .and_then(|community_id| {
                        self.request_fetcher::<source::CommunityViewerRemoved>()
                            .get(community_id)
                    })
                    .copied()
                    .unwrap_or_default(),
                article_has_media_entities: self
                    .key(KeyOrigin::TweetArticle, row)
                    .and_then(|article_id| {
                        self.request_fetcher::<source::ArticleMediaEntities>()
                            .get(article_id)
                    })
                    .copied()
                    .unwrap_or_default(),
                article_has_cover_media: self
                    .key(KeyOrigin::TweetArticle, row)
                    .and_then(|article_id| {
                        self.request_fetcher::<source::ArticleCoverMedia>()
                            .get(article_id)
                    })
                    .copied()
                    .unwrap_or_default(),
                article_lifecycle: self
                    .key(KeyOrigin::TweetArticle, row)
                    .and_then(|article_id| {
                        self.request_fetcher::<source::ArticleLifecycle>()
                            .get(article_id)
                    })
                    .copied(),
                reacted_to_post_safety_labels: self
                    .read_key(Hydrator::ReactedToPostSafetyLabels, position)
                    .and_then(|post_id| {
                        self.fetcher_of::<source::SafetyLabels>(
                            self.owner(Hydrator::ReactedToPostSafetyLabels),
                        )
                        .get(post_id)
                    })
                    .map(|labels| SafetyLabelMap::from_proto_label_types(labels))
                    .unwrap_or_default(),
                reacted_to_author_is_protected: self
                    .read_key(Hydrator::ReactedToAuthorSafety, position)
                    .and_then(|user_id| {
                        self.fetcher_of::<source::GizmoduckAuthor>(
                            self.owner(Hydrator::ReactedToAuthorSafety),
                        )
                        .get(user_id)
                    })
                    .is_some_and(|author| author.features.is_protected),
                conversation_root_author_blocks_viewer:
                    Hydrators::BLOCKED_BY_CONVERSATION_ROOT_AUTHOR
                        .iter()
                        .any(|node| edges.contains(node)),
            },
            failed: self.failed(row, position, unclaimed),
        }
    }
}
