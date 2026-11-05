pub mod batch;
pub(crate) mod community_source;
mod decode;
pub(crate) mod execute;
pub(crate) mod fallback_cache;
mod fetcher;
mod hydrator;
pub(crate) mod inventory;
pub mod metrics;
pub(crate) mod plan;
mod relation;
pub(crate) mod sources;
mod store;
pub mod tweet_source;

use crate::metrics::RequestMetrics;
use crate::models::{
    AuthorId, Deletion, EdgeSet, Placement, PostCandidate, Quote, RawCandidate, RequestFacts,
    RetweetMerge, TweetId, UserCandidate, UserEntity, ViewerFeatures,
};
pub(crate) use decode::author::AuthorNotFound;
pub(crate) use decode::author::fallback_cache as author_fallback_cache;
pub(crate) use decode::tweet::tweet_fallback_cache;
use decode::viewer::ViewerState;
use fallback_cache::FallbackCaches;
pub(crate) use hydrator::RequestRows;
pub use hydrator::{Hydrator, Hydrators};
pub(crate) use plan::HydrationPlan;
pub(crate) use relation::Relation;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) const HYDRATION_TIMEOUT: Duration = Duration::from_secs(1);
pub(crate) const BEST_EFFORT_HYDRATION_TIMEOUT: Duration = Duration::from_millis(100);
pub(crate) const INBOUND_ALLOWANCE: Duration = Duration::from_millis(10);

pub(crate) fn request_context(
    entered: tokio::time::Instant,
    grpc_timeout: Option<Duration>,
) -> xai_x_rpc::CallContext {
    xai_x_rpc::CallContext {
        deadline: Some(
            entered
                + grpc_timeout
                    .unwrap_or(HYDRATION_TIMEOUT)
                    .saturating_sub(INBOUND_ALLOWANCE),
        ),
    }
}

pub(crate) struct HydrationRequest<'a> {
    facts: RequestFacts,
    raw_candidates: Vec<RawCandidate>,
    users: Vec<AuthorId>,
    quotes: Vec<QuoteEntry>,
    retweet_merge: RetweetMerge,
    fallback_caches: &'a FallbackCaches,
    metrics: &'a RequestMetrics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct QuoteEntry {
    pub(crate) inner: TweetId,
    pub(crate) outer: TweetId,
}

impl<'a> HydrationRequest<'a> {
    pub(crate) fn new(
        facts: RequestFacts,
        raw_candidates: Vec<RawCandidate>,
        retweet_merge: RetweetMerge,
        fallback_caches: &'a FallbackCaches,
        metrics: &'a RequestMetrics,
    ) -> Self {
        Self {
            facts,
            raw_candidates,
            users: Vec::new(),
            quotes: Vec::new(),
            retweet_merge,
            fallback_caches,
            metrics,
        }
    }

    pub(crate) fn for_users(
        facts: RequestFacts,
        users: Vec<AuthorId>,
        fallback_caches: &'a FallbackCaches,
        metrics: &'a RequestMetrics,
    ) -> Self {
        Self {
            users,
            ..Self::new(
                facts,
                Vec::new(),
                RetweetMerge::Unmerged,
                fallback_caches,
                metrics,
            )
        }
    }

    pub(crate) fn with_quotes(mut self, quotes: Vec<QuoteEntry>) -> Self {
        self.quotes = quotes;
        self
    }
}

fn candidate_count_by_key<K: Eq + Hash>(keys: impl Iterator<Item = K>) -> FxHashMap<K, usize> {
    let mut candidate_count_by_key =
        FxHashMap::with_capacity_and_hasher(keys.size_hint().0, Default::default());
    for key in keys {
        *candidate_count_by_key.entry(key).or_default() += 1;
    }
    candidate_count_by_key
}

pub(crate) struct Hydration {
    viewer: ViewerFeatures,
    viewer_state: Option<ViewerState>,
    planned: Hydrators,
    viewer_failed: Hydrators,
    tweets: FxHashMap<TweetId, HydratedTweet>,
    users: FxHashMap<AuthorId, HydratedUser>,
    quotes: FxHashMap<QuoteEntry, Quoted>,
    has_fetched_retweet_sources: bool,
    retweet_merge: RetweetMerge,
}

pub(crate) enum HydratedUser {
    Resolved {
        user: UserEntity,
        edges: EdgeSet,
        failed: Hydrators,
    },
    Unresolved(Cause),
}

#[derive(Clone, Copy, Default)]
struct Quoted {
    quote: Option<Quote>,
    failed: Hydrators,
}

impl Hydration {
    pub(crate) fn viewer(&self) -> &ViewerFeatures {
        &self.viewer
    }

            pub(crate) fn viewer_found(&self) -> bool {
        self.viewer_state.is_some()
    }

    pub(crate) fn planned(&self) -> Hydrators {
        self.planned
    }

    pub(crate) fn viewer_failed(&self) -> Hydrators {
        self.viewer_failed
    }

    pub(crate) fn tweet(&self, id: TweetId) -> Option<&HydratedTweet> {
        self.tweets.get(&id)
    }

    pub(crate) fn user(&self, id: AuthorId) -> Option<&HydratedUser> {
        self.users.get(&id)
    }

    pub(crate) fn author(
        &self,
        id: TweetId,
        placement: Placement,
        reads: Hydrators,
    ) -> Option<UserCandidate> {
        let post = self.tweet(id)?.candidate()?;
        let quoted = match placement {
            Placement::InnerQuoted { outer } => self
                .quotes
                .get(&QuoteEntry { inner: id, outer })
                .copied()
                .unwrap_or_default(),
            Placement::Standalone | Placement::RetweetSource => Quoted::default(),
        };
        Some(UserCandidate {
            user: post.author.clone(),
            edges: post.edges,
            quote: quoted.quote,
            failed: post.failed.intersection(reads).union(quoted.failed),
        })
    }

    pub(crate) fn user_candidate(&self, id: AuthorId, reads: Hydrators) -> Option<UserCandidate> {
        let HydratedUser::Resolved {
            user,
            edges,
            failed,
        } = self.user(id)?
        else {
            return None;
        };
        Some(UserCandidate {
            user: user.clone(),
            edges: *edges,
            quote: None,
            failed: failed.intersection(reads),
        })
    }

    pub(crate) fn is_on(&self, relation: Relation) -> bool {
        relation.is_on(self.retweet_merge, self.planned)
    }

            pub(crate) fn candidate(&self, id: TweetId, copied: bool) -> Option<Cow<'_, PostCandidate>> {
        let tweet = self.tweet(id)?;
        let candidate = tweet.candidate()?;
        let copy = copied
            .then(|| self.tweet(tweet.source_tweet_id()?)?.candidate())
            .flatten()
            .map(|source| source.post.tweet_features.media.has_uploaded_media)
            .filter(|&has_media| has_media != candidate.post.tweet_features.media.has_media);
        Some(match copy {
            None => Cow::Borrowed(candidate),
            Some(has_media) => {
                let mut candidate = candidate.clone();
                candidate.post.tweet_features.media.has_media = has_media;
                Cow::Owned(candidate)
            }
        })
    }

    pub(crate) fn failed_nodes(&self, post: TweetId) -> Hydrators {
        let Some(candidate) = self.tweet(post).and_then(HydratedTweet::candidate) else {
            return Hydrators::empty();
        };
        let source = candidate
            .post
            .source_tweet_id
            .and_then(|source| self.tweet(TweetId(source))?.candidate())
            .map_or(Hydrators::empty(), |source| source.failed);
        candidate.failed.union(source)
    }

    pub(crate) fn has_fetched_retweet_sources(&self) -> bool {
        self.has_fetched_retweet_sources
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "nearly every tweet resolves, so a box would cost an allocation per tweet"
)]
pub(crate) enum HydratedTweet {
    Resolved {
        candidate: PostCandidate,
        safety_labels: Option<Arc<vf_pb::SafetyLabelMap>>,
    },
    Unresolved {
        reason: Unresolved,
        deletion: Option<Deletion>,
        safety_labels: Option<Arc<vf_pb::SafetyLabelMap>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Unresolved {
    pub(crate) lookup: Lookup,
    pub(crate) cause: Cause,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum Lookup {
    Tweet,
    Author,
    SharedTweet,
    SharedAuthor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Cause {
    NotFound,
    Failed,
}

impl HydratedTweet {
    pub(crate) fn candidate(&self) -> Option<&PostCandidate> {
        match self {
            Self::Resolved { candidate, .. } => Some(candidate),
            Self::Unresolved { .. } => None,
        }
    }

    pub(crate) fn source_tweet_id(&self) -> Option<TweetId> {
        self.candidate()?.post.source_tweet_id.map(TweetId)
    }

    pub(crate) fn source_to_merge(&self) -> Option<TweetId> {
        self.candidate()
            .filter(|candidate| !candidate.failed.has_must_have())?
            .post
            .source_tweet_id
            .map(TweetId)
    }

    pub(crate) fn safety_labels(&self) -> Option<&Arc<vf_pb::SafetyLabelMap>> {
        match self {
            Self::Resolved { safety_labels, .. } | Self::Unresolved { safety_labels, .. } => {
                safety_labels.as_ref()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_context_budget_is_header_or_hang_guard_minus_allowance() {
        let entered = tokio::time::Instant::now();

        let header = Duration::from_millis(400);
        assert_eq!(
            request_context(entered, Some(header)).deadline,
            Some(entered + header - INBOUND_ALLOWANCE)
        );
        assert_eq!(
            request_context(entered, None).deadline,
            Some(entered + HYDRATION_TIMEOUT - INBOUND_ALLOWANCE)
        );
    }
}
