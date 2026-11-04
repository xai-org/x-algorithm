use crate::clients::socialgraph_client::{EdgeFrom, EdgeQuery};
use crate::hydration::community_source::CommunityPost;
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::fetcher::{AnyFetcher, Fetcher, Owner};
use crate::hydration::hydrator::{Edge, EdgeKey, NodeSpec, RequestRows};
use crate::hydration::inventory::{Answer, Effect, Stake, claimed_answer_fails, classify};
use crate::hydration::plan::Group;
use crate::hydration::relation::{Related, Relation, Role, Trigger};
use crate::hydration::sources::{Input, Registered, Reply, Source, SourceFetchers, Value, source};
use crate::hydration::{
    Cause, HydratedTweet, HydratedUser, Hydration, HydrationPlan, Hydrator, Hydrators, QuoteEntry,
};
use crate::metrics::RequestMetrics;
use crate::metrics::families::HYDRATOR_UNASKED_KEYS;
use crate::models::{
    AuthorId, NotificationFilters, RawCandidate, RequestFacts, RetweetMerge, TweetId, Viewer,
    ViewerFeatures,
};
pub(super) use edges::Lands;
pub(super) use key_origin::{ControlArms, KeyOrigin, Names};
use quotes::Quotes;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::time::Duration;
use strum::VariantArray;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};

mod conversation_root;
mod edges;
mod key_origin;
mod quotes;
mod resolve;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowSubject {
    Post(TweetId),
    User,
}

struct RequestRow {
    subject: RowSubject,
    request_author: Option<AuthorId>,
    author: Option<AuthorId>,
}

impl RequestRow {
    fn user(user: AuthorId) -> Self {
        Self {
            subject: RowSubject::User,
            request_author: Some(user),
            author: Some(user),
        }
    }

    fn tweet_id(&self) -> Option<TweetId> {
        match self.subject {
            RowSubject::Post(tweet_id) => Some(tweet_id),
            RowSubject::User => None,
        }
    }
}

pub(super) struct CallRequest<'p> {
    pub(super) group: &'p Group,
    pub(super) is_first: bool,
    pub(super) keys: Vec<u64>,
    pub(super) key_count: usize,
    queries: Vec<EdgeQuery>,
    queried_edges: Vec<Edge>,
    pub(super) community_posts: Vec<CommunityPost>,
    pub(super) viewer_id: Option<u64>,
        pub(super) candidate_count_by_claimed_key: FxHashMap<u64, usize>,
}

impl CallRequest<'_> {
    pub(super) fn queries(&self) -> &[EdgeQuery] {
        &self.queries
    }

    pub(super) fn input(&self) -> Input<'_> {
        Input {
            keys: &self.keys,
            viewer_id: self.viewer_id,
            fields: self.group.fields(),
            queries: &self.queries,
            community_posts: &self.community_posts,
            #[cfg(test)]
            fetcher: self.group.fetcher,
        }
    }

    fn edge_queries(&self) -> impl Iterator<Item = (Edge, &EdgeQuery)> {
        debug_assert_eq!(self.queried_edges.len(), self.queries.len());
        self.queried_edges.iter().copied().zip(&self.queries)
    }

    pub(super) fn claimed_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.edge_queries()
            .filter(|(_, query)| !query.destination_ids.is_empty())
            .map(|(edge, _)| edge)
    }
}

fn claimed<'a>(keys: &'a [u64], queries: &'a [EdgeQuery]) -> impl Iterator<Item = u64> + 'a {
    let destinations = queries.iter().flat_map(|query| &query.destination_ids);
    keys.iter().chain(destinations).copied()
}

pub(super) enum FetcherRef<'a> {
    Keyed(&'a dyn AnyFetcher),
    Edges(&'a Fetcher<bool, EdgeKey>),
}

pub(super) enum FetcherMut<'a> {
    Keyed(&'a mut dyn AnyFetcher),
    Edges(&'a mut Fetcher<bool, EdgeKey>),
}

pub(super) trait FetcherKey<V>: Sized {
    fn view(fetcher: &Fetcher<V, Self>) -> FetcherRef<'_>;
    fn view_mut(fetcher: &mut Fetcher<V, Self>) -> FetcherMut<'_>;
}

impl<V> FetcherKey<V> for u64 {
    fn view(fetcher: &Fetcher<V, Self>) -> FetcherRef<'_> {
        FetcherRef::Keyed(fetcher)
    }

    fn view_mut(fetcher: &mut Fetcher<V, Self>) -> FetcherMut<'_> {
        FetcherMut::Keyed(fetcher)
    }
}

impl FetcherKey<bool> for EdgeKey {
    fn view(fetcher: &Fetcher<bool, Self>) -> FetcherRef<'_> {
        FetcherRef::Edges(fetcher)
    }

    fn view_mut(fetcher: &mut Fetcher<bool, Self>) -> FetcherMut<'_> {
        FetcherMut::Edges(fetcher)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Landing {
    Answered,
    KeysJoined,
}

#[derive(Default)]
pub(super) struct Store {
    facts: RequestFacts,
    rows: Vec<RequestRow>,
    related: FxHashMap<(Relation, usize), TweetId>,
    paired: Vec<(usize, TweetId)>,
        requested_tweet_ids: FxHashSet<TweetId>,
        joined_tweet_ids: FxHashSet<TweetId>,
    request_rows: RequestRows,
    quotes: Quotes,
    retweet_merge: RetweetMerge,
    callable: Hydrators,
    fetchers: SourceFetchers,
    has_called: Vec<bool>,
    pub(super) core_elapsed: Duration,
    pub(super) tweets_elapsed: Option<Duration>,
}

impl Store {
    pub(super) fn new(
        plan: &HydrationPlan,
        facts: RequestFacts,
        raw: &[RawCandidate],
        retweet_merge: RetweetMerge,
    ) -> Self {
        Self {
            callable: plan.callable(facts.viewer_id()),
            facts,
            retweet_merge,
            rows: raw
                .iter()
                .map(|candidate| RequestRow {
                    subject: RowSubject::Post(candidate.tweet_id),
                    request_author: candidate.request_author_id.map(AuthorId),
                    author: None,
                })
                .collect(),
            requested_tweet_ids: raw.iter().map(|candidate| candidate.tweet_id).collect(),
            request_rows: plan.rows(),
            has_called: vec![false; plan.groups().count()],
            ..Self::default()
        }
    }

    pub(super) fn with_quotes(mut self, quotes: &[QuoteEntry]) -> Self {
        self.quotes = Quotes::new(quotes);
        self
    }

    pub(super) fn with_users(mut self, users: &[AuthorId]) -> Self {
        self.rows
            .extend(users.iter().copied().map(RequestRow::user));
        self
    }

    pub(super) fn request_fetcher<S: Registered>(&self) -> &Fetcher<Value<S>, S::Key> {
        S::fetchers(&self.fetchers).request()
    }

    fn fetcher_of<S: Registered>(&self, owner: Owner) -> &Fetcher<Value<S>, S::Key> {
        S::fetchers(&self.fetchers).of(owner)
    }

    fn take<S: Registered<Key = u64>>(&mut self, key: u64) -> Option<Value<S>> {
        S::fetchers_mut(&mut self.fetchers)
            .of_mut(Owner::Request)
            .take(key)
    }

        pub(super) fn land_reply<S: Registered>(
        &mut self,
        call: &CallRequest<'_>,
        reply: S::Reply,
        elapsed: Duration,
    ) where
        S::Reply: Lands<S::Key>,
    {
        let fetcher = S::fetchers_mut(&mut self.fetchers).of_mut(call.group.fetcher.owner);
        reply.land(fetcher, call, S::EDGE);
        let first = call.is_first;
        if S::SOURCE == Source::TesPureCoreData
            && first
            && call.group.nodes.contains(Hydrator::PureCore)
        {
            self.core_elapsed = elapsed;
        }
        if S::SOURCE == Source::TesTweetForVisibility && first {
            self.tweets_elapsed = Some(elapsed);
        }
        if matches!(S::SOURCE, Source::TesPureCoreData | Source::TesQuotedTweet) {
            self.land_authors();
        }
    }

    fn spec(&self, node: Hydrator) -> NodeSpec {
        node.spec_at(self.request_rows)
    }

    fn stake(&self, node: Hydrator) -> Stake {
        self.spec(node).stake()
    }

    pub(super) fn existence_miss(&self, node: Hydrator, key: u64) -> Option<Cause> {
        match classify(self.stake(node), self.fetcher(node)?.answer(key)) {
            Effect::Unresolves(cause) => Some(cause),
            Effect::Value | Effect::Absent | Effect::Fails => None,
        }
    }

    fn owner(&self, node: Hydrator) -> Owner {
        self.spec(node).fetcher().owner
    }

    fn origin(&self, node: Hydrator) -> KeyOrigin {
        node.key_at(self.request_rows)
    }

    fn viewer_filters(&self, filters: impl FnOnce(NotificationFilters) -> bool) -> bool {
        self.facts
            .viewer_id()
            .and_then(|viewer_id| {
                self.request_fetcher::<source::GizmoduckViewer>()
                    .get(viewer_id)
            })
            .is_some_and(|viewer| filters(viewer.profile.notification_filters))
    }

    fn has_joined_retweet_sources(&self) -> bool {
        !self.joined_tweet_ids.is_empty()
    }

    fn is_on(&self, relation: Relation) -> bool {
        relation.is_on(self.retweet_merge, self.callable)
    }

    fn named(&self, relation: Relation, parent: &RequestRow) -> Option<Related> {
        match relation.spec().trigger {
            Trigger::PureCore(named) => named(
                self.request_fetcher::<source::TesPureCoreData>()
                    .get(parent.tweet_id()?.0)?,
            ),
            Trigger::Tweet(named) => named(
                self.request_fetcher::<source::TesTweetForVisibility>()
                    .get(parent.tweet_id()?.0)?,
            ),
            Trigger::Key(origin) => Some(Related {
                tweet_id: TweetId(self.key(origin, parent)?),
                author: None,
            }),
            Trigger::QuoteEntry => None,
        }
    }

    fn related_post(&self, relation: Relation, parent: usize) -> Option<Related> {
        let tweet_id = *self.related.get(&(relation, parent))?;
        let named = self.named(relation, self.rows.get(parent)?)?;
        (named.tweet_id == tweet_id).then_some(named)
    }

    fn related_posts(&self, relation: Relation) -> impl Iterator<Item = Related> + '_ {
        self.related
            .keys()
            .filter(move |&&(joined, _)| joined == relation)
            .filter_map(|&(relation, parent)| self.related_post(relation, parent))
    }

    fn read_key(&self, node: Hydrator, parent: usize) -> Option<u64> {
        let related = self.related_post(node.spec().relation?, parent)?;
        self.related_key(self.origin(node), related)
    }

    fn parent_key(&self, node: Hydrator, row: &RequestRow, position: usize) -> Option<u64> {
        match node.spec().relation {
            None => self.key(self.origin(node), row),
            Some(_) => self.read_key(node, position),
        }
    }

    fn controls(&self) -> impl Iterator<Item = &ConversationControl> {
        self.rows.iter().filter_map(|row| self.control(row))
    }

    fn control(&self, row: &RequestRow) -> Option<&ConversationControl> {
        self.request_fetcher::<source::TesConversationControl>()
            .get(row.tweet_id()?.0)
    }

    fn source_tweet_id(&self, tweet_id: TweetId) -> Option<TweetId> {
        self.request_fetcher::<source::TesPureCoreData>()
            .get(tweet_id.0)?
            .source_tweet_id
    }

        fn community_post(&self, row: &RequestRow) -> Option<CommunityPost> {
        let tweet_id = row.tweet_id()?.0;
        let author_id = row.author?.get();
        if self.facts.viewer_id() == Some(author_id) {
            return None;
        }
        Some(CommunityPost {
            tweet_id,
            author_id,
            community_id: self
                .request_fetcher::<source::TesTweetForVisibility>()
                .get(tweet_id)?
                .community_id?
                .get(),
        })
    }

    fn community_posts(&self, tweet_ids: &[u64]) -> Vec<CommunityPost> {
        let mut posts: Vec<CommunityPost> = self
            .rows
            .iter()
            .filter_map(|row| self.community_post(row))
            .filter(|post| tweet_ids.binary_search(&post.tweet_id).is_ok())
            .collect();
        posts.sort_unstable_by_key(|post| post.tweet_id);
        posts.dedup_by_key(|post| post.tweet_id);
        posts
    }

    fn keys(&self, nodes: Hydrators) -> Vec<u64> {
        let mut keys = Vec::new();
        for (position, node) in nodes.iter().enumerate() {
            let origin = self.origin(node);
            if nodes
                .iter()
                .take(position)
                .any(|earlier| self.origin(earlier) == origin)
            {
                continue;
            }
            keys.extend(self.node_keys(node));
        }
        if nodes.contains(Hydrator::PureCore) {
            keys.extend(self.paired.iter().map(|&(_, tweet_id)| tweet_id.0));
        }
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    fn claim(&mut self, nodes: Hydrators) -> Vec<u64> {
        let keys = self.keys(nodes);
        let Some(node) = nodes.iter().next() else {
            return Vec::new();
        };
        if node.is_edge() {
            return self.claim_viewer_edge(node, keys);
        }
        match self.fetcher_mut(node) {
            Some(fetcher) => fetcher.claim(keys),
            None => Vec::new(),
        }
    }

    fn fetcher(&self, node: Hydrator) -> Option<&dyn AnyFetcher> {
        match self.fetchers.get(self.spec(node).fetcher()) {
            FetcherRef::Keyed(fetcher) => Some(fetcher),
            FetcherRef::Edges(_) => None,
        }
    }

    fn fetcher_mut(&mut self, node: Hydrator) -> Option<&mut dyn AnyFetcher> {
        match self.fetchers.get_mut(self.spec(node).fetcher()) {
            FetcherMut::Keyed(fetcher) => Some(fetcher),
            FetcherMut::Edges(_) => None,
        }
    }

    fn candidate_count_by_key(&self, nodes: Hydrators) -> FxHashMap<u64, usize> {
        let mut counts = self.candidate_count_by_pair(nodes);
        let (viewer_nodes, row_nodes): (Vec<Hydrator>, Vec<Hydrator>) = nodes
            .iter()
            .partition(|&node| self.origin(node).names() == Names::Viewer);
        for node in viewer_nodes {
            counts.extend(self.node_keys(node).into_iter().map(|key| (key, 1)));
        }
        for (parent, row) in self.rows.iter().enumerate() {
            for (position, &node) in row_nodes.iter().enumerate() {
                let Some(key) = self.parent_key(node, row, parent) else {
                    continue;
                };
                let is_candidate = row.author.is_some() || node.precedes_author(self.request_rows);
                let counted = row_nodes
                    .iter()
                    .take(position)
                    .any(|&earlier| self.parent_key(earlier, row, parent) == Some(key));
                if is_candidate && !counted {
                    *counts.entry(key).or_default() += 1;
                }
            }
        }
        counts
    }

    pub(super) fn offer<'p>(&mut self, group: &'p Group) -> Option<CallRequest<'p>> {
        if self.facts.viewer_id().is_none() && group.nodes.iter().all(Hydrator::needs_viewer) {
            return None;
        }
        let is_first = self.has_called.get(group.position) != Some(&true);
        let mut queries = Vec::new();
        let mut queried_edges = Vec::new();
        for &(edge, graph, direction, nodes) in group.edges() {
            let query = |from, destination_ids| EdgeQuery {
                graph,
                direction,
                from,
                destination_ids,
            };
            if nodes.iter().any(Hydrator::is_quote_pair_edge) {
                for (source, destination_ids) in self.claim_quote_pairs(nodes) {
                    queried_edges.push(edge);
                    queries.push(query(EdgeFrom::User(source), destination_ids));
                }
            } else {
                queried_edges.push(edge);
                queries.push(query(EdgeFrom::Viewer, self.claim(nodes)));
            }
        }
        let keys = if queries.is_empty() {
            self.claim(group.nodes)
        } else {
            Vec::new()
        };
        let key_count = claimed(&keys, &queries).count();
        if key_count == 0 {
            return None;
        }
        let community_posts = if group.fetcher.source == Source::CommunityModeration {
            self.community_posts(&keys)
        } else {
            Vec::new()
        };
        if let Some(has_called) = self.has_called.get_mut(group.position) {
            *has_called = true;
        }
        let candidate_count_by_key = self.candidate_count_by_key(group.nodes);
        let candidate_count_by_claimed_key = claimed(&keys, &queries)
            .map(|key| (key, candidate_count_by_key.get(&key).copied().unwrap_or(0)))
            .collect();
        Some(CallRequest {
            group,
            is_first,
            candidate_count_by_claimed_key,
            viewer_id: self.facts.viewer_id(),
            keys,
            key_count,
            queries,
            queried_edges,
            community_posts,
        })
    }

    pub(super) fn is_country_lookup_skipped(&self) -> bool {
        self.callable.contains(Hydrator::ViewerCountry)
            && !self
                .request_fetcher::<source::ViewerCountry>()
                .has_claimed()
            && self
                .controls()
                .any(|control| control.arm == ConversationControlArm::Co)
    }

    pub(super) fn land(
        &mut self,
        call: &CallRequest<'_>,
        reply: Reply,
        elapsed: Duration,
    ) -> Landing {
        reply.land(self, call, elapsed);
        if self.join(call.group.nodes) {
            Landing::KeysJoined
        } else {
            Landing::Answered
        }
    }

    fn land_authors(&mut self) {
        let pure_cores = source::TesPureCoreData::fetchers(&self.fetchers).request();
        let quoted_tweets = source::TesQuotedTweet::fetchers(&self.fetchers).request();
        for row in &mut self.rows {
            row.author = row
                .request_author
                .or_else(|| Some(pure_cores.get(row.tweet_id()?.0)?.author_id))
                .or_else(|| {
                    self.quotes
                        .quoted_author(quoted_tweets, row.tweet_id()?)
                        .ok()
                });
        }
    }

    fn join(&mut self, landed: Hydrators) -> bool {
        let mut has_joined = false;
        for &relation in Relation::VARIANTS {
            let spec = relation.spec();
            if spec.trigger.reads().intersection(landed).is_empty() || !self.is_on(relation) {
                continue;
            }
            let named: Vec<(usize, Related)> = match spec.trigger {
                Trigger::QuoteEntry => self.quote_outers(),
                Trigger::PureCore(_) | Trigger::Tweet(_) | Trigger::Key(_) => self
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(parent, row)| Some((parent, self.named(relation, row)?)))
                    .collect(),
            };
            for (parent, related) in named {
                match (spec.trigger, spec.role) {
                    (Trigger::QuoteEntry, _) => {
                        let paired = (parent, related.tweet_id);
                        if self.paired.contains(&paired) {
                            continue;
                        }
                        self.paired.push(paired);
                        if self.requested_tweet_ids.contains(&related.tweet_id)
                            || self.joined_tweet_ids.contains(&related.tweet_id)
                        {
                            continue;
                        }
                    }
                    (_, Role::Judged) => {
                        if self.requested_tweet_ids.contains(&related.tweet_id)
                            || !self.joined_tweet_ids.insert(related.tweet_id)
                        {
                            continue;
                        }
                        self.rows.push(RequestRow {
                            subject: RowSubject::Post(related.tweet_id),
                            request_author: None,
                            author: related.author,
                        });
                    }
                    (_, Role::Read) => match self.related.entry((relation, parent)) {
                        Entry::Vacant(entry) => {
                            entry.insert(related.tweet_id);
                        }
                        Entry::Occupied(_) => continue,
                    },
                }
                has_joined = true;
            }
        }
        has_joined
    }

    pub(super) fn assemble(mut self, metrics: &RequestMetrics) -> Hydration {
        let viewer_failed = self.viewer_failed();
        let mut unclaimed = Vec::new();
        let mut tweet_misses: FxHashMap<TweetId, Option<Cause>> =
            FxHashMap::with_capacity_and_hasher(self.rows.len(), Default::default());
        for tweet_id in self.rows.iter().filter_map(RequestRow::tweet_id) {
            if let Entry::Vacant(entry) = tweet_misses.entry(tweet_id) {
                entry.insert(self.tweet_miss(tweet_id, &mut unclaimed));
            }
        }
        let mut tweets: FxHashMap<TweetId, HydratedTweet> =
            FxHashMap::with_capacity_and_hasher(self.rows.len(), Default::default());
        let mut users: FxHashMap<AuthorId, HydratedUser> = FxHashMap::default();
        for (position, row) in self.rows.iter().enumerate() {
            match row.subject {
                RowSubject::Post(tweet_id) => {
                    if let Entry::Vacant(entry) = tweets.entry(tweet_id) {
                        entry.insert(self.hydrated_tweet(
                            row,
                            position,
                            tweet_id,
                            &tweet_misses,
                            &mut unclaimed,
                        ));
                    }
                }
                RowSubject::User => {
                    if let Some(user) = row.author
                        && let Entry::Vacant(entry) = users.entry(user)
                    {
                        entry.insert(self.hydrated_user(row, position, user, &mut unclaimed));
                    }
                }
            }
        }
        for ((backend, call), keys) in unclaimed_keys_by_label(unclaimed) {
            metrics
                .plain()
                .count(&HYDRATOR_UNASKED_KEYS, [backend, call], keys as u64);
        }
        for (id, tweet) in &mut tweets {
            match tweet {
                HydratedTweet::Resolved { candidate, .. } => {
                    candidate.post.tweet_features = self
                        .take::<source::TesTweetForVisibility>(id.0)
                        .unwrap_or_default();
                }
                HydratedTweet::Unresolved { .. } => {}
            }
        }
        let (viewer, viewer_state) = match self.facts.viewer_id() {
            None => (Viewer::LoggedOut, None),
            Some(id) => {
                let DecodedViewer {
                    profile,
                    has_age_verified_18_label,
                    state,
                } = self.take::<source::GizmoduckViewer>(id).unwrap_or_default();
                let viewer = Viewer::LoggedIn {
                    id,
                    profile,
                    has_age_verified_18_label,
                };
                (viewer, state)
            }
        };
        let quotes = self.quoted();
        let has_fetched_retweet_sources = self.has_joined_retweet_sources();
        Hydration {
            viewer: ViewerFeatures {
                viewer,
                facts: self.facts,
            },
            viewer_state,
            planned: self.callable,
            viewer_failed,
            tweets,
            users,
            quotes,
            has_fetched_retweet_sources,
            retweet_merge: self.retweet_merge,
        }
    }

    fn viewer_failed(&self) -> Hydrators {
        let Some(viewer_id) = self.facts.viewer_id() else {
            return Hydrators::empty();
        };
        self.callable
            .iter()
            .filter(|&node| self.origin(node).names() == Names::Viewer)
            .filter(|&node| {
                self.answer(node, viewer_id)
                    .is_some_and(|answer| claimed_answer_fails(self.stake(node), answer))
            })
            .fold(Hydrators::empty(), Hydrators::with)
    }

    fn failed(
        &self,
        row: &RequestRow,
        position: usize,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> Hydrators {
        let has_joined_retweet_sources = self.has_joined_retweet_sources();
        let mut answered_incompletely = Hydrators::empty();
        for node in self.callable.iter() {
            let Some(key) = self.parent_key(node, row, position) else {
                continue;
            };
            let Some(answer) = self.answer(node, key) else {
                continue;
            };
            if answer == Answer::Unclaimed {
                if !has_joined_retweet_sources {
                    continue;
                }
                unclaimed.push((node, key));
            }
            if classify(self.stake(node), answer).fails_node() {
                answered_incompletely = answered_incompletely.with(node);
            }
        }
        let root_author_block = self.root_author_block(row);
        answered_incompletely =
            answered_incompletely.union(root_author_block.failed().intersection(self.callable));
        if answered_incompletely.is_empty() {
            return answered_incompletely;
        }
        self.callable
            .iter()
            .fold(answered_incompletely, |failed, node| {
                match node.input_at(self.request_rows) {
                    Some(input_node)
                        if failed.contains(input_node)
                            && self.parent_key(node, row, position).is_none() =>
                    {
                        failed.with(node)
                    }
                    _ => failed,
                }
            })
            .difference(root_author_block.unread())
    }
}

fn unclaimed_keys_by_label(
    mut unclaimed: Vec<(Hydrator, u64)>,
) -> HashMap<(&'static str, &'static str), usize> {
    unclaimed.sort_unstable_by_key(|&(node, key)| (node as u8, key));
    unclaimed.dedup();
    let mut by_label = HashMap::new();
    for (node, _) in &unclaimed {
        *by_label.entry(node.spec().label).or_default() += 1;
    }
    by_label
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeQuery, Graph};
    use crate::hydration::batch::HydrationBatch;
    use crate::hydration::execute::tests::{
        VIEWER, article_tweet, exclusive_tweet, executor, ids, tweet_failed,
    };
    use crate::hydration::sources::{Fault, InMemorySources, Source, control, suspended};
    use crate::hydration::{Cause, HydrationPlan, Hydrator, Hydrators, Lookup, Unresolved};
    use crate::models::PureCore;
    use crate::models::{RetweetMerge, TweetId, Viewer, ViewerProfile};
    use crate::rules::SafetyLevel;
    use crate::scenario::raw;
    use std::collections::HashSet;
    use xai_core_entities::entities::{
        ConversationControlArm, ExtendedProfile, GizmoduckUser, GizmoduckUserResult,
        UserResponseState,
    };

        #[test]
    fn each_edge_is_answered_by_one_source() {
        let mut sources: HashMap<Edge, Source> = HashMap::new();
        for &node in Hydrator::VARIANTS {
            if let Some(edge) = node.edge() {
                let source = node.spec().source;
                assert_eq!(*sources.entry(edge).or_insert(source), source, "{edge:?}");
            }
        }
    }

    #[test]
    fn each_viewer_keyed_edge_node_reads_its_sources_edge() {
        for &node in Hydrator::VARIANTS {
            let source = node.spec().source;
            if source.edge().is_some() || (node.is_viewer_edge() && source != Source::Flock) {
                assert_eq!(source.edge(), node.edge(), "{node:?}");
            }
        }
    }

    #[test]
    fn authors_prefer_the_request_author_and_leave_unresolved_tweets_without_one() {
        let core = |author| PureCore {
            author_id: AuthorId(author),
            source_tweet_id: None,
            source_author_id: None,
            reply_to: None,
        };
        let raw =
            [(1, Some(10)), (2, None), (3, None), (4, Some(41))].map(|(id, author)| RawCandidate {
                tweet_id: TweetId(id),
                request_author_id: author,
            });
        let plan = HydrationPlan::new(SafetyLevel::FilterAll, Hydrators::empty());
        let mut store = Store::new(&plan, RequestFacts::default(), &raw, RetweetMerge::Unmerged);
        let group = plan.groups().next().unwrap();
        let call = store
            .offer(group)
            .expect("pure core claims the request tweets");
        let pure_cores = HydrationBatch::from_results(
            [1, 2, 3, 4],
            HashMap::from([(2, Ok::<_, &str>(Some(core(20)))), (4, Ok(Some(core(40))))]),
        );
        store.land(&call, Reply::TesPureCoreData(pure_cores), Duration::ZERO);
        let resolved: Vec<(TweetId, u64)> = store
            .rows
            .iter()
            .filter_map(|row| Some((row.tweet_id()?, row.author?.get())))
            .collect();
        assert_eq!(
            resolved,
            vec![(TweetId(1), 10), (TweetId(2), 20), (TweetId(4), 41)]
        );
    }

    #[test]
    fn a_level_planning_only_the_follow_pair_edge_joins_its_quote_outers() {
        let raw = [RawCandidate {
            tweet_id: TweetId(1),
            request_author_id: None,
        }];
        let plan = HydrationPlan::new(
            SafetyLevel::FilterAll,
            Hydrators::of(Hydrator::OuterAuthorFollowsAuthor),
        );
        let mut store = Store::new(&plan, RequestFacts::default(), &raw, RetweetMerge::Unmerged)
            .with_quotes(&[QuoteEntry {
                inner: TweetId(1),
                outer: TweetId(9),
            }]);
        let pure_core = plan.groups().next().unwrap();
        let call = store.offer(pure_core).unwrap();
        let inner = PureCore {
            author_id: AuthorId(10),
            source_tweet_id: None,
            source_author_id: None,
            reply_to: None,
        };
        let cores =
            HydrationBatch::from_results([1], HashMap::from([(1, Ok::<_, ()>(Some(inner)))]));
        assert_eq!(
            store.land(&call, Reply::TesPureCoreData(cores), Duration::ZERO),
            Landing::KeysJoined
        );
        assert_eq!(store.offer(pure_core).unwrap().keys, [9]);
    }

    #[test]
    fn a_key_no_call_claimed_fails_its_candidate_once_sources_joined() {
        let raw = [RawCandidate {
            tweet_id: TweetId(1),
            request_author_id: None,
        }];
        let plan = HydrationPlan::new(SafetyLevel::FilterAll, Hydrators::empty());
        let mut store = Store::new(&plan, RequestFacts::default(), &raw, RetweetMerge::Merged);
        let call = store.offer(plan.groups().next().unwrap()).unwrap();
        let retweet = PureCore {
            author_id: AuthorId(10),
            source_tweet_id: Some(TweetId(5)),
            source_author_id: Some(AuthorId(20)),
            reply_to: None,
        };
        let cores =
            HydrationBatch::from_results([1], HashMap::from([(1, Ok::<_, ()>(Some(retweet)))]));
        assert_eq!(
            store.land(&call, Reply::TesPureCoreData(cores), Duration::ZERO),
            Landing::KeysJoined
        );
        let mut unclaimed = Vec::new();
        let source = &store.rows[1];
        for _ in 0..2 {
            let candidate = store.candidate(source, 1, TweetId(5), AuthorId(20), &mut unclaimed);
            assert_eq!(candidate.failed, Hydrators::of(Hydrator::PureCore));
        }
        assert_eq!(
            unclaimed_keys_by_label(unclaimed),
            HashMap::from([(Hydrator::PureCore.spec().label, 1)])
        );
    }

    #[tokio::test]
    async fn failed_ids_reports_exactly_the_candidates_each_node_flags() {
        use ConversationControlArm::{Co, Subscribers};
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let rows = [
            ("healthy home", TimelineHome, world(), vec![], vec![]),
            (
                "healthy home hydration",
                TimelineHomeHydration,
                world(),
                vec![],
                vec![],
            ),
            (
                "failed pure core",
                TimelineHome,
                world().fault(Source::TesPureCoreData, Fault::Fails),
                vec![],
                vec![(TweetId(1), tweet_failed()), (TweetId(2), tweet_failed())],
            ),
            (
                "incomplete author",
                TimelineHome,
                world().user(
                    10,
                    GizmoduckUserResult {
                        response_state: Some(UserResponseState::Failed),
                        ..suspended()
                    },
                ),
                vec![],
                vec![(
                    TweetId(1),
                    Unresolved {
                        lookup: Lookup::Author,
                        cause: Cause::Failed,
                    },
                )],
            ),
            (
                "failed author-keyed select",
                TimelineHome,
                world().fail_graph(Graph::Mutes),
                vec![1, 2],
                vec![],
            ),
            (
                "failed blocked-by select",
                TimelineHomeHydration,
                world().fail_graph(Graph::Blocks),
                vec![1, 2],
                vec![],
            ),
            (
                "failed tweet row",
                TimelineHome,
                world().fail_key(Source::TesTweetForVisibility, 1),
                vec![],
                vec![(TweetId(1), tweet_failed())],
            ),
            (
                "failed label lookup",
                TimelineHome,
                world().fail_key(Source::SafetyLabels, 1),
                vec![1],
                vec![],
            ),
            (
                "failed exclusive select",
                TimelineHome,
                world()
                    .tweet_features(1, exclusive_tweet())
                    .fail_graph(Graph::SuperFollows),
                vec![1],
                vec![],
            ),
            (
                "failed root-edge select",
                TimelineHomeHydration,
                world()
                    .control(1, control(Subscribers, 30, &[]))
                    .fail_graph(Graph::SuperFollows),
                vec![1],
                vec![],
            ),
            (
                "failed conversation-control row",
                TimelineHomeHydration,
                world().fail_key(Source::TesConversationControl, 1),
                vec![1],
                vec![],
            ),
            (
                "failed country lookup",
                TimelineHomeHydration,
                world()
                    .control(1, control(Co, 30, &["us"]))
                    .control(2, control(Co, 30, &[]))
                    .fault(Source::ViewerCountry, Fault::Fails),
                vec![1],
                vec![],
            ),
            (
                "failed viewer",
                TimelineHome,
                world().fault(Source::GizmoduckViewer, Fault::Fails),
                vec![1, 2],
                vec![],
            ),
            (
                "failed article lifecycle",
                TimelineHomeHydration,
                world()
                    .tweet_features(1, article_tweet(70))
                    .fault(Source::ArticleLifecycle, Fault::Fails),
                vec![1],
                vec![],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, sources, failed, unresolved) in rows {
            let hydrated = executor(sources, level, Some(VIEWER)).hydrate(&raw).await;
            assert_eq!(
                (hydrated.failed_ids, hydrated.unresolved),
                (ids(&failed), unresolved.into_iter().collect()),
                "{name}"
            );
        }
    }

    #[tokio::test]
    async fn each_candidate_carries_the_nodes_that_failed_for_it() {
        use ConversationControlArm::MyNetwork;
        use Hydrator::{
            BlockedByAuthor, Blocks, Follows, MuteRetweets, Mutes, RootFollowsViewer,
            RootFollowsViewerSecondDegree,
        };
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let relationships = Hydrators::of(Follows).with(Blocks).with(Mutes);
        let rows = [
            (
                "failed tweet row",
                TimelineHome,
                Some(VIEWER),
                world()
                    .tweet_features(2, exclusive_tweet())
                    .fail_key(Source::TesTweetForVisibility, 1),
                [Err(tweet_failed()), Ok(Hydrators::empty())],
            ),
            (
                "failed root edge on a MyNetwork reply",
                TimelineHomeHydration,
                Some(VIEWER),
                world()
                    .control(1, control(MyNetwork, 30, &[]))
                    .fail_graph(Graph::Follows),
                [
                    Ok(Hydrators::of(RootFollowsViewer)
                        .with(RootFollowsViewerSecondDegree)
                        .with(Follows)
                        .with(BlockedByAuthor)),
                    Ok(Hydrators::of(Follows).with(BlockedByAuthor)),
                ],
            ),
            (
                "failed pure core, authors from the request",
                TimelineHome,
                Some(VIEWER),
                world().fault(Source::TesPureCoreData, Fault::Fails),
                [Err(tweet_failed()); 2],
            ),
            (
                "failed relationships select on a retweet and an original",
                TimelineHome,
                Some(VIEWER),
                world().retweet(1, 10, 5, Some(30)).fail_graph(Graph::Mutes),
                [Ok(relationships.with(MuteRetweets)), Ok(relationships)],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, viewer_id, sources, expected) in rows {
            let hydrated = executor(sources, level, viewer_id).hydrate(&raw).await;
            assert_eq!(hydrated.failed_nodes, expected, "{name}");
        }
    }

    #[tokio::test]
    async fn safety_labels_carry_only_the_tweets_whose_labels_were_found() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .tweet(3, 30)
            .labels(2, Default::default())
            .fail_key(Source::SafetyLabels, 1);
        let raw = [raw(1, None), raw(2, None), raw(3, None)];
        let hydrated = executor(sources, SafetyLevel::TimelineHome, None)
            .hydrate(&raw)
            .await;
        assert_eq!(
            hydrated
                .safety_labels
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            ids(&[2])
        );
    }

    #[tokio::test]
    async fn authors_share_one_key_and_a_missing_user_is_complete() {
        let sources = InMemorySources::default().tweet(1, 10).tweet(2, 10);
        let raw = [raw(1, None), raw(2, None)];
        let scenario = executor(sources, SafetyLevel::TimelineHome, None);
        let sources = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert!(
            hydrated
                .candidates
                .iter()
                .all(|c| !c.author.features.is_suspended)
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn repeated_ids_share_the_first_occurrences_resolution() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended());
        let raw = [raw(1, Some(10)), raw(1, Some(20)), raw(1, None)];
        let hydrated = executor(sources, SafetyLevel::TimelineHome, None)
            .hydrate(&raw)
            .await;
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (c.author.id, c.author.features.is_suspended))
                .collect::<Vec<_>>(),
            [(10, true); 3]
        );
        assert!(hydrated.failed_ids.is_empty());
        assert!(hydrated.unresolved.is_empty());
    }

    #[tokio::test]
    async fn a_failed_viewer_call_fails_both_viewer_nodes_unverified() {
        use crate::hydration::plan::HydrationPlan;
        use xai_core_entities::entities::{Label, Labels};
        use xai_x_thrift::user_labels::LabelValue;
        let viewer_nodes = Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels);
        let labeled = GizmoduckUser {
            labels: Labels {
                labels: vec![Label {
                    label_value: LabelValue::AGE_VERIFIED_18.0,
                    created_at_msec: 0,
                }],
            },
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .viewer(VIEWER, labeled)
            .fault(Source::GizmoduckViewer, Fault::Fails);
        let level = SafetyLevel::TimelineHomeHydration;
        let plan = HydrationPlan::new(level, viewer_nodes);
        let hydrated = executor(sources, level, Some(VIEWER))
            .hydrate_with_plan(&plan, &[raw(1, None)])
            .await;
        assert_eq!(hydrated.candidates[0].failed, viewer_nodes);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
    }

    #[tokio::test]
    async fn a_level_that_plans_the_viewer_profile_decodes_it() {
        let sources = InMemorySources::default().viewer(
            VIEWER,
            GizmoduckUser {
                extended_profile: Some(ExtendedProfile {
                    age_in_years: Some(30),
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        let hydrated = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER))
            .hydrate(&[])
            .await;
        let Viewer::LoggedIn { profile, .. } = hydrated.viewer_features.viewer else {
            panic!("a logged-in request stays logged in")
        };
        assert_ne!(profile, ViewerProfile::default());
    }

    #[tokio::test]
    async fn only_retweeters_are_asked_for_the_mute_retweets_edge() {
        let relationships = |authors: &[u64], retweeters: &[u64]| {
            vec![
                EdgeQuery::forward(Graph::Follows, authors.to_vec()),
                EdgeQuery::forward(Graph::Blocks, authors.to_vec()),
                EdgeQuery::forward(Graph::Mutes, authors.to_vec()),
                EdgeQuery::forward(Graph::MuteRetweets, retweeters.to_vec()),
            ]
        };
        let sources = InMemorySources::default()
            .retweet(1, 10, 5, Some(20))
            .tweet(2, 30)
            .tweet(5, 20);
        let scenario = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER))
            .retweet_merge(RetweetMerge::Merged);
        scenario.hydrate(&[raw(1, None), raw(2, None)]).await;
        assert_eq!(
            scenario.sources().selects(),
            [relationships(&[10, 20, 30], &[10])]
        );

        let originals = InMemorySources::default().tweet(2, 30);
        let scenario = executor(originals, SafetyLevel::TimelineHome, Some(VIEWER));
        let originals = scenario.sources();
        scenario.hydrate(&[raw(2, None)]).await;
        assert_eq!(originals.selects(), [relationships(&[30], &[])]);
    }

    #[tokio::test]
    async fn exclusive_edges_dedup_conversation_authors() {
        let sources = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .tweet(3, 40)
                .authors(&[10, 20, 40])
                .tweet_features(1, exclusive_tweet())
                .tweet_features(2, exclusive_tweet())
                .edge(Graph::SuperFollows, VIEWER, 30)
        };
        let raw = [raw(1, None), raw(2, None), raw(1, None), raw(3, Some(40))];
        for viewer_id in [Some(VIEWER), None] {
            let sources = sources();
            let scenario = executor(sources, SafetyLevel::TimelineHome, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            assert_eq!(sources.keys(Source::TesPureCoreData), [vec![1, 2, 3]]);
            assert_eq!(sources.keys(Source::TesTweetForVisibility), [vec![1, 2, 3]]);
            let exclusive = (Some(30), viewer_id.is_some());
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| (
                        c.post.tweet_features.exclusive_conversation_author_id,
                        c.edges.contains(Hydrator::SuperFollowsExclusive)
                    ))
                    .collect::<Vec<_>>(),
                [exclusive, exclusive, exclusive, (None, false)]
            );
            let super_follows: Vec<Vec<u64>> = sources
                .selects()
                .into_iter()
                .flatten()
                .filter(|query| query.graph == Graph::SuperFollows)
                .map(|query| query.destination_ids)
                .collect();
            let expected: &[Vec<u64>] = if viewer_id.is_some() {
                &[vec![30]]
            } else {
                &[]
            };
            assert_eq!(super_follows, expected);
        }
    }
}
