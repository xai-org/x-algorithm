use crate::clients::socialgraph_client::{EdgeFrom, EdgeQuery};
use crate::hydration::community_source::CommunityPost;
use crate::hydration::decode::author::{AuthorNotFound, DecodedActor, DecodedAuthor};
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::execute::Reply;
use crate::hydration::fetcher::{AnyFetcher, Fetcher, FetcherId, Fetchers, Owner};
use crate::hydration::hydrator::{Edge, NodeSpec, RequestRows, UserPair};
use crate::hydration::inventory::{Answer, Effect, Source, Stake, claimed_answer_fails, classify};
use crate::hydration::plan::Group;
use crate::hydration::relation::{Related, Relation, Role, Trigger};
use crate::hydration::tweet_source::TweetState;
use crate::hydration::{
    Cause, HydratedTweet, HydratedUser, Hydration, HydrationPlan, HydrationRequest, Hydrator,
    Hydrators, QuoteEntry,
};
use crate::metrics::families::HYDRATOR_UNASKED_KEYS;
use crate::models::{
    ArticleLifecycle, AuthorId, ClientCapability, CommunityModeration, NotificationFilters,
    PureCore, RawCandidate, RetweetMerge, TweetFeatures, TweetId, Viewer, ViewerFeatures,
};
pub(super) use key_origin::{ControlArms, KeyOrigin, Names};
use quotes::Quotes;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::Duration;
use strum::VariantArray;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};
use xai_visibility_filtering_proto as vf_pb;

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

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Landing {
    Answered,
    KeysJoined,
}

#[derive(Default)]
pub(super) struct Store {
    viewer_id: Option<u64>,
    client_capability: ClientCapability,
    rows: Vec<RequestRow>,
    related: FxHashMap<(Relation, usize), TweetId>,
    paired: Vec<(usize, TweetId)>,
        requested_tweet_ids: FxHashSet<TweetId>,
        joined_tweet_ids: FxHashSet<TweetId>,
    request_rows: RequestRows,
    quotes: Quotes,
    retweet_merge: RetweetMerge,
    callable: Hydrators,
    pure_cores: Fetchers<PureCore>,
    tweets: Fetcher<TweetFeatures>,
    controls: Fetcher<ConversationControl>,
    tweet_states: Fetcher<TweetState>,
    labels: Fetchers<Arc<vf_pb::SafetyLabelMap>>,
    viewer: Fetcher<DecodedViewer>,
    authors: Fetchers<DecodedAuthor>,
    author_not_found: AuthorNotFound,
    actors: Fetcher<DecodedActor>,
    actor_emails: Fetcher<bool>,
    actor_phones: Fetcher<bool>,
    edges: [Fetchers<bool, UserPair>; Edge::VARIANTS.len()],
    has_called: Vec<bool>,
    viewer_country: Fetcher<Arc<str>>,
    community_moderations: Fetcher<CommunityModeration>,
    community_moderators: Fetcher<bool>,
    community_viewer_removals: Fetcher<bool>,
    article_lifecycles: Fetcher<ArticleLifecycle>,
    article_media_entities: Fetcher<bool>,
    article_cover_media: Fetcher<bool>,
    pub(super) core_elapsed: Duration,
    pub(super) tweets_elapsed: Option<Duration>,
}

impl Store {
    pub(super) fn new(
        plan: &HydrationPlan,
        viewer_id: Option<u64>,
        client_capability: ClientCapability,
        raw: &[RawCandidate],
        retweet_merge: RetweetMerge,
    ) -> Self {
        Self {
            viewer_id,
            client_capability,
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
            callable: plan.callable(viewer_id),
            author_not_found: plan.level().spec().author_not_found(),
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
        self.viewer_id
            .and_then(|viewer_id| self.viewer.get(viewer_id))
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
            Trigger::PureCore(named) => named(self.pure_cores.request().get(parent.tweet_id()?.0)?),
            Trigger::Tweet(named) => named(self.tweets.get(parent.tweet_id()?.0)?),
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
        self.controls.get(row.tweet_id()?.0)
    }

    fn source_tweet_id(&self, tweet_id: TweetId) -> Option<TweetId> {
        self.pure_cores.request().get(tweet_id.0)?.source_tweet_id
    }

        fn community_post(&self, row: &RequestRow) -> Option<CommunityPost> {
        let tweet_id = row.tweet_id()?.0;
        let author_id = row.author?.get();
        if self.viewer_id == Some(author_id) {
            return None;
        }
        Some(CommunityPost {
            tweet_id,
            author_id,
            community_id: self.tweets.get(tweet_id)?.community_id?.get(),
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
        self.fetcher_by_id(self.spec(node).fetcher())
    }

    fn fetcher_mut(&mut self, node: Hydrator) -> Option<&mut dyn AnyFetcher> {
        self.fetcher_by_id_mut(self.spec(node).fetcher())
    }

    fn fetcher_by_id(&self, FetcherId { source, owner }: FetcherId) -> Option<&dyn AnyFetcher> {
        let fetcher: &dyn AnyFetcher = match source {
            Source::TesPureCoreData => self.pure_cores.of(owner),
            Source::TesTweetForVisibility => &self.tweets,
            Source::TesConversationControl => &self.controls,
            Source::TesQuotedTweet => &self.quotes.quoted_tweets,
            Source::TesTweetState => &self.tweet_states,
            Source::SafetyLabels => self.labels.of(owner),
            Source::GizmoduckViewer => &self.viewer,
            Source::GizmoduckAuthor => self.authors.of(owner),
            Source::GizmoduckActor => &self.actors,
            Source::ActorEmails => &self.actor_emails,
            Source::ActorPhones => &self.actor_phones,
            Source::ViewerCountry => &self.viewer_country,
            Source::CommunityModeration => &self.community_moderations,
            Source::CommunityModerator => &self.community_moderators,
            Source::CommunityViewerRemoved => &self.community_viewer_removals,
            Source::ArticleLifecycle => &self.article_lifecycles,
            Source::ArticleMediaEntities => &self.article_media_entities,
            Source::ArticleCoverMedia => &self.article_cover_media,
            Source::Flock | Source::Wingman | Source::TrustedFriends | Source::UserLocation => {
                return None;
            }
        };
        Some(fetcher)
    }

    fn fetcher_by_id_mut(
        &mut self,
        FetcherId { source, owner }: FetcherId,
    ) -> Option<&mut dyn AnyFetcher> {
        let fetcher: &mut dyn AnyFetcher = match source {
            Source::TesPureCoreData => self.pure_cores.of_mut(owner),
            Source::TesTweetForVisibility => &mut self.tweets,
            Source::TesConversationControl => &mut self.controls,
            Source::TesQuotedTweet => &mut self.quotes.quoted_tweets,
            Source::TesTweetState => &mut self.tweet_states,
            Source::SafetyLabels => self.labels.of_mut(owner),
            Source::GizmoduckViewer => &mut self.viewer,
            Source::GizmoduckAuthor => self.authors.of_mut(owner),
            Source::GizmoduckActor => &mut self.actors,
            Source::ActorEmails => &mut self.actor_emails,
            Source::ActorPhones => &mut self.actor_phones,
            Source::ViewerCountry => &mut self.viewer_country,
            Source::CommunityModeration => &mut self.community_moderations,
            Source::CommunityModerator => &mut self.community_moderators,
            Source::CommunityViewerRemoved => &mut self.community_viewer_removals,
            Source::ArticleLifecycle => &mut self.article_lifecycles,
            Source::ArticleMediaEntities => &mut self.article_media_entities,
            Source::ArticleCoverMedia => &mut self.article_cover_media,
            Source::Flock | Source::Wingman | Source::TrustedFriends | Source::UserLocation => {
                return None;
            }
        };
        Some(fetcher)
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
        if self.viewer_id.is_none() && group.nodes.iter().all(Hydrator::needs_viewer) {
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
        let community_posts = match group.fetcher.source {
            Source::CommunityModeration => self.community_posts(&keys),
            Source::TesPureCoreData
            | Source::TesTweetForVisibility
            | Source::TesConversationControl
            | Source::TesQuotedTweet
            | Source::TesTweetState
            | Source::SafetyLabels
            | Source::GizmoduckViewer
            | Source::GizmoduckAuthor
            | Source::Flock
            | Source::ViewerCountry
            | Source::Wingman
            | Source::CommunityModerator
            | Source::CommunityViewerRemoved
            | Source::ArticleLifecycle
            | Source::ArticleMediaEntities
            | Source::ArticleCoverMedia
            | Source::TrustedFriends
            | Source::UserLocation
            | Source::GizmoduckActor
            | Source::ActorEmails
            | Source::ActorPhones => Vec::new(),
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
            viewer_id: self.viewer_id,
            keys,
            key_count,
            queries,
            queried_edges,
            community_posts,
        })
    }

    pub(super) fn is_country_lookup_skipped(&self) -> bool {
        self.callable.contains(Hydrator::ViewerCountry)
            && !self.viewer_country.has_claimed()
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
        let keys = &call.keys;
        let owner = call.group.fetcher.owner;
        let is_first_request_core = call.is_first && call.group.nodes.contains(Hydrator::PureCore);
        match reply {
            Reply::PureCores(pure_cores) => {
                if is_first_request_core {
                    self.core_elapsed = elapsed;
                }
                self.pure_cores.of_mut(owner).land(keys, pure_cores);
                self.land_authors();
            }
            Reply::Tweets(tweets) => {
                if call.is_first {
                    self.tweets_elapsed = Some(elapsed);
                }
                self.tweets.land(keys, tweets);
            }
            Reply::Controls(controls) => self.controls.land(keys, controls),
            Reply::QuotedTweets(quoted_tweets) => {
                self.quotes.quoted_tweets.land(keys, quoted_tweets);
                self.land_authors();
            }
            Reply::TweetStates(states) => self.tweet_states.land(keys, states),
            Reply::Labels(labels) => self.labels.of_mut(owner).land(keys, labels),
            Reply::Viewer(viewer) => self.viewer.land(keys, viewer),
            Reply::Authors(authors) => {
                let authors = if owner == Owner::Request {
                    self.author_not_found.read(authors)
                } else {
                    authors
                };
                self.authors.of_mut(owner).land(keys, authors);
            }
            Reply::Actors(actors) => self.actors.land(keys, actors),
            Reply::ActorEmails(emails) => self.actor_emails.land(keys, emails),
            Reply::ActorPhones(phones) => self.actor_phones.land(keys, phones),
            Reply::Select(answers) => {
                debug_assert_eq!(answers.len(), call.queries.len(), "one answer per query");
                for ((edge, query), answer) in call.edge_queries().zip(answers) {
                    if let Some(source) = query.from.id(call.viewer_id) {
                        self.land_edge(owner, edge, source, &query.destination_ids, answer);
                    }
                }
            }
            Reply::Edge(edge, answers) => {
                if let Some(source) = call.viewer_id {
                    self.land_edge(owner, edge, source, keys, answers);
                }
            }
            Reply::ViewerCountry(country) => self.viewer_country.land(keys, country),
            Reply::CommunityModerations(moderations) => {
                self.community_moderations.land(keys, moderations);
            }
            Reply::CommunityModerators(moderators) => {
                self.community_moderators.land(keys, moderators);
            }
            Reply::CommunityViewerRemovals(removals) => {
                self.community_viewer_removals.land(keys, removals);
            }
            Reply::ArticleLifecycles(lifecycles) => self.article_lifecycles.land(keys, lifecycles),
            Reply::ArticleMediaEntities(rows) => self.article_media_entities.land(keys, rows),
            Reply::ArticleCoverMedia(rows) => self.article_cover_media.land(keys, rows),
        }
        if self.join(call.group.nodes) {
            Landing::KeysJoined
        } else {
            Landing::Answered
        }
    }

    fn land_authors(&mut self) {
        let (pure_cores, quotes) = (self.pure_cores.request(), &self.quotes);
        for row in &mut self.rows {
            row.author = row
                .request_author
                .or_else(|| Some(pure_cores.get(row.tweet_id()?.0)?.author_id))
                .or_else(|| quotes.quoted_author(row.tweet_id()?).ok());
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

    pub(super) fn assemble(mut self, request: HydrationRequest<'_>) -> Hydration {
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
            let metrics = request.metrics.plain();
            metrics.count(&HYDRATOR_UNASKED_KEYS, [backend, call], keys as u64);
        }
        for (id, tweet) in &mut tweets {
            match tweet {
                HydratedTweet::Resolved { candidate, .. } => {
                    candidate.post.tweet_features = self.tweets.take(id.0).unwrap_or_default();
                }
                HydratedTweet::Unresolved { .. } => {}
            }
        }
        let (viewer, viewer_state) = match request.viewer_id {
            None => (Viewer::LoggedOut, None),
            Some(id) => {
                let DecodedViewer {
                    profile,
                    has_age_verified_18_label,
                    state,
                } = self.viewer.take(id).unwrap_or_default();
                let viewer = Viewer::LoggedIn {
                    id,
                    profile,
                    has_age_verified_18_label,
                };
                (viewer, state)
            }
        };
        let quotes = self.quoted();
        Hydration {
            viewer: ViewerFeatures::from_request(
                viewer,
                request.country_code,
                request.client_capability,
            ),
            viewer_state,
            planned: self.callable,
            viewer_failed,
            tweets,
            users,
            quotes,
            has_fetched_retweet_sources: self.has_joined_retweet_sources(),
            retweet_merge: self.retweet_merge,
        }
    }

    fn viewer_failed(&self) -> Hydrators {
        let Some(viewer_id) = self.viewer_id else {
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
    use crate::hydration::batch::HydrationBatch;
    use crate::rules::SafetyLevel;

    #[test]
    fn every_fetcher_id_off_an_edge_has_a_fetcher_of_its_own() {
        let mut store = Store::default();
        let is_edge_only = |source: Source| {
            let readers: Vec<&Hydrator> = Hydrator::VARIANTS
                .iter()
                .filter(|node| node.spec().source == source)
                .collect();
            !readers.is_empty() && readers.iter().all(|node| node.is_edge())
        };
        let mut fetchers = Vec::new();
        for id in FetcherId::all() {
            let fetcher = store
                .fetcher_by_id(id)
                .map(|fetcher| std::ptr::from_ref(fetcher).cast::<()>());
            let fetcher_mut = store
                .fetcher_by_id_mut(id)
                .map(|fetcher| std::ptr::from_mut(fetcher).cast_const().cast::<()>());
            assert_eq!(fetcher, fetcher_mut, "{id:?}");
            match fetcher {
                Some(fetcher) => fetchers.push((id, fetcher)),
                None => assert!(is_edge_only(id.source), "{id:?} has no fetcher"),
            }
        }
        for (position, (id, fetcher)) in fetchers.iter().enumerate() {
            for (other, other_fetcher) in fetchers.iter().skip(position + 1) {
                assert_ne!(
                    fetcher, other_fetcher,
                    "{id:?} shares the fetcher of {other:?}"
                );
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
        let mut store = Store::new(
            &plan,
            None,
            ClientCapability::default(),
            &raw,
            RetweetMerge::Unmerged,
        );
        let group = plan.groups().next().unwrap();
        let call = store
            .offer(group)
            .expect("pure core claims the request tweets");
        let pure_cores = HydrationBatch::from_results(
            [1, 2, 3, 4],
            HashMap::from([(2, Ok::<_, &str>(Some(core(20)))), (4, Ok(Some(core(40))))]),
        );
        store.land(&call, Reply::PureCores(pure_cores), Duration::ZERO);
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
        let mut store = Store::new(
            &plan,
            None,
            ClientCapability::default(),
            &raw,
            RetweetMerge::Unmerged,
        )
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
            store.land(&call, Reply::PureCores(cores), Duration::ZERO),
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
        let mut store = Store::new(
            &plan,
            None,
            ClientCapability::default(),
            &raw,
            RetweetMerge::Merged,
        );
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
            store.land(&call, Reply::PureCores(cores), Duration::ZERO),
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
}
