use super::edges::edge_key;
use super::key_origin::PairKey;
use super::{RequestRow, Store};
use crate::hydration::fetcher::{AnyFetcher, Fetcher};
use crate::hydration::hydrator::{EdgeKey, UserPair};
use crate::hydration::inventory::classify;
use crate::hydration::relation::{quoting_author, Related};
use crate::hydration::sources::source;
use crate::hydration::{candidate_count_by_key, Cause, Hydrator, Hydrators, QuoteEntry, Quoted};
use crate::models::{AuthorId, EdgeSet, Quote, QuoteLink, TweetId};
use rustc_hash::FxHashMap;
use xai_core_entities::entities::QuotedTweet;

#[derive(Default)]
pub(super) struct Quotes {
    entries: Vec<QuoteEntry>,
}

impl Quotes {
    pub(super) fn new(entries: &[QuoteEntry]) -> Self {
        let mut entries = entries.to_vec();
        entries.sort_unstable();
        entries.dedup();
        Self { entries }
    }

    pub(super) fn outer_of(&self, inner: TweetId) -> Option<TweetId> {
        self.entries
            .iter()
            .find(|quote| quote.inner == inner)
            .map(|quote| quote.outer)
    }

    pub(super) fn quoted_author(
        &self,
        quoted_tweets: &Fetcher<QuotedTweet>,
        inner: TweetId,
    ) -> Result<AuthorId, Cause> {
        let quotes = || {
            self.entries
                .iter()
                .filter(move |quote| quote.inner == inner)
        };
        let named = quotes().find_map(|quote| {
            quoted_tweets
                .get(quote.outer.0)
                .filter(|quoted| quoted.tweet_id == inner.0 && quoted.user_id != 0)
        });
        match named {
            Some(quoted) => Ok(AuthorId(quoted.user_id)),
            None if quotes().any(|quote| quoted_tweets.is_incomplete(quote.outer.0)) => {
                Err(Cause::Failed)
            }
            None => Err(Cause::NotFound),
        }
    }
}

impl Store {
    pub(super) fn quote_outers(&self) -> Vec<(usize, Related)> {
        self.quotes
            .entries
            .iter()
            .filter_map(|quote| {
                let parent = self
                    .rows
                    .iter()
                    .position(|row| row.tweet_id() == Some(quote.inner))?;
                let related = Related {
                    tweet_id: quote.outer,
                    author: None,
                };
                Some((parent, related))
            })
            .collect()
    }

    fn paired_rows(&self) -> impl Iterator<Item = (&RequestRow, TweetId)> {
        self.paired
            .iter()
            .filter_map(|&(parent, outer)| Some((self.rows.get(parent)?, outer)))
    }

    fn quoting_author(&self, outer: TweetId) -> Option<AuthorId> {
        self.request_fetcher::<source::TesPureCoreData>()
            .get(outer.0)
            .map(quoting_author)
    }

    pub(super) fn quote_pair(
        &self,
        parent: &RequestRow,
        outer: TweetId,
        ask: impl FnOnce(AuthorId, AuthorId) -> PairKey,
    ) -> PairKey {
        let (Some(quoted), Some(quoting)) = (parent.author, self.quoting_author(outer)) else {
            return PairKey::InputFailed;
        };
        if quoted == quoting {
            return PairKey::Unasked;
        }
        ask(quoted, quoting)
    }

    fn pair_key(&self, node: Hydrator, parent: &RequestRow, outer: TweetId) -> PairKey {
        self.quote_pair_key(self.origin(node), parent, outer)
    }

    fn asked_pairs(&self, node: Hydrator) -> Vec<UserPair> {
        let mut pairs: Vec<UserPair> = self
            .paired_rows()
            .filter_map(|(parent, outer)| match self.pair_key(node, parent, outer) {
                PairKey::Asked(asked) => Some(asked),
                PairKey::Unasked | PairKey::InputFailed => None,
            })
            .collect();
        pairs.sort_unstable_by_key(|pair| (pair.source, pair.destination));
        pairs.dedup();
        pairs
    }

    pub(super) fn claim_quote_pairs(&mut self, nodes: Hydrators) -> Vec<(u64, Vec<u64>)> {
        let mut claimed: Vec<(u64, Vec<u64>)> = Vec::new();
        for node in nodes.iter() {
            let edges = self
                .asked_pairs(node)
                .into_iter()
                .filter_map(|pair| edge_key(node, pair))
                .collect();
            let Some(fetcher) = self.edge_fetcher_mut(node) else {
                continue;
            };
            for pair in fetcher.claim(edges).into_iter().map(|edge| edge.pair) {
                match claimed.last_mut() {
                    Some((source, destinations)) if *source == pair.source => {
                        destinations.push(pair.destination);
                    }
                    _ => claimed.push((pair.source, vec![pair.destination])),
                }
            }
        }
        claimed
    }

    pub(super) fn candidate_count_by_pair(&self, nodes: Hydrators) -> FxHashMap<u64, usize> {
        let destinations = |node| {
            self.paired_rows().filter_map(move |(parent, outer)| {
                match self.pair_key(node, parent, outer) {
                    PairKey::Asked(asked) => Some(asked.destination),
                    PairKey::Unasked | PairKey::InputFailed => None,
                }
            })
        };
        candidate_count_by_key(
            nodes
                .iter()
                .filter(|node| node.is_quote_pair_edge())
                .flat_map(destinations),
        )
    }

    pub(super) fn quoted(&self) -> FxHashMap<QuoteEntry, Quoted> {
        self.paired_rows()
            .filter_map(|(parent, outer)| {
                let quote = QuoteEntry {
                    inner: parent.tweet_id()?,
                    outer,
                };
                Some((quote, self.quoted_entry(parent, outer)))
            })
            .collect()
    }

    fn quoted_entry(&self, parent: &RequestRow, outer: TweetId) -> Quoted {
        let mut pair_edges = EdgeSet::default();
        let mut failed = Hydrators::empty();
        for node in self
            .callable
            .iter()
            .filter(|node| node.is_quote_pair_edge())
        {
            let (Some(fetcher), Some(edge)) = (self.edge_fetcher(node), node.edge()) else {
                continue;
            };
            let holds = match self.pair_key(node, parent, outer) {
                PairKey::Asked(pair) => {
                    let asked = EdgeKey { edge, pair };
                    if classify(self.stake(node), fetcher.answer(asked)).fails_node() {
                        failed = failed.with(node);
                    }
                    fetcher.get(asked).copied().unwrap_or_default()
                }
                PairKey::Unasked => false,
                PairKey::InputFailed => {
                    failed = failed.with(node);
                    false
                }
            };
            if holds {
                pair_edges = pair_edges.with(node);
            }
        }
        Quoted {
            quote: self.quoting_author(outer).map(|quoting| Quote {
                link: QuoteLink {
                    outer,
                    quoting_author: quoting.get(),
                },
                pair_edges,
            }),
            failed,
        }
    }
}
