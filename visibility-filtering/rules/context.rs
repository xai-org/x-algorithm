use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    tweet_timestamp_ms, ClientCapability, Placement, PostCandidate, PostEntity, Quote,
    RequestFacts, UserEntity, ViewerFeatures,
};
use crate::params::{CountryList, CountryLists};
use crate::rules::rule_spec::engine::Candidate;
use crate::rules::rule_spec::features::Reads;

pub(crate) struct RuleContext<'a, C = PostCandidate> {
    facts: CoreFacts<'a>,
    viewer: &'a ViewerFeatures,
    candidate: &'a C,
    #[cfg(test)]
    hydrated: Hydrators,
}

#[derive(Clone, Copy)]
pub(super) struct CoreFacts<'a> {
    viewer_id: Option<u64>,
    request_facts: &'a RequestFacts,
    placement: Placement,
    country_lists: &'a CountryLists,
}

#[derive(Clone, Copy)]
pub(super) struct PostCoreFacts<'a>(&'a PostCandidate);

impl<'a, C: Candidate> RuleContext<'a, C> {
    pub(super) fn new(
        viewer: &'a ViewerFeatures,
        candidate: &'a C,
        placement: Placement,
        country_lists: &'a CountryLists,
    ) -> Self {
        Self {
            facts: CoreFacts {
                viewer_id: viewer.viewer.user_id(),
                request_facts: &viewer.facts,
                placement,
                country_lists,
            },
            viewer,
            candidate,
            #[cfg(test)]
            hydrated: Hydrators::all(),
        }
    }

    #[cfg(test)]
    pub(super) fn hydrated_by(self, hydrated: Hydrators) -> Self {
        Self { hydrated, ..self }
    }

    #[inline]
    fn reads(&self, required: Hydrators) -> &Self {
        #[cfg(test)]
        for hydrator in required.iter() {
            assert!(
                self.hydrated.contains(hydrator),
                "a rule reads {hydrator:?}, which its policy does not derive"
            );
        }
        #[cfg(not(test))]
        let _ = required;
        self
    }

    #[inline]
    pub(super) fn facts(&self) -> CoreFacts<'a> {
        self.facts
    }

    #[inline]
    pub(super) fn failed(&self) -> Hydrators {
        self.candidate.failed()
    }

    #[inline]
    pub(super) fn acquire<F: Reads<C>>(&self) -> F::Value<'a> {
        F::read(At(self.reads(F::REQUIRED)))
    }

    #[inline]
    pub(super) fn acquire_pair_edge(&self, node: Hydrator) -> bool {
        self.reads(Hydrators::of(node))
            .candidate
            .quote()
            .is_some_and(|quote| quote.pair_edges.contains(node))
    }

    #[inline]
    pub(super) fn acquire_edge(&self, node: Hydrator) -> bool {
        At(self.reads(Hydrators::of(node))).edge(node)
    }
}

pub(super) struct At<'c, 'a, C = PostCandidate>(&'c RuleContext<'a, C>);

impl<C> Clone for At<'_, '_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for At<'_, '_, C> {}

impl<'a, C: Candidate> At<'_, 'a, C> {
    #[inline]
    pub(super) fn author(self) -> &'a UserEntity {
        self.0.candidate.author()
    }

    #[inline]
    pub(super) fn viewer(self) -> &'a ViewerFeatures {
        self.0.viewer
    }

    #[inline]
    pub(super) fn quote(self) -> Option<Quote> {
        self.0.candidate.quote()
    }

    #[inline]
    pub(super) fn edge(self, node: Hydrator) -> bool {
        debug_assert!(node.is_viewer_edge(), "{node:?} is not a viewer edge node");
        self.0.candidate.edges().contains(node)
    }
}

impl<'a> At<'_, 'a> {
    #[inline]
    pub(super) fn post(self) -> &'a PostEntity {
        &self.0.candidate.post
    }

    #[inline]
    pub(super) fn post_facts(self) -> PostCoreFacts<'a> {
        PostCoreFacts(self.0.candidate)
    }
}

impl<'a> CoreFacts<'a> {
    #[inline]
    pub(super) fn viewer_id(self) -> Option<u64> {
        self.viewer_id
    }

    #[inline]
    pub(super) fn request_country(self) -> Option<&'a str> {
        self.request_facts.request_country()
    }

    #[inline]
    pub(super) fn client_capability(self) -> ClientCapability {
        self.request_facts.client_capability()
    }

    #[inline]
    pub(super) fn soft_nsfw_new_user_drop(self) -> bool {
        self.request_facts.soft_nsfw_new_user_drop()
    }

    #[inline]
    pub(super) fn placement(self) -> Placement {
        self.placement
    }

    #[inline]
    pub(super) fn in_country_list(self, list: CountryList, country_code: &str) -> bool {
        self.country_lists.contains(list, country_code)
    }
}

impl PostCoreFacts<'_> {
    #[inline]
    pub(super) fn tweet_id(self) -> u64 {
        self.0.post.tweet_id
    }

    #[inline]
    pub(super) fn created_after(self, unix_ms: u64) -> bool {
        tweet_timestamp_ms(self.0.post.tweet_id) > unix_ms
    }
}
