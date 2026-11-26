use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    AuthorZeroFavCodivertIsControl, AuthorZeroFavCodivertIsTreatment,
    ViewerZeroFavCodivertIsTreatment,
};
use std::collections::HashMap;
use std::sync::Arc;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_feature_switches::AuthorRulesEvaluator;
use xai_home_mixer_proto::ServedType;
use xai_stats_receiver::global_stats_receiver;

const METRIC: &str = "ZeroFavCodivert";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthorBucket {
    Treatment,
    Control,
    Unbucketed,
}

impl AuthorBucket {
    const ALL: [AuthorBucket; 3] = [
        AuthorBucket::Treatment,
        AuthorBucket::Control,
        AuthorBucket::Unbucketed,
    ];

    fn label(self) -> &'static str {
        match self {
            AuthorBucket::Treatment => "treatment",
            AuthorBucket::Control => "control",
            AuthorBucket::Unbucketed => "unbucketed",
        }
    }
}

pub struct ZeroFavCodivertFilter {
    pub author_rules: Arc<AuthorRulesEvaluator>,
}

impl ZeroFavCodivertFilter {
    fn author_bucket(&self, author_id: u64) -> AuthorBucket {
        let is_treatment = self
            .author_rules
            .get(author_id, AuthorZeroFavCodivertIsTreatment);
        let is_control = self
            .author_rules
            .get(author_id, AuthorZeroFavCodivertIsControl);
        match (is_treatment, is_control) {
            (true, _) => AuthorBucket::Treatment,
            (false, true) => AuthorBucket::Control,
            (false, false) => AuthorBucket::Unbucketed,
        }
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for ZeroFavCodivertFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(ViewerZeroFavCodivertIsTreatment)
    }

    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let mut bucket_by_author = HashMap::<u64, AuthorBucket>::new();
        let mut counts = [0u64; AuthorBucket::ALL.len()];
        let (kept, removed): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            let bucket = *bucket_by_author
                .entry(c.author_id)
                .or_insert_with(|| self.author_bucket(c.author_id));
            if !is_zero_fav_only(c) {
                return true;
            }
            counts[bucket as usize] += 1;
            bucket == AuthorBucket::Treatment
        });

        if let Some(receiver) = global_stats_receiver() {
            let mut unique_authors = [0u64; AuthorBucket::ALL.len()];
            for bucket in bucket_by_author.values() {
                unique_authors[*bucket as usize] += 1;
            }
            for bucket in AuthorBucket::ALL {
                let labels = [("author_bucket", bucket.label())];
                let value = counts[bucket as usize];
                if value > 0 {
                    receiver.incr(METRIC, &labels, value);
                }
                receiver.observe_vm(
                    &format!("{METRIC}.UniqueAuthors"),
                    &labels,
                    unique_authors[bucket as usize] as f64,
                );
            }
        }

        FilterResult { kept, removed }
    }
}

fn is_zero_fav_only(c: &PostCandidate) -> bool {
    c.served_type == Some(ServedType::ForYouPhoenixRetrievalZeroFav)
        && c.retrieval_sources
            .iter()
            .all(|s| s.served_type == ServedType::ForYouPhoenixRetrievalZeroFav)
}
