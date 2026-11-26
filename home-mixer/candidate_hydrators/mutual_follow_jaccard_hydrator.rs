use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableMutualFollowJaccardHydration;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

const MIN_HASHES: usize = 256;

pub struct MutualFollowJaccardHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

fn jaccard_from_minhash(a: &[i64], b: &[i64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let matching = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matching as f64 / len as f64
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for MutualFollowJaccardHydrator {
    type Fetched = Vec<Option<Option<f64>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableMutualFollowJaccardHydration) && query.viewer_minhash.is_some()
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Option<f64>>>, String> {
        let viewer_minhash = match &query.viewer_minhash {
            Some(mh) if mh.len() >= MIN_HASHES => mh,
            _ => return Ok(candidates.iter().map(|_| Some(None)).collect()),
        };

        let unique_author_ids: Vec<i64> = candidates
            .iter()
            .map(|c| c.author_id as i64)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let results = self
            .strato_client
            .batch_get_minhash_with_count(&unique_author_ids)
            .await;

        let mut author_minhash: HashMap<i64, Option<Vec<i64>>> = HashMap::new();
        for (uid, result) in unique_author_ids.iter().zip(results) {
            match result {
                Ok(Some((minhash, _count))) if minhash.len() >= MIN_HASHES => {
                    author_minhash.insert(*uid, Some(minhash));
                }
                Ok(None) => {
                    author_minhash.insert(*uid, None);
                }
                Ok(Some(_)) | Err(_) => {}
            }
        }

        Ok(candidates
            .iter()
            .map(|c| {
                author_minhash.get(&(c.author_id as i64)).map(|author_mh| {
                    author_mh
                        .as_ref()
                        .map(|author_mh| jaccard_from_minhash(viewer_minhash, author_mh))
                })
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        jaccards: Vec<Option<Option<f64>>>,
    ) {
        for (candidate, jaccard) in candidates.iter_mut().zip(jaccards) {
            if let Some(jaccard) = jaccard {
                candidate.mutual_follow_jaccard = jaccard;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.mutual_follow_jaccard = hydrated.mutual_follow_jaccard;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_signatures_yield_1() {
        let sig: Vec<i64> = (0..MIN_HASHES as i64).collect();
        assert!((jaccard_from_minhash(&sig, &sig) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn completely_different_signatures_yield_0() {
        let a: Vec<i64> = (0..MIN_HASHES as i64).collect();
        let b: Vec<i64> = (MIN_HASHES as i64..2 * MIN_HASHES as i64).collect();
        assert!((jaccard_from_minhash(&a, &b)).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_signatures_yield_0() {
        assert!((jaccard_from_minhash(&[], &[])).abs() < f64::EPSILON);
    }

    #[test]
    fn partial_overlap() {
        let a: Vec<i64> = (0..MIN_HASHES as i64).collect();
        let mut b = a.clone();
        for i in MIN_HASHES / 2..MIN_HASHES {
            b[i] = a[i] + 1000;
        }
        let j = jaccard_from_minhash(&a, &b);
        assert!((j - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn mixed_length_256_vs_1024_compares_shared_prefix() {
        let short: Vec<i64> = (0..256_i64).collect();
        let mut long: Vec<i64> = (0..1024_i64).collect();
        for val in &mut long[256..1024] {
            *val += 9999;
        }
        let j = jaccard_from_minhash(&short, &long);
        assert!((j - 1.0).abs() < f64::EPSILON);
    }
}
