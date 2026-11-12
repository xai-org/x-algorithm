use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct DropDuplicatesFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for DropDuplicatesFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        deduplicate(candidates, |candidate| candidate.tweet_id)
    }
}

pub(crate) fn deduplicate(
    mut candidates: Vec<PostCandidate>,
    key: impl Fn(&PostCandidate) -> u64,
) -> FilterResult<PostCandidate> {
    let mut first: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(candidates.len(), Default::default());
    let mut keep = Vec::with_capacity(candidates.len());
    for index in 0..candidates.len() {
        match first.entry(key(&candidates[index])) {
            Entry::Occupied(entry) => {
                let (head, tail) = candidates.split_at_mut(index);
                head[*entry.get()]
                    .retrieval_sources
                    .append(&mut tail[0].retrieval_sources);
                keep.push(false);
            }
            Entry::Vacant(entry) => {
                entry.insert(index);
                keep.push(true);
            }
        }
    }
    let mut keep = keep.into_iter();
    FilterResult::partition(candidates, |_| keep.next().unwrap_or(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::retweet_deduplication_filter::RetweetDeduplicationFilter;
    use crate::models::candidate::RetrievalSource;
    use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
    use xai_home_mixer_proto::ServedType;

    fn sources(positions: &[u32]) -> Vec<RetrievalSource> {
        positions
            .iter()
            .map(|&position| RetrievalSource {
                served_type: ServedType::ForYouPhoenixRetrieval,
                cluster: None,
                score: Some(position as f32 / 8.0),
                position: Some(position),
            })
            .collect()
    }

    fn candidate(id: u64, retweet: Option<u64>, payload: u64, positions: &[u32]) -> PostCandidate {
        PostCandidate {
            tweet_id: id,
            retweeted_tweet_id: retweet,
            author_id: payload,
            tweet_text: format!("payload-{payload} café"),
            ancestors: vec![payload + 100],
            retrieval_sources: sources(positions),
            ..Default::default()
        }
    }

    fn assert_result(
        filter: &impl Filter<ScoredPostsQuery, PostCandidate>,
        cached: bool,
        input: Vec<PostCandidate>,
        kept: &[PostCandidate],
        removed: &[PostCandidate],
    ) {
        let json = |candidates: &[PostCandidate]| serde_json::to_value(candidates).unwrap();
        let query = ScoredPostsQuery {
            has_cached_posts: cached,
            ..Default::default()
        };
        let pointer = input.as_ptr();
        let result = filter.run(&query, input, PipelineStage::Filter);
        assert_eq!(json(&result.kept), json(kept));
        assert_eq!(json(&result.removed), json(removed));
        if !kept.is_empty() {
            assert_eq!(result.kept.as_ptr(), pointer);
        }
    }

    #[test]
    fn keeps_first_payload_and_appends_provenance_in_encounter_order() {
        let input = vec![
            candidate(10, Some(100), 1, &[1, 2]),
            candidate(20, None, 2, &[3]),
            candidate(10, Some(999), 3, &[4]),
            candidate(11, Some(100), 4, &[5, 6]),
            candidate(100, None, 5, &[7]),
            candidate(20, None, 6, &[]),
            candidate(10, None, 7, &[8, 8]),
        ];
        let at = |index: usize, positions: &[u32]| PostCandidate {
            retrieval_sources: sources(positions),
            ..input[index].clone()
        };
        for cached in [false, true] {
            assert_result(
                &DropDuplicatesFilter,
                cached,
                input.clone(),
                &[
                    at(0, &[1, 2, 4, 8, 8]),
                    at(1, &[3]),
                    at(3, &[5, 6]),
                    at(4, &[7]),
                ],
                &[at(2, &[]), at(5, &[]), at(6, &[])],
            );
            assert_result(
                &RetweetDeduplicationFilter,
                cached,
                input.clone(),
                &[
                    at(0, &[1, 2, 5, 6, 7]),
                    at(1, &[3]),
                    at(2, &[4]),
                    at(6, &[8, 8]),
                ],
                &[at(3, &[]), at(4, &[]), at(5, &[])],
            );
        }
    }

    #[test]
    fn empty_and_unique_batches_keep_every_candidate() {
        for input in [
            vec![],
            vec![
                candidate(0, Some(0), 1, &[1, 1]),
                candidate(u64::MAX, None, 2, &[]),
            ],
        ] {
            assert_result(&DropDuplicatesFilter, false, input.clone(), &input, &[]);
            assert_result(
                &RetweetDeduplicationFilter,
                false,
                input.clone(),
                &input,
                &[],
            );
        }
    }

    #[test]
    fn large_batches_deduplicate_in_the_input_allocation() {
        for retweets in [false, true] {
            let input: Vec<_> = (0..3000u64)
                .map(|i| match (i % 10 == 9, retweets) {
                    (true, false) => candidate(i - 5, None, i, &[i as u32]),
                    (true, true) => candidate(100_000 + i, Some(i - 5), i, &[i as u32]),
                    _ => candidate(i, None, i, &[i as u32]),
                })
                .collect();
            let (mut kept, mut removed): (Vec<_>, Vec<_>) =
                input.iter().cloned().partition(|c| c.author_id % 10 != 9);
            for candidate in &mut kept {
                if candidate.author_id % 10 == 4 && candidate.author_id + 5 < 3000 {
                    candidate
                        .retrieval_sources
                        .extend(sources(&[candidate.author_id as u32 + 5]));
                }
            }
            for candidate in &mut removed {
                candidate.retrieval_sources.clear();
            }
            if retweets {
                assert_result(&RetweetDeduplicationFilter, false, input, &kept, &removed);
            } else {
                assert_result(&DropDuplicatesFilter, false, input, &kept, &removed);
            }
        }
    }
}
