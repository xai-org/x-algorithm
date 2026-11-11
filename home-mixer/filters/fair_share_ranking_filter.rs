use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableFairShareRanking, PhoenixScoresResultSize};
use crate::scorers::phoenix_scorer::PHOENIX_RANKER_KILL_SWITCH_DECIDER;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_home_mixer_proto::ServedType;

pub struct FairShareRankingFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for FairShareRankingFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        if !query.params.get(EnableFairShareRanking)
            || query.has_cached_posts
            || query.scoring_sequence.is_none()
        {
            return false;
        }
        !query
            .decider
            .as_ref()
            .is_some_and(|decider| decider.enabled(PHOENIX_RANKER_KILL_SWITCH_DECIDER))
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let budget = query.params.get(PhoenixScoresResultSize) as usize;
        let mut selected = selection_mask(&candidates, budget).into_iter();
        FilterResult::partition(candidates, |_| selected.next().unwrap_or(false))
    }
}

fn served_type(candidate: &PostCandidate) -> Option<ServedType> {
    candidate.served_type.or_else(|| {
        candidate
            .retrieval_sources
            .first()
            .map(|source| source.served_type)
    })
}

fn group_by_served_type(candidates: &[PostCandidate]) -> Vec<Vec<usize>> {
    let mut keys: Vec<Option<ServedType>> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let key = served_type(candidate);
        if let Some(group) = keys.iter().position(|seen| *seen == key) {
            groups[group].push(index);
        } else {
            keys.push(key);
            groups.push(vec![index]);
        }
    }
    groups
}

fn fair_share_counts(counts: &[usize], budget: usize) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total == 0 || total <= budget {
        return counts.to_vec();
    }
    let total = total as u128;
    let budget_u = budget as u128;
    let mut alloc = Vec::with_capacity(counts.len());
    let mut remainders = Vec::with_capacity(counts.len());
    let mut assigned = 0usize;
    for (index, &count) in counts.iter().enumerate() {
        let product = count as u128 * budget_u;
        let keep = (product / total) as usize;
        alloc.push(keep);
        assigned += keep;
        remainders.push((product % total, index));
    }
    remainders.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    let mut leftover = budget - assigned;
    for (_, index) in remainders {
        if leftover == 0 {
            break;
        }
        if alloc[index] < counts[index] {
            alloc[index] += 1;
            leftover -= 1;
        }
    }
    alloc
}

fn selection_mask(candidates: &[PostCandidate], budget: usize) -> Vec<bool> {
    let groups = group_by_served_type(candidates);
    let demands: Vec<usize> = groups.iter().map(|group| group.len()).collect();
    let keep = fair_share_counts(&demands, budget);
    let mut selected = vec![false; candidates.len()];
    for (group, &count) in groups.iter().zip(&keep) {
        for &index in group.iter().take(count) {
            selected[index] = true;
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::RetrievalSource;

    fn candidates(ids: std::ops::Range<u64>, served: ServedType) -> Vec<PostCandidate> {
        ids.map(|tweet_id| PostCandidate {
            tweet_id,
            served_type: Some(served),
            retrieval_sources: vec![RetrievalSource::from_served_type(served)],
            ..Default::default()
        })
        .collect()
    }

    #[test]
    fn drops_the_same_fraction_of_each_served_type() {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_fair_share_ranking".to_string(),
            "true",
        );
        results.override_fs(
            "rust_home_mixer_phoenix_scores_result_size".to_string(),
            "1000",
        );
        let query = ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        };
        let mut pool = candidates(0..440, ServedType::ForYouInNetwork);
        pool.extend(candidates(10_000..10_220, ServedType::ForYouSimclusters));
        pool.extend(candidates(
            20_000..20_440,
            ServedType::ForYouPhoenixRetrieval,
        ));

        let result = FairShareRankingFilter.filter(&query, pool);
        let count = |pred: fn(u64) -> bool| {
            result
                .kept
                .iter()
                .filter(|candidate| pred(candidate.tweet_id))
                .count()
        };
        assert_eq!(
            (
                count(|id| id < 10_000),
                count(|id| (10_000..10_220).contains(&id)),
                count(|id| id >= 20_000),
            ),
            (400, 200, 400)
        );
        assert_eq!(result.kept.len(), 1000);
        assert_eq!(result.kept.len() + result.removed.len(), 1100);
    }
}
