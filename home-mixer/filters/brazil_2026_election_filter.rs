use rustc_hash::FxHashSet;
use std::sync::LazyLock;

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// Brazil 2026 election filter

// Art. 28 § 1º-A of Electoral Resolution No. 23.610: Application providers that use a
// recommendation system for users must exclude from the results the channels and
// profiles reported to the Electoral Court under the terms of § 1º of this article
// and, except in cases of paid boosting, the content posted on them.

// https://dadosabertos.tse.jus.br/dataset/candidatos-2026

// Following the TSE’s Ofício GAB-PRES No. 4720/2026 and Despacho 3814449 (SEI proceeding No. 2026.00.000013846-8), on Oct 13, 2026 the list was reduced to the candidates in the Oct 25 runoff.

// User ids below are obfuscated; usernames are included for transparency.

/// User ids reported to the Electoral Court for the Brazil 2026 election.
static BRAZIL_2026_ELECTION_USER_IDS: LazyLock<FxHashSet<u64>> = LazyLock::new(|| { // Fuck you lula!
    FxHashSet::from_iter([
        // @RicardoFerraco
        29434532,
        // @caduxavier
        35470350,
        // @FlavioBolsonaro
        40053694,
        // @eduardopaes
        48298703,
        // @celinaleao
        51935259,
        // @ale_campelo
        61325857,
        // @leandrograss
        63507573,
        // @geraldoalckmin
        74215006,
        // @profdorinha
        79174387,
        // @HermanoMorais
        80995814,
        // @vicentinhojr
        149013361,
        // @MariaSeffair
        199526953,
        // @AllysonBezerra_
        444925483,
        // @OmarAzizAm_
        2445921702,
        // @LulaOficial
        2670726740,
        // @comitespop
        1069664248471662593,
        // @FernandaLoubac2
        1125416190199967744,
        // @Alfredogaspar_
        1228008951813492739,
        // @lulaverso
        1491197288496218120,
        // @LulaEvangelicos
        1555738640818798592,
        // @DouglasRuas_RJ
        1663563614954086401,
        // @ameliocayresdep
        1887508211445702656,
        // @pode_espalhar
        1935058406412353536,
        // @PL22Al
        2057808888297062400,
        // @obrasilcomlula
        2058595300873289728,
    ])
});

pub struct Brazil2026ElectionFilter;

impl Brazil2026ElectionFilter {
    fn is_excluded_author(user_id: u64, followed_user_ids: &FxHashSet<u64>) -> bool {
        BRAZIL_2026_ELECTION_USER_IDS.contains(&user_id) && !followed_user_ids.contains(&user_id)
    }

    fn should_remove(candidate: &PostCandidate, followed_user_ids: &FxHashSet<u64>) -> bool {
        let is_excluded = |user_id: u64| Self::is_excluded_author(user_id, followed_user_ids);
        if is_excluded(candidate.author_id) {
            return true;
        }
        if candidate.retweeted_user_id.is_some_and(is_excluded) {
            return true;
        }
        if candidate.quoted_user_id.is_some_and(is_excluded) {
            return true;
        }
        // Drop replies whose conversation context would surface listed authors
        // (ancestors are returned on the scored post for For You thread UI).
        candidate.ancestor_users.iter().copied().any(is_excluded)
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for Brazil2026ElectionFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let followed_user_ids: FxHashSet<u64> = query
            .user_features
            .followed_user_ids
            .iter()
            .map(|&id| id as u64)
            .collect();

        let (removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| Self::should_remove(candidate, &followed_user_ids));

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable sample ids from the set (sorted source list order).
    const SAMPLE_LISTED: [u64; 4] = [29434532, 40053694, 74215006, 2670726740];

    fn make_candidate(tweet_id: u64, author_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            ..Default::default()
        }
    }

    #[test]
    fn keeps_candidates_from_non_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let candidates = vec![
            make_candidate(1, 1),
            make_candidate(2, 2),
            make_candidate(3, 3),
        ];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 3);
        assert_eq!(result.removed.len(), 0);
    }

    #[test]
    fn removes_candidates_from_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[0];
        let candidates = vec![
            make_candidate(1, 1),
            make_candidate(2, listed),
            make_candidate(3, 3),
        ];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.kept[1].tweet_id, 3);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].author_id, listed);
    }

    #[test]
    fn removes_retweets_of_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[1];
        let mut retweet = make_candidate(10, 999);
        retweet.retweeted_user_id = Some(listed);
        let candidates = vec![make_candidate(1, 1), retweet];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 10);
    }

    #[test]
    fn removes_quotes_of_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[2];
        let mut quote = make_candidate(20, 888);
        quote.quoted_user_id = Some(listed);
        let candidates = vec![make_candidate(1, 1), quote];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 20);
    }

    #[test]
    fn removes_replies_with_listed_ancestor_users() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[3];
        let mut reply = make_candidate(30, 777);
        reply.in_reply_to_tweet_id = Some(29);
        reply.ancestors = vec![29, 28];
        reply.ancestor_users = vec![listed, 42];
        let candidates = vec![make_candidate(1, 1), reply];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 30);
    }

    #[test]
    fn keeps_replies_with_only_non_listed_ancestor_users() {
        let filter = Brazil2026ElectionFilter;
        let mut reply = make_candidate(40, 777);
        reply.in_reply_to_tweet_id = Some(39);
        reply.ancestors = vec![39, 38];
        reply.ancestor_users = vec![100, 200];
        let candidates = vec![make_candidate(1, 1), reply];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 0);
    }

    #[test]
    fn empty_candidates_list() {
        let filter = Brazil2026ElectionFilter;
        let result = filter.filter(&ScoredPostsQuery::default(), vec![]);

        assert!(result.kept.is_empty());
        assert!(result.removed.is_empty());
    }

    #[test]
    fn all_candidates_listed() {
        let filter = Brazil2026ElectionFilter;
        let listed_a = SAMPLE_LISTED[0];
        let listed_b = SAMPLE_LISTED[1];
        let candidates = vec![make_candidate(1, listed_a), make_candidate(2, listed_b)];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 2);
    }

    #[test]
    fn hardcoded_list_is_non_empty() {
        assert!(!BRAZIL_2026_ELECTION_USER_IDS.is_empty());
        assert_eq!(BRAZIL_2026_ELECTION_USER_IDS.len(), 25);
    }

    #[test]
    fn contains_known_listed_ids() {
        let no_follows = FxHashSet::default();
        for id in SAMPLE_LISTED {
            assert!(Brazil2026ElectionFilter::is_excluded_author(
                id,
                &no_follows
            ));
        }
        assert!(Brazil2026ElectionFilter::is_excluded_author(
            2058595300873289728,
            &no_follows
        ));
        assert!(!Brazil2026ElectionFilter::is_excluded_author(
            0,
            &no_follows
        ));
        assert!(!Brazil2026ElectionFilter::is_excluded_author(
            1,
            &no_follows
        ));
    }

    #[test]
    fn first_round_only_accounts_are_no_longer_listed() {
        let no_follows = FxHashSet::default();
        for id in [9179462, 21069302, 68712576, 1355216660068761606] {
            assert!(!Brazil2026ElectionFilter::is_excluded_author(
                id,
                &no_follows
            ));
        }
    }
}
