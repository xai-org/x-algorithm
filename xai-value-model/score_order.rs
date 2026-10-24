use std::cmp::Ordering;

/// Compare ranking scores in descending order, with non-finite scores last.
///
/// All non-finite scores tie with each other. Equal finite scores (including
/// signed zero) also tie; callers retain control over stable sorting/tie-breaks.
pub fn compare_scores_desc(a: f64, b: f64) -> Ordering {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => b.partial_cmp(&a).expect("finite scores are comparable"),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_scores_sort_after_even_negative_finite_scores() {
        let mut scores = [5.0, f64::NAN, 10.0, f64::INFINITY, -5.0, f64::NEG_INFINITY];
        scores.sort_by(|a, b| compare_scores_desc(*a, *b));
        assert_eq!(&scores[..3], &[10.0, 5.0, -5.0]);
        assert!(scores[3].is_nan());
        assert_eq!(scores[4], f64::INFINITY);
        assert_eq!(scores[5], f64::NEG_INFINITY);
    }

    #[test]
    fn finite_order_and_stable_ties_are_preserved() {
        let mut scores = [0.0, -0.0, -4.0, 3.0, f64::MIN, 3.0, f64::MAX];
        let mut expected = scores;
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());
        scores.sort_by(|a, b| compare_scores_desc(*a, *b));
        assert_eq!(scores.map(f64::to_bits), expected.map(f64::to_bits));
    }

    #[test]
    fn empty_singleton_and_all_invalid_inputs() {
        for mut scores in [
            vec![],
            vec![f64::NAN],
            vec![f64::NEG_INFINITY, f64::NAN, -f64::NAN, f64::INFINITY],
        ] {
            let expected: Vec<_> = scores.iter().map(|s| s.to_bits()).collect();
            scores.sort_by(|a, b| compare_scores_desc(*a, *b));
            assert_eq!(
                scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn comparison_is_consistent_and_transitive() {
        let scores = [
            f64::MIN,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::MAX,
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for a in scores {
            assert_eq!(compare_scores_desc(a, a), Ordering::Equal);
            for b in scores {
                let ab = compare_scores_desc(a, b);
                assert_eq!(ab, compare_scores_desc(b, a).reverse());
                for c in scores {
                    if ab != Ordering::Greater && compare_scores_desc(b, c) != Ordering::Greater {
                        assert_ne!(compare_scores_desc(a, c), Ordering::Greater);
                    }
                }
            }
        }
    }
}
