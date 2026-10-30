use crate::inputs::{CandidateScoringInputs, QueryScoringContext};
use crate::scoring::{
    ValueScores, compute_value_scores, compute_weighted_score, offset_score, unoffset_score,
};
use crate::weights::ValueModelWeights;
use std::fmt;

/// Optional, calibrated probabilities of explicit viewer-reported outcomes.
/// These are not raw feedback counts, creator labels, or inferred app exits.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UserValuePredictions {
    pub worth_my_time: Option<f64>,
    pub helped_my_goal: Option<f64>,
    pub inspired_exit: Option<f64>,
    pub regretted_time: Option<f64>,
}

/// Nonnegative weights in the raw weighted-score units, with a symmetric cap.
/// The all-zero default disables adjustment. No model heads are enabled by this type.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UserValueWeights {
    pub worth_my_time: f64,
    pub helped_my_goal: f64,
    pub inspired_exit: f64,
    pub regretted_time: f64,
    pub max_abs_adjustment: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserValueError {
    LengthMismatch,
    InvalidWeights,
    InvalidProbability,
    NonFiniteScore,
}

impl fmt::Display for UserValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LengthMismatch => "user-value predictions must align one-to-one with candidates",
            Self::InvalidWeights => "user-value weights and cap must be finite and nonnegative, with a finite weight sum",
            Self::InvalidProbability => "user-value probabilities must be finite and between zero and one",
            Self::NonFiniteScore => "user-value adjustment produced a non-finite score",
        })
    }
}

impl std::error::Error for UserValueError {}

impl UserValueWeights {
    fn validate(&self) -> Result<(), UserValueError> {
        let weights = [
            self.worth_my_time,
            self.helped_my_goal,
            self.inspired_exit,
            self.regretted_time,
        ];
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || !weights.iter().sum::<f64>().is_finite()
            || !self.max_abs_adjustment.is_finite()
            || self.max_abs_adjustment < 0.0
        {
            return Err(UserValueError::InvalidWeights);
        }
        Ok(())
    }

    fn adjustment(&self, predictions: &UserValuePredictions) -> Result<f64, UserValueError> {
        fn probability(p: Option<f64>) -> Result<f64, UserValueError> {
            match p {
                None => Ok(0.0),
                Some(p) if p.is_finite() && (0.0..=1.0).contains(&p) => Ok(p),
                Some(_) => Err(UserValueError::InvalidProbability),
            }
        }
        let delta = probability(predictions.worth_my_time)? * self.worth_my_time
            + probability(predictions.helped_my_goal)? * self.helped_my_goal
            + probability(predictions.inspired_exit)? * self.inspired_exit
            - probability(predictions.regretted_time)? * self.regretted_time;
        Ok(delta.clamp(-self.max_abs_adjustment, self.max_abs_adjustment))
    }
}

/// Add bounded explicit-outcome utility before the existing offset/diversity path.
///
/// Missing predictions contribute zero; ordinary app closure is not a positive label.
/// Candidates and predictions must share the same order. Cached weighted scores must
/// be baseline scores under `weights`, without any previous user-value adjustment.
/// This function does not collect feedback, train heads, or override eligibility filters.
pub fn compute_value_scores_with_user_value(
    weights: &ValueModelWeights,
    context: &QueryScoringContext,
    candidates: &[CandidateScoringInputs],
    predictions: &[UserValuePredictions],
    user_value_weights: &UserValueWeights,
) -> Result<ValueScores, UserValueError> {
    if predictions.len() != candidates.len() {
        return Err(UserValueError::LengthMismatch);
    }
    user_value_weights.validate()?;
    if user_value_weights.max_abs_adjustment == 0.0 {
        return Ok(compute_value_scores(weights, context, candidates));
    }
    let mut adjusted = candidates.to_vec();
    for (candidate, prediction) in adjusted.iter_mut().zip(predictions) {
        let delta = user_value_weights.adjustment(prediction)?;
        if delta == 0.0 {
            continue;
        }
        let baseline = match candidate.weighted_score {
            Some(score) => unoffset_score(score, weights),
            None => compute_weighted_score(weights, candidate),
        };
        let raw = baseline + delta;
        let offset = offset_score(raw, weights);
        if !baseline.is_finite() || !raw.is_finite() || !offset.is_finite() {
            return Err(UserValueError::NonFiniteScore);
        }
        candidate.weighted_score = Some(offset);
    }
    Ok(compute_value_scores(weights, context, &adjusted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoenix_scores::PhoenixScores;

    fn outcome_weights() -> UserValueWeights {
        UserValueWeights {
            worth_my_time: 4.0,
            helped_my_goal: 3.0,
            inspired_exit: 10.0,
            regretted_time: 8.0,
            max_abs_adjustment: 20.0,
        }
    }

    fn candidate(dwell: f64) -> CandidateScoringInputs {
        CandidateScoringInputs {
            author_id: 1,
            in_network: Some(false),
            phoenix_scores: PhoenixScores {
                dwell_time: Some(dwell),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn disabled_and_missing_outcomes_preserve_existing_scoring_paths() {
        for multiplier_pre_offset in [false, true] {
            let weights = ValueModelWeights {
                cont_dwell_time: 1.0,
                favorite: 2.0,
                report: -10.0,
                enable_author_diversity: true,
                author_diversity_decay: 0.5,
                author_diversity_floor: 0.1,
                multiplier_pre_offset,
                ..Default::default()
            };
            let context = QueryScoringContext {
                effective_oon_weight: 0.5,
            };
            let candidates = [
                candidate(5.0),
                CandidateScoringInputs {
                    weighted_score: Some(2.001),
                    author_exploration_bonus: 0.2,
                    cold_start_lift_to_rank: Some(0),
                    ..candidate(1.0)
                },
            ];
            let expected = compute_value_scores(&weights, &context, &candidates);
            for policy in [UserValueWeights::default(), outcome_weights()] {
                let actual = compute_value_scores_with_user_value(
                    &weights,
                    &context,
                    &candidates,
                    &[UserValuePredictions::default(); 2],
                    &policy,
                )
                .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn an_explicit_inspired_exit_can_reward_a_short_successful_visit() {
        let weights = ValueModelWeights {
            cont_dwell_time: 1.0,
            ..Default::default()
        };
        let candidates = [candidate(5.0), candidate(0.5)];
        let baseline = compute_value_scores(&weights, &QueryScoringContext::default(), &candidates);
        assert!(baseline.scores[0] > baseline.scores[1]);
        let result = compute_value_scores_with_user_value(
            &weights,
            &QueryScoringContext::default(),
            &candidates,
            &[
                UserValuePredictions::default(),
                UserValuePredictions {
                    inspired_exit: Some(0.9),
                    ..Default::default()
                },
            ],
            &outcome_weights(),
        )
        .unwrap();
        assert!(result.scores[1] > result.scores[0]);
        assert!((result.weighted[1] - baseline.weighted[1] - 9.0).abs() < 1e-12);
        // The caller's original candidate remains unadjusted.
        assert_eq!(candidates[1].weighted_score, None);
    }

    #[test]
    fn regret_is_a_penalty_and_adjustments_are_capped() {
        let policy = UserValueWeights {
            max_abs_adjustment: 0.25,
            ..outcome_weights()
        };
        assert_eq!(
            policy.adjustment(&UserValuePredictions {
                worth_my_time: Some(1.0),
                helped_my_goal: Some(1.0),
                inspired_exit: Some(1.0),
                ..Default::default()
            }),
            Ok(0.25)
        );
        assert_eq!(
            policy.adjustment(&UserValuePredictions {
                regretted_time: Some(1.0),
                ..Default::default()
            }),
            Ok(-0.25)
        );
        let weights = ValueModelWeights {
            cont_dwell_time: 1.0,
            report: -10.0,
            ..Default::default()
        };
        let candidates = [candidate(0.1)];
        let baseline = compute_value_scores(&weights, &QueryScoringContext::default(), &candidates);
        let result = compute_value_scores_with_user_value(
            &weights,
            &QueryScoringContext::default(),
            &candidates,
            &[UserValuePredictions {
                regretted_time: Some(1.0),
                ..Default::default()
            }],
            &policy,
        )
        .unwrap();
        assert!(result.scores[0] < baseline.scores[0]);
    }

    #[test]
    fn cached_and_fresh_baselines_receive_the_same_adjustment() {
        let weights = ValueModelWeights {
            favorite: 2.0,
            report: -10.0,
            cont_dwell_time: 1.0,
            ..Default::default()
        };
        let fresh = candidate(2.0);
        let cached = CandidateScoringInputs {
            weighted_score: Some(offset_score(2.0, &weights)),
            ..fresh.clone()
        };
        let predictions = [UserValuePredictions {
            helped_my_goal: Some(0.5),
            ..Default::default()
        }];
        let result = |c| {
            compute_value_scores_with_user_value(
                &weights,
                &QueryScoringContext::default(),
                &[c],
                &predictions,
                &outcome_weights(),
            )
            .unwrap()
        };
        assert_eq!(result(fresh), result(cached));
    }

    #[test]
    fn adjusted_scores_still_use_existing_diversity_and_oon_multipliers() {
        let weights = ValueModelWeights {
            cont_dwell_time: 1.0,
            enable_author_diversity: true,
            author_diversity_decay: 0.5,
            author_diversity_floor: 0.0,
            ..Default::default()
        };
        let context = QueryScoringContext {
            effective_oon_weight: 0.5,
        };
        let result = compute_value_scores_with_user_value(
            &weights,
            &context,
            &[candidate(2.0), candidate(1.0)],
            &[
                UserValuePredictions::default(),
                UserValuePredictions {
                    helped_my_goal: Some(1.0),
                    ..Default::default()
                },
            ],
            &outcome_weights(),
        )
        .unwrap();
        assert_eq!(result.weighted, vec![2.0, 4.0]);
        assert_eq!(result.scores, vec![0.5, 2.0]);
    }

    #[test]
    fn validates_prediction_alignment_and_probabilities() {
        let weights = ValueModelWeights::default();
        let context = QueryScoringContext::default();
        assert_eq!(
            compute_value_scores_with_user_value(
                &weights,
                &context,
                &[candidate(1.0)],
                &[],
                &outcome_weights()
            ),
            Err(UserValueError::LengthMismatch)
        );
        for p in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for prediction in [
                UserValuePredictions {
                    worth_my_time: Some(p),
                    ..Default::default()
                },
                UserValuePredictions {
                    helped_my_goal: Some(p),
                    ..Default::default()
                },
                UserValuePredictions {
                    inspired_exit: Some(p),
                    ..Default::default()
                },
                UserValuePredictions {
                    regretted_time: Some(p),
                    ..Default::default()
                },
            ] {
                assert_eq!(
                    compute_value_scores_with_user_value(
                        &weights,
                        &context,
                        &[candidate(1.0)],
                        &[prediction],
                        &outcome_weights()
                    ),
                    Err(UserValueError::InvalidProbability)
                );
            }
        }
    }

    #[test]
    fn validates_configuration_and_score_overflow() {
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for policy in [
                UserValueWeights {
                    worth_my_time: value,
                    ..outcome_weights()
                },
                UserValueWeights {
                    helped_my_goal: value,
                    ..outcome_weights()
                },
                UserValueWeights {
                    inspired_exit: value,
                    ..outcome_weights()
                },
                UserValueWeights {
                    regretted_time: value,
                    ..outcome_weights()
                },
                UserValueWeights {
                    max_abs_adjustment: value,
                    ..outcome_weights()
                },
            ] {
                assert_eq!(policy.validate(), Err(UserValueError::InvalidWeights));
            }
        }
        assert_eq!(
            UserValueWeights {
                worth_my_time: f64::MAX,
                helped_my_goal: f64::MAX,
                ..Default::default()
            }
            .validate(),
            Err(UserValueError::InvalidWeights)
        );
        let candidate = CandidateScoringInputs {
            weighted_score: Some(f64::MAX),
            ..Default::default()
        };
        let policy = UserValueWeights {
            worth_my_time: f64::MAX,
            max_abs_adjustment: f64::MAX,
            ..Default::default()
        };
        assert_eq!(
            compute_value_scores_with_user_value(
                &ValueModelWeights::default(),
                &QueryScoringContext::default(),
                &[candidate],
                &[UserValuePredictions {
                    worth_my_time: Some(1.0),
                    ..Default::default()
                }],
                &policy
            ),
            Err(UserValueError::NonFiniteScore)
        );
    }
}
