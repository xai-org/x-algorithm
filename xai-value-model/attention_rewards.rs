use crate::weights::ValueModelWeights;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionRewardScaleError {
    InvalidScale,
    NonFiniteWeight,
}

impl fmt::Display for AttentionRewardScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidScale => "attention reward scale must be finite and between zero and one",
            Self::NonFiniteWeight => "attention reward weights must be finite",
        })
    }
}

impl std::error::Error for AttentionRewardScaleError {}

impl ValueModelWeights {
    /// Reduce positive rewards for consuming attention, without weakening negative feedback.
    ///
    /// A scale of one preserves these weights; zero removes their positive contribution.
    /// Dwell probability, quality video views, continuous time predictions, and the
    /// mutual-follow dwell boost are covered. Clicks, likes, replies, shares, follows,
    /// not-interested, blocks, mutes, reports, and not-dwelled weights are unchanged.
    /// This must be applied to the original weights, once, before scoring or caching.
    /// It represents an explicit viewer preference, not an inferred well-being label.
    pub fn with_attention_reward_scale(
        mut self,
        scale: f64,
    ) -> Result<Self, AttentionRewardScaleError> {
        if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
            return Err(AttentionRewardScaleError::InvalidScale);
        }
        let attention_weights = [
            &mut self.dwell,
            &mut self.vqv,
            &mut self.quoted_vqv,
            &mut self.cont_dwell_time,
            &mut self.cont_click_dwell_time,
            &mut self.video_continuation,
            &mut self.user_video_continuation,
            &mut self.profile_visit_secs,
            &mut self.bidirectional_follow_dwell_weight_boost,
        ];
        for weight in attention_weights {
            if !weight.is_finite() {
                return Err(AttentionRewardScaleError::NonFiniteWeight);
            }
            if *weight > 0.0 {
                *weight *= scale;
            }
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::CandidateScoringInputs;
    use crate::phoenix_scores::PhoenixScores;
    use crate::scoring::compute_weighted_score;

    fn weights() -> ValueModelWeights {
        ValueModelWeights {
            favorite: 2.0,
            dwell: 4.0,
            vqv: 6.0,
            quoted_vqv: 8.0,
            cont_dwell_time: 10.0,
            cont_click_dwell_time: 12.0,
            video_continuation: 14.0,
            user_video_continuation: 16.0,
            profile_visit_secs: 18.0,
            bidirectional_follow_dwell_weight_boost: 20.0,
            not_interested: -5.0,
            block_author: -6.0,
            mute_author: -7.0,
            report: -8.0,
            not_dwelled: -9.0,
            enable_author_diversity: true,
            author_diversity_decay: 0.5,
            author_diversity_floor: 0.1,
            ..Default::default()
        }
    }

    #[test]
    fn one_preserves_the_entire_configuration() {
        assert_eq!(
            weights().with_attention_reward_scale(1.0).unwrap(),
            weights()
        );
    }

    #[test]
    fn zero_removes_every_positive_attention_reward_only() {
        let original = weights();
        let scaled = original.clone().with_attention_reward_scale(0.0).unwrap();
        assert_eq!(
            scaled,
            ValueModelWeights {
                dwell: 0.0,
                vqv: 0.0,
                quoted_vqv: 0.0,
                cont_dwell_time: 0.0,
                cont_click_dwell_time: 0.0,
                video_continuation: 0.0,
                user_video_continuation: 0.0,
                profile_visit_secs: 0.0,
                bidirectional_follow_dwell_weight_boost: 0.0,
                ..original
            }
        );
    }

    #[test]
    fn fractional_scale_covers_all_attention_channels() {
        let original = weights();
        let scaled = original.clone().with_attention_reward_scale(0.25).unwrap();
        assert_eq!(
            scaled,
            ValueModelWeights {
                dwell: 1.0,
                vqv: 1.5,
                quoted_vqv: 2.0,
                cont_dwell_time: 2.5,
                cont_click_dwell_time: 3.0,
                video_continuation: 3.5,
                user_video_continuation: 4.0,
                profile_visit_secs: 4.5,
                bidirectional_follow_dwell_weight_boost: 5.0,
                ..original
            }
        );
    }

    #[test]
    fn negative_attention_weights_are_not_weakened() {
        let original = ValueModelWeights {
            dwell: -2.0,
            vqv: -3.0,
            quoted_vqv: -4.0,
            cont_dwell_time: -5.0,
            cont_click_dwell_time: -6.0,
            video_continuation: -7.0,
            user_video_continuation: -8.0,
            profile_visit_secs: -9.0,
            bidirectional_follow_dwell_weight_boost: -10.0,
            ..weights()
        };
        assert_eq!(
            original.clone().with_attention_reward_scale(0.0).unwrap(),
            original
        );
    }

    #[test]
    fn rejects_invalid_scales_and_weights() {
        for scale in [-0.01, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                weights().with_attention_reward_scale(scale),
                Err(AttentionRewardScaleError::InvalidScale)
            );
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = ValueModelWeights {
                cont_dwell_time: value,
                ..weights()
            };
            assert_eq!(
                invalid.with_attention_reward_scale(0.0),
                Err(AttentionRewardScaleError::NonFiniteWeight)
            );
        }
    }

    #[test]
    fn time_alone_no_longer_outranks_a_useful_action_at_zero() {
        let original = ValueModelWeights {
            favorite: 2.0,
            cont_dwell_time: 1.0,
            ..Default::default()
        };
        let long_view = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                dwell_time: Some(30.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let liked = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.8),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            compute_weighted_score(&original, &long_view)
                > compute_weighted_score(&original, &liked)
        );
        let scaled = original.with_attention_reward_scale(0.0).unwrap();
        assert!(
            compute_weighted_score(&scaled, &long_view) < compute_weighted_score(&scaled, &liked)
        );
    }

    #[test]
    fn negative_feedback_contribution_is_identical() {
        let original = weights();
        let scaled = original.clone().with_attention_reward_scale(0.0).unwrap();
        let candidate = CandidateScoringInputs {
            phoenix_scores: PhoenixScores {
                not_interested_score: Some(0.5),
                block_author_score: Some(0.4),
                mute_author_score: Some(0.3),
                report_score: Some(0.2),
                not_dwelled_score: Some(0.1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            compute_weighted_score(&original, &candidate),
            compute_weighted_score(&scaled, &candidate)
        );
    }
}
