use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableSimclustersSource;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_home_mixer_proto::ServedType;
use xai_visibility_filtering::models::SafetyLabelType;

pub struct OONNsfwSimclustersFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for OONNsfwSimclustersFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableSimclustersSource)
    }

    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            // Keep post-level restrictions focused on the post:
            // For out-of-network SimClusters candidates, if the author has an nsfw_author flag,
            // drop the candidate if the post itself carries sensitive/nsfw safety labels,
            // or if the post contains media. Compliant non-media posts from the creator
            // are retained to give subsequent compliant content a fresh evaluation.
            let post_has_nsfw = c.safety_labels.iter().any(|l| {
                matches!(
                    l.label_type,
                    SafetyLabelType::NSFW_HIGH_PRECISION
                        | SafetyLabelType::NSFW_HIGH_RECALL
                        | SafetyLabelType::NSFW_TEXT
                        | SafetyLabelType::NSFW_CARD_IMAGE
                        | SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION
                )
            });

            c.served_type == Some(ServedType::ForYouSimclusters)
                && c.in_network == Some(false)
                && c.nsfw_author == Some(true)
                && (post_has_nsfw || c.has_media.unwrap_or(false))
        });

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::SafetyLabelInfo;

    #[test]
    fn removes_oon_simclusters_when_author_nsfw_and_has_media() {
        let filter = OONNsfwSimclustersFilter;
        let query = ScoredPostsQuery::default();
        let candidates = vec![PostCandidate {
            served_type: Some(ServedType::ForYouSimclusters),
            in_network: Some(false),
            nsfw_author: Some(true),
            has_media: Some(true),
            ..Default::default()
        }];

        let result = filter.filter(&query, candidates);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.kept.len(), 0);
    }

    #[test]
    fn removes_oon_simclusters_when_author_nsfw_and_has_nsfw_label() {
        let filter = OONNsfwSimclustersFilter;
        let query = ScoredPostsQuery::default();
        let candidates = vec![PostCandidate {
            served_type: Some(ServedType::ForYouSimclusters),
            in_network: Some(false),
            nsfw_author: Some(true),
            has_media: Some(false),
            safety_labels: vec![SafetyLabelInfo {
                label_type: SafetyLabelType::NSFW_TEXT,
            }],
            ..Default::default()
        }];

        let result = filter.filter(&query, candidates);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.kept.len(), 0);
    }

    #[test]
    fn keeps_compliant_post_from_nsfw_author_without_media() {
        let filter = OONNsfwSimclustersFilter;
        let query = ScoredPostsQuery::default();
        let candidates = vec![PostCandidate {
            served_type: Some(ServedType::ForYouSimclusters),
            in_network: Some(false),
            nsfw_author: Some(true),
            has_media: Some(false),
            safety_labels: vec![],
            ..Default::default()
        }];

        let result = filter.filter(&query, candidates);
        assert_eq!(result.removed.len(), 0);
        assert_eq!(result.kept.len(), 1);
    }

    #[test]
    fn keeps_compliant_post_when_not_nsfw_author() {
        let filter = OONNsfwSimclustersFilter;
        let query = ScoredPostsQuery::default();
        let candidates = vec![PostCandidate {
            served_type: Some(ServedType::ForYouSimclusters),
            in_network: Some(false),
            nsfw_author: Some(false),
            has_media: Some(true),
            ..Default::default()
        }];

        let result = filter.filter(&query, candidates);
        assert_eq!(result.removed.len(), 0);
        assert_eq!(result.kept.len(), 1);
    }
}
