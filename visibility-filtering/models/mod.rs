pub mod author;
pub mod exclusive_content;
pub mod relationship;
pub mod safety_labels;
pub mod tweet;
pub mod viewer;
pub mod high_risk_tags;

pub use author::{AuthorFeatures, UserLabelSet};
pub use exclusive_content::ExclusiveContentFeatures;
pub use relationship::ViewerAuthorRelationship;
pub use safety_labels::{SafetyLabel, SafetyLabelMap, SafetyLabelType};
pub use tweet::{CoreFeature, MediaFeature, NsfwFeature, TakedownFeature, TweetFeatures};
pub use viewer::{Viewer, ViewerAge, ViewerFeatures, ADULT_AGE_YEARS};

use std::collections::HashMap;
use xai_core_entities::entities::PureCoreData;
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::user_labels::LabelValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TweetId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AuthorId(u64);

impl AuthorId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawCandidate {
    pub tweet_id: TweetId,
    pub request_author_id: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct TweetCandidateInput {
    pub tweet_id: TweetId,
    pub author_id: AuthorId,
}

pub fn resolve_candidate(
    raw: &RawCandidate,
    core: &HashMap<TweetId, PureCoreData>,
) -> Option<TweetCandidateInput> {
    let author = raw
        .request_author_id
        .or_else(|| core.get(&raw.tweet_id).map(|c| c.author_id))?;
    Some(TweetCandidateInput {
        tweet_id: raw.tweet_id,
        author_id: AuthorId(author),
    })
}

pub fn resolve_candidates(
    raw: &[RawCandidate],
    core: &HashMap<TweetId, PureCoreData>,
) -> Vec<TweetCandidateInput> {
    raw.iter()
        .filter_map(|c| resolve_candidate(c, core))
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct HydratedTweetCandidate {
    pub tweet_id: u64,
    pub author_id: u64,
    pub tweet_features: TweetFeatures,
    pub author_features: AuthorFeatures,
    pub safety_labels: SafetyLabelMap,
    pub relationship: ViewerAuthorRelationship,
    pub exclusive_content: Option<ExclusiveContentFeatures>,
}

impl HydratedTweetCandidate {
    pub fn is_author_viewer(&self, viewer: Viewer) -> bool {
        matches!(viewer, Viewer::LoggedIn(viewer_id) if viewer_id == self.author_id)
    }

    pub fn has_safety_label(&self, label: SafetyLabelType) -> bool {
        self.safety_labels.has_label(label)
    }

    pub fn viewer_follows_author(&self) -> bool {
        self.relationship.viewer_follows_author
    }

    pub fn has_media(&self) -> bool {
        self.tweet_features.media.has_media
    }

    pub fn is_retweet(&self) -> bool {
        self.tweet_features.core.source_tweet_id.is_some()
    }

    pub fn is_nsfw_flagged(&self) -> bool {
        self.author_features.is_nsfw_user
            || self.author_features.is_nsfw_admin
            || self.tweet_features.nsfw.user
            || self.tweet_features.nsfw.admin
    }

    pub fn has_dmca_media(&self) -> bool {
        self.tweet_features.media.has_dmca_media
    }

    pub fn is_nullcast(&self) -> bool {
        self.tweet_features.is_nullcast
    }

    pub fn is_community_tweet(&self) -> bool {
        self.tweet_features.is_community_tweet
    }

    pub fn author_has_user_label(&self, label: LabelValue) -> bool {
        self.author_features.user_labels.has_label(label)
    }

    pub fn is_stale_tweet(&self) -> bool {
        let Some(ec) = &self.tweet_features.edit_control else {
            return false;
        };
        let edit_tweet_ids = match ec {
            xai_core_entities::entities::EditControl::Initial(initial) => &initial.edit_tweet_ids,
            xai_core_entities::entities::EditControl::Edit(edit) => {
                match &edit.edit_control_initial {
                    Some(initial) => &initial.edit_tweet_ids,
                    None => return false,
                }
            }
        };
        edit_tweet_ids
            .last()
            .is_some_and(|&last| last != self.tweet_id)
    }
}

pub fn assemble(
    candidate: &TweetCandidateInput,
    tweet_features: TweetFeatures,
    author_features: AuthorFeatures,
    safety_labels: SafetyLabelMap,
    relationship: ViewerAuthorRelationship,
    exclusive_content: Option<ExclusiveContentFeatures>,
) -> HydratedTweetCandidate {
    HydratedTweetCandidate {
        tweet_id: candidate.tweet_id.0,
        author_id: candidate.author_id.get(),
        tweet_features,
        author_features,
        safety_labels,
        relationship,
        exclusive_content,
    }
}

#[derive(Clone, Debug)]
pub enum VfAction {
    Allow,
    Drop(FilteredReason),
    Interstitial(FilteredReason),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_request_author_id_without_touching_core() {
        let core: HashMap<TweetId, PureCoreData> = HashMap::new();
        let raw = RawCandidate {
            tweet_id: TweetId(10),
            request_author_id: Some(100),
        };
        let resolved = resolve_candidate(&raw, &core).unwrap();
        assert_eq!(resolved.author_id.get(), 100);
    }

    #[test]
    fn resolve_falls_back_to_core_when_request_author_missing() {
        let core = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 200,
                ..Default::default()
            },
        )]);
        let raw = RawCandidate {
            tweet_id: TweetId(10),
            request_author_id: None,
        };
        let resolved = resolve_candidate(&raw, &core).unwrap();
        assert_eq!(resolved.author_id.get(), 200);
    }

    #[test]
    fn resolve_returns_none_when_unresolvable() {
        let core: HashMap<TweetId, PureCoreData> = HashMap::new();
        let raw = RawCandidate {
            tweet_id: TweetId(10),
            request_author_id: None,
        };
        assert!(resolve_candidate(&raw, &core).is_none());
    }

    #[test]
    fn resolve_candidates_drops_unresolved_preserving_order() {
        let core = HashMap::from([(
            TweetId(2),
            PureCoreData {
                author_id: 20,
                ..Default::default()
            },
        )]);
        let raw = vec![
            RawCandidate {
                tweet_id: TweetId(1),
                request_author_id: None,
            },
            RawCandidate {
                tweet_id: TweetId(2),
                request_author_id: None,
            },
        ];
        let resolved = resolve_candidates(&raw, &core);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].tweet_id, TweetId(2));
        assert_eq!(resolved[0].author_id.get(), 20);
    }

    fn candidate() -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 10,
            author_id: 100,
            ..Default::default()
        }
    }

    #[test]
    fn is_stale_tweet_when_superseded() {
        use xai_core_entities::entities::{EditControl, EditControlInitial};
        let mut c = candidate();
        c.tweet_features.edit_control = Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![10, 20],
            ..Default::default()
        }));
        assert!(c.is_stale_tweet());
    }

    #[test]
    fn is_stale_tweet_when_current() {
        use xai_core_entities::entities::{EditControl, EditControlInitial};
        let mut c = candidate();
        c.tweet_features.edit_control = Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![10],
            ..Default::default()
        }));
        assert!(!c.is_stale_tweet());
    }

    #[test]
    fn is_stale_tweet_when_no_edit_control() {
        assert!(!candidate().is_stale_tweet());
    }

    #[test]
    fn is_stale_tweet_edit_variant_superseded() {
        use xai_core_entities::entities::{EditControl, EditControlEdit, EditControlInitial};
        let mut c = HydratedTweetCandidate {
            tweet_id: 20,
            author_id: 100,
            ..Default::default()
        };
        c.tweet_features.edit_control = Some(EditControl::Edit(EditControlEdit {
            initial_tweet_id: 10,
            edit_control_initial: Some(EditControlInitial {
                edit_tweet_ids: vec![10, 20, 30],
                ..Default::default()
            }),
        }));
        assert!(c.is_stale_tweet());
    }

    #[test]
    fn is_stale_tweet_edit_variant_no_initial() {
        use xai_core_entities::entities::{EditControl, EditControlEdit};
        let mut c = HydratedTweetCandidate {
            tweet_id: 20,
            author_id: 100,
            ..Default::default()
        };
        c.tweet_features.edit_control = Some(EditControl::Edit(EditControlEdit {
            initial_tweet_id: 10,
            edit_control_initial: None,
        }));
        assert!(!c.is_stale_tweet());
    }
}
