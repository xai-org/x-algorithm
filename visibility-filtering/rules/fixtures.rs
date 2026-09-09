use crate::models::{
    AuthorFeatures, HydratedTweetCandidate, NsfwFeature, SafetyLabelMap, SafetyLabelType,
    TweetFeatures, UserLabelSet, VfAction, Viewer, ViewerAuthorRelationship, ViewerFeatures,
};
use crate::rules::rule_spec::RuleSpec;
use crate::rules::test_context;
use std::collections::{HashMap, HashSet};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::user_labels::LabelValue;

const TWEET_ID: u64 = 1;
const AUTHOR_ID: u64 = 100;
pub(crate) const VIEWER_ID: u64 = 999;

pub(super) fn assert_drops(
    spec: &RuleSpec,
    viewer: &ViewerFeatures,
    candidate: &HydratedTweetCandidate,
    expected: &FilteredReason,
) {
    let action = spec.evaluate(&test_context(viewer, candidate));
    assert!(
        matches!(&action, VfAction::Drop(reason) if reason == expected),
        "{} should drop with {expected:?}, got {action:?}",
        spec.name()
    );
}

pub(super) fn assert_allows(
    spec: &RuleSpec,
    viewer: &ViewerFeatures,
    candidate: &HydratedTweetCandidate,
) {
    let action = spec.evaluate(&test_context(viewer, candidate));
    assert!(
        matches!(action, VfAction::Allow),
        "{} should allow, got {action:?}",
        spec.name()
    );
}

pub(crate) fn viewer(id: u64) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn(id),
        ..Default::default()
    }
}

pub(crate) fn author_viewer() -> ViewerFeatures {
    viewer(AUTHOR_ID)
}

pub(crate) fn logged_out_viewer() -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedOut,
        ..Default::default()
    }
}

pub(crate) fn sensitive_opt_in_viewer() -> ViewerFeatures {
    ViewerFeatures {
        allows_sensitive_media: true,
        ..viewer(VIEWER_ID)
    }
}

pub(crate) fn candidate() -> CandidateBuilder {
    CandidateBuilder {
        candidate: HydratedTweetCandidate {
            tweet_id: TWEET_ID,
            author_id: AUTHOR_ID,
            ..Default::default()
        },
        labels: HashSet::new(),
        label_countries: HashMap::new(),
        user_labels: HashSet::new(),
    }
}

pub(crate) struct CandidateBuilder {
    candidate: HydratedTweetCandidate,
    labels: HashSet<SafetyLabelType>,
    label_countries: HashMap<SafetyLabelType, Vec<String>>,
    user_labels: HashSet<LabelValue>,
}

impl CandidateBuilder {
    pub(crate) fn tweet_id(mut self, id: u64) -> Self {
        self.candidate.tweet_id = id;
        self
    }

    pub(crate) fn author_id(mut self, id: u64) -> Self {
        self.candidate.author_id = id;
        self
    }

    pub(crate) fn with_label(mut self, label: SafetyLabelType) -> Self {
        self.labels.insert(label);
        self
    }

    pub(crate) fn with_label_countries(
        mut self,
        label: SafetyLabelType,
        countries: Vec<String>,
    ) -> Self {
        self.labels.insert(label);
        self.label_countries.insert(label, countries);
        self
    }

    pub(crate) fn with_author_user_label(mut self, label: LabelValue) -> Self {
        self.user_labels.insert(label);
        self
    }

    pub(crate) fn with_tweet_features(mut self, features: TweetFeatures) -> Self {
        self.candidate.tweet_features = features;
        self
    }

    pub(crate) fn with_author_features(mut self, features: AuthorFeatures) -> Self {
        self.candidate.author_features = features;
        self
    }

    pub(crate) fn with_relationship(mut self, relationship: ViewerAuthorRelationship) -> Self {
        self.candidate.relationship = relationship;
        self
    }

    pub(crate) fn followed(mut self) -> Self {
        self.candidate.relationship.viewer_follows_author = true;
        self
    }

    pub(crate) fn with_media(mut self) -> Self {
        self.candidate.tweet_features.media.has_media = true;
        self
    }

    pub(crate) fn retweet_of(mut self, source_tweet_id: u64) -> Self {
        self.candidate.tweet_features.core.source_tweet_id = Some(source_tweet_id);
        self
    }

    pub(crate) fn build(self) -> HydratedTweetCandidate {
        let mut candidate = self.candidate;
        if !self.labels.is_empty() || !self.label_countries.is_empty() {
            let mut map = SafetyLabelMap::new(self.labels);
            for (label, countries) in self.label_countries {
                map = map.with_country_scope(label, countries);
            }
            candidate.safety_labels = map;
        }
        if !self.user_labels.is_empty() {
            candidate.author_features.user_labels = UserLabelSet::new(self.user_labels);
        }
        candidate
    }
}

pub(super) fn nsfw_flag_media_candidates() -> [HydratedTweetCandidate; 4] {
    let author_user = candidate()
        .with_media()
        .with_author_features(AuthorFeatures {
            is_nsfw_user: true,
            ..Default::default()
        })
        .build();
    let tweet_user = candidate()
        .with_tweet_features(TweetFeatures {
            nsfw: NsfwFeature {
                user: true,
                admin: false,
            },
            ..Default::default()
        })
        .with_media()
        .build();
    let tweet_admin = candidate()
        .with_tweet_features(TweetFeatures {
            nsfw: NsfwFeature {
                user: false,
                admin: true,
            },
            ..Default::default()
        })
        .with_media()
        .build();
    let both = candidate()
        .with_author_features(AuthorFeatures {
            is_nsfw_admin: true,
            ..Default::default()
        })
        .with_tweet_features(TweetFeatures {
            nsfw: NsfwFeature {
                user: true,
                admin: false,
            },
            ..Default::default()
        })
        .with_media()
        .build();
    [author_user, tweet_user, tweet_admin, both]
}
