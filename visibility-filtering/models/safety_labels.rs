pub use xai_x_thrift::tweet_safety_label::SafetyLabelType;

use std::collections::{HashMap, HashSet};
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap {
    types: HashSet<SafetyLabelType>,
    /// Per-type viewer scope from proto. Missing or empty = every viewer.
    users: HashMap<SafetyLabelType, Vec<u64>>,
}

impl SafetyLabelMap {
    #[cfg(test)]
    pub fn new(label_types: HashSet<SafetyLabelType>) -> Self {
        Self {
            types: label_types,
            users: HashMap::new(),
        }
    }

    pub fn with_user_scope(mut self, label: SafetyLabelType, users: Vec<u64>) -> Self {
        self.types.insert(label);
        self.users.insert(label, users);
        self
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        let mut types = HashSet::with_capacity(proto.labels.len());
        let mut users = HashMap::with_capacity(proto.labels.len());
        for (label_type, label) in &proto.labels {
            let lt = SafetyLabelType(*label_type);
            types.insert(lt);
            users.insert(lt, label.applicable_users.clone());
        }
        Self { types, users }
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.types.contains(&label_type)
    }

    /// Type is present and in scope for this viewer.
    /// Empty applicable_users is everyone (current behavior).
    /// A scoped label does not apply when the viewer is missing or unlisted.
    /// This is not country scope (PR 110) and not expiry (PR 106).
    pub fn applies(&self, label_type: SafetyLabelType, viewer_id: Option<u64>) -> bool {
        if !self.types.contains(&label_type) {
            return false;
        }
        let Some(ids) = self.users.get(&label_type) else {
            return true;
        };
        if ids.is_empty() {
            return true;
        }
        let Some(uid) = viewer_id else {
            return false;
        };
        ids.contains(&uid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dna() -> SafetyLabelType {
        SafetyLabelType::DO_NOT_AMPLIFY
    }

    fn proto_with_users(users: Vec<u64>) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                dna().0,
                vf_pb::SafetyLabel {
                    score: None,
                    applicable_users: users,
                    holdback_experiment: None,
                    source: None,
                    created_at_msec: None,
                    expires_at_msec: None,
                    applicable_countries: Vec::new(),
                    safety_label_source: None,
                },
            )]),
        }
    }

    #[test]
    fn empty_users_applies_to_every_viewer() {
        let map = SafetyLabelMap::from_proto_label_types(&proto_with_users(vec![]));
        assert!(map.has_label(dna()));
        assert!(map.applies(dna(), Some(1)));
        assert!(map.applies(dna(), Some(99)));
        assert!(map.applies(dna(), None));
    }

    #[test]
    fn scoped_label_applies_only_to_listed_viewer() {
        let map = SafetyLabelMap::from_proto_label_types(&proto_with_users(vec![42]));
        assert!(map.has_label(dna()));
        assert!(map.applies(dna(), Some(42)));
        assert!(!map.applies(dna(), Some(99)));
        assert!(!map.applies(dna(), None));
    }
}
