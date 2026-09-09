pub use xai_x_thrift::tweet_safety_label::SafetyLabelType;

use std::collections::{HashMap, HashSet};
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap {
    types: HashSet<SafetyLabelType>,
    /// Per-type country scope from proto. Missing or empty = worldwide.
    countries: HashMap<SafetyLabelType, Vec<String>>,
}

impl SafetyLabelMap {
    #[cfg(test)]
    pub fn new(label_types: HashSet<SafetyLabelType>) -> Self {
        Self {
            types: label_types,
            countries: HashMap::new(),
        }
    }

    pub fn with_country_scope(mut self, label: SafetyLabelType, countries: Vec<String>) -> Self {
        self.types.insert(label);
        self.countries.insert(label, countries);
        self
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        let mut types = HashSet::with_capacity(proto.labels.len());
        let mut countries = HashMap::with_capacity(proto.labels.len());
        for (label_type, label) in &proto.labels {
            let lt = SafetyLabelType(*label_type);
            types.insert(lt);
            countries.insert(lt, label.applicable_countries.clone());
        }
        Self { types, countries }
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.types.contains(&label_type)
    }

    /// Type is present and in scope for the viewer country.
    /// Empty applicable_countries is worldwide (current behavior).
    /// A scoped label does not apply when the viewer country is missing
    /// or outside the list. This is not expiry filtering (PR 106).
    pub fn applies(&self, label_type: SafetyLabelType, viewer_country: Option<&str>) -> bool {
        if !self.types.contains(&label_type) {
            return false;
        }
        let Some(cs) = self.countries.get(&label_type) else {
            return true;
        };
        if cs.is_empty() {
            return true;
        }
        let Some(country) = viewer_country else {
            return false;
        };
        cs.iter().any(|c| c.eq_ignore_ascii_case(country))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dna() -> SafetyLabelType {
        SafetyLabelType::DO_NOT_AMPLIFY
    }

    fn proto_with_countries(countries: Vec<String>) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                dna().0,
                vf_pb::SafetyLabel {
                    score: None,
                    applicable_users: Vec::new(),
                    holdback_experiment: None,
                    source: None,
                    created_at_msec: None,
                    expires_at_msec: None,
                    applicable_countries: countries,
                    safety_label_source: None,
                },
            )]),
        }
    }

    #[test]
    fn empty_countries_is_worldwide() {
        let map = SafetyLabelMap::from_proto_label_types(&proto_with_countries(vec![]));
        assert!(map.has_label(dna()));
        assert!(map.applies(dna(), Some("us")));
        assert!(map.applies(dna(), Some("br")));
        assert!(map.applies(dna(), None));
    }

    #[test]
    fn scoped_label_applies_only_in_listed_country() {
        let map = SafetyLabelMap::from_proto_label_types(&proto_with_countries(vec![
            "br".to_string(),
        ]));
        assert!(map.has_label(dna()));
        assert!(map.applies(dna(), Some("br")));
        assert!(map.applies(dna(), Some("BR")));
        assert!(!map.applies(dna(), Some("us")));
        assert!(!map.applies(dna(), None));
    }

    #[test]
    fn scoped_label_does_not_apply_to_unlisted_country() {
        let map = SafetyLabelMap::from_proto_label_types(&proto_with_countries(vec![
            "gb".to_string(),
            "de".to_string(),
        ]));
        assert!(!map.applies(dna(), Some("us")));
        assert!(map.applies(dna(), Some("de")));
    }
}
