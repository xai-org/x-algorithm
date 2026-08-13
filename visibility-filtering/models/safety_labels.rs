pub use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};

use std::collections::{HashMap, HashSet};
use xai_visibility_filtering_proto as vf_pb;

/// Removes labels whose absolute expiry has been reached.
///
/// This is intentionally applied at the serving boundary as well as relying on
/// upstream cleanup: a cached label map can outlive an individual label's TTL.
pub(crate) fn retain_unexpired_proto_labels(
    labels: &mut vf_pb::SafetyLabelMap,
    now_msec: i64,
) -> usize {
    let before = labels.labels.len();
    labels.labels.retain(|_, label| {
        label
            .expires_at_msec
            .map_or(true, |expires_at| expires_at > now_msec)
    });
    before - labels.labels.len()
}

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap {
    pub labels: HashMap<SafetyLabelType, SafetyLabel>,
    label_types: HashSet<SafetyLabelType>,
}

impl SafetyLabelMap {
    pub fn new(labels: HashMap<SafetyLabelType, SafetyLabel>) -> Self {
        let label_types = labels.keys().copied().collect();
        Self {
            labels,
            label_types,
        }
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        let labels = proto
            .labels
            .keys()
            .map(|label_type| (SafetyLabelType(*label_type), SafetyLabel::default()))
            .collect();
        Self::new(labels)
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.label_types.contains(&label_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proto_map(expiries: &[(i32, Option<i64>)]) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: expiries
                .iter()
                .map(|(label_type, expires_at_msec)| {
                    (
                        *label_type,
                        vf_pb::SafetyLabel {
                            expires_at_msec: *expires_at_msec,
                            ..Default::default()
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn keeps_labels_without_expiry_and_with_future_expiry() {
        let mut labels = proto_map(&[(1, None), (2, Some(101))]);

        let removed = retain_unexpired_proto_labels(&mut labels, 100);

        assert_eq!(removed, 0);
        assert_eq!(labels.labels.len(), 2);
    }

    #[test]
    fn removes_labels_expiring_at_or_before_now() {
        let mut labels = proto_map(&[(1, Some(99)), (2, Some(100)), (3, Some(101)), (4, None)]);

        let removed = retain_unexpired_proto_labels(&mut labels, 100);

        assert_eq!(removed, 2);
        assert!(!labels.labels.contains_key(&1));
        assert!(!labels.labels.contains_key(&2));
        assert!(labels.labels.contains_key(&3));
        assert!(labels.labels.contains_key(&4));
    }
}
