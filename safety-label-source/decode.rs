use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tracing::debug;
use xai_safety_label_store::mval_safety_label_map::decode_mval;
use xai_safety_label_store::types::{decode_lkey_bytes, SafetyLabelMap};
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::servo_repo::{CachedValue, CachedValueStatus};
use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelSource};

#[derive(Debug, PartialEq)]
pub(crate) enum CachedLabels {
    Found(vf_pb::SafetyLabelMap),
    NotFound,
    Miss,
}

#[derive(Debug, Clone)]
pub(crate) struct RawSafetyLabel {
    pub(crate) lkey: [u8; 4],
    pub(crate) mval: Vec<u8>,
}

pub(crate) fn cached_value(bytes: &[u8]) -> Option<CachedLabels> {
    contain_decode_panic(|| cached_value_inner(bytes)).flatten()
}

fn cached_value_inner(bytes: &[u8]) -> Option<CachedLabels> {
    let cv: CachedValue = xai_x_thrift::deserialize_binary(bytes).ok()?;
    match cv.status {
        Some(s) if s == CachedValueStatus::NOT_FOUND || s == CachedValueStatus::DELETED => {
            Some(CachedLabels::NotFound)
        }
        Some(s) if s == CachedValueStatus::FOUND => cv
            .value
            .as_deref()
            .and_then(decode_mval)
            .map(|labels| CachedLabels::Found(label_map_to_proto(labels))),
        _ => Some(CachedLabels::Miss),
    }
}

pub(crate) fn raw_labels(items: &[RawSafetyLabel]) -> Option<vf_pb::SafetyLabelMap> {
    contain_decode_panic(|| raw_labels_inner(items)).map(label_map_to_proto)
}

fn contain_decode_panic<T>(decode: impl FnOnce() -> T) -> Option<T> {
    catch_unwind(AssertUnwindSafe(decode)).ok()
}

fn raw_labels_inner(items: &[RawSafetyLabel]) -> SafetyLabelMap {
    let mut map = HashMap::with_capacity(items.len());
    for item in items {
        let label_type = decode_lkey_bytes(item.lkey);
        let label = match xai_x_thrift::deserialize_mval(&item.mval) {
            Ok(l) => l,
            Err(e) => {
                debug!(label_type = ?label_type, error = %e, "skipping label value with corrupt mval from manhattan");
                SafetyLabel::default()
            }
        };
        map.insert(label_type, label);
    }
    map
}

fn label_to_proto(label: SafetyLabel) -> vf_pb::SafetyLabel {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "SafetyLabelSource is a generated Thrift union; vf_pb::SafetyLabel carries only the BotMaker, tool and Grok annotation sources, so it omits every other source, including any the IDL adds"
    )]
    let safety_label_source = label.safety_label_source.and_then(|src| match src {
        SafetyLabelSource::BotMakerAction(a) => {
            Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                vf_pb::BotmakerAction { rule_id: a.rule_id },
            ))
        }
        SafetyLabelSource::ToolAction(a) => Some(
            vf_pb::safety_label::SafetyLabelSource::ToolAction(vf_pb::ToolAction {
                agent_tool: i32::from(a.agent_tool),
                actor_ldap: a.actor_ldap,
            }),
        ),
        SafetyLabelSource::GrokAnnotationAction(a) => Some(
            vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(
                vf_pb::GrokAnnotationAction {
                    source: i32::from(a.grok_annotation_source),
                },
            ),
        ),
        _ => None,
    });

    vf_pb::SafetyLabel {
        score: label.score.map(|f| f.into_inner()),
        applicable_users: label.applicable_users.map_or_else(Vec::new, |users| {
            users.into_iter().map(|u| u.user_id).collect()
        }),
        holdback_experiment: label.holdback_experiment,
        source: label.source,
        created_at_msec: label.created_at_msec,
        expires_at_msec: label.expires_at_msec,
        applicable_countries: label.applicable_countries.unwrap_or_default(),
        safety_label_source,
    }
}

fn label_map_to_proto(labels: SafetyLabelMap) -> vf_pb::SafetyLabelMap {
    vf_pb::SafetyLabelMap {
        labels: labels
            .into_iter()
            .map(|(lt, label)| (i32::from(lt), label_to_proto(label)))
            .collect(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use thrift::protocol::{TBinaryOutputProtocol, TSerializable};
    use xai_safety_label_store::types::encode_lkey;
    use xai_x_thrift::tweet_safety_label::SafetyLabelType;

    fn cached_value_blob(status: CachedValueStatus, inner: Option<&[u8]>) -> Vec<u8> {
        let cv = CachedValue::new(
            inner.map(|b| b.to_vec()),
            Some(status),
            None::<i64>,
            None::<i64>,
            None::<i64>,
            None::<i64>,
            None::<i16>,
        );
        let mut buf = Vec::new();
        cv.write_to_out_protocol(&mut TBinaryOutputProtocol::new(&mut buf, false))
            .unwrap();
        buf
    }

    pub(crate) fn cached_value_not_found() -> Vec<u8> {
        crate::testing::cached_value_not_found().unwrap()
    }

    pub(crate) fn cached_value_found_mval() -> Vec<u8> {
        let mval_blob: &[u8] = &[
            0x0c, 0x00, 0x03, 0x0c, 0x5f, 0xff, 0x0d, 0x69, 0x14, 0x0c, 0x0c, 0x00, 0x00, 0x00,
            0x01, 0x01, 0xe7, 0x7c, 0x00, 0x00,
        ];
        cached_value_blob(CachedValueStatus::FOUND, Some(mval_blob))
    }

    fn mval(label: &SafetyLabel) -> Vec<u8> {
        crate::testing::mval(label).unwrap()
    }

    pub(crate) fn minimal_mval() -> Vec<u8> {
        mval(&SafetyLabel::default())
    }

    pub(crate) fn mval_with_score(score: f64) -> Vec<u8> {
        mval(&SafetyLabel {
            score: Some(score.into()),
            ..Default::default()
        })
    }

    pub(crate) fn raw_label(lt: SafetyLabelType, mval: Vec<u8>) -> RawSafetyLabel {
        RawSafetyLabel {
            lkey: encode_lkey(lt),
            mval,
        }
    }

    #[test]
    fn cached_value_status_selects_cache_disposition() {
        let found = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                i32::from(SafetyLabelType::NSFA_HIGH_PRECISION),
                vf_pb::SafetyLabel::default(),
            )]),
        };
        for (blob, expected) in [
            (cached_value_found_mval(), Some(CachedLabels::Found(found))),
            (cached_value_not_found(), Some(CachedLabels::NotFound)),
            (
                cached_value_blob(CachedValueStatus::DELETED, None),
                Some(CachedLabels::NotFound),
            ),
            (
                cached_value_blob(CachedValueStatus::DO_NOT_CACHE, None),
                Some(CachedLabels::Miss),
            ),
            (
                cached_value_blob(CachedValueStatus::FOUND, Some(&[0xff; 8])),
                None,
            ),
            (cached_value_blob(CachedValueStatus::FOUND, None), None),
        ] {
            assert_eq!(cached_value(&blob), expected);
        }
    }

    #[test]
    fn raw_labels_preserves_label_type_on_corrupt_mval() {
        let items = vec![
            raw_label(SafetyLabelType::SPAM, minimal_mval()),
            raw_label(SafetyLabelType::BOUNCE, vec![0xDE, 0xAD]),
            raw_label(SafetyLabelType::NSFW_HIGH_PRECISION, mval_with_score(0.9)),
        ];
        let map = raw_labels(&items).unwrap();
        assert_eq!(map.labels.len(), 3);
        assert!(map.labels.contains_key(&i32::from(SafetyLabelType::SPAM)));
        assert!(map.labels.contains_key(&i32::from(SafetyLabelType::BOUNCE)));
        assert!(map
            .labels
            .contains_key(&i32::from(SafetyLabelType::NSFW_HIGH_PRECISION)));
        assert_eq!(
            map.labels[&i32::from(SafetyLabelType::BOUNCE)],
            vf_pb::SafetyLabel::default()
        );
    }
}
