use std::collections::BTreeSet;

use xai_abuse_enforcement_types::action_control::{
    ActionControlListResponse, ControlMode, PodView,
};
use xai_abuse_enforcement_types::off_switch::OffSwitchPutRequest;

pub const REASON: &str =
    "action-control rollback: page-managed scope switched off before an image rollback";

pub fn plan(list: &ActionControlListResponse, updated_by: &str) -> Vec<OffSwitchPutRequest> {
    let mut scopes: BTreeSet<(String, Option<String>)> = list
        .records
        .iter()
        .filter(|v| v.record.mode != ControlMode::Rules)
        .map(|v| (v.record.head.clone(), v.record.classifier.clone()))
        .collect();
    for u in &list.unreadable {
        let parts: Vec<&str> = u.lkey.split('/').collect();
        match parts.as_slice() {
            [_entity, "h" | "b" | "l", head, ..] => {
                scopes.insert(((*head).to_owned(), None));
            }
            [_entity, "c" | "cb" | "cl", head, classifier, ..] => {
                scopes.insert(((*head).to_owned(), Some((*classifier).to_owned())));
            }
            _ => {}
        }
    }
    let whole_heads: BTreeSet<String> = scopes
        .iter()
        .filter(|(_, c)| c.is_none())
        .map(|(h, _)| h.clone())
        .collect();
    scopes
        .into_iter()
        .filter(|(h, c)| c.is_none() || !whole_heads.contains(h))
        .map(|(head, classifier)| OffSwitchPutRequest {
            head,
            classifier,
            off: true,
            reason: REASON.to_owned(),
            auto_on_at_ms: None,
            updated_by: Some(updated_by.to_owned()),
        })
        .collect()
}

pub fn switch_lkey(s: &OffSwitchPutRequest) -> String {
    crate::off_switch::lkey_for(&s.head, s.classifier.as_deref())
}

pub fn verify(planned: &[OffSwitchPutRequest], pods: &[PodView]) -> Vec<String> {
    let mut missing = Vec::new();
    for pod in pods {
        let active: BTreeSet<&str> = pod.off_switch.active.iter().map(String::as_str).collect();
        for s in planned {
            let head_key = crate::off_switch::lkey_for(&s.head, None);
            if !active.contains(switch_lkey(s).as_str()) && !active.contains(head_key.as_str()) {
                missing.push(format!("{}: {} not active", pod.pod, switch_lkey(s)));
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_control::test_support::record;
    use xai_abuse_enforcement_types::action_control::{
        ActionControlView, AllowedActions, CseLists, PageAction, PodActionControlView,
        PodOffSwitchView, SafetyChecksByEntity, SafetyChecksStatus, UnreadableRecord,
    };

    fn list(records: Vec<ActionControlView>, unreadable: Vec<&str>) -> ActionControlListResponse {
        let ok = SafetyChecksStatus {
            ok: true,
            problems: vec![],
            checks: vec![],
        };
        ActionControlListResponse {
            feature_version: 1,
            env: "prod".into(),
            pod: "p".into(),
            enabled: true,
            ignore_store: false,
            source: "store".into(),
            error: None,
            snapshot_age_ms: 1,
            stale: false,
            stale_after_ms: 60_000,
            records,
            unreadable: unreadable
                .into_iter()
                .map(|l| UnreadableRecord {
                    lkey: l.into(),
                    error: "x".into(),
                    version: 1,
                })
                .collect(),
            seen: vec![],
            seen_error: None,
            seen_rows: vec![],
            safety_checks: SafetyChecksByEntity {
                user: ok.clone(),
                post: ok,
            },
            allowed_actions: AllowedActions {
                user: vec![],
                post: vec![],
            },
            cse: CseLists {
                heads: vec![],
                labels: vec![],
            },
            confidence: None,
            follower_skip: None,
        }
    }

    fn view(head: &str, c: Option<&str>, b: Option<&str>, mode: ControlMode) -> ActionControlView {
        let actions = if mode == ControlMode::Live {
            vec![PageAction::SUSPEND]
        } else {
            vec![]
        };
        let r = record(head, c, b, mode, actions);
        ActionControlView {
            scope: r.scope().into(),
            lkey: crate::action_control::record_lkey(&r),
            effective_mode: mode,
            forced_recording_reason: None,
            record: r,
        }
    }

    fn pod(name: &str, active: &[&str]) -> PodView {
        PodView {
            pod: name.into(),
            env: "prod".into(),
            image: "x".into(),
            action_control: PodActionControlView {
                loaded: true,
                snapshot_age_ms: 1,
                stale: false,
                records: 0,
                page_managed: vec![],
            },
            off_switch: PodOffSwitchView {
                snapshot_age_ms: 1,
                active: active.iter().map(|s| (*s).to_owned()).collect(),
            },
        }
    }

    #[test]
    fn plan_switches_off_every_page_managed_scope_and_nothing_else() {
        let l = list(
            vec![
                view("Rec", None, None, ControlMode::Recording),
                view("Live", Some("c1"), None, ControlMode::Live),
                view("Band", None, Some("b1"), ControlMode::Live),
                view("Both", None, None, ControlMode::Recording),
                view("Both", Some("c2"), None, ControlMode::Live),
                view("RulesOnly", None, None, ControlMode::Rules),
            ],
            vec!["user/c/Broken/c9", "post/b/BrokenBand/x"],
        );
        let p = plan(&l, "operator@example.com");
        let keys: Vec<String> = p.iter().map(switch_lkey).collect();
        assert_eq!(
            keys,
            vec![
                "h/Band",
                "h/Both",
                "p/Broken/c9",
                "h/BrokenBand",
                "p/Live/c1",
                "h/Rec"
            ]
        );
        assert!(p.iter().all(|s| s.off && s.reason == REASON));
        assert_eq!(p[0].updated_by.as_deref(), Some("operator@example.com"));
    }

    #[test]
    fn plan_switches_off_heads_with_rows() {
        let mut level = view("Level", None, None, ControlMode::Live);
        level.record.confidence = Some(crate::confidence::LevelRow::High);
        let l = list(
            vec![level, view("Defaults", None, None, ControlMode::Rows)],
            vec!["user/l/BrokenLevel/high", "post/cl/BrokenClf/c3/mid"],
        );
        let keys: Vec<String> = plan(&l, "operator@example.com")
            .iter()
            .map(switch_lkey)
            .collect();
        assert_eq!(
            keys,
            vec!["p/BrokenClf/c3", "h/BrokenLevel", "h/Defaults", "h/Level"]
        );
    }

    #[test]
    fn verify_needs_every_pod_to_show_every_switch() {
        let l = list(
            vec![
                view("Rec", None, None, ControlMode::Recording),
                view("Live", Some("c1"), None, ControlMode::Live),
            ],
            vec![],
        );
        let p = plan(&l, "w");
        assert!(verify(
            &p,
            &[
                pod("a", &["h/Rec", "p/Live/c1"]),
                pod("b", &["h/Rec", "h/Live"])
            ]
        )
        .is_empty());
        let missing = verify(
            &p,
            &[pod("a", &["h/Rec", "p/Live/c1"]), pod("b", &["h/Rec"])],
        );
        assert_eq!(missing, vec!["b: p/Live/c1 not active"]);
    }
}
