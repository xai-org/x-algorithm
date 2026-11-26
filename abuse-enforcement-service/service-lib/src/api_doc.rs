use serde::{Deserialize, Serialize};
use utoipa::{
    openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme},
    Modify, OpenApi, ToSchema,
};

use crate::service;

pub struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "BearerAuth",
            SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiError {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DryRunStatus {
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RateLimitStats {
    pub used: u64,
    pub cap: u64,
    pub remaining: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct RulesResponse {
    pub user: crate::rules::RuleStatus,
    pub post: crate::rules::RuleStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AllowlistDeleteResponse {
    pub user_id: i64,
    pub removed: bool,
}

#[derive(OpenApi)]
#[openapi(
    modifiers(&SecurityAddon),
    paths(
        service::handle_action,
        service::handle_action_controls_get,
        service::handle_action_controls_put,
        service::handle_action_controls_history,
        service::handle_action_controls_pod,
        service::handle_config,
        service::handle_dry_run,
        service::handle_rate_limit,
        service::handle_rate_limit_increment,
        service::handle_rate_limit_reset,
        service::handle_dedup_get,
        service::handle_dedup_get_entity,
        service::handle_allowlist_list,
        service::handle_allowlist_get,
        service::handle_config_put,
        service::handle_config_patch_field,
        service::handle_rules_get,
        service::handle_allowlist_put,
        service::handle_allowlist_delete,
        service::handle_allowlist_bulk,
        service::handle_allowlist_get_entity,
        service::handle_allowlist_put_entity,
        service::handle_allowlist_delete_entity,
        service::handle_off_switches_get,
        service::handle_off_switches_put,
        service::handle_head_guards_get,
        service::handle_head_guards_put,
        service::handle_backfill_runs_list,
        service::handle_backfill_runs_budget,
        service::handle_backfill_run_get,
        service::handle_backfill_run_start,
        service::handle_backfill_run_change,
        service::handle_backfill_settings_get,
        service::handle_backfill_settings_put,
    ),
    components(schemas(
        xai_abuse_enforcement_types::action_control::ActionControlRecord,
        xai_abuse_enforcement_types::action_control::ActionControlView,
        xai_abuse_enforcement_types::action_control::ActionControlPutRequest,
        xai_abuse_enforcement_types::action_control::ActionControlSeen,
        xai_abuse_enforcement_types::action_control::ActionControlListResponse,
        xai_abuse_enforcement_types::action_control::ActionControlHistoryEntry,
        xai_abuse_enforcement_types::action_control::ActionControlHistoryResponse,
        xai_abuse_enforcement_types::action_control::PodView,
        xai_abuse_enforcement_types::action_control::PageAction,
        xai_abuse_enforcement_types::action_control::ControlMode,
        xai_abuse_enforcement_types::action_control::ControlEntity,
        xai_abuse_enforcement_types::action_control::ConfidenceLevel,
        xai_abuse_enforcement_types::action_control::ConfidenceEvidence,
        xai_abuse_enforcement_types::action_control::EvidenceOverride,
        xai_abuse_enforcement_types::action_control::ConfidenceView,
        xai_abuse_enforcement_types::action_control::LevelRow,
        xai_abuse_enforcement_types::action_control::RowView,
        xai_abuse_enforcement_types::action_control::RowsRequest,
        xai_abuse_enforcement_types::action_control::RowSetting,
        xai_abuse_enforcement_types::action_control::ActionControlSeenRow,
        xai_abuse_enforcement_types::action_control::FollowerSkipView,
        xai_abuse_enforcement_types::allowlist::AllowlistEntry,
        xai_abuse_enforcement_types::allowlist::AllowlistRecord,
        xai_abuse_enforcement_types::allowlist::AllowlistAddRequest,
        xai_abuse_enforcement_types::allowlist::AllowlistBulkRequest,
        xai_abuse_enforcement_types::allowlist::AllowlistBulkEntry,
        xai_abuse_enforcement_types::allowlist::BulkResponse,
        xai_abuse_enforcement_types::allowlist::BulkRowResult,
        xai_abuse_enforcement_types::config::ConfigPatchRequest,
        xai_abuse_enforcement_types::config::ConfigPendingReview,
        xai_abuse_enforcement_types::off_switch::OffSwitchRecord,
        xai_abuse_enforcement_types::off_switch::PreviousOff,
        xai_abuse_enforcement_types::off_switch::OffSwitchPutRequest,
        xai_abuse_enforcement_types::off_switch::OffSwitchView,
        xai_abuse_enforcement_types::off_switch::OffSwitchSeen,
        xai_abuse_enforcement_types::off_switch::OffSwitchListResponse,
        xai_abuse_enforcement_types::head_guard::HeadGuardRecord,
        xai_abuse_enforcement_types::head_guard::OverturnGuard,
        xai_abuse_enforcement_types::head_guard::HeadGuardPutRequest,
        xai_abuse_enforcement_types::head_guard::HeadGuardPutResponse,
        xai_abuse_enforcement_types::head_guard::HeadGuardView,
        xai_abuse_enforcement_types::head_guard::HeadGuardListResponse,
        xai_abuse_enforcement_types::action_control::UnreadableRecord,
        xai_abuse_enforcement_types::backfill::BackfillRunRecord,
        xai_abuse_enforcement_types::backfill::BackfillRunStartRequest,
        xai_abuse_enforcement_types::backfill::BackfillRunChangeRequest,
        xai_abuse_enforcement_types::backfill::BackfillRunListResponse,
        xai_abuse_enforcement_types::backfill::RunState,
        xai_abuse_enforcement_types::backfill::RunSource,
        xai_abuse_enforcement_types::backfill::PausedReason,
        xai_abuse_enforcement_types::backfill::RunTransition,
        xai_abuse_enforcement_types::backfill::BackfillSettingsRecord,
        xai_abuse_enforcement_types::backfill::BackfillSettingsChange,
        xai_abuse_enforcement_types::backfill::BackfillSettingsRequest,
        xai_abuse_enforcement_types::backfill::BackfillSettingsResponse,
        crate::rules::RuleStatus,
        crate::rules::RuleSource,
        ApiError,
        DryRunStatus,
        RateLimitStats,
        RulesResponse,
        AllowlistDeleteResponse,
    )),
    tags(
        (name = "config", description = "🔒 PROTECTED (bearer token). GrowthBook config — read, plus writes that become GrowthBook draft revisions awaiting review (never published here)."),
        (name = "allowlist", description = "🔒 PROTECTED (bearer token). Allowlist — read + write (single + bulk)."),
        (name = "action_control", description = "🔒 PROTECTED (bearer token). Page-owned action control — per entity (user or post) and head / classifier / band: follow the GrowthBook rules, record only, or act with the page's actions; history."),
        (name = "off_switch", description = "🔒 PROTECTED (bearer token). Enforcement off switches — turn enforcement off or back on for a whole head or one head + classifier."),
        (name = "head_guard", description = "🔒 PROTECTED (bearer token). Head guards — hourly and daily action ceilings for a whole head or one head + classifier (reaching one turns that scope's off switch on; resume by turning the switch back on), plus the appeal-overturn limit the factory backend evaluates. A PUT with no limits clears the guard."),
        (name = "backfill_runs", description = "🔒 PROTECTED (bearer token). Runs that act on past decisions: start, pause, resume and stop (page key only), done (the sender), reads and claims."),
        (name = "debug", description = "Open (no auth) ops/debug endpoints — dry-run flag, rate-limit usage, dedup lookup — EXCEPT POST /action and both rate-limit writes (POST /rate_limit and POST /rate_limit/reset), which are 🔒 protected because they mutate the enforcement budget. Look for the lock icon per endpoint."),
    ),
    servers(
        (url = "/api", description = "API base"),
    ),
    info(
        title = "xai-abuse-enforcement-service",
        description = "HTTP API for the xAI abuse-enforcement system. Used by the slackbot + the embedded admin UI at /admin.\n\n\
        **Auth:** endpoints with a 🔒 lock icon require `Authorization: Bearer <token>` — click **Authorize** and paste a key. \
        PROTECTED: everything under `config` + `allowlist` + `off_switch` + `head_guard` + `action_control` + `backfill_runs`, plus `POST /action`, `POST /rate_limit` and `POST /rate_limit/reset`. \
        OPEN (no auth): `GET /dry_run`, `GET /rate_limit`, `GET /dedup/{user_id}`.",
    ),
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    const EXPECTED_OPERATIONS: &[(&str, &str, bool)] = &[
        ("/action", "post", true),
        ("/config", "get", true),
        ("/config", "put", true),
        ("/config/{field}", "patch", true),
        ("/rules", "get", true),
        ("/dry_run", "get", false),
        ("/rate_limit/{entity_type}", "get", false),
        ("/rate_limit/{entity_type}", "post", true),
        ("/rate_limit/{entity_type}/reset", "post", true),
        ("/dedup/{user_id}", "get", false),
        ("/dedup/{entity_type}/{entity_id}", "get", false),
        ("/allowlist", "get", true),
        ("/allowlist/bulk", "post", true),
        ("/allowlist/{user_id}", "get", true),
        ("/allowlist/{user_id}", "put", true),
        ("/allowlist/{user_id}", "delete", true),
        ("/allowlist/{entity_type}/{entity_id}", "get", true),
        ("/allowlist/{entity_type}/{entity_id}", "put", true),
        ("/allowlist/{entity_type}/{entity_id}", "delete", true),
        ("/off_switches", "get", true),
        ("/off_switches", "put", true),
        ("/head_guards", "get", true),
        ("/head_guards", "put", true),
        ("/action_controls", "get", true),
        ("/action_controls", "put", true),
        ("/action_controls/history", "get", true),
        ("/action_controls/pod", "get", true),
        ("/backfill_runs", "get", true),
        ("/backfill_runs", "post", true),
        ("/backfill_runs/budget", "get", true),
        ("/backfill_runs/{run_id}", "get", true),
        ("/backfill_runs/{run_id}/{change}", "post", true),
        ("/backfill_settings", "get", true),
        ("/backfill_settings", "put", true),
    ];

    #[test]
    fn openapi_doc_is_well_formed() {
        let doc = ApiDoc::openapi();
        let json = serde_json::to_value(&doc).expect("doc serializes to JSON");

        let paths = json
            .get("paths")
            .and_then(|v| v.as_object())
            .expect("`paths` is an object");

        let mut got: std::collections::BTreeMap<(String, String), bool> =
            std::collections::BTreeMap::new();
        for (path, item) in paths {
            for (method, op) in item.as_object().expect("path item is an object") {
                let has_security = op
                    .get("security")
                    .and_then(|v| v.as_array())
                    .map(|a| !a.is_empty())
                    .unwrap_or(false);
                got.insert((path.clone(), method.clone()), has_security);
            }
        }

        let expected_keys: std::collections::BTreeSet<(String, String)> = EXPECTED_OPERATIONS
            .iter()
            .map(|(p, m, _)| ((*p).to_owned(), (*m).to_owned()))
            .collect();
        let got_keys: std::collections::BTreeSet<(String, String)> = got.keys().cloned().collect();
        let missing: Vec<_> = expected_keys.difference(&got_keys).collect();
        let extra: Vec<_> = got_keys.difference(&expected_keys).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "ApiDoc operations drift — missing: {missing:?}, extra: {extra:?}",
        );

        let mut wrong: Vec<String> = Vec::new();
        for (path, method, want_protected) in EXPECTED_OPERATIONS {
            let key = ((*path).to_owned(), (*method).to_owned());
            let got_protected = *got.get(&key).expect("checked above");
            if got_protected != *want_protected {
                wrong.push(format!(
                    "{method} {path}: expected security={want_protected}, got security={got_protected}",
                ));
            }
        }
        assert!(
            wrong.is_empty(),
            "ApiDoc `security(...)` annotations disagree with the protected/public split \
             in `build_api_router`:\n  {}",
            wrong.join("\n  "),
        );

        let schemes = json
            .pointer("/components/securitySchemes")
            .and_then(|v| v.as_object())
            .expect("componentss.securitySchemes must be an object");
        assert!(
            schemes.contains_key("BearerAuth"),
            "BearerAuth security scheme missing — `SecurityAddon` modifier didn't run? \
             schemes={schemes:?}",
        );

        let declared_tags: std::collections::HashSet<String> = doc
            .tags
            .as_ref()
            .map(|t| t.iter().map(|tag| tag.name.clone()).collect())
            .unwrap_or_default();
        let mut handler_tags: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (_path, item) in paths {
            for (_method, op) in item.as_object().expect("path item is an object") {
                if let Some(tags) = op.get("tags").and_then(|v| v.as_array()) {
                    for t in tags {
                        if let Some(s) = t.as_str() {
                            handler_tags.insert(s.to_string());
                        }
                    }
                }
            }
        }
        let undeclared: Vec<&String> = handler_tags.difference(&declared_tags).collect();
        assert!(
            undeclared.is_empty(),
            "handlers reference tags not declared in tags(...): {undeclared:?}",
        );
    }
}
