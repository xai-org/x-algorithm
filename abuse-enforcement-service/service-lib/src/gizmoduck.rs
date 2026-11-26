use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use tracing::info;
use xai_core_entities::gizmoduck_client::{LookupContext, QueryFields};
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants};
use xai_strato::{Column, StratoFetchError, StratoGrpc, StratoGrpcConfig, Thrift};
use xai_x_thrift::user::{EmailConfirmedState as WireEmailConfirmed, TestUserStatus, User};
use xai_x_thrift::user_result::UserResult;

use crate::facts::{EmailConfirmedState, GizmoduckFacts};
use crate::gizmoduck_labels::label_ordinal_to_name;
use crate::metrics;
use crate::safety_checks::TestAccountPolicy;

const GET_V2_USER: &str = "gizmoduck/internal_get/get-V2.User";

const QUERY_FIELDS: &[QueryFields] = &[
    QueryFields::SAFETY,
    QueryFields::LABELS,
    QueryFields::PROFILE,
    QueryFields::NOTIFICATION,
    QueryFields::ROLES,
    QueryFields::ACCOUNT,
];

const REDACTED_EMAIL: &str = "<redacted>";

#[async_trait]
pub(crate) trait UserLookup: Send + Sync {
    async fn get_user(
        &self,
        user_id: i64,
        context: LookupContext,
        fields: &HashSet<QueryFields>,
    ) -> Result<Option<UserResult>>;
}

struct StratoUserLookup {
    grpc: StratoGrpc,
}

#[async_trait]
impl UserLookup for StratoUserLookup {
    async fn get_user(
        &self,
        user_id: i64,
        context: LookupContext,
        fields: &HashSet<QueryFields>,
    ) -> Result<Option<UserResult>> {
        let get_v2: Column<(i64, (LookupContext, &HashSet<QueryFields>)), UserResult, Thrift> =
            Column::fetch(GET_V2_USER);
        self.grpc
            .fetch_many(&get_v2, [(user_id, (context, fields))])
            .await
            .pop()
            .ok_or_else(|| anyhow!("gizmoduck: empty batch for user_id={user_id}"))?
            .map_err(StratoFetchError::into_anyhow)
    }
}

#[derive(Debug, Clone)]
pub struct GizmoduckCoreClientConfig {
    pub zone: String,
    pub ca_cert_path: String,
    pub client_cert_path: String,
    pub client_key_path: String,
    pub client_id: String,
}

#[derive(Clone)]
pub struct GizmoduckCoreClient {
    inner: Arc<dyn UserLookup>,
    policy: Arc<TestAccountPolicy>,
}

impl GizmoduckCoreClient {
    #[cfg(test)]
    pub(crate) fn from_client(
        inner: Arc<dyn xai_core_entities::gizmoduck_client::GizmoduckClient + Send + Sync>,
    ) -> Self {
        Self::from_lookup(Arc::new(test_support::CoreEntitiesLookup(inner)))
    }

    #[cfg(test)]
    pub(crate) fn from_lookup(inner: Arc<dyn UserLookup>) -> Self {
        Self {
            inner,
            policy: Arc::default(),
        }
    }

    pub fn with_policy(mut self, policy: TestAccountPolicy) -> Self {
        self.policy = Arc::new(policy);
        self
    }

    pub async fn connect(cfg: &GizmoduckCoreClientConfig) -> Result<Self> {
        let service_url = format!("gizmoduck.gizmoduck.prod.{}.s2s.twttr.net", cfg.zone);
        let strato_cfg = StratoGrpcConfig {
            ca_cert_path: cfg.ca_cert_path.clone(),
            client_cert_path: cfg.client_cert_path.clone(),
            client_key_path: cfg.client_key_path.clone(),
            num_endpoints: Some(GizmoduckRpcConstants::num_endpoints()),
            max_batch_size: GizmoduckRpcConstants::max_batch_size(),
            connect_timeout_ms: GizmoduckRpcConstants::connect_timeout_msec(),
            request_timeout_ms: GizmoduckRpcConstants::request_timeout_msec(),
            client_id: Some(cfg.client_id.clone()),
            wily_ns_path: "/s/gizmoduck/gizmoduck:fed-grpc".to_string(),
            service_url,
            dns_probe_interval_ms: GizmoduckRpcConstants::dns_probe_interval_msec(),
            zone: cfg.zone.clone(),
            readiness_probe_port: Some(8081),
            filter_to_k8s_only: true,
            ..Default::default()
        };
        let grpc = StratoGrpc::new(strato_cfg)
            .await
            .context("failed to create gizmoduck StratoGrpc client")?;
        info!(
            zone = %cfg.zone,
            client_id = %cfg.client_id,
            "gizmoduck client initialised (get-V2 fed-grpc)"
        );
        Ok(Self {
            inner: Arc::new(StratoUserLookup { grpc }),
            policy: Arc::default(),
        })
    }

    #[tracing::instrument(
        skip(self),
        fields(present, suspended, deactivated, label_count, user_type)
    )]
    pub async fn fetch_user(&self, user_id: i64) -> Result<GizmoduckFacts> {
        let start = Instant::now();
        let context = LookupContext {
            for_user_id: Some(user_id),
            include_soft_users: true,
            include_login_email: true,
            ..Default::default()
        };
        let fields: HashSet<QueryFields> = QUERY_FIELDS.iter().copied().collect();
        let result = match self.inner.get_user(user_id, context, &fields).await {
            Ok(result) => result,
            Err(e) => {
                metrics::GIZMODUCK_LATENCY
                    .with_label_values(&["error"])
                    .observe(start.elapsed().as_secs_f64());
                return Err(e.context(format!("gizmoduck get-V2 failed for user_id={user_id}")));
            }
        };
        let mut facts = match result.and_then(|r| r.user) {
            Some(user) => facts_of(&user),
            None => GizmoduckFacts::default(),
        };

        metrics::GIZMODUCK_LATENCY
            .with_label_values(&["ok"])
            .observe(start.elapsed().as_secs_f64());

        let span = tracing::Span::current();
        if !facts.present {
            metrics::GIZMODUCK_CHECK_TOTAL
                .with_label_values(&["user_not_found"])
                .inc();
            span.record("present", false);
            return Ok(facts);
        }
        facts.test_signal = self.policy.signal(&facts);
        metrics::GIZMODUCK_LOGIN_EMAIL_TOTAL
            .with_label_values(&[login_email_state(&facts)])
            .inc();

        span.record("present", true);
        span.record("suspended", facts.suspended);
        span.record("deactivated", facts.deactivated);
        span.record("label_count", facts.labels.len());
        span.record("user_type", tracing::field::debug(&facts.user_type));

        let state_label = if facts.suspended {
            "suspended"
        } else if facts.deactivated {
            "deactivated"
        } else {
            "active"
        };
        metrics::GIZMODUCK_CHECK_TOTAL
            .with_label_values(&[state_label])
            .inc();

        info!(
            state = state_label,
            has_spam_high_recall = facts.labels.iter().any(|l| l == "SpamHighRecall"),
            label_count = facts.labels.len(),
            test_signal = facts.test_signal.map(|s| s.as_str()),
            "gizmoduck fetch ok",
        );

        Ok(facts)
    }
}

fn facts_of(user: &User) -> GizmoduckFacts {
    let safety = user.safety.clone().unwrap_or_default();
    let labels = user
        .labels
        .as_ref()
        .and_then(|l| l.labels.as_ref())
        .map(|labels| {
            labels
                .iter()
                .map(|l| label_ordinal_to_name(l.label_value.map_or(0, |v| v.0)))
                .collect()
        })
        .unwrap_or_default();
    GizmoduckFacts {
        present: true,
        suspended: safety.suspended.unwrap_or(false),
        deactivated: safety.deactivated.unwrap_or(false),
        labels,
        is_blue_verified: safety.is_blue_verified.unwrap_or(false),
        verified_type: safety
            .verified_type
            .and_then(|v| verified_type_name(v.0))
            .map(str::to_owned),
        is_affiliate: safety
            .verified_organization_details
            .as_ref()
            .and_then(|d| d.is_verified_organization_affiliate)
            .unwrap_or(false),
        user_type: user
            .user_type
            .and_then(|t| user_type_name(t.0))
            .map(str::to_owned),
        test_user_status: user.test_user_status.map(test_user_status_name),
        roles: user
            .roles
            .as_ref()
            .and_then(|r| r.roles.as_ref())
            .map(|roles| roles.iter().cloned().collect())
            .unwrap_or_default(),
        login_email: user
            .profile
            .as_ref()
            .and_then(|p| p.auth.as_ref())
            .and_then(|a| a.login_email.as_deref())
            .map(str::trim)
            .filter(|e| !e.is_empty() && !e.eq_ignore_ascii_case(REDACTED_EMAIL))
            .map(str::to_owned),
        email_confirmed: user
            .notification
            .as_ref()
            .and_then(|n| n.email_confirmed)
            .and_then(email_confirmed_state),
        creation_ip: user
            .account
            .as_ref()
            .and_then(|a| a.creation_ip.as_deref())
            .map(str::trim)
            .filter(|ip| !ip.is_empty())
            .map(str::to_owned),
        test_signal: None,
        permanently_suspended: safety
            .suspension_details
            .as_ref()
            .and_then(|d| d.permanently_suspended)
            .unwrap_or(false),
    }
}

fn login_email_state(facts: &GizmoduckFacts) -> &'static str {
    match (&facts.login_email, facts.email_confirmed) {
        (None, _) => "missing_or_redacted",
        (Some(_), Some(EmailConfirmedState::Confirmed)) => "confirmed_present",
        (Some(_), _) => "unconfirmed_present",
    }
}

fn verified_type_name(v: i32) -> Option<&'static str> {
    Some(match v {
        1 => "User",
        2 => "Notable",
        3 => "Business",
        4 => "Government",
        _ => return None,
    })
}

fn user_type_name(t: i32) -> Option<&'static str> {
    Some(match t {
        0 => "Normal",
        1 => "Frictionless",
        2 => "Email",
        3 => "Soft",
        4 => "NoScreenName",
        5 => "Periscope",
        6 => "LoggedOut",
        _ => return None,
    })
}

fn test_user_status_name(s: TestUserStatus) -> String {
    match s {
        TestUserStatus::NONE => "None".to_owned(),
        TestUserStatus::TEMPORARY => "Temporary".to_owned(),
        TestUserStatus::PERMANENT => "Permanent".to_owned(),
        TestUserStatus(n) => format!("Unknown({n})"),
    }
}

fn email_confirmed_state(s: WireEmailConfirmed) -> Option<EmailConfirmedState> {
    Some(match s {
        WireEmailConfirmed::GRANDFATHERED => EmailConfirmedState::Grandfathered,
        WireEmailConfirmed::EXCUSED => EmailConfirmedState::Excused,
        WireEmailConfirmed::UNCONFIRMED => EmailConfirmedState::Unconfirmed,
        WireEmailConfirmed::CONFIRMED => EmailConfirmedState::Confirmed,
        _ => return None,
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::{BTreeSet, HashSet};
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use async_trait::async_trait;
    use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult};
    use xai_core_entities::gizmoduck_client::{GizmoduckClient, LookupContext, QueryFields};
    use xai_x_thrift::user::{
        Account, Auth, EmailConfirmedState, Label, Labels, Notification, Profile, Roles, Safety,
        TestUserStatus, User, VerifiedOrganizationDetails, VerifiedType,
    };
    use xai_x_thrift::user_labels::LabelValue;
    use xai_x_thrift::user_result::UserResult;
    use xai_x_thrift::user_type::UserType;

    use super::UserLookup;

    pub(crate) struct CoreEntitiesLookup(pub Arc<dyn GizmoduckClient + Send + Sync>);

    #[async_trait]
    impl UserLookup for CoreEntitiesLookup {
        async fn get_user(
            &self,
            user_id: i64,
            context: LookupContext,
            fields: &HashSet<QueryFields>,
        ) -> Result<Option<UserResult>> {
            let fields: Vec<QueryFields> = fields.iter().copied().collect();
            let mut rows = self
                .0
                .get_users_with_context(vec![user_id], Some(context), &fields)
                .await;
            let row = rows.remove(&user_id).ok_or_else(|| {
                anyhow::anyhow!("gizmoduck core: empty batch for user_id={user_id}")
            })?;
            Ok(row?.map(|r: GizmoduckUserResult| UserResult {
                user: r.user.as_ref().map(thrift_user),
                ..Default::default()
            }))
        }
    }

    fn thrift_user(u: &GizmoduckUser) -> User {
        User {
            id: Some(u.user_id as i64),
            user_type: u.user_type.map(|t| UserType(t as i32)),
            safety: Some(Safety {
                suspended: Some(u.safety.suspended),
                deactivated: Some(u.safety.deactivated),
                is_blue_verified: Some(u.safety.is_blue_verified),
                verified_type: u.safety.verified_type.map(|v| VerifiedType(v as i32 + 1)),
                verified_organization_details: u.safety.verified_organization_details.as_ref().map(
                    |d| VerifiedOrganizationDetails {
                        is_verified_organization: Some(d.is_verified_organization),
                        is_verified_organization_affiliate: Some(
                            d.is_verified_organization_affiliate,
                        ),
                    },
                ),
                ..Default::default()
            }),
            labels: Some(Labels {
                labels: Some(
                    u.labels
                        .labels
                        .iter()
                        .map(|l| Label {
                            label_value: Some(LabelValue(l.label_value)),
                            created_at_msec: Some(l.created_at_msec),
                            ..Default::default()
                        })
                        .collect(),
                ),
                ..Default::default()
            }),
            roles: u.roles.as_ref().map(|r| Roles {
                roles: Some(r.roles.iter().cloned().collect()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[derive(Clone, Default)]
    pub(crate) struct AccountShape {
        pub test_user_status: Option<i32>,
        pub roles: Vec<&'static str>,
        pub login_email: Option<&'static str>,
        pub email_confirmed: Option<i32>,
        pub creation_ip: Option<&'static str>,
    }

    impl AccountShape {
        pub(crate) fn user(&self, user_id: i64) -> User {
            User {
                id: Some(user_id),
                safety: Some(Safety::default()),
                test_user_status: self.test_user_status.map(TestUserStatus),
                roles: (!self.roles.is_empty()).then(|| Roles {
                    roles: Some(
                        self.roles
                            .iter()
                            .map(|r| (*r).to_owned())
                            .collect::<BTreeSet<_>>(),
                    ),
                    ..Default::default()
                }),
                profile: self.login_email.map(|e| Profile {
                    auth: Some(Auth {
                        login_email: Some(e.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                notification: self.email_confirmed.map(|s| Notification {
                    email_confirmed: Some(EmailConfirmedState(s)),
                    ..Default::default()
                }),
                account: self.creation_ip.map(|ip| Account {
                    creation_ip: Some(ip.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }
        }
    }

    #[derive(Default)]
    pub(crate) struct FixedUsers {
        pub users: std::collections::HashMap<i64, User>,
        pub fail: bool,
        pub lookups: Mutex<Vec<(i64, LookupContext, Vec<QueryFields>)>>,
    }

    #[async_trait]
    impl UserLookup for FixedUsers {
        async fn get_user(
            &self,
            user_id: i64,
            context: LookupContext,
            fields: &HashSet<QueryFields>,
        ) -> Result<Option<UserResult>> {
            let mut fields: Vec<QueryFields> = fields.iter().copied().collect();
            fields.sort_by_key(|f| *f as i32);
            self.lookups
                .lock()
                .unwrap()
                .push((user_id, context, fields));
            if self.fail {
                anyhow::bail!("gizmoduck unavailable");
            }
            Ok(Some(UserResult {
                user: self.users.get(&user_id).cloned(),
                ..Default::default()
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use xai_core_entities::entities::{
        GizmoduckUser, GizmoduckUserResult, GizmoduckUserType, PCFLabel, Safety,
    };
    use xai_core_entities::gizmoduck_client::{
        GizmoduckClient, LookupContext, QueryFields, UserFields, ViewerData,
    };

    use super::test_support::{AccountShape, FixedUsers};
    use super::GizmoduckCoreClient;
    use crate::facts::EmailConfirmedState;
    use crate::gizmoduck_labels::label_ordinal_to_name;
    use crate::safety_checks::{TestAccountPolicy, TestAccountSignal};

    const SOFT_USER: i64 = 1001;

    #[derive(Default)]
    struct SoftUserGizmoduck {
        contexts: Mutex<Vec<Option<LookupContext>>>,
    }

    #[async_trait::async_trait]
    impl GizmoduckClient for SoftUserGizmoduck {
        async fn get_users_with_context(
            &self,
            user_ids: Vec<i64>,
            context: Option<LookupContext>,
            _query_fields: &[QueryFields],
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            let include_soft_users = context.as_ref().is_some_and(|c| c.include_soft_users);
            self.contexts.lock().unwrap().push(context);
            user_ids
                .into_iter()
                .map(|id| {
                    let user = (id == SOFT_USER && include_soft_users).then(|| GizmoduckUser {
                        user_id: id as u64,
                        user_type: Some(GizmoduckUserType::Soft),
                        safety: Safety {
                            suspended: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    });
                    let result = GizmoduckUserResult {
                        user,
                        response_state: None,
                    };
                    (id, Ok(Some(result)))
                })
                .collect()
        }
        async fn get_users(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            unimplemented!()
        }
        async fn get_users_with_perspective(
            &self,
            _viewer_id: i64,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            unimplemented!()
        }
        async fn get_viewer_roles(&self, _user_id: u64) -> anyhow::Result<Vec<String>> {
            unimplemented!()
        }
        async fn get_viewer_data(&self, _user_id: u64) -> anyhow::Result<ViewerData> {
            unimplemented!()
        }
        async fn get_viewer_data_with_fields(
            &self,
            _user_id: u64,
            _query_fields: &[QueryFields],
        ) -> anyhow::Result<ViewerData> {
            unimplemented!()
        }
        async fn get_pcf_labels(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<PCFLabel>> {
            unimplemented!()
        }
        async fn get_profile_description_languages(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<String>>> {
            unimplemented!()
        }
        async fn get_user_fields(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<UserFields>> {
            unimplemented!()
        }
        async fn get_by_screen_name(
            &self,
            _screen_name: &str,
        ) -> anyhow::Result<Option<GizmoduckUserResult>> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn fetch_user_includes_soft_users() {
        let gizmoduck = Arc::new(SoftUserGizmoduck::default());
        let client = GizmoduckCoreClient::from_client(gizmoduck.clone());

        let facts = client.fetch_user(SOFT_USER).await.unwrap();

        assert!(facts.present);
        assert!(facts.suspended);
        assert_eq!(facts.user_type.as_deref(), Some("Soft"));
        assert_eq!(
            gizmoduck.contexts.lock().unwrap().as_slice(),
            [Some(LookupContext {
                for_user_id: Some(SOFT_USER),
                include_soft_users: true,
                include_login_email: true,
                ..Default::default()
            })],
        );
    }

    #[test]
    fn spam_high_recall_maps_to_pascal_case() {
        assert_eq!(label_ordinal_to_name(23), "SpamHighRecall");
    }

    #[test]
    fn unmapped_ordinals_render_as_unknown_label() {
        assert_eq!(label_ordinal_to_name(3), "UnknownLabel(3)");
        assert_eq!(label_ordinal_to_name(9999), "UnknownLabel(9999)");
    }

    const UID: i64 = 4242;

    fn client_with(
        shape: AccountShape,
        policy: TestAccountPolicy,
    ) -> (GizmoduckCoreClient, Arc<FixedUsers>) {
        let users = Arc::new(FixedUsers {
            users: [(UID, shape.user(UID))].into(),
            ..Default::default()
        });
        (
            GizmoduckCoreClient::from_lookup(users.clone()).with_policy(policy),
            users,
        )
    }

    fn policy() -> TestAccountPolicy {
        TestAccountPolicy::new(
            &["test.example.com".into()],
            &["qa_test_role".into()],
            &["203.0.113.0/24".into()],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn one_lookup_asks_for_every_field_the_checks_read() {
        let (client, users) = client_with(AccountShape::default(), policy());
        client.fetch_user(UID).await.unwrap();
        let lookups = users.lookups.lock().unwrap();
        assert_eq!(lookups.len(), 1);
        let (id, context, fields) = &lookups[0];
        assert_eq!(*id, UID);
        assert_eq!(
            *context,
            LookupContext {
                for_user_id: Some(UID),
                include_soft_users: true,
                include_login_email: true,
                ..Default::default()
            }
        );
        assert_eq!(
            fields.as_slice(),
            [
                QueryFields::PROFILE,
                QueryFields::ACCOUNT,
                QueryFields::SAFETY,
                QueryFields::NOTIFICATION,
                QueryFields::ROLES,
                QueryFields::LABELS,
            ]
        );
    }

    #[tokio::test]
    async fn the_account_fields_map_from_the_thrift_user() {
        let (client, _) = client_with(
            AccountShape {
                test_user_status: Some(1),
                roles: vec!["qa_test_role", "other"],
                login_email: Some("  someone@Test.Example.com \n"),
                email_confirmed: Some(3),
                creation_ip: Some(" 203.0.113.9 "),
            },
            TestAccountPolicy::default(),
        );
        let f = client.fetch_user(UID).await.unwrap();
        assert!(f.present);
        assert_eq!(f.test_user_status.as_deref(), Some("Temporary"));
        assert_eq!(f.roles, ["other", "qa_test_role"]);
        assert_eq!(f.login_email.as_deref(), Some("someone@Test.Example.com"));
        assert_eq!(f.email_confirmed, Some(EmailConfirmedState::Confirmed));
        assert_eq!(f.creation_ip.as_deref(), Some("203.0.113.9"));
        assert_eq!(
            f.test_signal,
            Some(TestAccountSignal::TestUserStatus),
            "a test status needs no configuration"
        );
    }

    #[tokio::test]
    async fn every_wire_value_maps() {
        for (wire, name) in [
            (0, "None"),
            (1, "Temporary"),
            (2, "Permanent"),
            (7, "Unknown(7)"),
        ] {
            let (client, _) = client_with(
                AccountShape {
                    test_user_status: Some(wire),
                    ..Default::default()
                },
                TestAccountPolicy::default(),
            );
            let f = client.fetch_user(UID).await.unwrap();
            assert_eq!(f.test_user_status.as_deref(), Some(name));
        }
        for (wire, state) in [
            (0, Some(EmailConfirmedState::Grandfathered)),
            (1, Some(EmailConfirmedState::Excused)),
            (2, Some(EmailConfirmedState::Unconfirmed)),
            (3, Some(EmailConfirmedState::Confirmed)),
            (9, None),
        ] {
            let (client, _) = client_with(
                AccountShape {
                    email_confirmed: Some(wire),
                    ..Default::default()
                },
                TestAccountPolicy::default(),
            );
            assert_eq!(client.fetch_user(UID).await.unwrap().email_confirmed, state);
        }
    }

    #[tokio::test]
    async fn a_redacted_or_blank_email_and_a_blank_ip_are_absent() {
        for email in [
            "<redacted>",
            "  <redacted> ",
            "<REDACTED>",
            "<Redacted>",
            "",
            "   ",
        ] {
            let (client, _) = client_with(
                AccountShape {
                    login_email: Some(email),
                    email_confirmed: Some(3),
                    creation_ip: Some("  "),
                    ..Default::default()
                },
                policy(),
            );
            let f = client.fetch_user(UID).await.unwrap();
            assert_eq!(f.login_email, None, "{email:?}");
            assert_eq!(super::login_email_state(&f), "missing_or_redacted");
            assert_eq!(f.creation_ip, None);
            assert_eq!(f.test_signal, None);
        }
    }

    #[tokio::test]
    async fn the_lookup_classifies_with_the_policy() {
        let cases = [
            (
                AccountShape {
                    roles: vec!["qa_test_role"],
                    ..Default::default()
                },
                Some(TestAccountSignal::Role),
            ),
            (
                AccountShape {
                    login_email: Some("a@test.example.com"),
                    email_confirmed: Some(3),
                    ..Default::default()
                },
                Some(TestAccountSignal::EmailDomain),
            ),
            (
                AccountShape {
                    login_email: Some("a@test.example.com"),
                    email_confirmed: Some(2),
                    ..Default::default()
                },
                None,
            ),
            (
                AccountShape {
                    creation_ip: Some("203.0.113.200"),
                    ..Default::default()
                },
                Some(TestAccountSignal::CreationCidr),
            ),
            (AccountShape::default(), None),
        ];
        for (shape, want) in cases {
            let (client, _) = client_with(shape, policy());
            assert_eq!(client.fetch_user(UID).await.unwrap().test_signal, want);
        }
    }

    #[tokio::test]
    async fn a_missing_user_is_not_found_and_unclassified() {
        let users = Arc::new(FixedUsers::default());
        let client = GizmoduckCoreClient::from_lookup(users).with_policy(policy());
        let f = client.fetch_user(UID).await.unwrap();
        assert!(!f.present);
        assert_eq!(f.test_signal, None);
    }

    #[tokio::test]
    async fn a_failed_lookup_is_an_error() {
        let users = Arc::new(FixedUsers {
            fail: true,
            ..Default::default()
        });
        let client = GizmoduckCoreClient::from_lookup(users).with_policy(policy());
        assert!(client.fetch_user(UID).await.is_err());
    }

    #[tokio::test]
    async fn the_lookup_reads_whether_a_suspension_is_permanent() {
        let suspended = |permanently: Option<bool>| {
            let mut user = AccountShape::default().user(UID);
            user.safety = Some(xai_x_thrift::user::Safety {
                suspended: Some(true),
                suspension_details: permanently.map(|p| xai_x_thrift::user::SuspensionDetails {
                    permanently_suspended: Some(p),
                    ..Default::default()
                }),
                ..Default::default()
            });
            user
        };
        for (permanently, expected) in [(Some(true), true), (Some(false), false), (None, false)] {
            let users = Arc::new(FixedUsers {
                users: [(UID, suspended(permanently))].into(),
                ..Default::default()
            });
            let facts = GizmoduckCoreClient::from_lookup(users)
                .fetch_user(UID)
                .await
                .unwrap();
            assert!(facts.suspended);
            assert_eq!(facts.permanently_suspended, expected, "{permanently:?}");
        }
    }
}
