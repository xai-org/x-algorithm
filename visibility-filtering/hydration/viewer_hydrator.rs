use crate::hydration::metrics::{record_hydrator_request, HydratorOutcome};
use crate::models::{Viewer, ViewerAge, ViewerFeatures};
use crate::rules::SafetyLevel;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::warn;
use xai_core_entities::gizmoduck_client::{GizmoduckClient, QueryFields};

const CLIENT_TIMEOUT: Duration = Duration::from_millis(150);

const VIEWER_QUERY_FIELDS: [QueryFields; 2] = [QueryFields::ACCOUNT, QueryFields::EXTENDED_PROFILE];

pub struct ViewerHydrator {
    pub gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
}

impl ViewerHydrator {
    pub async fn hydrate(
        &self,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
    ) -> ViewerFeatures {
        let viewer = match viewer_id {
            Some(id) => Viewer::LoggedIn(id),
            None => Viewer::LoggedOut,
        };
        let (allows_sensitive_media, viewer_age, account_country_code) = match viewer_id {
            Some(vid) => {
                let start = Instant::now();
                let result = tokio::time::timeout(
                    CLIENT_TIMEOUT,
                    self.gizmoduck_client
                        .get_viewer_data_with_fields(vid, &VIEWER_QUERY_FIELDS),
                )
                .await;
                let outcome = match &result {
                    Ok(Ok(_)) => HydratorOutcome::Success,
                    Ok(Err(_)) => HydratorOutcome::Error,
                    Err(_) => HydratorOutcome::Timeout,
                };
                record_hydrator_request(
                    "gizmoduck",
                    "get_viewer_data",
                    safety_level,
                    outcome,
                    1,
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                match result {
                    Ok(Ok(data)) => {
                        let age = match data.age_in_years {
                            Some(age) => ViewerAge::Known(age),
                            None if data.user_exists => ViewerAge::NotStated,
                            None => ViewerAge::Unknown,
                        };
                        (
                            data.nsfw_view.unwrap_or(false),
                            age,
                            data.account_country_code,
                        )
                    }
                    Ok(Err(e)) => {
                        warn!(
                            error = %e,
                            "Gizmoduck viewer lookup failed; treating logged-in viewer as no stated age"
                        );
                        (false, ViewerAge::NotStated, None)
                    }
                    Err(_) => {
                        warn!(
                            "Gizmoduck viewer lookup timed out; treating logged-in viewer as no stated age"
                        );
                        (false, ViewerAge::NotStated, None)
                    }
                }
            }
            None => (false, ViewerAge::Unknown, None),
        };

        ViewerFeatures {
            viewer,
            allows_sensitive_media,
            country_code: country_code.map(|c| c.to_ascii_lowercase()),
            account_country_code: account_country_code.map(|c| c.to_ascii_lowercase()),
            viewer_age,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::collections::HashMap;
    use xai_core_entities::entities::{GizmoduckUserResult, PCFLabel};
    use xai_core_entities::gizmoduck_client::{MockGizmoduckClient, UserFields, ViewerData};

    fn hydrator_with_viewer_data(user_id: u64, data: ViewerData) -> ViewerHydrator {
        let mut client = MockGizmoduckClient::default();
        client.viewer_data.insert(user_id, data);
        ViewerHydrator {
            gizmoduck_client: Arc::new(client),
        }
    }

    enum ViewerLookup {
        Fails,
        Hangs,
    }

    struct BrokenViewerClient(ViewerLookup);

    #[tonic::async_trait]
    impl GizmoduckClient for BrokenViewerClient {
        async fn get_viewer_data_with_fields(
            &self,
            _user_id: u64,
            _query_fields: &[QueryFields],
        ) -> Result<ViewerData> {
            match self.0 {
                ViewerLookup::Fails => Err(anyhow::anyhow!("gizmoduck unavailable")),
                ViewerLookup::Hangs => std::future::pending().await,
            }
        }

        async fn get_users(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<GizmoduckUserResult>>> {
            unreachable!()
        }

        async fn get_users_with_perspective(
            &self,
            _viewer_id: i64,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<GizmoduckUserResult>>> {
            unreachable!()
        }

        async fn get_viewer_roles(&self, _user_id: u64) -> Result<Vec<String>> {
            unreachable!()
        }

        async fn get_viewer_data(&self, _user_id: u64) -> Result<ViewerData> {
            unreachable!()
        }

        async fn get_pcf_labels(&self, _user_ids: Vec<i64>) -> HashMap<i64, Result<PCFLabel>> {
            unreachable!()
        }

        async fn get_profile_description_languages(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<String>>> {
            unreachable!()
        }

        async fn get_user_fields(&self, _user_ids: Vec<i64>) -> HashMap<i64, Result<UserFields>> {
            unreachable!()
        }

        async fn get_by_screen_name(
            &self,
            _screen_name: &str,
        ) -> Result<Option<GizmoduckUserResult>> {
            unreachable!()
        }
    }

    async fn hydrate_with_broken_client(lookup: ViewerLookup) -> ViewerFeatures {
        let hydrator = ViewerHydrator {
            gizmoduck_client: Arc::new(BrokenViewerClient(lookup)),
        };
        hydrator
            .hydrate(Some(123), Some("US".to_string()), SafetyLevel::FilterAll)
            .await
    }

    #[tokio::test]
    async fn rpc_error_treats_logged_in_viewer_as_no_stated_age() {
        let viewer = hydrate_with_broken_client(ViewerLookup::Fails).await;

        assert_eq!(viewer.viewer, Viewer::LoggedIn(123));
        assert!(!viewer.allows_sensitive_media);
        assert_eq!(viewer.viewer_age, ViewerAge::NotStated);
        assert!(viewer.viewer_has_no_stated_age());
        assert_eq!(viewer.account_country_code, None);
        assert_eq!(viewer.country_code.as_deref(), Some("us"));
    }

    #[tokio::test]
    async fn rpc_timeout_treats_logged_in_viewer_as_no_stated_age() {
        let viewer = hydrate_with_broken_client(ViewerLookup::Hangs).await;

        assert_eq!(viewer.viewer, Viewer::LoggedIn(123));
        assert!(!viewer.allows_sensitive_media);
        assert_eq!(viewer.viewer_age, ViewerAge::NotStated);
        assert!(viewer.viewer_has_no_stated_age());
        assert_eq!(viewer.account_country_code, None);
        assert_eq!(viewer.country_code.as_deref(), Some("us"));
    }

    #[tokio::test]
    async fn logged_out_viewer_defaults_sensitive_media_to_false() {
        let hydrator = ViewerHydrator {
            gizmoduck_client: Arc::new(MockGizmoduckClient::default()),
        };

        let viewer = hydrator
            .hydrate(None, Some("US".to_string()), SafetyLevel::FilterAll)
            .await;

        assert_eq!(viewer.viewer, Viewer::LoggedOut);
        assert!(!viewer.allows_sensitive_media);
        assert_eq!(viewer.country_code.as_deref(), Some("us"));
        assert_eq!(viewer.viewer_age, ViewerAge::Unknown);
    }

    #[tokio::test]
    async fn account_country_code_is_hydrated_lowercased() {
        let hydrator = hydrator_with_viewer_data(
            123,
            ViewerData {
                account_country_code: Some("KR".to_string()),
                ..Default::default()
            },
        );

        let viewer = hydrator
            .hydrate(Some(123), Some("US".to_string()), SafetyLevel::FilterAll)
            .await;

        assert_eq!(viewer.account_country_code.as_deref(), Some("kr"));
        assert_eq!(viewer.country_code.as_deref(), Some("us"));
    }

    #[tokio::test]
    async fn existing_viewer_without_age_is_not_stated() {
        let hydrator = hydrator_with_viewer_data(
            123,
            ViewerData {
                user_exists: true,
                nsfw_view: Some(false),
                age_in_years: None,
                ..Default::default()
            },
        );

        let viewer = hydrator
            .hydrate(Some(123), None, SafetyLevel::FilterAll)
            .await;

        assert_eq!(viewer.viewer_age, ViewerAge::NotStated);
    }

    #[tokio::test]
    async fn nonexistent_viewer_is_unknown() {
        let hydrator = ViewerHydrator {
            gizmoduck_client: Arc::new(MockGizmoduckClient::default()),
        };

        let viewer = hydrator
            .hydrate(Some(123), Some("FR".to_string()), SafetyLevel::FilterAll)
            .await;

        assert_eq!(viewer.viewer, Viewer::LoggedIn(123));
        assert!(!viewer.allows_sensitive_media);
        assert_eq!(viewer.country_code.as_deref(), Some("fr"));
        assert_eq!(viewer.viewer_age, ViewerAge::Unknown);
    }
}
