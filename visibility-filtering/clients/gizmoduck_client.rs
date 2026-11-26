use anyhow::anyhow;
use std::collections::HashMap;
use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, UserResponseState};
use xai_core_entities::gizmoduck_client::{GizmoduckClient, LookupContext, QueryFields};

fn author_hydration_lookup_context() -> LookupContext {
    LookupContext {
        for_user_id: None,
        include_deactivated: true,
        include_failed: true,
        include_erased: true,
        include_no_screen_name_users: true,
        include_offboarded: true,
        include_soft_users: true,
        ..Default::default()
    }
}

fn viewer_lookup_context(viewer_id: i64) -> LookupContext {
    LookupContext {
        for_user_id: Some(viewer_id),
        ..author_hydration_lookup_context()
    }
}

pub async fn get_users(
    gizmoduck: &(dyn GizmoduckClient + Send + Sync),
    user_ids: &[u64],
    fields: &[QueryFields],
) -> HashMap<u64, anyhow::Result<Option<GizmoduckUserResult>>> {
    let ids: Vec<i64> = user_ids.iter().map(|&id| id.cast_signed()).collect();
    gizmoduck
        .get_users_with_context(ids, Some(author_hydration_lookup_context()), fields)
        .await
        .into_iter()
        .map(|(id, result)| (id.cast_unsigned(), result))
        .collect()
}

pub async fn get_viewer(
    gizmoduck: &(dyn GizmoduckClient + Send + Sync),
    viewer_id: u64,
    fields: &[QueryFields],
) -> anyhow::Result<Option<GizmoduckUser>> {
    let viewer_id = viewer_id.cast_signed();
    let result = gizmoduck
        .get_users_with_context(
            vec![viewer_id],
            Some(viewer_lookup_context(viewer_id)),
            fields,
        )
        .await
        .into_values()
        .next()
        .ok_or_else(|| anyhow!("Empty batch response from gizmoduck"))??;
    match result {
        Some(GizmoduckUserResult {
            response_state: Some(state @ (UserResponseState::Failed | UserResponseState::Filtered)),
            ..
        }) => Err(anyhow!("Gizmoduck answered the viewer {state:?}")),
        result => Ok(result.and_then(|result| result.user)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tonic::async_trait;
    use xai_core_entities::entities::PCFLabel;
    use xai_core_entities::gizmoduck_client::{UserFields, ViewerData};

    type Users = HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>>;

    #[derive(Default)]
    struct Recording {
        contexts: Mutex<Vec<Option<LookupContext>>>,
        state_without_user: Option<UserResponseState>,
    }

    #[async_trait]
    impl GizmoduckClient for Recording {
        async fn get_users_with_context(
            &self,
            ids: Vec<i64>,
            context: Option<LookupContext>,
            _: &[QueryFields],
        ) -> Users {
            self.contexts.lock().unwrap().push(context);
            let Some(state) = self.state_without_user else {
                return HashMap::new();
            };
            let answer = || {
                Ok(Some(GizmoduckUserResult {
                    user: None,
                    response_state: Some(state),
                }))
            };
            ids.into_iter().map(|id| (id, answer())).collect()
        }

        async fn get_users(&self, _: Vec<i64>) -> Users {
            unreachable!()
        }

        async fn get_users_with_perspective(&self, _: i64, _: Vec<i64>) -> Users {
            unreachable!()
        }

        async fn get_viewer_roles(&self, _: u64) -> anyhow::Result<Vec<String>> {
            unreachable!()
        }

        async fn get_viewer_data(&self, _: u64) -> anyhow::Result<ViewerData> {
            unreachable!()
        }

        async fn get_viewer_data_with_fields(
            &self,
            _: u64,
            _: &[QueryFields],
        ) -> anyhow::Result<ViewerData> {
            unreachable!()
        }

        async fn get_pcf_labels(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<PCFLabel>> {
            unreachable!()
        }

        async fn get_profile_description_languages(
            &self,
            _: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<String>>> {
            unreachable!()
        }

        async fn get_user_fields(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<UserFields>> {
            unreachable!()
        }

        async fn get_by_screen_name(&self, _: &str) -> anyhow::Result<Option<GizmoduckUserResult>> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn author_lookups_include_deactivated_erased_offboarded_and_soft_users() {
        let client = Recording::default();
        get_users(&client, &[10], &[QueryFields::SAFETY]).await;
        let contexts = client.contexts.lock().unwrap();
        let [Some(context)] = contexts.as_slice() else {
            panic!("one lookup with a context: {contexts:?}")
        };
        assert!(context.include_deactivated);
        assert!(context.include_erased);
        assert!(context.include_offboarded);
        assert!(context.include_soft_users);
    }

    #[tokio::test]
    async fn viewer_lookups_include_every_user_state_tweetypie_reads() {
        let client = Recording::default();
        let viewer = get_viewer(&client, 10, &[QueryFields::SAFETY]).await;
        assert!(viewer.is_err(), "an empty batch is an error");
        let contexts = client.contexts.lock().unwrap();
        let [Some(context)] = contexts.as_slice() else {
            panic!("one lookup with a context: {contexts:?}")
        };
        assert!(context.include_deactivated);
        assert!(context.include_failed);
        assert!(context.include_erased);
        assert!(context.include_offboarded);
        assert!(context.include_no_screen_name_users);
        assert!(context.include_soft_users);
        assert_eq!(context.for_user_id, Some(10));
    }

    #[tokio::test]
    async fn a_viewer_gizmoduck_failed_or_filtered_is_a_failed_lookup() {
        for state in [UserResponseState::Failed, UserResponseState::Filtered] {
            let client = Recording {
                state_without_user: Some(state),
                ..Recording::default()
            };
            let viewer = get_viewer(&client, 10, &[]).await;
            assert!(viewer.is_err(), "{state:?}: {viewer:?}");
        }
    }
}
