use std::sync::Arc;

use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableUnderTheHoodFrame, EnableUnderTheHoodSetting};

pub struct UnderTheHoodSettingQueryHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for UnderTheHoodSettingQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableUnderTheHoodSetting) && query.params.get(EnableUnderTheHoodFrame)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let under_the_hood_setting_enabled = self
            .strato_client
            .get_under_the_hood_home_enabled(query.user_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(ScoredPostsQuery {
            under_the_hood_setting_enabled,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.under_the_hood_setting_enabled = hydrated.under_the_hood_setting_enabled;
    }
}
