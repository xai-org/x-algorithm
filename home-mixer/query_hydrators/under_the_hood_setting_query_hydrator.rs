use std::sync::Arc;

use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableUnderTheHood, EnableUnderTheHoodFrame, EnableUnderTheHoodLimit, EnableUnderTheHoodSetting,
};
use crate::util::under_the_hood;

pub struct UnderTheHoodSettingQueryHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for UnderTheHoodSettingQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableUnderTheHood)
            && query.params.get(EnableUnderTheHoodSetting)
            && query.params.get(EnableUnderTheHoodFrame)
            && under_the_hood::is_viewer_visible_request(query)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        if is_prod() && query.params.get(EnableUnderTheHoodLimit) {
            match self
                .strato_client
                .get_under_the_hood_trace_gate(query.user_id)
                .await
            {
                Ok((under_the_hood_setting_enabled, under_the_hood_trace_limited)) => {
                    return Ok(ScoredPostsQuery {
                        under_the_hood_setting_enabled,
                        under_the_hood_trace_limited,
                        ..Default::default()
                    });
                }
                Err(e) => warn!(
                    user_id = query.user_id,
                    error = %e,
                    "under-the-hood trace gate fetch failed; using the setting read"
                ),
            }
        }
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
        query.under_the_hood_trace_limited = hydrated.under_the_hood_trace_limited;
    }
}
