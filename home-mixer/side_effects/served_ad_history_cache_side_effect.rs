use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAdsHistoryXdcCacheWrite;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_ad_index_history::{
    build_history_record, UserAdHistoryCacheStore, UserAdHistoryStore, WriteScope,
};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::feed_item::Item as FeedItemKind;
use xai_home_mixer_proto::FeedItem;
use xai_recsys_proto::AdIndexInfo;

const AD_HISTORY_CACHE_TTL: Duration = Duration::from_secs(5 * 60);

pub struct ServedAdHistoryCacheSideEffect {
    store: Arc<dyn UserAdHistoryStore>,
}

impl ServedAdHistoryCacheSideEffect {
    pub fn new(store: Arc<dyn UserAdHistoryStore>) -> Self {
        Self { store }
    }

    pub async fn prod(datacenter: &str) -> Self {
        let store = UserAdHistoryCacheStore::connect(datacenter).await;
        Self::new(Arc::new(store))
    }
}

fn ads_in_feed(items: &[FeedItem]) -> Vec<&AdIndexInfo> {
    items
        .iter()
        .filter_map(|item| match item.item.as_ref() {
            Some(FeedItemKind::Ad(ad)) => Some(ad),
            _ => None,
        })
        .collect()
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for ServedAdHistoryCacheSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && !query.is_preview
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let scope = if query.params.get(EnableAdsHistoryXdcCacheWrite) {
            WriteScope::LocalAndPeer
        } else {
            WriteScope::LocalOnly
        };
        let ads = ads_in_feed(&input.selected_candidates);
        if ads.is_empty() {
            return Ok(());
        }

        let record = build_history_record(query.request_id as i64, ads)?;
        self.store
            .put_user_value_scoped(
                query.user_id as i64,
                query.request_time_ms,
                record,
                AD_HISTORY_CACHE_TTL,
                scope,
            )
            .await
            .map_err(|e| format!("ad history cache write failed: {}", e.message()))
    }
}
