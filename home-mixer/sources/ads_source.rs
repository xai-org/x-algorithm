use crate::clients::ad_index_client::{AdIndexClient, SR_ROUTER_DECIDER};
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::EnableAdsSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::client_utils::{
    feed_cursor_direction, RequestContext,
};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::{feed_item, FeedItem};
use xai_recsys_proto::{AdIndexRequest, ClientContext, FeedRequestContext, ProductSurface};

pub struct AdsSource {
    pub ad_index_client: Arc<dyn AdIndexClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for AdsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableAdsSource) && !query.is_preview
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let request = build_ad_index_request(query);
        let sr_router_enabled = query
            .decider
            .as_ref()
            .is_some_and(|d| d.enabled(SR_ROUTER_DECIDER));

        let response = self
            .ad_index_client
            .get_eligible_ads(request, sr_router_enabled)
            .await
            .map_err(|e| format!("AdsSource: {e}"))?;

        let feed_items = response
            .ad_info
            .into_iter()
            .map(|ad| FeedItem {
                position: 0,
                item: Some(feed_item::Item::Ad(ad)),
            })
            .collect();
        Ok(feed_items)
    }
}

fn build_ad_index_request(query: &ScoredPostsQuery) -> AdIndexRequest {
    let product_surface = match query.request_type {
        RequestType::RankedFollowing => ProductSurface::HomeTimelineRankedFollowing,
        RequestType::Following => ProductSurface::HomeTimelineLatest,
        _ => ProductSurface::HomeTimelineRanking,
    };

    let dsp_client_context = query.dsp_client_context.clone().filter(|dcc| {
        dcc.google_sdk.as_ref().is_some_and(|g| {
            !g.gsig_native.is_empty() || g.gsigs_native.iter().any(|s| !s.is_empty())
        })
    });

    AdIndexRequest {
        user_id: query.user_id as i64,
        product_surface: product_surface as i32,
        dsp_client_context,
        client_seen_post_ids: query.seen_ids.iter().map(|&id| id as i64).collect(),
        feed_request_context: FeedRequestContext::from(RequestContext::parse(
            &query.request_context,
        )) as i32,
        feed_cursor_direction: feed_cursor_direction(query.cursor.as_ref()) as i32,
        client_context: Some(ClientContext {
            user_id: query.user_id as i64,
            app_id: query.client_app_id as i64,
            country_code: query.country_code.clone(),
            language_code: query.language_code.clone(),
            ip_address: query.ip_address.clone(),
            user_agent: query.user_agent.clone(),
            user_roles: query.user_roles.clone(),
            device_id: query.device_id.clone(),
            mobile_device_id: query.mobile_device_id.clone(),
            mobile_device_ad_id: query.mobile_device_ad_id.clone(),
            client_version: query.client_version.clone(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::ad_index_client::MockAdIndexClient;
    use xai_candidate_pipeline::source::Source;

    fn ads_source() -> AdsSource {
        AdsSource {
            ad_index_client: Arc::new(MockAdIndexClient),
        }
    }

    #[test]
    fn enable_returns_false_when_is_preview() {
        let query = ScoredPostsQuery {
            is_preview: true,
            ..Default::default()
        };
        assert!(!ads_source().enable(&query));
    }

    #[test]
    fn forwards_seen_ids_into_ad_index_request() {
        let query = ScoredPostsQuery {
            user_id: 42,
            seen_ids: vec![7, 9],
            ..Default::default()
        };
        assert_eq!(
            build_ad_index_request(&query).client_seen_post_ids,
            vec![7, 9]
        );
    }

    #[test]
    fn forwards_dsp_client_context_into_ad_index_request() {
        let query = ScoredPostsQuery {
            user_id: 42,
            dsp_client_context: Some(xai_recsys_proto::DspClientContext {
                session_id: "sess-1".to_string(),
                webkit_user_agent: "Mozilla/5.0 (wv)".to_string(),
                google_sdk: Some(xai_recsys_proto::GoogleSdk {
                    gsig_native: "DUMMY_GSIG_ABC123".to_string(),
                    gsigs_native: vec!["g1".to_string(), "g2".to_string()],
                }),
            }),
            ..Default::default()
        };
        let req = build_ad_index_request(&query);
        let dcc = req
            .dsp_client_context
            .expect("dsp_client_context set when gsig present");
        assert_eq!(dcc.session_id, "sess-1");
        assert_eq!(dcc.webkit_user_agent, "Mozilla/5.0 (wv)");
        let sdk = dcc.google_sdk.expect("google_sdk set");
        assert_eq!(sdk.gsig_native, "DUMMY_GSIG_ABC123");
        assert_eq!(sdk.gsigs_native, vec!["g1".to_string(), "g2".to_string()]);
    }

    #[test]
    fn omits_dsp_client_context_when_gsig_empty() {
        let query = ScoredPostsQuery {
            dsp_client_context: Some(xai_recsys_proto::DspClientContext {
                session_id: "sess-1".to_string(),
                webkit_user_agent: String::new(),
                google_sdk: Some(xai_recsys_proto::GoogleSdk {
                    gsig_native: String::new(),
                    gsigs_native: vec![String::new()],
                }),
            }),
            ..Default::default()
        };
        assert!(build_ad_index_request(&query).dsp_client_context.is_none());
    }

    #[test]
    fn omits_dsp_client_context_when_no_gsig() {
        let req = build_ad_index_request(&ScoredPostsQuery::default());
        assert!(req.dsp_client_context.is_none());
    }
}
