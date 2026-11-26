use crate::models::candidate::{PostCandidate, RetrievalSource};
use crate::models::in_network_reply::InNetworkReply;
use crate::models::query::ScoredPostsQuery;
use crate::params::{ThunderAlgorithm, ThunderMaxResults};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::{ThunderCapiClient, ThunderClient};
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_thunder_proto::in_network_posts_service_client::InNetworkPostsServiceClient;
use xai_thunder_proto::GetInNetworkPostsRequest;

pub const THUNDER_CAPI_DECIDER: &str = "enable_thunder_capi_home_mixer";

pub struct ThunderSource {
    pub thunder_client: Arc<ThunderClient>,
    pub thunder_capi_client: Option<Arc<dyn ThunderCapiClient + Send + Sync>>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for ThunderSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let following_list = &query.user_features.followed_user_ids;
        let excluded_ids = query.seen_ids.to_vec();
        let following_user_ids: Vec<u64> = following_list.iter().map(|&id| id as u64).collect();

        let request = GetInNetworkPostsRequest {
            user_id: query.user_id,
            following_user_ids,
            max_results: quality_factor::apply(query.params.get(ThunderMaxResults)),
            exclude_tweet_ids: excluded_ids,
            algorithm: query.params.get(ThunderAlgorithm),
            debug: false,
            is_video_request: false,
            per_author_limits: None,
        };

        let capi = self.thunder_capi_client.as_ref().filter(|_| {
            query
                .decider
                .as_ref()
                .is_some_and(|d| d.enabled(THUNDER_CAPI_DECIDER))
        });
        let posts = if let Some(capi) = capi {
            capi.get_in_network_posts(request)
                .await
                .map_err(|e| format!("ThunderSource(capi): {e}"))?
                .posts
        } else {
            let channel = self
                .thunder_client
                .get_random_channel()
                .ok_or_else(|| "ThunderSource: no available channel".to_string())?;

            let mut client = InNetworkPostsServiceClient::new(channel.clone());
            client
                .get_in_network_posts(request)
                .await
                .map_err(|e| format!("ThunderSource: {}", e))?
                .into_inner()
                .posts
        };

        let replies: Vec<InNetworkReply> = posts
            .iter()
            .filter_map(|post| {
                post.in_reply_to_post_id.map(|reply_to_id| InNetworkReply {
                    author_id: post.author_id as u64,
                    in_reply_to_tweet_id: reply_to_id as u64,
                })
            })
            .collect();

        let _ = query.in_network_replies.set(replies);

        let candidates: Vec<PostCandidate> = posts
            .into_iter()
            .map(|post| {
                let in_reply_to_tweet_id = post
                    .in_reply_to_post_id
                    .and_then(|id| u64::try_from(id).ok());
                let conversation_id = post.conversation_id.and_then(|id| u64::try_from(id).ok());

                let mut ancestors = Vec::new();
                if let Some(reply_to) = in_reply_to_tweet_id {
                    ancestors.push(reply_to);
                    if let Some(root) = conversation_id.filter(|&root| root != reply_to) {
                        ancestors.push(root);
                    }
                }

                let served_type = if !query.in_network_only {
                    pb::ServedType::ForYouInNetwork
                } else {
                    pb::ServedType::RankedFollowing
                };

                let retweeted_tweet_id = post.source_post_id.and_then(|id| u64::try_from(id).ok());

                PostCandidate {
                    tweet_id: post.post_id as u64,
                    author_id: post.author_id as u64,
                    in_reply_to_tweet_id,
                    retweeted_tweet_id,
                    ancestors,
                    served_type: Some(served_type),
                    retrieval_sources: vec![RetrievalSource::from_served_type(served_type)],
                    ..Default::default()
                }
            })
            .collect();

        Ok(candidates)
    }
}
