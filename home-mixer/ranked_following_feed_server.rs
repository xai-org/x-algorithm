use crate::candidate_pipeline::ranked_following_candidate_pipeline::RankedFollowingCandidatePipeline;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::server::QueryBuilder;
use crate::util::feed_log;
use crate::util::urt;
use tonic::Status;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_home_mixer_proto::FeedItem;

pub(crate) struct RankedFollowingFeedOutput {
    pub items: Vec<FeedItem>,
}

pub struct RankedFollowingFeedServer {
    pub(crate) query_builder: QueryBuilder,
    pipeline: RankedFollowingCandidatePipeline,
}

impl RankedFollowingFeedServer {
    pub fn new(query_builder: QueryBuilder, pipeline: RankedFollowingCandidatePipeline) -> Self {
        Self {
            query_builder,
            pipeline,
        }
    }

    pub(crate) async fn get_ranked_following_feed(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<RankedFollowingFeedOutput, Status> {
        if params::TEST_USER_IDS.contains(&query.user_id) {
            return Ok(RankedFollowingFeedOutput { items: vec![] });
        }

        let result = self.pipeline.execute(query).await;
        feed_log::log_feed_response(&result);

        Ok(RankedFollowingFeedOutput {
            items: result.selected_candidates,
        })
    }

    pub(crate) async fn get_ranked_following_feed_urt(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<Vec<u8>, Status> {
        feed_log::log_feed_request(&query);

        let cursor = query.cursor.clone();
        let request_context = query.request_context.clone();
        let client_app_id = query.client_app_id;
        let viewer_id = query.user_id;
        let language_code = query.language_code.clone();
        let country_code = query.country_code.clone();
        let request_type = query.request_type;
        let request_id = query.request_id;

        let output = self.get_ranked_following_feed(query).await?;

        let timeline_response = urt::make_urt_timeline(
            &output.items,
            cursor.as_ref(),
            &request_context,
            client_app_id,
            viewer_id,
            &language_code,
            if country_code.is_empty() {
                None
            } else {
                Some(&country_code)
            },
            None,
            request_type,
            request_id,
            false,
        );

        xai_urt_thrift::serialize_binary(&timeline_response)
            .map_err(|e| Status::internal(format!("failed to serialize URT: {e}")))
    }
}
