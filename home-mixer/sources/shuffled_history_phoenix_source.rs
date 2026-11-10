use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableShuffledHistoryRetrieval, PhoenixMaxResults, PhoenixXdsRetrievalMaxRetries,
    RetrievalValueModelWeights,
};
use crate::sources::phoenix_source::PhoenixSource;
use crate::util::phoenix_request::{
    build_client_context, build_user_context, candidates_from_retrieval_response,
    retrieval_value_model_weights,
};
use arrow::array::UInt32Array;
use arrow::compute::{concat_batches, take_record_batch};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use rand::seq::SliceRandom;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalClient;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub struct ShuffledHistoryPhoenixSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for ShuffledHistoryPhoenixSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableShuffledHistoryRetrieval)
            && !query.is_topic_request()
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or("ShuffledHistoryPhoenixSource: missing retrieval_sequence")?;
        let shuffled = query
            .columnar_retrieval_sequence
            .as_ref()
            .and_then(shuffle_rows)
            .ok_or("ShuffledHistoryPhoenixSource: failed to shuffle history")?;

        let cluster = PhoenixSource::resolve_cluster(query);
        let response = self
            .phoenix_retrieval_client
            .retrieve_excluding(
                cluster,
                query.user_id,
                sequence.clone(),
                Some(shuffled),
                quality_factor::apply(query.params.get(PhoenixMaxResults)),
                0,
                vec![],
                None,
                build_client_context(query),
                build_user_context(query),
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                vec![],
                PhoenixSource::seen_post_exclusions(query),
                retrieval_value_model_weights(&query.params.get(RetrievalValueModelWeights)),
            )
            .await
            .map_err(|e| format!("ShuffledHistoryPhoenixSource: {e}"))?;

        Ok(candidates_from_retrieval_response(
            response,
            |_| pb::ServedType::ForYouPhoenixRetrievalShuffledHistory,
            cluster,
        ))
    }
}

fn shuffle_rows(columnar_sequence: &bytes::Bytes) -> Option<bytes::Bytes> {
    let reader =
        StreamReader::try_new(std::io::Cursor::new(columnar_sequence.as_ref()), None).ok()?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().ok()?;
    let batch = concat_batches(&schema, &batches).ok()?;
    let mut order: Vec<u32> = (0..batch.num_rows() as u32).collect();
    order.shuffle(&mut rand::rng());
    let shuffled = take_record_batch(&batch, &UInt32Array::from(order)).ok()?;
    let mut writer = StreamWriter::try_new(Vec::new(), &schema).ok()?;
    writer.write(&shuffled).ok()?;
    writer.into_inner().ok().map(bytes::Bytes::from)
}
