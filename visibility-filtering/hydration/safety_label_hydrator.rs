use crate::hydration::metrics::{batch_outcome, record_batch_size, record_hydrator_request};
use crate::models::{SafetyLabelMap, TweetId};
use crate::rules::SafetyLevel;
use crate::safety_label_source::SafetyLabelSource;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use xai_visibility_filtering_proto as vf_pb;

const CLIENT: &str = "safety_labels";

pub struct SafetyLabelHydrator {
    pub source: Arc<SafetyLabelSource>,
}

pub struct SafetyLabelHydration {
    pub label_types: HashMap<TweetId, SafetyLabelMap>,
    pub label_response: HashMap<TweetId, Arc<vf_pb::SafetyLabelMap>>,
}

impl SafetyLabelHydrator {
    pub async fn hydrate(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> SafetyLabelHydration {
        let raw_ids: Vec<u64> = tweet_ids.iter().map(|t| t.0).collect();
        let candidate_count = raw_ids.len();
        record_batch_size(CLIENT, candidate_count);
        let start = Instant::now();
        let resolved = self.source.get(&raw_ids).await;
        record_hydrator_request(
            CLIENT,
            "get",
            safety_level,
            batch_outcome(&resolved),
            candidate_count,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        let mut label_types = HashMap::with_capacity(tweet_ids.len());
        let mut label_response = HashMap::with_capacity(tweet_ids.len());
        for tweet_id in tweet_ids {
            match resolved
                .get(&tweet_id.0)
                .and_then(|result| result.as_ref().ok())
            {
                Some(label_map) => {
                    label_types
                        .insert(*tweet_id, SafetyLabelMap::from_proto_label_types(label_map));
                    label_response.insert(*tweet_id, Arc::clone(label_map));
                }
                None => {}
            }
        }
        SafetyLabelHydration {
            label_types,
            label_response,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SafetyLabelType, TweetId};
    use crate::rules::SafetyLevel;
    use crate::safety_label_source::lookup::RemoteSource;
    use crate::safety_label_source::manhattan::ManhattanSource;
    use crate::safety_label_source::mh_client::{FetchResult, ManhattanLabelFetcher};
    use crate::safety_label_source::twemcache::{CacheRead, TwemcacheSource};
    use crate::twemcache::{Key, Value};
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tonic::async_trait;
    use xai_manhattan::ManhattanError;
    use xai_safety_label_store::types::encode_lkey;

    struct FakeTwemcache {
        results: HashMap<Key, crate::twemcache::Result<Option<Value>>>,
    }

    #[async_trait]
    impl CacheRead for FakeTwemcache {
        async fn multi_get(
            &self,
            _keys: &[Key],
        ) -> HashMap<Key, crate::twemcache::Result<Option<Value>>> {
            self.results.clone()
        }
    }

    struct FakeLabelFetcher {
        items: HashMap<i64, Vec<crate::safety_label_source::codec::RawSafetyLabel>>,
        batch_error: Mutex<Option<ManhattanError>>,
    }

    #[async_trait]
    impl ManhattanLabelFetcher for FakeLabelFetcher {
        async fn fetch_labels(
            &self,
            tweet_ids: &[i64],
        ) -> Result<Vec<FetchResult>, ManhattanError> {
            if let Some(error) = self.batch_error.lock().unwrap().take() {
                return Err(error);
            }
            Ok(tweet_ids
                .iter()
                .map(|id| Ok(self.items.get(id).cloned().unwrap_or_default()))
                .collect())
        }
    }

    fn cache_key(tweet_id: u64) -> Key {
        Key::new(format!("slm_{tweet_id}").into_bytes()).unwrap()
    }

    fn cached_label() -> Value {
        vec![
            0x0b, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x0c, 0x00, 0x03, 0x0c, 0x5f, 0xff, 0x0d,
            0x69, 0x14, 0x0c, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x01, 0xe7, 0x7c, 0x00, 0x00,
        ]
    }

    fn raw_label(label_type: SafetyLabelType) -> crate::safety_label_source::codec::RawSafetyLabel {
        crate::safety_label_source::codec::RawSafetyLabel {
            lkey: crate::safety_label_source::codec::LkeyBytes(encode_lkey(label_type)),
            mval: crate::safety_label_source::codec::MvalBytes(vec![0x0C, 0x00, 0x04, 0x00, 0x00]),
        }
    }

    fn hydrator(
        cache_results: HashMap<Key, crate::twemcache::Result<Option<Value>>>,
        mh_items: HashMap<i64, Vec<crate::safety_label_source::codec::RawSafetyLabel>>,
        batch_error: Option<ManhattanError>,
    ) -> SafetyLabelHydrator {
        let twemcache = Arc::new(TwemcacheSource::with_cache(Arc::new(FakeTwemcache {
            results: cache_results,
        })));
        let manhattan = Arc::new(ManhattanSource::new(Arc::new(FakeLabelFetcher {
            items: mh_items,
            batch_error: Mutex::new(batch_error),
        })));
        let source = SafetyLabelSource::new(Arc::new(RemoteSource::new(twemcache, manhattan)));
        SafetyLabelHydrator {
            source: Arc::new(source),
        }
    }

    #[tokio::test]
    async fn hydrate_keys_results_by_tweet_id() {
        let tweet_ids = vec![TweetId(1), TweetId(2)];
        let hydrator = hydrator(
            HashMap::from([(cache_key(2), Ok(Some(cached_label())))]),
            HashMap::from([(1, vec![raw_label(SafetyLabelType::NSFW_HIGH_PRECISION)])]),
            None,
        );

        let result = hydrator
            .hydrate(&tweet_ids, SafetyLevel::TimelineHome)
            .await;

        assert!(result.label_types[&TweetId(1)].has_label(SafetyLabelType::NSFW_HIGH_PRECISION));
        assert!(!result.label_types[&TweetId(2)].has_label(SafetyLabelType::SPAM));
        assert!(result.label_response.contains_key(&TweetId(1)));
        assert!(result.label_response.contains_key(&TweetId(2)));
    }

    #[tokio::test]
    async fn hydrate_fails_closed_on_lookup_errors() {
        let tweet_ids = vec![TweetId(1)];
        let hydrator = hydrator(
            HashMap::new(),
            HashMap::new(),
            Some(ManhattanError::NativeProtocol("decode".into())),
        );

        let result = hydrator
            .hydrate(&tweet_ids, SafetyLevel::TimelineHome)
            .await;

        assert!(!result.label_types.contains_key(&TweetId(1)));
        assert!(!result.label_response.contains_key(&TweetId(1)));
    }

    #[tokio::test]
    async fn hydrate_fails_open_on_missing_results() {
        let tweet_ids = vec![TweetId(1)];
        let hydrator = hydrator(HashMap::new(), HashMap::new(), None);

        let result = hydrator
            .hydrate(&tweet_ids, SafetyLevel::TimelineHome)
            .await;

        assert!(!result.label_types[&TweetId(1)].has_label(SafetyLabelType::SPAM));
        assert!(result.label_response[&TweetId(1)].labels.is_empty());
    }
}
