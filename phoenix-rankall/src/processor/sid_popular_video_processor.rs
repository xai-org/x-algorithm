use std::sync::Arc;

use log::warn;

use super::record::IndexRecord;
use super::thrift_types::{deserialize_binary, PhoenixRankAllCandidateFlat};
use super::{ProcessorStats, RecordProcessor};
use crate::sid::client::SidClient;

pub const POPULAR_VIDEO_INDEX_NAME: &str = "popular_video";

pub struct SidPopularVideoProcessor {
    stats: ProcessorStats,
    warn_limit: u64,
    min_view_count: i64,
    min_fav_count: i64,
    sid_client: Arc<SidClient>,
    runtime: tokio::runtime::Handle,
}

impl SidPopularVideoProcessor {
    pub fn new(
        min_view_count: i64,
        min_fav_count: i64,
        sid_client: Arc<SidClient>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            stats: ProcessorStats::default(),
            warn_limit: 5,
            min_view_count,
            min_fav_count,
            sid_client,
            runtime,
        }
    }

    fn keeps(&self, obj: &PhoenixRankAllCandidateFlat) -> bool {
        let counts = obj.engagement_count.as_ref();
        obj.index_name.as_deref().unwrap_or("metadata") == "metadata"
            && obj.has_video.unwrap_or(false)
            && counts.and_then(|c| c.view_count).unwrap_or(0) >= self.min_view_count
            && counts.and_then(|c| c.favorite_count).unwrap_or(0) >= self.min_fav_count
    }
}

impl RecordProcessor for SidPopularVideoProcessor {
    fn process_batch(&mut self, raw: &[Vec<u8>]) -> Vec<IndexRecord> {
        let mut staged: Vec<(i64, i64)> = Vec::new();
        for payload in raw {
            self.stats.total_processed += 1;
            let obj: PhoenixRankAllCandidateFlat = match deserialize_binary(payload) {
                Ok(obj) => obj,
                Err(e) => {
                    self.stats.total_deser_error += 1;
                    if self.stats.total_deser_error <= self.warn_limit {
                        warn!(
                            "sid-popular-video: failed to deserialize PhoenixRankAllCandidateFlat: {e}"
                        );
                    }
                    continue;
                }
            };
            let post_id = obj.post_id.unwrap_or(0);
            let author_id = obj.author_id.unwrap_or(0);
            if post_id == 0 || author_id == 0 {
                self.stats.total_invalid += 1;
                continue;
            }
            if !self.keeps(&obj) {
                self.stats.total_filtered += 1;
                continue;
            }
            staged.push((post_id, author_id));
        }
        if staged.is_empty() {
            return Vec::new();
        }

        let ids: Vec<i64> = staged.iter().map(|&(post_id, _)| post_id).collect();
        let client = Arc::clone(&self.sid_client);
        let sid_map = self
            .runtime
            .block_on(async move { client.lookup_sids(&ids, None).await });
        self.stats.total_success += staged.len() as u64;
        staged
            .into_iter()
            .map(|(post_id, author_id)| IndexRecord::Sid {
                post_id,
                author_id,
                index_name: POPULAR_VIDEO_INDEX_NAME.to_string(),
                post_sid: sid_map.get(&post_id).cloned().unwrap_or_default(),
            })
            .collect()
    }

    fn stats(&self) -> &ProcessorStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::thrift_types::{
        serialize_binary, EngagementCount, MultiModalEmbeddings, PhoenixRankAllCandidateFlat,
    };
    use crate::sid::metrics::SidMetrics;
    use std::time::Duration;

    fn event(post_id: i64, has_video: bool, views: i64, favs: i64, index: Option<&str>) -> Vec<u8> {
        let counts = EngagementCount::new(
            None::<i64>,
            None::<i64>,
            Some(favs),
            None::<i64>,
            None::<i64>,
            Some(views),
            None::<i64>,
            None::<i64>,
            None::<i64>,
        );
        serialize_binary(&PhoenixRankAllCandidateFlat::new(
            Some(post_id),
            Some(post_id * 10),
            Some(has_video),
            Some(false),
            Some(500),
            Some(counts),
            None::<i64>,
            None::<MultiModalEmbeddings>,
            index.map(str::to_string),
        ))
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn keeps_videos_over_both_thresholds() {
        let client = Arc::new(
            SidClient::new(
                "127.0.0.1:1",
                0.0,
                Duration::from_millis(50),
                SidMetrics::for_tests(),
            )
            .unwrap(),
        );
        let mut processor =
            SidPopularVideoProcessor::new(10_000, 100, client, tokio::runtime::Handle::current());
        let batch = vec![
            event(1, true, 10_000, 100, None),
            event(2, true, 9_999, 5_000, Some("metadata")),
            event(3, true, 50_000, 99, Some("metadata")),
            event(4, false, 50_000, 5_000, Some("metadata")),
            event(5, true, 50_000, 5_000, Some("other")),
            event(6, true, 2_000_000, 40_000, Some("metadata")),
        ];
        let records = tokio::task::spawn_blocking(move || processor.process_batch(&batch))
            .await
            .unwrap();
        let ids: Vec<i64> = records.iter().map(|r| r.post_id()).collect();
        assert_eq!(ids, vec![1, 6]);
        assert!(records
            .iter()
            .all(|r| r.index_name() == POPULAR_VIDEO_INDEX_NAME));
    }
}
