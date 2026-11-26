use crate::models::candidate::PostCandidate;
use std::ops::Range;
use xai_proto::engagement_counter::PostEngagers;
use xai_recsys_proto::{PostEngagerFeatures, TweetInfo};
use xai_stats_receiver::global_stats_receiver;

const MAX_POSITIVE_ENGAGERS: usize = 32;
const MAX_NEGATIVE_ENGAGERS: usize = 8;
const MILLIS_PER_SECOND: i64 = 1000;

const SENT_CANDIDATES_METRIC: &str = "PostEngagers.sent_candidates";
const SENT_ENGAGERS_METRIC: &str = "PostEngagers.sent_engagers";

pub fn set_post_engagers(
    tweet_info: &mut TweetInfo,
    record: &PostEngagers,
    viewer_id: u64,
    serve_time_ms: i64,
) {
    let serve_secs = serve_time_ms.div_euclid(MILLIS_PER_SECOND);
    let entry_count = record
        .user_ids
        .len()
        .min(record.action_bits.len())
        .min(record.engaged_at_secs.len());
    let num_positive = (record.num_positive as usize).min(entry_count);
    let mut features = PostEngagerFeatures::default();
    let positives_kept = push_visible_engagers(
        &mut features,
        record,
        0..num_positive,
        MAX_POSITIVE_ENGAGERS,
        viewer_id,
        serve_secs,
    );
    push_visible_engagers(
        &mut features,
        record,
        num_positive..entry_count,
        MAX_NEGATIVE_ENGAGERS,
        viewer_id,
        serve_secs,
    );
    features.num_positive = positives_kept as u32;
    tweet_info.post_engagers = Some(features);
}

fn push_visible_engagers(
    features: &mut PostEngagerFeatures,
    record: &PostEngagers,
    entries: Range<usize>,
    max_engagers: usize,
    viewer_id: u64,
    serve_secs: i64,
) -> usize {
    let visible_entries = entries
        .filter(|&entry| {
            record.user_ids[entry] != viewer_id
                && i64::from(record.engaged_at_secs[entry]) <= serve_secs
        })
        .take(max_engagers);
    let mut kept = 0;
    for entry in visible_entries {
        let age_secs = serve_secs - i64::from(record.engaged_at_secs[entry]);
        features.user_ids.push(record.user_ids[entry]);
        features.action_bits.push(record.action_bits[entry]);
        features
            .age_secs
            .push(age_secs.clamp(0, i64::from(i32::MAX)) as u32);
        kept += 1;
    }
    kept
}

pub fn clear_post_engagers(tweet_info: &mut TweetInfo) {
    tweet_info.post_engagers = None;
}

pub fn record_sent_post_engagers(candidates: &[PostCandidate], tweet_infos: &[TweetInfo]) {
    let mut hydrated = 0u64;
    let mut with_engagers = 0u64;
    let mut positives = 0u64;
    let mut negatives = 0u64;
    for (candidate, tweet_info) in candidates.iter().zip(tweet_infos) {
        if candidate.post_engagers.is_none() {
            continue;
        }
        let (engagers, engager_positives) =
            tweet_info
                .post_engagers
                .as_ref()
                .map_or((0, 0), |features| {
                    (
                        features.user_ids.len() as u64,
                        u64::from(features.num_positive),
                    )
                });
        hydrated += 1;
        with_engagers += u64::from(engagers > 0);
        positives += engager_positives;
        negatives += engagers - engager_positives;
    }
    if hydrated == 0 {
        return;
    }
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    receiver.incr(
        SENT_CANDIDATES_METRIC,
        &[("engagers", "present")],
        with_engagers,
    );
    receiver.incr(
        SENT_CANDIDATES_METRIC,
        &[("engagers", "absent")],
        hydrated - with_engagers,
    );
    receiver.incr(SENT_ENGAGERS_METRIC, &[("group", "positive")], positives);
    receiver.incr(SENT_ENGAGERS_METRIC, &[("group", "negative")], negatives);
}
