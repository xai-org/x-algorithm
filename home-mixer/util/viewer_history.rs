use arrow::array::{Array, BooleanArray, FixedSizeListArray, Int32Array, Int64Array};
use arrow::ipc::reader::StreamReader;
use std::collections::{HashMap, HashSet};
use xai_recsys_proto::ActionName;
use xai_vm_ranker_proto::experiment::{HistoryTopics, L2TopicCount};

const COL_AUTHOR_ID: &str = "authorId";
const COL_ACTION_NAME_MULTI_HOT: &str = "actionNameMultiHot";
const COL_IMPRESSED_TIME_MS: &str = "impressedTimeMs";
const COL_SEMANTIC_ID: &str = "semanticId";
const COL_POST_LANGUAGE_CODE: &str = "postLanguageCode";

const POSITIVE_ENGAGEMENTS: [ActionName; 18] = [
    ActionName::ServerTweetFav,
    ActionName::ServerTweetReply,
    ActionName::ServerTweetQuote,
    ActionName::ServerTweetRetweet,
    ActionName::ClientTweetClick,
    ActionName::ClientQuotedTweetClick,
    ActionName::ClientTweetOpenLink,
    ActionName::ClientTweetClickProfile,
    ActionName::ClientTweetPhotoExpand,
    ActionName::ClientQuotedTweetPhotoExpand,
    ActionName::ClientTweetVideoOpen,
    ActionName::ClientTweetVideoQualityView,
    ActionName::ClientQuotedTweetVideoQualityView,
    ActionName::ClientTweetShare,
    ActionName::ClientTweetClickSendViaDirectMessage,
    ActionName::ClientTweetShareViaCopyLink,
    ActionName::ClientTweetBookmark,
    ActionName::ClientTweetFollowAuthor,
];

pub fn positively_engaged_author_ids(columnar_sequence: &bytes::Bytes) -> Option<HashSet<u64>> {
    let reader =
        StreamReader::try_new(std::io::Cursor::new(columnar_sequence.as_ref()), None).ok()?;
    let mut authors = HashSet::new();
    for batch in reader {
        let batch = batch.ok()?;
        let author_ids = batch
            .column_by_name(COL_AUTHOR_ID)?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let multi_hot = batch
            .column_by_name(COL_ACTION_NAME_MULTI_HOT)?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()?;
        let hot_values = multi_hot.values().as_any().downcast_ref::<BooleanArray>()?;
        let vocab = multi_hot.value_length() as usize;
        for row in 0..batch.num_rows() {
            if author_ids.is_null(row) || author_ids.value(row) <= 0 {
                continue;
            }
            if is_engaged(hot_values, row * vocab, vocab) {
                authors.insert(author_ids.value(row) as u64);
            }
        }
    }
    Some(authors)
}

pub fn engaged_history_topics(
    columnar_sequence: &bytes::Bytes,
    since_ms: i64,
) -> Option<HistoryTopics> {
    let reader =
        StreamReader::try_new(std::io::Cursor::new(columnar_sequence.as_ref()), None).ok()?;
    let mut l1_counts: HashMap<u32, u32> = HashMap::new();
    let mut l2_counts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut language_counts: HashMap<i32, u32> = HashMap::new();
    let mut engaged_rows = 0;
    let mut sid_rows = 0;
    let mut oldest_row_ms: Option<i64> = None;
    for batch in reader {
        let batch = batch.ok()?;
        let multi_hot = batch
            .column_by_name(COL_ACTION_NAME_MULTI_HOT)?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()?;
        let hot_values = multi_hot.values().as_any().downcast_ref::<BooleanArray>()?;
        let vocab = multi_hot.value_length() as usize;
        let impressed_ms = batch
            .column_by_name(COL_IMPRESSED_TIME_MS)?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let sids = batch
            .column_by_name(COL_SEMANTIC_ID)
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());
        let sid_values = sids.and_then(|s| s.values().as_any().downcast_ref::<Int32Array>());
        let sid_width = sids.map_or(0, |s| s.value_length() as usize);
        let languages = batch
            .column_by_name(COL_POST_LANGUAGE_CODE)
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        for row in 0..batch.num_rows() {
            if impressed_ms.is_null(row) || impressed_ms.value(row) <= 0 {
                continue;
            }
            let ts = impressed_ms.value(row);
            oldest_row_ms = Some(oldest_row_ms.map_or(ts, |oldest| oldest.min(ts)));
            if ts < since_ms || !is_engaged(hot_values, row * vocab, vocab) {
                continue;
            }
            engaged_rows += 1;
            if let Some(languages) = languages
                && languages.is_valid(row)
                && languages.value(row) > 0
            {
                *language_counts.entry(languages.value(row)).or_default() += 1;
            }
            let (Some(sids), Some(values)) = (sids, sid_values) else {
                continue;
            };
            if sid_width < 2 || sids.is_null(row) {
                continue;
            }
            let base = row * sid_width;
            let (Ok(l1), Ok(l2)) = (
                u32::try_from(values.value(base)),
                u32::try_from(values.value(base + 1)),
            ) else {
                continue;
            };
            sid_rows += 1;
            *l1_counts.entry(l1).or_default() += 1;
            *l2_counts.entry((l1, l2)).or_default() += 1;
        }
    }
    Some(HistoryTopics {
        l1_counts,
        l2_counts: l2_counts
            .into_iter()
            .map(|((l1, l2), count)| L2TopicCount { l1, l2, count })
            .collect(),
        engaged_rows,
        sid_rows,
        oldest_row_ms,
        language_counts,
    })
}

fn is_engaged(hot_values: &BooleanArray, base: usize, vocab: usize) -> bool {
    POSITIVE_ENGAGEMENTS.iter().any(|action| {
        let idx = *action as usize;
        idx < vocab && hot_values.value(base + idx)
    })
}
