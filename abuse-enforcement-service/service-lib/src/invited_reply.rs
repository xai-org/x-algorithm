use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use serde_json::Value;
use tracing::warn;
#[allow(deprecated)]
use xai_strato::Strato;

use crate::facts::InvitedReply;
use crate::metrics;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentPost {
    pub typed_mentions: Vec<i64>,
}

impl ParentPost {
    pub fn parse(v: &Value) -> Option<Self> {
        v.get("coreData")?.as_object()?;
        let visible_from =
            id(v.get("visibleTextRange").and_then(|r| r.get("fromIndex"))).unwrap_or(0);
        let typed_mentions = v
            .get("mentions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|m| id(m.get("fromIndex")).unwrap_or(0) >= visible_from)
            .filter_map(|m| id(m.get("user")))
            .filter(|u| *u > 0)
            .collect();
        Some(Self { typed_mentions })
    }

    pub fn answer(&self, replier: i64) -> InvitedReply {
        if self.typed_mentions.contains(&replier) {
            InvitedReply::Mentioned
        } else {
            InvitedReply::NotInvited
        }
    }
}

fn id(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

const KNOWN_TTL: Duration = Duration::from_secs(600);
pub const FAILED_READ_TTL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
enum Read {
    Found(Arc<ParentPost>),
    Missing,
    Failed,
}

type InFlight = Arc<Mutex<HashMap<i64, Shared<BoxFuture<'static, Read>>>>>;

#[derive(Clone)]
#[allow(deprecated)]
pub struct ReplyParents {
    strato: Arc<Strato>,
    known: moka::sync::Cache<i64, Option<Arc<ParentPost>>>,
    failed: moka::sync::Cache<i64, ()>,
    in_flight: InFlight,
    timeout: Duration,
}

impl ReplyParents {
    #[allow(deprecated)]
    pub fn new(strato: Arc<Strato>, timeout: Duration) -> Self {
        Self {
            strato,
            known: moka::sync::Cache::builder()
                .max_capacity(200_000)
                .time_to_live(KNOWN_TTL)
                .build(),
            failed: moka::sync::Cache::builder()
                .max_capacity(50_000)
                .time_to_live(FAILED_READ_TTL)
                .build(),
            in_flight: InFlight::default(),
            timeout,
        }
    }

    pub async fn invited(&self, parent_id: i64, replier: i64) -> InvitedReply {
        if parent_id <= 0 {
            return InvitedReply::NotAReply;
        }
        let answer = match self.parent(parent_id).await {
            Some(parent) => parent.answer(replier),
            None => InvitedReply::Unknown,
        };
        metrics::INVITED_REPLY_TOTAL
            .with_label_values(&[answer.as_str()])
            .inc();
        answer
    }

    async fn parent(&self, parent_id: i64) -> Option<Arc<ParentPost>> {
        if let Some(known) = self.known.get(&parent_id) {
            return known;
        }
        if self.failed.contains_key(&parent_id) {
            return None;
        }
        let pending = {
            let mut in_flight = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            in_flight
                .entry(parent_id)
                .or_insert_with(|| {
                    read(self.strato.clone(), parent_id, self.timeout)
                        .boxed()
                        .shared()
                })
                .clone()
        };
        let parent = match pending.await {
            Read::Found(parent) => {
                self.known.insert(parent_id, Some(parent.clone()));
                Some(parent)
            }
            Read::Missing => {
                self.known.insert(parent_id, None);
                None
            }
            Read::Failed => {
                self.failed.insert(parent_id, ());
                None
            }
        };
        self.in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&parent_id);
        parent
    }
}

#[allow(deprecated)]
async fn read(strato: Arc<Strato>, parent_id: i64, timeout: Duration) -> Read {
    let fetched = tokio::time::timeout(
        timeout,
        crate::strato::strato_fetch::<i64, (), Value>(&strato, parent_id, None, "reply_parent"),
    )
    .await;
    match fetched {
        Ok(Ok(Some(v))) => {
            ParentPost::parse(&v).map_or(Read::Missing, |p| Read::Found(Arc::new(p)))
        }
        Ok(Ok(None)) => Read::Missing,
        Ok(Err(e)) => {
            warn!(parent_id, "reply parent read failed: {e:#}");
            Read::Failed
        }
        Err(_) => {
            warn!(parent_id, "reply parent read timed out");
            Read::Failed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPLIER: i64 = 1_000_000_001;
    const POSTER: i64 = 1_000_000_002;
    const OTHER: i64 = 1_000_000_003;

    fn record(prefix: &[i64], typed: &[i64], reply_to: Option<i64>) -> Value {
        let mut mentions = Vec::new();
        let mut at = 0;
        for u in prefix {
            mentions.push(
                serde_json::json!({"fromIndex": at, "toIndex": at + 10, "user": u.to_string()}),
            );
            at += 11;
        }
        let visible = at;
        for u in typed {
            at += 5;
            mentions.push(
                serde_json::json!({"fromIndex": at, "toIndex": at + 10, "user": u.to_string()}),
            );
            at += 11;
        }
        let mut core = serde_json::json!({"user": POSTER.to_string(), "text": "x"});
        if let Some(r) = reply_to {
            core["reply"] =
                serde_json::json!({"inReplyToStatus": "7", "inReplyToUser": r.to_string()});
        }
        serde_json::json!({
            "coreData": core,
            "mentions": mentions,
            "visibleTextRange": {"fromIndex": visible, "toIndex": at + 20},
        })
    }

    fn answer(v: &Value) -> InvitedReply {
        ParentPost::parse(v).unwrap().answer(REPLIER)
    }

    #[test]
    fn a_mention_typed_in_the_visible_text_invites() {
        assert_eq!(
            answer(&record(&[], &[REPLIER], None)),
            InvitedReply::Mentioned
        );
        assert_eq!(
            answer(&record(&[OTHER, REPLIER], &[OTHER, REPLIER], Some(OTHER))),
            InvitedReply::Mentioned,
            "named in the prefix and typed again"
        );
        assert_eq!(
            answer(&record(&[OTHER], &[OTHER, REPLIER], Some(OTHER))),
            InvitedReply::Mentioned
        );
    }

    #[test]
    fn a_thank_you_that_tags_the_replier_in_its_text_invites() {
        let v = serde_json::json!({
            "coreData": {
                "user": POSTER.to_string(),
                "reply": {"inReplyToStatus": "5", "inReplyToUser": POSTER.to_string()},
            },
            "mentions": [
                {"fromIndex": 0, "toIndex": 15, "user": REPLIER.to_string()},
                {"fromIndex": 16, "toIndex": 28, "user": OTHER.to_string()},
                {"fromIndex": 39, "toIndex": 54, "user": REPLIER.to_string()},
            ],
            "visibleTextRange": {"fromIndex": 29, "toIndex": 276},
        });
        assert_eq!(answer(&v), InvitedReply::Mentioned);
    }

    #[test]
    fn a_parent_that_only_replies_to_the_replier_does_not_invite() {
        assert_eq!(
            answer(&record(&[REPLIER], &[], Some(REPLIER))),
            InvitedReply::NotInvited
        );
        assert_eq!(
            answer(&record(&[], &[OTHER], Some(REPLIER))),
            InvitedReply::NotInvited
        );
    }

    #[test]
    fn a_mention_only_in_the_automatic_prefix_does_not_invite() {
        assert_eq!(
            answer(&record(&[OTHER, REPLIER], &[], Some(OTHER))),
            InvitedReply::NotInvited
        );
        assert_eq!(
            answer(&record(&[OTHER], &[OTHER], Some(OTHER))),
            InvitedReply::NotInvited
        );
        assert_eq!(answer(&record(&[], &[], None)), InvitedReply::NotInvited);
    }

    #[test]
    fn a_record_without_ranges_counts_every_mention_as_typed() {
        let v = serde_json::json!({
            "coreData": {"user": POSTER},
            "mentions": [{"fromIndex": 0, "user": REPLIER}],
        });
        assert_eq!(answer(&v), InvitedReply::Mentioned);
    }

    #[test]
    fn a_record_without_core_data_is_unreadable() {
        assert_eq!(
            ParentPost::parse(&serde_json::json!({"mentions": []})),
            None
        );
        assert_eq!(ParentPost::parse(&Value::Null), None);
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn concurrent_missing_and_failed_reads_reach_the_column_once() {
        use wiremock::matchers::{body_string_contains, method};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        const FOUND: i64 = 7_000_001;
        const MISSING: i64 = 7_000_002;
        const FAILING: i64 = 7_000_003;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains(FOUND.to_string()))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"v": record(&[], &[REPLIER], None)}))
                    .set_delay(Duration::from_millis(200)),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_string_contains(MISSING.to_string()))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"v": null})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_string_contains(FAILING.to_string()))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let strato = Strato::with_client(
            "reply_parent",
            server.uri(),
            Strato::build_default_client(&Default::default()).unwrap(),
        );
        let parents = ReplyParents::new(Arc::new(strato), Duration::from_millis(500));
        let reads = |id: i64| {
            let server = &server;
            async move {
                server
                    .received_requests()
                    .await
                    .unwrap_or_default()
                    .iter()
                    .filter(|r| String::from_utf8_lossy(&r.body).contains(&id.to_string()))
                    .count()
            }
        };

        let together =
            futures::future::join_all((0..5).map(|_| parents.invited(FOUND, REPLIER))).await;
        assert!(
            together.iter().all(|a| *a == InvitedReply::Mentioned),
            "{together:?}"
        );
        assert_eq!(
            parents.invited(FOUND, OTHER).await,
            InvitedReply::NotInvited
        );
        assert_eq!(reads(FOUND).await, 1);

        for _ in 0..3 {
            assert_eq!(
                parents.invited(MISSING, REPLIER).await,
                InvitedReply::Unknown
            );
        }
        assert_eq!(reads(MISSING).await, 1);

        assert_eq!(
            parents.invited(FAILING, REPLIER).await,
            InvitedReply::Unknown
        );
        let first = reads(FAILING).await;
        assert!(first >= 1);
        for _ in 0..3 {
            assert_eq!(
                parents.invited(FAILING, REPLIER).await,
                InvitedReply::Unknown
            );
        }
        assert_eq!(
            reads(FAILING).await,
            first,
            "a failed parent waits out its retry window"
        );
        assert_eq!(parents.invited(0, REPLIER).await, InvitedReply::NotAReply);
    }

    #[test]
    fn ids_read_as_strings_or_numbers_and_bad_ones_are_dropped() {
        let v = serde_json::json!({
            "coreData": {"reply": {"inReplyToUser": "not a number"}},
            "mentions": [{"fromIndex": "0", "user": "0"}, {"fromIndex": 3, "user": -4}, {"user": OTHER}],
        });
        let p = ParentPost::parse(&v).unwrap();
        assert_eq!(p.typed_mentions, [OTHER]);
    }
}
