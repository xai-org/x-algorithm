use super::{
    ENV_IMAGE, ENV_TWEETYPIE_CLIENT_ID, ENV_TWEETYPIE_TLS_DOMAIN, ENV_TWEETYPIE_XDS_LISTENER,
    Sample,
};
use crate::config::ENV_REFERENCE;
use crate::hydration::HYDRATION_TIMEOUT;
use crate::metrics::Metrics;
use crate::metrics::families::{TWEETYPIE_REFERENCE_BATCHES, TWEETYPIE_REFERENCE_COMPARE};
use crate::models::{Evaluation, Placement, TweetId};
use crate::params::LimitedActionsPolicies;
use crate::post_evaluator::PostEvaluation;
use crate::quote::QuotePass;
use crate::result_label::{
    Class, Failure, Label, NONE, failed, label, quote_label, renders_alike, state_label,
};
use crate::rules::SafetyLevel;
use crate::server_deps::init_client_with_retry;
use crate::treatment;
use anyhow::Context;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::spawn;
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tonic::metadata::MetadataMap;
use tracing::warn;
use xai_build_version::current_build_information;
use xai_strato::strato_thrift::{StratoResult, strato_decode};
use xai_strato::{StratoGrpc, encode};
use xai_twittercontext_proto::TwitterContextViewer;
use xai_x_rpc::balanced_channel::LbPolicy;
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::retry::RetryConfig;
use xai_x_rpc::timed_buffer::DEFAULT_BUFFER_MAX_WAIT;
use xai_x_rpc::total_timeout::DEFAULT_TOTAL_TIMEOUT;
use xai_x_thrift::get_tweet_fields::{
    GetTweetFieldsOptions, GetTweetFieldsResult, VISIBILITY_POLICY_USER_VISIBLE,
};
use xai_x_thrift::safety_level::SafetyLevel as ThriftSafetyLevel;
use xai_xds_client::StartFrom;

pub(crate) mod quotes;

const COLUMN: &str = "tweetypie/getTweetFields.Tweet";
const TWEETYPIE_MAX_BATCH_SIZE: usize = 250;
const MAX_IN_FLIGHT: usize = 256;
const TWEETYPIE_TIMEOUT: Duration = Duration::from_millis(1500);
const RECORD_LINE_MARKER: &str = "vf_tweetypie_reference_record";
pub(crate) const SAFETY_LEVEL: SafetyLevel = SafetyLevel::TimelineHomeHydration;
const RECORD_CAP_PER_MINUTE: u32 = 20;
const BEST_EFFORT_DEFAULTS: &str = "best_effort_defaults";
#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum BatchOutcome {
    Compared,
    TweetypieTimeout,
    Busy,
}

#[derive(Clone)]
pub(crate) struct TpResult {
    pub(crate) label: Label,
    result: Option<String>,
    strato_error: Option<String>,
    quoted: Option<Box<TpResult>>,
}

fn tp_failed(failure: Failure) -> TpResult {
    TpResult {
        label: failed(failure),
        result: None,
        strato_error: None,
        quoted: None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum Lens {
    Standalone,
    Outer,
    InnerQuoted,
    QuotePass,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Bucket {
    lens: Lens,
    tp: Label,
    vf: Label,
    vf_rule: &'static str,
    quoted_tweet_rule: &'static str,
    quote_pass: &'static str,
    renders_alike: bool,
}

impl Bucket {
    fn counter_values<'a>(&'a self, build_sha: &'a str) -> [&'a str; 11] {
        [
            SAFETY_LEVEL.into(),
            self.lens.into(),
            &self.tp.0,
            &self.tp.1,
            &self.vf.0,
            &self.vf.1,
            self.vf_rule,
            self.quoted_tweet_rule,
            self.quote_pass,
            if self.renders_alike { "true" } else { "false" },
            build_sha,
        ]
    }
}

pub(super) async fn build(
    init_deadline: Instant,
    sample: Sample,
    metrics: Metrics,
    metadata: Option<&MetadataMap>,
) -> Arc<TweetypieReference> {
    let tweetypie = connect(init_deadline, metadata).await;
    warn!("reference comparator: tweetypie enabled, sample {sample:?}");
    Arc::new(TweetypieReference {
        tweetypie,
        sample,
        metrics,
        permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
        sampler: Mutex::default(),
        build_sha: resolve_build_sha(
            &current_build_information().git_commit_sha,
            env::var(ENV_IMAGE).ok().as_deref(),
        ),
        pod: env::var("HOSTNAME").ok(),
    })
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub(crate) async fn connect(init_deadline: Instant, metadata: Option<&MetadataMap>) -> StratoGrpc {
    let [xds_listener, tls_domain, client_id] = &[
        ENV_TWEETYPIE_XDS_LISTENER,
        ENV_TWEETYPIE_TLS_DOMAIN,
        ENV_TWEETYPIE_CLIENT_ID,
    ]
    .map(required_env);
    init_client_with_retry("tweetypie_reference", init_deadline, || async move {
        let channel = ChannelBuilder::new("tweetypie-xds")
            .tls(
                TlsMode::mtls_from_env()
                    .context("S2S cert env vars required for mTLS")?
                    .with_domain_override(tls_domain.clone()),
            )
            .request_timeout(HYDRATION_TIMEOUT)
            .connect_timeout(Duration::from_millis(400))
            .xds(StartFrom::Lds(xds_listener.clone()))
            .await
            .with_context(|| format!("failed to initialize xDS for {xds_listener}"))?
            .eager_resolution(Duration::from_secs(5))
            .aperture(12)
            .buffer_max_wait(DEFAULT_BUFFER_MAX_WAIT)
            .total_timeout(DEFAULT_TOTAL_TIMEOUT)
            .build_load_balanced(LbPolicy::least_request())
            .await
            .context("failed to build xDS LoadBalancedChannel for tweetypie-xds")?;
        anyhow::Ok(StratoGrpc::from_load_balanced_channel(
            channel,
            metadata.cloned(),
            Some(client_id.clone()),
            Some(RetryConfig::for_idempotent()),
            TWEETYPIE_MAX_BATCH_SIZE,
        ))
    })
    .await
    .expect("Failed to initialize Tweetypie client (reference comparator)")
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
fn required_env(name: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("{name} must be set when {ENV_REFERENCE}=tweetypie"))
}

const BUILD_SHA_LEN: usize = 12;

fn resolve_build_sha(compiled: &str, image: Option<&str>) -> String {
    if let Some(sha) = sha_prefix(compiled) {
        return sha.to_owned();
    }
    if let Some(image) = image
        && let Some(tag) = image.rsplit(':').next()
        && let Some(sha) = sha_prefix(tag)
    {
        return sha.to_owned();
    }
    let mut fallback = compiled.to_owned();
    fallback.truncate(BUILD_SHA_LEN);
    fallback
}

fn sha_prefix(s: &str) -> Option<&str> {
    s.get(..BUILD_SHA_LEN)
        .filter(|prefix| prefix.bytes().all(|b| b.is_ascii_hexdigit()))
}

pub(super) struct TweetypieReference {
    tweetypie: StratoGrpc,
    sample: Sample,
    metrics: Metrics,
    permits: Arc<Semaphore>,
    sampler: Mutex<(u64, HashMap<Bucket, u32>)>,
    build_sha: String,
    pod: Option<String>,
}

impl TweetypieReference {
    fn detach<F>(self: &Arc<Self>, comparison: impl FnOnce(Arc<Self>) -> F)
    where
        F: Future<Output = BatchOutcome> + Send + 'static,
    {
        let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
            self.count(BatchOutcome::Busy);
            return;
        };
        let reference = Arc::clone(self);
        let comparison = comparison(Arc::clone(self));
        spawn(async move {
            let _permit = permit;
            let outcome = comparison.await;
            reference.count(outcome);
        });
    }

    fn record<'a>(&self, comparisons: impl Iterator<Item = Compared<'a>>) {
        let minute = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() / 60);
        let mut counts: HashMap<Bucket, u64> = HashMap::new();
        let mut sampler = self.sampler.lock().unwrap_or_else(PoisonError::into_inner);
        if sampler.0 != minute {
            *sampler = (minute, HashMap::new());
        }
        for compared in comparisons {
            let bucket = compared.bucket();
            let seen = sampler.1.entry(bucket.clone()).or_default();
            *seen += 1;
            if *seen <= RECORD_CAP_PER_MINUTE {
                #[expect(clippy::print_stdout, reason = "stdout is the record sink")]
                {
                    println!(
                        "{}",
                        compared.to_json(&bucket, &self.build_sha, self.pod.as_deref())
                    );
                }
            }
            *counts.entry(bucket).or_default() += 1;
        }
        drop(sampler);
        for (bucket, count) in counts {
            self.metrics.count(
                &TWEETYPIE_REFERENCE_COMPARE,
                bucket.counter_values(&self.build_sha),
                count,
            );
        }
    }

    fn count(&self, outcome: BatchOutcome) {
        self.metrics.count(
            &TWEETYPIE_REFERENCE_BATCHES,
            [SAFETY_LEVEL.into(), outcome.into(), &self.build_sha],
            1,
        );
    }
}

pub(crate) async fn get_tweet_fields(
    tweetypie: &StratoGrpc,
    client: &TwitterContextViewer,
    tweet_ids: &[TweetId],
) -> Vec<TpResult> {
    let view = GetTweetFieldsOptions {
        for_user_id: Some(client.user_id).filter(|&user_id| user_id != 0),
        language_tag: Some("en"),
        safety_level: ThriftSafetyLevel::TIMELINE_HOME_HYDRATION,
        visibility_policy: VISIBILITY_POLICY_USER_VISIBLE,
        ..Default::default()
    };
    let calls = tweet_ids
        .iter()
        .map(|id| {
            (
                COLUMN.to_string(),
                "fetch".to_string(),
                vec![encode(&(id.0.cast_signed(), view))],
            )
        })
        .collect();
    tweetypie
        .batch_call(calls, Some(client))
        .await
        .into_iter()
        .map(|result| match result {
            Ok(bytes) => match strato_decode::<GetTweetFieldsResult>(&bytes) {
                Ok(result) => tp_result(result),
                Err(_) => tp_failed(Failure::DecodeError),
            },
            Err(_) => tp_failed(Failure::RpcError),
        })
        .collect()
}

struct Compared<'a> {
    lens: Lens,
    tweet_id: TweetId,
    viewer_id: u64,
    country_code: Option<&'a str>,
    tp: &'a TpResult,
    vf: Option<&'a PostEvaluation>,
    pass: Option<&'a QuotePass>,
}

impl Compared<'_> {
    fn bucket(&self) -> Bucket {
        let vf = match self.vf {
            None => label(Class::Missing, NONE),
            Some(PostEvaluation {
                evaluation:
                    Evaluation::Complete {
                        fail_open_defaults, ..
                    },
                ..
            }) if !fail_open_defaults.is_empty() => label(Class::Failed, BEST_EFFORT_DEFAULTS),
            Some(outcome) => vf_label(outcome, SAFETY_LEVEL),
        };
        let (vf, quote_pass) = match (self.lens, self.pass) {
            (Lens::QuotePass, Some(pass)) => (
                quote_label(pass.outcome.rendering(), vf),
                (&pass.outcome).into(),
            ),
            (Lens::Standalone | Lens::Outer | Lens::InnerQuoted | Lens::QuotePass, _) => (vf, NONE),
        };
        let allow: &str = Class::Allow.into();
        let (tp, vf) = if self.tp.label.0 == allow && vf.0 == allow {
            (label(Class::Allow, NONE), label(Class::Allow, NONE))
        } else {
            (self.tp.label.clone(), vf)
        };
        let vf_rule = self
            .vf
            .and_then(|outcome| treatment::decided_by(outcome.evaluation.verdict()))
            .unwrap_or(NONE);
        let quoted_tweet_rule = self
            .pass
            .and_then(|pass| pass.author.as_ref())
            .and_then(|evaluation| treatment::decided_by(evaluation.verdict()))
            .unwrap_or(NONE);
        Bucket {
            lens: self.lens,
            renders_alike: renders_alike(&vf, &tp),
            tp,
            vf,
            vf_rule,
            quoted_tweet_rule,
            quote_pass,
        }
    }

    fn to_json(&self, bucket: &Bucket, build_sha: &str, pod: Option<&str>) -> Value {
        let vf_rules: Vec<String> = self
            .vf
            .into_iter()
            .flat_map(|outcome| treatment::decided_rows(outcome.evaluation.verdict()))
            .map(|(rule, action)| format!("{rule}:{}", <&str>::from(action)))
            .collect();
        let labels: BTreeMap<i32, Option<i64>> = self
            .vf
            .and_then(|outcome| outcome.safety_labels.as_ref())
            .into_iter()
            .flat_map(|map| &map.labels)
            .map(|(&label_type, label)| (label_type, label.created_at_msec))
            .collect();
        let outer = self.vf.and_then(|outcome| match outcome.placement {
            Placement::InnerQuoted { outer } => Some(outer.0),
            Placement::Standalone | Placement::RetweetSource => None,
        });
        json!({
            "h": RECORD_LINE_MARKER,
            "v": 1,
            "level": <&str>::from(SAFETY_LEVEL),
            "placement": <&str>::from(self.lens),
            "outer": outer,
            "build": build_sha,
            "pod": pod,
            "tweet": self.tweet_id.0,
            "viewer": self.viewer_id,
            "country": self.country_code,
            "tp": [&bucket.tp.0, &bucket.tp.1],
            "tp_result": self.tp.result,
            "tp_strato_error": self.tp.strato_error,
            "vf": [&bucket.vf.0, &bucket.vf.1],
            "vf_status": self.vf.map(|outcome| vf_status(&outcome.evaluation)),
            "vf_rules": vf_rules,
            "quoted_tweet_rule": bucket.quoted_tweet_rule,
            "quote_pass": bucket.quote_pass,
            "renders_alike": bucket.renders_alike,
            "labels": labels,
        })
    }
}

fn tp_result(result: StratoResult<GetTweetFieldsResult>) -> TpResult {
    match result {
        StratoResult::Ok { value: None, .. } => tp_failed(Failure::NoValue),
        StratoResult::Ok {
            value: Some(value), ..
        } => TpResult {
            label: state_label(&value.tweet_result),
            result: Some(format!("{:?}", value.tweet_result)),
            strato_error: None,
            quoted: value.quoted_tweet_result.map(|quoted| {
                Box::new(TpResult {
                    label: state_label(&quoted),
                    result: Some(format!("{quoted:?}")),
                    strato_error: None,
                    quoted: None,
                })
            }),
        },
        StratoResult::Err { code, message } => TpResult {
            strato_error: Some(format!("{code}: {message}")),
            ..tp_failed(if message.contains("RateLimited") {
                Failure::RateLimited
            } else {
                Failure::StratoError
            })
        },
    }
}

fn vf_status(evaluation: &Evaluation) -> &'static str {
    match evaluation {
        Evaluation::Complete { .. } => "Evaluated",
        Evaluation::Partial { .. } => "Failed",
        Evaluation::NotFound(_) | Evaluation::Deleted(_) => "NotFound",
        Evaluation::Failed(_) => "LookupFailed",
    }
}

fn vf_label(outcome: &PostEvaluation, level: SafetyLevel) -> Label {
    treatment::evaluate_tweets_label(
        &outcome.evaluation,
        level,
        &LimitedActionsPolicies::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{Hydrators, Lookup};
    use crate::metrics::{Metrics, Recorded};
    use crate::models::viewer::Client;
    use crate::models::{Decided, DropReason, Verdict, Withholding};
    use crate::params::ClientSwitches;
    use crate::rules::RuleId;
    use crate::rules::fixtures::{CLIENT_CLASSES, allow};
    use strum::VariantArray;
    use xai_visibility_filtering::models::FilteredReason as VfFilteredReason;
    use xai_x_thrift::action::{
        Action, AnyInterstitial, BlockedViewer, Drop as ThriftDrop, Interstitial,
        InterstitialReason, LimitedEngagementReason, LimitedEngagements, LocalizedMessage,
        TweetInterstitial,
    };
    use xai_x_thrift::get_tweet_fields::{TweetFieldsResultFound, TweetFieldsResultState};
    use xai_x_thrift::safety_result::{FilteredReason, SafetyResult};
    use xai_x_thrift::tweet_service::{
        TweetFieldsResultFailed, TweetFieldsResultFiltered, TweetFieldsResultNotFound,
    };

    const GOLDEN_RECORD: &str = "scripts/tests/fixtures/tweetypie_reference_record.json";

    pub(super) fn outcome(verdict: Verdict) -> PostEvaluation {
        PostEvaluation {
            tweet_id: TweetId(1),
            placement: Placement::Standalone,
            source_tweet_id: None,
            evaluation: Evaluation::complete(verdict),
            safety_labels: None,
        }
    }

    pub(super) fn pair(class: &str, reason: &str) -> Label {
        (class.to_string(), reason.to_string())
    }

    pub(super) fn compared<'a>(tp: &'a TpResult, vf: Option<&'a PostEvaluation>) -> Compared<'a> {
        Compared {
            lens: Lens::Standalone,
            tweet_id: TweetId(1),
            viewer_id: 2,
            country_code: Some("us"),
            tp,
            vf,
            pass: None,
        }
    }

    #[test]
    fn a_post_resting_on_best_effort_defaults_is_labelled_uncomparable() {
        use crate::hydration::Hydrator;
        let tp = TpResult {
            label: pair("allow", NONE),
            result: None,
            strato_error: None,
            quoted: None,
        };
        let vf = PostEvaluation {
            evaluation: Evaluation::Complete {
                verdict: allow(),
                fail_open_defaults: Hydrators::of(Hydrator::Follows),
            },
            ..outcome(allow())
        };
        assert_eq!(
            compared(&tp, Some(&vf)).bucket().vf,
            pair("failed", "best_effort_defaults")
        );
    }

    #[test]
    fn the_record_names_each_evaluation() {
        assert_eq!(
            [
                Evaluation::complete(allow()),
                Evaluation::Partial {
                    verdict: allow(),
                    fail_open_defaults: Hydrators::empty(),
                },
                Evaluation::NotFound(Lookup::Tweet),
                Evaluation::Failed(Lookup::Author),
            ]
            .iter()
            .map(vf_status)
            .collect::<Vec<_>>(),
            ["Evaluated", "Failed", "NotFound", "LookupFailed"]
        );
    }

    #[test]
    fn tweetypie_labels_and_buckets() {
        let safety_result = |action| FilteredReason::SafetyResult(SafetyResult::new(None, action));
        let blocked_viewer = safety_result(Action::LimitedEngagements(LimitedEngagements::new(
            LimitedEngagementReason::BlockedViewer(BlockedViewer::default()),
            None,
            None,
        )));
        let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
        let filtered =
            |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
        let cases = [
            (found(None), ("allow", "none")),
            (
                found(Some(blocked_viewer)),
                ("limited_engagements", "blocked_viewer"),
            ),
            (
                found(Some(FilteredReason::ReportedTweet(true))),
                ("suppressed", "reported_tweet"),
            ),
            (
                filtered(FilteredReason::AuthorIsSuspended(true)),
                ("drop", "author_is_suspended"),
            ),
            (
                filtered(safety_result(Action::Drop(ThriftDrop::new(None, None)))),
                ("bare_drop", "none"),
            ),
            (
                TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(
                    true,
                    None,
                    None::<FilteredReason>,
                )),
                ("not_found", "none"),
            ),
            (
                TweetFieldsResultState::Failed(TweetFieldsResultFailed::new(true, None::<String>)),
                ("failed", "over_capacity"),
            ),
            (
                TweetFieldsResultState::Failed(TweetFieldsResultFailed::new(false, None::<String>)),
                ("failed", "tweetypie_failed"),
            ),
        ];
        for (state, (class, reason)) in cases {
            assert_eq!(state_label(&state), pair(class, reason), "{state:?}");
        }
        let rate_limited = tp_result(StratoResult::Err {
            code: 305,
            message: "ClientError(RateLimited: exceeded the rate limit)".to_string(),
        });
        assert_eq!(rate_limited.label, pair("failed", "rate_limited"));

        let shown = outcome(allow());
        let allow = TpResult {
            label: pair("allow", "safety_result_reason_14"),
            result: None,
            strato_error: None,
            quoted: None,
        };
        assert_eq!(
            compared(&allow, Some(&shown)).bucket(),
            Bucket {
                lens: Lens::Standalone,
                tp: pair("allow", NONE),
                vf: pair("allow", NONE),
                vf_rule: NONE,
                quoted_tweet_rule: NONE,
                quote_pass: NONE,
                renders_alike: true,
            }
        );
        let suspended = outcome(Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(VfFilteredReason::AuthorIsSuspended)),
            id: RuleId::SuspendedAuthor,
            by: "suspended_author/drop",
        }));
        let bucket = compared(&allow, Some(&suspended)).bucket();
        assert_eq!(
            (bucket.vf, bucket.vf_rule),
            (pair("drop", "author_is_suspended"), "suspended_author/drop")
        );
        let not_found = TpResult {
            label: pair("not_found", NONE),
            result: None,
            strato_error: None,
            quoted: None,
        };
        let deleted = PostEvaluation {
            evaluation: Evaluation::NotFound(Lookup::Tweet),
            ..shown
        };
        let bucket = compared(&not_found, Some(&deleted)).bucket();
        assert_eq!(
            (bucket.tp, bucket.vf, bucket.vf_rule),
            (
                pair("not_found", NONE),
                pair("not_found", NONE),
                "not_found"
            )
        );
    }

    #[test]
    fn each_client_resolves_its_fixture_class_capability() {
        let switches = ClientSwitches::for_tests();
        for &client in Client::VARIANTS {
            let name = serde_json::to_value(client).unwrap();
            let class = CLIENT_CLASSES
                .iter()
                .find(|class| name == class.name)
                .unwrap_or_else(|| panic!("no client class named {name}"));
            assert_eq!(
                switches.resolve(
                    Some(&client.context(Some(1), Some("fr"))),
                    Some(1),
                    Some("fr")
                ),
                class.capability,
                "{name}"
            );
        }
    }

    #[test]
    fn a_reason_after_multi_byte_text_keeps_its_label() {
        let message =
            LocalizedMessage::new("Contenu réservé ✓".to_string(), "fr".to_string(), None);
        let action = Action::TweetInterstitial(TweetInterstitial {
            interstitial: Some(AnyInterstitial::Interstitial(Interstitial::new(
                None::<InterstitialReason>,
                message,
            ))),
            limited_engagements: Some(LimitedEngagements::new(
                LimitedEngagementReason::BlockedViewer(BlockedViewer::default()),
                None,
                None,
            )),
            ..TweetInterstitial::default()
        });
        let state = TweetFieldsResultState::Found(TweetFieldsResultFound::new(Some(
            FilteredReason::SafetyResult(SafetyResult::new(None, action)),
        )));
        assert_eq!(
            state_label(&state),
            pair("tweet_interstitial", "blocked_viewer")
        );
    }

    #[test]
    fn each_lens_labels_its_counter_and_record() {
        let tp = TpResult {
            label: pair("tombstone", "bounced"),
            result: None,
            strato_error: None,
            quoted: None,
        };
        for (lens, placement, name, outer) in [
            (Lens::Standalone, Placement::Standalone, "standalone", None),
            (Lens::Outer, Placement::Standalone, "outer", None),
            (
                Lens::InnerQuoted,
                Placement::InnerQuoted { outer: TweetId(7) },
                "inner_quoted",
                Some(7),
            ),
        ] {
            let vf = PostEvaluation {
                placement,
                ..outcome(allow())
            };
            let compared = Compared {
                lens,
                ..compared(&tp, Some(&vf))
            };
            let bucket = compared.bucket();
            assert_eq!(counted(&bucket).label("placement"), Some(name));
            let record = compared.to_json(&bucket, "sha", None);
            assert_eq!(
                (record["placement"].as_str(), record["outer"].as_u64()),
                (Some(name), outer)
            );
        }
    }

    pub(super) fn counted(bucket: &Bucket) -> Recorded {
        let (metrics, recording) = Metrics::recording();
        metrics.count(
            &TWEETYPIE_REFERENCE_COMPARE,
            bucket.counter_values("sha"),
            1,
        );
        let [sample] = recording
            .take(&TWEETYPIE_REFERENCE_COMPARE.family)
            .try_into()
            .unwrap();
        sample
    }

    #[test]
    fn record_and_counter_labels_match_the_golden_fixture() {
        let tp = TpResult {
            label: pair("drop", "author_is_suspended"),
            result: None,
            strato_error: None,
            quoted: None,
        };
        let vf = outcome(allow());
        let compared = compared(&tp, Some(&vf));
        let bucket = compared.bucket();
        let mut emitted = json!({
            "compare_metric": TWEETYPIE_REFERENCE_COMPARE.family.name,
            "compare_labels": TWEETYPIE_REFERENCE_COMPARE.family.labels,
            "record": compared.to_json(&bucket, "abc123def456", Some("xai-vf-service-cand--abcde-0")),
        });
        emitted.sort_all_objects();
        crate::pinned::assert_pinned(
            GOLDEN_RECORD,
            &(serde_json::to_string_pretty(&emitted).unwrap() + "\n"),
        );
    }
}
