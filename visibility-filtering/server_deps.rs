use crate::clients::socialgraph_client::ProdSocialgraphClient;
use crate::filter::{FilterRequest, FilterResponse, FilterTweets};
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use crate::hydration::{FallbackCacheMode, HydrationPipeline};
use crate::models::{RawCandidate, TweetId};
use crate::rules::{SafetyLevel, Verdict};
use crate::safety_label_source::lookup::RemoteSource;
use crate::safety_label_source::manhattan::ManhattanSource;
use crate::safety_label_source::twemcache::TwemcacheSource;
use crate::safety_label_source::{ManhattanLabelFetcher, MhLabelClient, SafetyLabelSource};
use crate::server::VFServer;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use xai_core_entities::gizmoduck_client::{GizmoduckClientConfig, ProdGizmoduckClient};
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants, TESRpcConstants};
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_core_entities::tweet_entity_service_client::{ProdTESClient, TESClientConfig};
use xai_x_rpc::balanced_channel::LbPolicy;

const CACHE_PATH: &str = "/s/cache/safety_label_store:twemcaches";
const TES_READINESS_PROBE_PORT: u16 = 8081;
const GIZMODUCK_READINESS_PROBE_PORT: u16 = 8081;

const CLIENT_INIT_RETRY_BUDGET: Duration = Duration::from_secs(240);
const CLIENT_INIT_MAX_BACKOFF: Duration = Duration::from_secs(15);
const CLIENT_INIT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

async fn init_client_with_retry<T, E, Fut>(
    client: &str,
    deadline: tokio::time::Instant,
    mut build: impl FnMut() -> Fut,
) -> Result<T, String>
where
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut backoff = Duration::from_secs(1);
    let mut attempt = 1u32;
    loop {
        if tokio::time::Instant::now() >= deadline {
            error!(
                client,
                attempt, "Client init skipped; init deadline exhausted"
            );
            return Err("init deadline exhausted before attempt".to_string());
        }
        let error = match tokio::time::timeout(CLIENT_INIT_ATTEMPT_TIMEOUT, build()).await {
            Ok(Ok(t)) => {
                if attempt > 1 {
                    info!(client, attempt, "Client init succeeded after retry");
                }
                return Ok(t);
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ),
        };
        if tokio::time::Instant::now() + backoff >= deadline {
            error!(
                client,
                attempt,
                error = %error,
                "Client init failed; retry budget exhausted"
            );
            return Err(error);
        }
        warn!(
            client,
            attempt,
            backoff_secs = backoff.as_secs(),
            error = %error,
            "Client init failed; retrying"
        );
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(CLIENT_INIT_MAX_BACKOFF);
        attempt += 1;
    }
}

pub async fn build_prod_server(datacenter: &str) -> VFServer {
    info!("Initializing prod clients for datacenter={}", datacenter);

    let init_deadline = tokio::time::Instant::now() + CLIENT_INIT_RETRY_BUDGET;

    let deterministic_aperture = std::env::var("APP_ENV").as_deref() == Ok("prod");
    let fallback_cache_serve_stale = crate::config::fallback_cache_serve_stale_enabled();
    let fallback_cache_mode = if fallback_cache_serve_stale {
        FallbackCacheMode::ServeStale
    } else if crate::config::fallback_cache_populate_enabled() {
        FallbackCacheMode::Shadow
    } else {
        FallbackCacheMode::Disabled
    };
    let media_fallback_cache_mode = if crate::config::media_fallback_cache_serve_stale_enabled() {
        FallbackCacheMode::ServeStale
    } else if crate::config::media_fallback_cache_populate_enabled() {
        FallbackCacheMode::Shadow
    } else {
        FallbackCacheMode::Disabled
    };

    let tes_client: Arc<
        dyn xai_core_entities::tweet_entity_service_client::TESClient + Send + Sync,
    > = Arc::new(
        init_client_with_retry("tes", init_deadline, || {
            ProdTESClient::new_with_config(
                None,
                datacenter,
                None,
                tes_client_config(deterministic_aperture),
            )
        })
        .await
        .expect("Failed to initialize TES client"),
    );

    let gizmoduck_client_id = format!(
        "visibility-filtering-service.{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "prod".to_string())
    );
    let gizmoduck_client: Arc<
        dyn xai_core_entities::gizmoduck_client::GizmoduckClient + Send + Sync,
    > = Arc::new(
        init_client_with_retry("gizmoduck", init_deadline, || {
            ProdGizmoduckClient::new_with_config(
                None,
                datacenter,
                Some(gizmoduck_client_id.clone()),
                gizmoduck_client_config(deterministic_aperture),
            )
        })
        .await
        .expect("Failed to initialize Gizmoduck client"),
    );
    info!(
        deterministic_aperture,
        "TES client: aperture + penalized peak-EWMA P2C + readiness probe; Gizmoduck client: \
         aperture + least-request P2C + readiness probe"
    );

    let sg_client: Arc<dyn crate::clients::socialgraph_client::SocialgraphClient + Send + Sync> =
        Arc::new(
            init_client_with_retry("socialgraph", init_deadline, || {
                ProdSocialgraphClient::new(
                    datacenter,
                    &S2S_CHAIN_PATH,
                    &S2S_CRT_PATH,
                    &S2S_KEY_PATH,
                )
            })
            .await
            .expect("Failed to initialize SocialGraph client"),
        );

    let mh_label_client: Arc<dyn ManhattanLabelFetcher> = Arc::new(
        init_client_with_retry("manhattan", init_deadline, || {
            let s2s = xai_manhattan::s2s::S2sConfig {
                client_cert_path: S2S_CRT_PATH.clone(),
                client_key_path: S2S_KEY_PATH.clone(),
                ca_cert_path: S2S_CHAIN_PATH.clone(),
            };
            MhLabelClient::new(datacenter, s2s, deterministic_aperture)
        })
        .await
        .expect("Failed to initialize MhLabelClient"),
    );

    let twemcache = Arc::new(
        init_client_with_retry("twemcache", init_deadline, || {
            crate::twemcache::TwemcacheClient::new_with_tls_paths(
                CACHE_PATH,
                "visibility-filtering-service",
                datacenter,
                &S2S_CHAIN_PATH,
                &S2S_CRT_PATH,
                &S2S_KEY_PATH,
                Duration::from_millis(20),
                Duration::from_secs(5),
                crate::twemcache::client::default_connections_per_host(),
                crate::twemcache::client::default_depth_cap(),
            )
        })
        .await
        .expect("Failed to create twemcache client"),
    );
    info!("Cache client connected to {CACHE_PATH}");

    warm_cache(&twemcache).await;
    warm_manhattan(mh_label_client.as_ref()).await;

    let twemcache_source = Arc::new(TwemcacheSource::new(twemcache));
    let manhattan_source = Arc::new(ManhattanSource::new(mh_label_client));
    let remote = Arc::new(RemoteSource::new(twemcache_source, manhattan_source));
    let safety_label_source = Arc::new(SafetyLabelSource::new(remote));

    let hydration_pipeline = HydrationPipeline::new(
        tes_client,
        gizmoduck_client,
        sg_client,
        safety_label_source.clone(),
        fallback_cache_mode,
        media_fallback_cache_mode,
    );
    let policies = crate::rules::Policies::new();
    let (home_rule_count, recommendations_rule_count) = policies.rule_counts();
    let filter_tweets = FilterTweets::new(hydration_pipeline, policies);

    warm_filter_tweets(&filter_tweets).await;

    info!(
        hydrator_count = 5,
        ?fallback_cache_mode,
        ?media_fallback_cache_mode,
        home_rule_count,
        recommendations_rule_count,
        "VFServer initialized with prod clients"
    );

    VFServer::from_endpoints(
        FilterTweetsEndpoint::new(filter_tweets),
        GetSafetyLabelsEndpoint::new(safety_label_source),
    )
}

const TES_STRATO_REQUEST_TIMEOUT_MS: u64 = 100;

fn tes_client_config(deterministic_aperture: bool) -> TESClientConfig {
    TESClientConfig {
        aperture_size: Some(TESRpcConstants::num_endpoints()),
        deterministic_aperture,
        lb_policy: Some(LbPolicy::penalized_peak_ewma()),
        readiness_probe_port: Some(TES_READINESS_PROBE_PORT),
        request_timeout_ms: Some(TES_STRATO_REQUEST_TIMEOUT_MS),
        enable_serve_within: true,
        ..Default::default()
    }
}

const GIZMODUCK_STRATO_REQUEST_TIMEOUT_MS: u64 = 80;

fn gizmoduck_client_config(deterministic_aperture: bool) -> GizmoduckClientConfig {
    GizmoduckClientConfig {
        aperture_size: Some(GizmoduckRpcConstants::num_endpoints()),
        deterministic_aperture,
        lb_policy: Some(LbPolicy::least_request()),
        readiness_probe_port: Some(GIZMODUCK_READINESS_PROBE_PORT),
        request_timeout_ms: Some(GIZMODUCK_STRATO_REQUEST_TIMEOUT_MS),
        ..Default::default()
    }
}

async fn warm_cache(twemcache: &crate::twemcache::TwemcacheClient) {
    let start = Instant::now();
    let key = match crate::twemcache::Key::new(b"slm_warmup".to_vec()) {
        Ok(key) => key,
        Err(e) => {
            warn!(error = %e, "Cache warmup key construction failed (non-fatal)");
            return;
        }
    };
    let latency_ms = || start.elapsed().as_millis() as u64;
    match twemcache
        .multi_get(std::slice::from_ref(&key))
        .await
        .get(&key)
    {
        Some(Ok(_)) => info!(latency_ms = latency_ms(), "Cache warmup succeeded"),
        Some(Err(e)) => warn!(
            latency_ms = latency_ms(),
            error = %e,
            "Cache warmup failed (non-fatal)"
        ),
        None => warn!(
            latency_ms = latency_ms(),
            "Cache warmup returned no response (non-fatal)"
        ),
    }
}

const WARM_FILTER_TWEETS_TWEET_ID: u64 = 20;
const WARM_FILTER_TWEETS_CONSECUTIVE: u32 = 3;
const WARM_FILTER_TWEETS_DEADLINE: Duration = Duration::from_secs(20);
const WARM_FILTER_TWEETS_SLEEP: Duration = Duration::from_millis(500);

fn warm_filter_tweets_request() -> FilterRequest {
    FilterRequest {
        viewer_id: None,
        country_code: None,
        safety_level: SafetyLevel::TimelineHomeRecommendations,
        candidates: vec![RawCandidate {
            tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
            request_author_id: None,
        }],
    }
}

fn filter_tweets_response_is_warm(response: &FilterResponse) -> bool {
    response
        .outcomes
        .iter()
        .all(|outcome| outcome.verdict.decided_by != Verdict::unresolved_author().decided_by)
}

async fn warm_filter_tweets(filter_tweets: &FilterTweets) {
    let start = Instant::now();
    let mut attempts = 0u32;
    let mut consecutive_warm = 0u32;
    while start.elapsed() < WARM_FILTER_TWEETS_DEADLINE {
        attempts += 1;
        let response = filter_tweets.run(warm_filter_tweets_request()).await;
        if filter_tweets_response_is_warm(&response) {
            consecutive_warm += 1;
            if consecutive_warm >= WARM_FILTER_TWEETS_CONSECUTIVE {
                info!(
                    attempts,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "filter_tweets warmup succeeded"
                );
                return;
            }
        } else {
            consecutive_warm = 0;
        }
        if start.elapsed() + WARM_FILTER_TWEETS_SLEEP < WARM_FILTER_TWEETS_DEADLINE {
            tokio::time::sleep(WARM_FILTER_TWEETS_SLEEP).await;
        } else {
            break;
        }
    }
    warn!(
        attempts,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "filter_tweets warmup deadline expired (non-fatal)"
    );
}

async fn warm_manhattan(manhattan: &dyn ManhattanLabelFetcher) {
    let start = Instant::now();
    match manhattan.fetch_labels(&[0]).await {
        Ok(_) => info!(
            latency_ms = start.elapsed().as_millis() as u64,
            "Manhattan warmup succeeded"
        ),
        Err(e) => warn!(
            latency_ms = start.elapsed().as_millis() as u64,
            error = %e,
            "Manhattan warmup failed (non-fatal)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{FilterOutcome, FilterSummary};
    use crate::models::VfAction;
    use std::cell::Cell;
    use xai_visibility_filtering::models::FilteredReason;

    fn deadline_in(budget: Duration) -> tokio::time::Instant {
        tokio::time::Instant::now() + budget
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_recovers_from_transient_failures() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();

        let result = init_client_with_retry("test", deadline_in(Duration::from_secs(240)), || {
            attempts.set(attempts.get() + 1);
            let n = attempts.get();
            async move {
                if n < 3 {
                    Err("discovery timed out")
                } else {
                    Ok(n)
                }
            }
        })
        .await;

        assert_eq!(result, Ok(3));
        assert_eq!(attempts.get(), 3);
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_gives_up_at_deadline() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();
        let budget = Duration::from_secs(240);

        let result: Result<(), String> =
            init_client_with_retry("test", deadline_in(budget), || {
                attempts.set(attempts.get() + 1);
                async { Err("discovery timed out") }
            })
            .await;

        assert_eq!(result, Err("discovery timed out".to_string()));
        assert!(attempts.get() > 1);
        assert!(start.elapsed() < budget);
        assert!(start.elapsed() >= budget - CLIENT_INIT_MAX_BACKOFF);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_skips_attempt_when_deadline_already_passed() {
        let deadline = tokio::time::Instant::now();
        tokio::time::sleep(Duration::from_secs(1)).await;
        let attempts = Cell::new(0u32);

        let result: Result<(), String> = init_client_with_retry("test", deadline, || {
            attempts.set(attempts.get() + 1);
            async { Err("unreachable") }
        })
        .await;

        assert_eq!(
            result,
            Err("init deadline exhausted before attempt".to_string())
        );
        assert_eq!(attempts.get(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_times_out_hanging_attempt() {
        let start = tokio::time::Instant::now();

        let result: Result<(), String> = init_client_with_retry(
            "test",
            deadline_in(CLIENT_INIT_ATTEMPT_TIMEOUT / 2),
            || async {
                std::future::pending::<()>().await;
                Err("unreachable")
            },
        )
        .await;

        assert_eq!(
            result,
            Err(format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ))
        );
        assert_eq!(start.elapsed(), CLIENT_INIT_ATTEMPT_TIMEOUT);
    }

    fn response(verdicts: Vec<Verdict>) -> FilterResponse {
        let outcomes: Vec<FilterOutcome> = verdicts
            .into_iter()
            .map(|verdict| FilterOutcome {
                tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
                verdict,
                safety_labels: None,
            })
            .collect();
        FilterResponse {
            summary: FilterSummary {
                tweet_count: outcomes.len(),
                drop_count: outcomes
                    .iter()
                    .filter(|outcome| matches!(outcome.verdict.action, VfAction::Drop(_)))
                    .count(),
                unresolved_author_count: outcomes
                    .iter()
                    .filter(|outcome| {
                        outcome.verdict.decided_by == Verdict::unresolved_author().decided_by
                    })
                    .count(),
            },
            outcomes,
        }
    }

    #[test]
    fn allow_verdict_is_warm() {
        let response = response(vec![Verdict {
            action: VfAction::Allow,
            decided_by: None,
        }]);
        assert!(filter_tweets_response_is_warm(&response));
    }

    #[test]
    fn drop_by_real_rule_is_warm() {
        let response = response(vec![Verdict {
            action: VfAction::Drop(FilteredReason::AuthorIsSuspended),
            decided_by: Some("DropSuspendedAuthorRule"),
        }]);
        assert!(filter_tweets_response_is_warm(&response));
    }

    #[test]
    fn unresolved_author_verdict_is_not_warm() {
        let response = response(vec![Verdict::unresolved_author()]);
        assert!(!filter_tweets_response_is_warm(&response));
    }
}
