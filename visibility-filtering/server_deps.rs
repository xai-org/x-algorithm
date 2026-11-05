use crate::clients::about_this_account_client::AboutThisAccountClient;
use crate::clients::article_client::ArticleClient;
use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::clients::label_warm_client::StratoWarmFetcher;
use crate::clients::socialgraph_client::ProdSocialgraphClient;
use crate::clients::trusted_friends_client::TrustedFriendsClient;
use crate::clients::user_devices_client::UserDevicesClient;
use crate::clients::user_location_client::UserLocationClient;
use crate::clients::wingman_client::ProdWingmanClient;
use crate::evaluate_tweets::{EntryComparator, EvaluateTweetsEndpoint};
use crate::filter_notifications::FilterNotificationsEndpoint;
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use crate::hydration::community_source::CommunitySource;
use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::sources::{Backends, ProdSources};
use crate::hydration::tweet_source::TweetSource;
use crate::hydration::{author_fallback_cache, tweet_fallback_cache, Lookup};
use crate::metrics::{Endpoint, Metrics, RequestLevel};
use crate::models::{
    Evaluation, PlacedCandidate, RawCandidate, RequestFacts, RetweetMerge, TweetId,
};
use crate::params::limited_actions_copy::LimitedActionsCopy;
use crate::params::{ClientSwitches, CountryLists};
use crate::post_evaluator::{PostBatch, PostEvaluation, PostEvaluator};
use crate::quote::QuotePassRunner;
use crate::rules::{NoTrace, SafetyLevel};
use crate::server::VFServer;
use anyhow::Context;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tonic::metadata::MetadataMap;
use tracing::{error, info, warn};
use xai_core_entities::gizmoduck_client::ProdGizmoduckClient;
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants, TESRpcConstants};
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CLIENT_ID, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_core_entities::tweet_entity_service_client::ProdTESClient;
use xai_safety_label_source::{
    LabelClients, LabelSourceConfig, S2sPaths, SafetyLabelSource, WarmFetcher,
};
use xai_stats_receiver::StatsReceiverExt;
use xai_strato::StratoGrpc;
use xai_x_rpc::balanced_channel::LbPolicy;
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::retry::RetryConfig;
use xai_x_rpc::timed_buffer::DEFAULT_BUFFER_MAX_WAIT;
use xai_x_rpc::total_timeout::DEFAULT_TOTAL_TIMEOUT;
use xai_xds_client::StartFrom;

const READINESS_PROBE_PORT: u16 = 8081;

pub(crate) const CLIENT_INIT_RETRY_BUDGET: Duration = Duration::from_secs(240);
const CLIENT_INIT_MAX_BACKOFF: Duration = Duration::from_secs(15);
const CLIENT_INIT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) async fn init_client_with_retry<T, E, Fut>(
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
            return Err(format!("{client}: init deadline exhausted before attempt"));
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
            return Err(format!("{client}: {error}"));
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

pub(crate) struct ServerDeps {
    pub(crate) init_deadline: tokio::time::Instant,
    pub(crate) post_evaluator: Arc<PostEvaluator>,
    pub(crate) client_switches: ClientSwitches,
    safety_label_source: Arc<SafetyLabelSource>,
    pub(crate) metrics: Metrics,
}

#[derive(Clone, Copy)]
pub(crate) enum CacheWarming {
    AsConfigured,
    Off,
}

impl CacheWarming {
    fn warmer(
        self,
        build: impl FnOnce() -> Option<Arc<dyn WarmFetcher>>,
    ) -> Option<Arc<dyn WarmFetcher>> {
        match self {
            Self::AsConfigured => build(),
            Self::Off => None,
        }
    }

    fn label_cache_eds(self) -> Option<String> {
        match self {
            Self::AsConfigured => crate::config::label_cache_eds(),
            Self::Off => None,
        }
    }
}

pub(crate) struct Serving {
    pub(crate) init_deadline: tokio::time::Instant,
    pub(crate) sources: ProdSources,
    pub(crate) fallback_caches: FallbackCaches,
    pub(crate) safety_label_source: Arc<SafetyLabelSource>,
    pub(crate) client_switches: ClientSwitches,
    pub(crate) country_lists: Arc<CountryLists>,
    pub(crate) metrics: Metrics,
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub(crate) async fn serving(
    datacenter: &str,
    metadata: Option<&MetadataMap>,
    cache_warming: CacheWarming,
    experiments: crate::params::Experiments,
) -> Serving {
    let stats = xai_stats_receiver::global_stats_receiver();
    let metrics = Metrics::new(stats.clone());
    let (limited_actions_copy, copy_drift) = LimitedActionsCopy::load_baked(datacenter, &metrics);
    let init_deadline = tokio::time::Instant::now() + CLIENT_INIT_RETRY_BUDGET;
    let author_cache_capacity = crate::config::author_cache_capacity();
    let tweet_cache_capacity = crate::config::tweet_cache_capacity();
    let (sources, safety_label_source) = prod_sources(
        datacenter,
        init_deadline,
        deterministic_aperture(),
        metadata,
        cache_warming,
        stats,
    )
    .await;
    let fallback_caches = FallbackCaches {
        tweets: tweet_cache_capacity.map(tweet_fallback_cache),
        authors: author_cache_capacity.map(author_fallback_cache),
    };
    let mut switch_files =
        crate::params::SwitchFiles::beside(&crate::config::fs_path(), experiments);
    let feature_switches = Arc::new(arc_swap::ArcSwap::from_pointee(
        switch_files
            .load(&metrics)
            .expect("files that each built an engine build one together"),
    ));
    let country_lists = Arc::new(CountryLists::starting_at_default());
    country_lists.refresh(&feature_switches.load());
    safety_label_source.set_label_cache_switches(crate::params::label_cache_switches(
        &feature_switches.load(),
    ));
    let client_switches = ClientSwitches::new(Arc::clone(&feature_switches), limited_actions_copy);
    crate::params::spawn_refresh(
        switch_files,
        feature_switches,
        Arc::clone(&country_lists),
        Arc::clone(&safety_label_source),
        copy_drift,
        metrics.clone(),
    );
    Serving {
        init_deadline,
        sources,
        fallback_caches,
        safety_label_source,
        client_switches,
        country_lists,
        metrics,
    }
}

pub(crate) async fn build(datacenter: &str, metadata: Option<&MetadataMap>) -> ServerDeps {
    info!("Initializing prod clients for datacenter={}", datacenter);
    let Serving {
        init_deadline,
        sources,
        fallback_caches,
        safety_label_source,
        client_switches,
        country_lists,
        metrics,
    } = serving(
        datacenter,
        metadata,
        CacheWarming::AsConfigured,
        crate::params::Experiments::serving(datacenter).await,
    )
    .await;
    let rule_engine = crate::rules::RuleEngine::with_country_lists(country_lists);
    let post_evaluator = Arc::new(PostEvaluator::new(
        Arc::new(sources),
        rule_engine,
        Arc::new(fallback_caches),
        metrics.clone(),
    ));

    warm_post_evaluator(&post_evaluator).await;

    ServerDeps {
        init_deadline,
        post_evaluator,
        client_switches,
        safety_label_source,
        metrics,
    }
}

fn deterministic_aperture() -> bool {
    std::env::var(crate::config::ENV_APP_ENV).as_deref() == Ok("prod")
}

impl ServerDeps {
    pub(crate) fn into_server(self, comparator: Option<Box<dyn EntryComparator>>) -> VFServer {
        VFServer {
            evaluate_tweets: EvaluateTweetsEndpoint::new(
                Arc::clone(&self.post_evaluator),
                self.client_switches.clone(),
                comparator,
                QuotePassRunner::new(crate::config::quote_pass()),
            ),
            filter_notifications: FilterNotificationsEndpoint::new(Arc::clone(
                &self.post_evaluator,
            )),
            filter_tweets: FilterTweetsEndpoint::new(self.post_evaluator, self.client_switches),
            get_safety_labels: GetSafetyLabelsEndpoint::new(self.safety_label_source, self.metrics),
        }
    }
}

#[expect(clippy::panic, reason = "startup fail-fast: init failure is fatal")]
pub(crate) async fn prod_sources(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
    deterministic_aperture: bool,
    metadata: Option<&MetadataMap>,
    cache_warming: CacheWarming,
    stats: Option<Arc<dyn StatsReceiverExt>>,
) -> (ProdSources, Arc<SafetyLabelSource>) {
    let tes = init_client_with_retry("tes", init_deadline, || async move {
        let strato = build_xds_strato(
            XdsStratoParams {
                name: "tes-xds",
                xds_listener: "tweet-entity-service.prod.tweet-entity-service:fed-grpc",
                tls_domain: format!(
                    "tweet-entity-service.tweet-entity-service.prod.{datacenter}.s2s.twttr.net"
                ),
                lb_policy: LbPolicy::penalized_peak_ewma(),
                client_id: S2S_CLIENT_ID.clone(),
                retry_config: Some(RetryConfig::for_idempotent()),
                max_batch_size: TESRpcConstants::max_batch_size(),
                metadata: metadata.cloned(),
            },
            deterministic_aperture,
        )
        .await?;
        anyhow::Ok(ProdTESClient {
            grpc_client: Arc::new(strato),
        })
    });

    let gizmoduck_client_id = crate::config::gizmoduck_client_id();
    let gizmoduck = init_client_with_retry("gizmoduck", init_deadline, || {
        let client_id = gizmoduck_client_id.clone();
        async move {
            let strato = build_xds_strato(
                XdsStratoParams {
                    name: "gizmoduck-xds",
                    xds_listener: "gizmoduck.prod.gizmoduck:fed-grpc",
                    tls_domain: format!("gizmoduck.gizmoduck.prod.{datacenter}.s2s.twttr.net"),
                    lb_policy: LbPolicy::least_request(),
                    client_id,
                    retry_config: None,
                    max_batch_size: GizmoduckRpcConstants::max_batch_size(),
                    metadata: metadata.cloned(),
                },
                deterministic_aperture,
            )
            .await?;
            anyhow::Ok(ProdGizmoduckClient {
                grpc_client: Arc::new(strato),
            })
        }
    });

    let socialgraph = init_client_with_retry("socialgraph", init_deadline, || {
        ProdSocialgraphClient::new(
            datacenter,
            &S2S_CHAIN_PATH,
            &S2S_CRT_PATH,
            &S2S_KEY_PATH,
            deterministic_aperture,
            metadata.cloned(),
        )
    });

    let stratoserver = init_client_with_retry("stratoserver", init_deadline, || {
        build_xds_strato(
            XdsStratoParams {
                name: "stratoserver-xds",
                xds_listener: "stratoserver.prod.stratostore:grpc",
                tls_domain: format!("stratostore.stratoserver.prod.{datacenter}.s2s.twttr.net"),
                lb_policy: LbPolicy::least_request(),
                client_id: S2S_CLIENT_ID.clone(),
                retry_config: None,
                max_batch_size: STRATOSERVER_MAX_BATCH_SIZE,
                metadata: metadata.cloned(),
            },
            deterministic_aperture,
        )
    });

    let wingman = init_client_with_retry("wingman", init_deadline, || {
        ProdWingmanClient::new(datacenter)
    });

    let labels = async {
        LabelClients::connect(LabelSourceConfig {
            datacenter: datacenter.to_owned(),
            s2s: S2sPaths {
                cert: S2S_CRT_PATH.clone(),
                key: S2S_KEY_PATH.clone(),
                ca_chain: S2S_CHAIN_PATH.clone(),
            },
            deterministic_aperture,
            twemcache_client_name: crate::config::twemcache_client_name(),
            label_cache_eds: cache_warming.label_cache_eds(),
            cache_capacity: crate::config::safety_label_cache_capacity(),
            stats,
            init_deadline,
        })
        .await
        .map_err(|e| e.to_string())
    };

    let connecting = Box::pin(async {
        tokio::join!(tes, gizmoduck, socialgraph, stratoserver, wingman, labels)
    });
    let (tes, gizmoduck, socialgraph, stratoserver, wingman, labels) = match connecting.await {
        (Ok(tes), Ok(gizmoduck), Ok(socialgraph), Ok(stratoserver), Ok(wingman), Ok(labels)) => {
            (tes, gizmoduck, socialgraph, stratoserver, wingman, labels)
        }
        (tes, gizmoduck, socialgraph, stratoserver, wingman, labels) => {
            let failures: Vec<String> = [
                tes.err(),
                gizmoduck.err(),
                socialgraph.err(),
                stratoserver.err(),
                wingman.err(),
                labels.err(),
            ]
            .into_iter()
            .flatten()
            .collect();
            panic!(
                "Failed to initialize backend clients: {}",
                failures.join("; ")
            )
        }
    };
    info!(deterministic_aperture, "Backend clients ready");

    let tes_client = Arc::new(tes);
    let gizmoduck_client: Arc<
        dyn xai_core_entities::gizmoduck_client::GizmoduckClient + Send + Sync,
    > = Arc::new(gizmoduck);
    let sg_client: Arc<dyn crate::clients::socialgraph_client::SocialgraphClient + Send + Sync> =
        Arc::new(socialgraph);
    let wingman_client = Arc::new(wingman);

    let communities = CommunitySource {
        grpc_client: stratoserver.clone(),
    };
    let about_this_account_client = AboutThisAccountClient::new(stratoserver.clone());
    let trusted_friends_client = TrustedFriendsClient::new(stratoserver.clone());
    let user_location_client = UserLocationClient::new(stratoserver.clone());
    let article_client = ArticleClient::new(stratoserver.clone());
    let user_devices_client = UserDevicesClient::new(stratoserver.clone());

    let safety_label_source =
        Arc::new(labels.into_source(cache_warming.warmer(|| warm_fetcher(stratoserver))));

    let tweet_source = TweetSource {
        grpc_client: Arc::clone(&tes_client.grpc_client),
    };
    let sources = ProdSources::new(Backends {
        tes: tes_client,
        tweets: tweet_source,
        gizmoduck: GizmoduckLookup::new(gizmoduck_client),
        socialgraph: sg_client,
        about_this_account: about_this_account_client,
        wingman: wingman_client,
        articles: article_client,
        trusted_friends: trusted_friends_client,
        user_location: user_location_client,
        safety_labels: Arc::clone(&safety_label_source),
        communities,
        user_devices: user_devices_client,
    });
    (sources, safety_label_source)
}

fn warm_fetcher(stratoserver: StratoGrpc) -> Option<Arc<dyn WarmFetcher>> {
    crate::config::cache_warm_enabled()
        .then(|| Arc::new(StratoWarmFetcher::new(stratoserver)) as Arc<dyn WarmFetcher>)
}

const STRATO_REQUEST_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const STRATO_CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
const STRATO_APERTURE_SIZE: usize = 32;
const STRATOSERVER_MAX_BATCH_SIZE: usize = 250;
const XDS_EAGER_RESOLUTION_TIMEOUT: Duration = Duration::from_secs(5);

struct XdsStratoParams {
    name: &'static str,
    xds_listener: &'static str,
    tls_domain: String,
    lb_policy: LbPolicy,
    client_id: String,
    retry_config: Option<RetryConfig>,
    max_batch_size: usize,
    metadata: Option<MetadataMap>,
}

async fn build_xds_strato(
    params: XdsStratoParams,
    deterministic_aperture: bool,
) -> anyhow::Result<StratoGrpc> {
    let mut builder = ChannelBuilder::new(params.name)
        .tls(
            TlsMode::mtls_from_env()
                .context("S2S cert env vars required for mTLS")?
                .with_domain_override(params.tls_domain),
        )
        .request_timeout(STRATO_REQUEST_TIMEOUT)
        .connect_timeout(STRATO_CONNECT_TIMEOUT)
        .xds(StartFrom::Lds(params.xds_listener.to_string()))
        .await
        .with_context(|| format!("failed to initialize xDS for {}", params.xds_listener))?
        .eager_resolution(XDS_EAGER_RESOLUTION_TIMEOUT)
        .aperture(STRATO_APERTURE_SIZE)
        .readiness_probe(READINESS_PROBE_PORT, "/ready")
        .buffer_max_wait(DEFAULT_BUFFER_MAX_WAIT)
        .total_timeout(DEFAULT_TOTAL_TIMEOUT);
    if deterministic_aperture {
        builder = builder.deterministic();
    }

    let channel = builder
        .build_load_balanced(params.lb_policy)
        .await
        .with_context(|| {
            format!(
                "failed to build xDS LoadBalancedChannel for {}",
                params.name
            )
        })?;

    Ok(StratoGrpc::from_load_balanced_channel(
        channel,
        params.metadata,
        Some(params.client_id),
        params.retry_config,
        params.max_batch_size,
    ))
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

const WARM_FILTER_TWEETS_TWEET_ID: u64 = 20;
const WARM_FILTER_TWEETS_CONSECUTIVE: u32 = 3;
const WARM_FILTER_TWEETS_DEADLINE: Duration = Duration::from_secs(20);
const WARM_FILTER_TWEETS_SLEEP: Duration = Duration::from_millis(500);

fn warm_filter_tweets_request() -> PostBatch {
    PostBatch {
        facts: RequestFacts::default(),
        safety_level: SafetyLevel::TimelineHomeRecommendations,
        candidates: vec![PlacedCandidate::standalone(RawCandidate {
            tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
            request_author_id: None,
        })],
        retweet_merge: RetweetMerge::Unmerged,
    }
}

fn filter_tweets_response_is_warm(outcomes: &[PostEvaluation]) -> bool {
    outcomes.iter().all(|outcome| match outcome.evaluation {
        Evaluation::NotFound(lookup) | Evaluation::Failed(lookup) => match lookup {
            Lookup::Tweet => false,
            Lookup::Author | Lookup::SharedTweet | Lookup::SharedAuthor => true,
        },
        Evaluation::Deleted(_) => false,
        Evaluation::Complete { .. } | Evaluation::Partial { .. } => true,
    })
}

async fn warm_post_evaluator(post_evaluator: &PostEvaluator) {
    let start = Instant::now();
    let mut attempts = 0u32;
    let mut consecutive_warm = 0u32;
    while start.elapsed() < WARM_FILTER_TWEETS_DEADLINE {
        attempts += 1;
        let request = warm_filter_tweets_request();
        let metrics = post_evaluator.metrics().request(
            Endpoint::FilterTweets,
            RequestLevel::Named(request.safety_level),
        );
        let (_, outcomes) = post_evaluator.run(request, &metrics, &mut NoTrace).await;
        if filter_tweets_response_is_warm(&outcomes) {
            consecutive_warm += 1;
            if consecutive_warm >= WARM_FILTER_TWEETS_CONSECUTIVE {
                info!(
                    attempts,
                    elapsed_ms = elapsed_ms(start),
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
        elapsed_ms = elapsed_ms(start),
        "filter_tweets warmup deadline expired (non-fatal)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn only_as_configured_calls_the_warmer_builder() {
        let calls = Cell::new(0u32);
        let build = || {
            calls.set(calls.get() + 1);
            None
        };
        assert!(CacheWarming::Off.warmer(build).is_none());
        assert_eq!(calls.get(), 0);
        assert!(CacheWarming::AsConfigured.warmer(build).is_none());
        assert_eq!(calls.get(), 1);
    }

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

        assert_eq!(result, Err("test: discovery timed out".to_string()));
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
            Err("test: init deadline exhausted before attempt".to_string())
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
                "test: init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ))
        );
        assert_eq!(start.elapsed(), CLIENT_INIT_ATTEMPT_TIMEOUT);
    }
}
