use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::sources::Sources;
use crate::hydration::{request_context, HydratedTweet, Hydration, Hydrator, Hydrators, Lookup};
use crate::ingress;
use crate::metrics::{Endpoint, Metrics, RequestLevel};
use crate::models::viewer::Client;
use crate::models::{
    Evaluation, EvaluationKey, PlacedCandidate, RawCandidate, RequestViewer, TweetId,
};
use crate::params::{ClientSwitches, CountryLists, Experiments};
use crate::post_evaluator::{self, MergeSelection, PostEvaluator};
use crate::result_label::{evaluate_tweets_label, filter_label, label, Class, Label, NONE};
use crate::rules::{RequestExperiments, Rpc, RuleEngine, SafetyLevel};
use crate::server_deps::{self, CacheWarming};
use crate::treatment;
use futures::future::join_all;
use std::fmt;
use std::sync::Arc;
use strum::VariantArray;
use tokio::time::Instant;

pub struct Why {
    post_evaluator: PostEvaluator,
    client_switches: ClientSwitches,
}

#[derive(clap::Parser)]
pub struct Ask {
    pub post: u64,
    #[arg(long)]
    pub viewer: Option<u64>,
    #[arg(long, default_value = "us")]
    country: String,
    #[arg(long, default_value = "web")]
    client: Client,
    #[arg(long = "level", value_parser = Route::parse)]
    levels: Vec<Route>,
}

#[derive(Clone, Copy)]
struct Route {
    level: SafetyLevel,
    rpc: Rpc,
}

pub struct LevelWhy {
    route: Route,
    served: Label,
    evaluation: Evaluation,
    viewer: ViewerResolution,
    failed: Hydrators,
    retweet: Option<RetweetWhy>,
}

struct RetweetWhy {
    source: TweetId,
    selection: MergeSelection,
    source_decided_by: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum ViewerResolution {
    LoggedOut,
    NotRead,
    Found,
    NoUser,
    LookupFailed,
}

impl Route {
    fn of(level: SafetyLevel) -> Option<Self> {
        [Rpc::FilterTweets, Rpc::EvaluateTweets]
            .into_iter()
            .find(|&rpc| level.is_carried_by(rpc))
            .map(|rpc| Self { level, rpc })
    }

    fn parse(name: &str) -> Result<Self, String> {
        name.parse()
            .ok()
            .and_then(Self::of)
            .ok_or_else(|| format!("{name} is not a post level"))
    }
}

impl Why {
    pub async fn connect(datacenter: &str) -> Self {
        let serving =
            server_deps::serving(datacenter, None, CacheWarming::Off, Experiments::none()).await;
        Self::new(
            Arc::new(serving.sources),
            serving.fallback_caches,
            serving.country_lists,
            serving.client_switches,
            serving.metrics,
        )
    }

    fn new(
        sources: Arc<impl Sources + 'static>,
        fallback_caches: FallbackCaches,
        country_lists: Arc<CountryLists>,
        client_switches: ClientSwitches,
        metrics: Metrics,
    ) -> Self {
        Self {
            post_evaluator: PostEvaluator::new(
                sources,
                RuleEngine::with_country_lists(country_lists),
                Arc::new(fallback_caches),
                metrics,
            ),
            client_switches,
        }
    }

    pub async fn judge(&self, ask: &Ask) -> Vec<LevelWhy> {
        let routes = if ask.levels.is_empty() {
            SafetyLevel::VARIANTS
                .iter()
                .copied()
                .filter_map(Route::of)
                .collect()
        } else {
            ask.levels.clone()
        };
        join_all(routes.into_iter().map(|route| self.level(ask, route))).await
    }

    async fn level(&self, ask: &Ask, route: Route) -> LevelWhy {
        let Route { level, rpc } = route;
        let requested = PlacedCandidate::standalone(RawCandidate {
            tweet_id: TweetId(ask.post),
            request_author_id: None,
        });
        let (request, endpoint) = match rpc {
            Rpc::FilterTweets => {
                let mut batch = ingress::filter_tweets_batch(
                    self.post_evaluator.rule_engine(),
                    &self.client_switches,
                    RequestViewer::new(ask.viewer, Some(ask.country.clone())),
                    level,
                    vec![requested],
                );
                batch.facts = batch.facts.with_experiments(RequestExperiments::default());
                (batch, Endpoint::FilterTweets)
            }
            Rpc::EvaluateTweets => (
                ingress::evaluate_tweets_batch(
                    &self.client_switches,
                    RequestViewer::with_context(
                        ask.viewer,
                        Some(ask.client.context(ask.viewer, Some(ask.country.as_str()))),
                    ),
                    level,
                    vec![requested],
                ),
                Endpoint::EvaluateTweets,
            ),
        };
        let metrics = self
            .post_evaluator
            .metrics()
            .request(endpoint, RequestLevel::Named(level));
        let (hydrated, outcomes, mut traced) = request_context(Instant::now(), None)
            .scope(post_evaluator::hydrate_and_trace(
                &self.post_evaluator,
                request,
                &metrics,
            ))
            .await;
        let hydration = hydrated.hydration();
        let outcome = outcomes.into_iter().next();
        let evaluation = outcome
            .as_ref()
            .map_or(Evaluation::Failed(Lookup::Tweet), |outcome| {
                outcome.evaluation.clone()
            });
        let served = match rpc {
            Rpc::FilterTweets => outcome.map_or_else(
                || label(Class::Missing, NONE),
                |outcome| filter_label(&treatment::filter_tweets_result(outcome)),
            ),
            Rpc::EvaluateTweets => {
                let viewer = hydration.viewer().facts.viewer();
                let policies = self.client_switches.limited_actions_policies(
                    viewer.context(),
                    viewer.id(),
                    viewer.country_code(),
                    evaluation.verdict().limited_engagement_reasons(),
                    &Metrics::none(),
                );
                evaluate_tweets_label(&evaluation, level, &policies)
            }
        };
        let retweet = hydration
            .tweet(requested.tweet_id())
            .and_then(HydratedTweet::source_tweet_id)
            .map(|source| RetweetWhy {
                source,
                selection: traced
                    .selections
                    .get(&requested.key())
                    .copied()
                    .unwrap_or(MergeSelection::RequestedOnly),
                source_decided_by: traced
                    .policies
                    .remove(&EvaluationKey::retweet_source(source))
                    .and_then(|trace| treatment::decided_by(trace.evaluation.verdict())),
            });
        LevelWhy {
            route,
            served,
            evaluation,
            viewer: ViewerResolution::of(hydration),
            failed: hydration.failed_nodes(requested.tweet_id()),
            retweet,
        }
    }
}

impl ViewerResolution {
    fn of(hydration: &Hydration) -> Self {
        if hydration.viewer().facts.viewer_id().is_none() {
            Self::LoggedOut
        } else if !hydration.planned().contains(Hydrator::ViewerProfile) {
            Self::NotRead
        } else if hydration.viewer_failed().contains(Hydrator::ViewerProfile) {
            Self::LookupFailed
        } else if hydration.viewer_found() {
            Self::Found
        } else {
            Self::NoUser
        }
    }
}

fn listed(names: impl Iterator<Item = &'static str>) -> String {
    let names: Vec<&'static str> = names.collect();
    if names.is_empty() {
        NONE.to_owned()
    } else {
        names.join(", ")
    }
}

impl fmt::Display for LevelWhy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level: &'static str = self.route.level.into();
        let rpc: &'static str = self.route.rpc.into();
        let viewer: &'static str = self.viewer.into();
        let (class, reason) = &self.served;
        let rule = treatment::decided_by(self.evaluation.verdict());
        let failed_lookup: Option<&'static str> = match self.evaluation {
            Evaluation::Failed(lookup) => Some(lookup.into()),
            Evaluation::Complete { .. }
            | Evaluation::Partial { .. }
            | Evaluation::NotFound(_)
            | Evaluation::Deleted(_) => None,
        };
        let failed = failed_lookup
            .into_iter()
            .chain(self.failed.iter().map(Into::into));
        let fail_open = self.evaluation.fail_open_defaults();
        writeln!(f, "{level} ({rpc}): {class}/{reason}")?;
        writeln!(f, "  rule: {}", rule.unwrap_or(NONE))?;
        writeln!(f, "  viewer: {viewer}")?;
        writeln!(f, "  failed lookups: {}", listed(failed))?;
        writeln!(
            f,
            "  fail-open defaults: {}",
            listed(fail_open.iter().map(Into::into))
        )?;
        if let Some(retweet) = &self.retweet {
            let selection: &'static str = retweet.selection.into();
            writeln!(
                f,
                "  retweet of {}: {selection}, source rule: {}",
                retweet.source.0,
                retweet.source_decided_by.unwrap_or(NONE)
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
