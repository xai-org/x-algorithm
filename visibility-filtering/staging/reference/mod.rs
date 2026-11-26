use crate::config::{ENV_APP_ENV, ENV_REFERENCE};
use crate::evaluate_tweets::EntryComparator;
use crate::metrics::Metrics;
use crate::models::RequestViewer;
use crate::post_evaluator::PostEvaluation;
use crate::quote::Passes;
use crate::rules::SafetyLevel;
use std::env;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use strum::VariantNames;
use tokio::time::Instant;
use tonic::metadata::MetadataMap;
use tweetypie::TweetypieReference;

pub(crate) mod tweetypie;

pub const ENV_IMAGE: &str = "VF_IMAGE";
pub const ENV_TWEETYPIE_XDS_LISTENER: &str = "VF_TWEETYPIE_XDS_LISTENER";
pub const ENV_TWEETYPIE_TLS_DOMAIN: &str = "VF_TWEETYPIE_TLS_DOMAIN";
pub const ENV_TWEETYPIE_CLIENT_ID: &str = "VF_TWEETYPIE_CLIENT_ID";
pub const ENV_REFERENCE_SAMPLE: &str = "VF_REFERENCE_SAMPLE";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::VariantNames)]
#[strum(serialize_all = "snake_case")]
pub enum Reference {
    None,
        Tweetypie,
}

impl Reference {
    pub fn resolve(configured: Option<&str>, app_env: Option<&str>) -> Result<Self, String> {
        let reference = configured.map_or(Ok(Self::None), |value| {
            value.parse().map_err(|_| {
                format!(
                    "{ENV_REFERENCE}={value:?} is not one of {}",
                    Self::VARIANTS.join(", ")
                )
            })
        })?;
        match (reference, app_env) {
            (Self::None, _) => Ok(Self::None),
            (_, Some("prod")) => Err(format!(
                "{ENV_REFERENCE} selects {reference:?} but {ENV_APP_ENV}=prod; reference comparisons are staging-only"
            )),
            (reference, _) => Ok(reference),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample(f64);

impl Sample {
    const DEFAULT: Self = Self(0.5);

    fn resolve(configured: Option<&str>) -> Result<Self, String> {
        configured.map_or(Ok(Self::DEFAULT), |value| {
            value
                .parse()
                .ok()
                .filter(|share| (0.0..=1.0).contains(share))
                .map(Self)
                .ok_or_else(|| format!("{ENV_REFERENCE_SAMPLE}={value:?} is not a share in [0, 1]"))
        })
    }

    fn selects(self, viewer_id: u64, outcomes: &[PostEvaluation]) -> bool {
        let mut hasher = DefaultHasher::new();
        viewer_id.hash(&mut hasher);
        for outcome in outcomes {
            outcome.tweet_id.0.hash(&mut hasher);
        }
        let draw = (hasher.finish() >> 11) as f64 / (1_u64 << 53) as f64;
        draw < self.0
    }
}

fn comparison(
    configured: Option<&str>,
    app_env: Option<&str>,
    sample: Option<&str>,
) -> Result<Option<Sample>, String> {
    match Reference::resolve(configured, app_env)? {
        Reference::None => Ok(None),
        Reference::Tweetypie => Sample::resolve(sample).map(Some),
    }
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
pub(crate) async fn build(
    init_deadline: Instant,
    metrics: &Metrics,
    metadata: Option<&MetadataMap>,
) -> Option<Box<dyn EntryComparator>> {
    let sample = comparison(
        env::var(ENV_REFERENCE).ok().as_deref(),
        env::var(ENV_APP_ENV).ok().as_deref(),
        env::var(ENV_REFERENCE_SAMPLE).ok().as_deref(),
    )
    .unwrap_or_else(|misconfiguration| panic!("{misconfiguration}"))?;
    Some(Box::new(
        tweetypie::build(init_deadline, sample, metrics.clone(), metadata).await,
    ))
}

impl EntryComparator for Arc<TweetypieReference> {
    fn compare(
        &self,
        viewer: &RequestViewer,
        safety_level: SafetyLevel,
        outcomes: &[PostEvaluation],
        quote_passes: &Passes,
    ) {
        self.spawn_entry_comparison(viewer, safety_level, outcomes, quote_passes.ran());
    }
}

#[cfg(test)]
mod tests {
    use super::{Reference, Sample, comparison};
    use crate::models::{Evaluation, Placement, TweetId};
    use crate::post_evaluator::PostEvaluation;
    use crate::rules::fixtures::allow;
    use serde::Deserialize;

    #[test]
    fn reference_defaults_to_none_and_is_refused_in_prod() {
        assert_eq!(
            Reference::resolve(None, Some("staging")),
            Ok(Reference::None)
        );
        assert_eq!(
            Reference::resolve(Some("none"), Some("staging")),
            Ok(Reference::None)
        );
        assert_eq!(
            Reference::resolve(Some("tweetypie"), Some("staging")),
            Ok(Reference::Tweetypie)
        );
        assert!(Reference::resolve(Some("Tweetypie"), Some("staging")).is_err());
        assert!(Reference::resolve(Some("vf_service"), Some("staging")).is_err());
        assert!(Reference::resolve(Some("tweetypie"), Some("prod")).is_err());
        assert_eq!(
            Reference::resolve(Some("none"), Some("prod")),
            Ok(Reference::None)
        );
    }

    #[test]
    fn the_sample_defaults_to_half_and_selects_its_share_of_requests() {
        assert_eq!(
            comparison(Some("tweetypie"), Some("devel"), None),
            Ok(Some(Sample(0.5)))
        );
        assert!(comparison(Some("tweetypie"), Some("devel"), Some("half")).is_err());
        for reference in [None, Some("none")] {
            assert_eq!(
                comparison(reference, Some("staging"), Some("half")),
                Ok(None)
            );
        }
        assert_eq!(Sample::resolve(Some("0.25")), Ok(Sample(0.25)));
        for invalid in ["", "half", "-0.1", "1.5", "NaN"] {
            assert!(Sample::resolve(Some(invalid)).is_err(), "{invalid:?}");
        }
        let entries = |tweet_id| {
            [PostEvaluation {
                tweet_id: TweetId(tweet_id),
                placement: Placement::Standalone,
                source_tweet_id: None,
                evaluation: Evaluation::complete(allow()),
                safety_labels: None,
            }]
        };
        let selected = |share| {
            (1..=1000)
                .filter(|&request| Sample(share).selects(request, &entries(request + 7)))
                .count()
        };
        assert_eq!(selected(0.0), 0);
        assert_eq!(selected(1.0), 1000);
        assert!((400..=600).contains(&selected(0.5)), "{}", selected(0.5));
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Kind {
        Deliberate,
        Gap,
    }

        #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Divergence {
        divergence: String,
        kind: Kind,
        ticket: Option<String>,
        behavior: String,
    }

    #[test]
    fn each_divergence_states_its_behavior_and_a_gap_names_its_ticket() {
        let path = crate::pinned::crate_path("tests/divergences.json");
        let registry: Vec<Divergence> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for entry in registry {
            let name = &entry.divergence;
            assert!(!entry.behavior.is_empty(), "{name}: no behavior");
            assert!(
                matches!(entry.kind, Kind::Deliberate) || entry.ticket.is_some(),
                "{name}: a gap names its ticket"
            );
        }
    }
}
