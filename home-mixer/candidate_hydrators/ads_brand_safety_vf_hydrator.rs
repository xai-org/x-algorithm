use crate::models::brand_safety::{
    botmaker_rule_category, botmaker_rule_id_from, compute_verdict, compute_verdict_v2,
    truncate_description, worst_verdict, BrandSafetyVerdict,
};
use crate::models::candidate::{PostCandidate, SafetyLabelInfo};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAdsBrandSafetyVerdictV2;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_safety_label_store::types::SafetyLabelMap;
use xai_stats_receiver::global_stats_receiver;
use xai_visibility_filtering::tweet_safety_label::TweetSafetyLabelClient;

const NSFW_AUTHOR_METRIC: &str = "AdsBrandSafetyVf.nsfw_author";

pub struct AdsBrandSafetyVfHydrator {
    pub client: Arc<dyn TweetSafetyLabelClient>,
}

fn to_safety_label_infos(labels: &SafetyLabelMap) -> impl Iterator<Item = SafetyLabelInfo> {
    labels.iter().map(|(k, v)| SafetyLabelInfo {
        label_type: *k,
        description: v.source.as_deref().map(truncate_description),
        source: botmaker_rule_id_from(v).map(|id| botmaker_rule_category(id).to_string()),
    })
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for AdsBrandSafetyVfHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let mut all_ids: HashSet<u64> = HashSet::new();
        for c in candidates {
            all_ids.insert(c.retweeted_tweet_id.unwrap_or(c.tweet_id));
            if let Some(qt_id) = c.quoted_tweet_id {
                all_ids.insert(qt_id);
            }
            all_ids.extend(c.ancestors.iter().copied());
        }

        let tweet_ids: Vec<u64> = all_ids.into_iter().collect();
        let batch = match self.client.get_safety_labels(tweet_ids).await {
            Ok(batch) => batch,
            Err(e) => {
                let err = format!("VF get_safety_labels failed: {e}");
                return candidates.iter().map(|_| Err(err.clone())).collect();
            }
        };

        let failed_ids: HashSet<u64> = batch.failures.keys().copied().collect();
        let label_map = batch.labels;

        let compute = if query.params.get(EnableAdsBrandSafetyVerdictV2) {
            compute_verdict_v2
        } else {
            compute_verdict
        };

        let mut nsfw_author_seen: u64 = 0;
        let mut nsfw_author_dropped: u64 = 0;

        let results: Vec<Result<PostCandidate, String>> = candidates
            .iter()
            .map(|c| {
                let primary_id = c.retweeted_tweet_id.unwrap_or(c.tweet_id);

                if failed_ids.contains(&primary_id) {
                    return Err(format!("VF lookup failed for tweet {primary_id}"));
                }

                let empty = HashMap::new();
                let primary_labels = label_map.get(&primary_id).unwrap_or(&empty);
                let mut verdict = compute(primary_labels, primary_id);
                let mut safety_labels: Vec<SafetyLabelInfo> =
                    to_safety_label_infos(primary_labels).collect();

                if let Some(qt_id) = c.quoted_tweet_id {
                    if failed_ids.contains(&qt_id) {
                        verdict = worst_verdict(&verdict, &BrandSafetyVerdict::MediumRisk);
                    } else {
                        let qt_labels = label_map.get(&qt_id).unwrap_or(&empty);
                        verdict = worst_verdict(&verdict, &compute(qt_labels, qt_id));
                        safety_labels.extend(to_safety_label_infos(qt_labels));
                    }
                }

                for &ancestor_id in &c.ancestors {
                    if failed_ids.contains(&ancestor_id) {
                        verdict = worst_verdict(&verdict, &BrandSafetyVerdict::MediumRisk);
                    } else {
                        let ancestor_labels = label_map.get(&ancestor_id).unwrap_or(&empty);
                        verdict = worst_verdict(&verdict, &compute(ancestor_labels, ancestor_id));
                        safety_labels.extend(to_safety_label_infos(ancestor_labels));
                    }
                }

                if c.nsfw_author_ads == Some(true) {
                    nsfw_author_seen += 1;
                    let before = verdict;
                    verdict = worst_verdict(&verdict, &BrandSafetyVerdict::MediumRisk);
                    if verdict != before {
                        nsfw_author_dropped += 1;
                    }
                }

                safety_labels.sort_unstable_by_key(|l| i32::from(l.label_type));
                safety_labels.dedup_by(|a, b| a.label_type == b.label_type);

                Ok(PostCandidate {
                    brand_safety_verdict: Some(verdict),
                    safety_labels,
                    ..Default::default()
                })
            })
            .collect();

        if let Some(receiver) = global_stats_receiver() {
            if nsfw_author_seen > 0 {
                receiver.incr(NSFW_AUTHOR_METRIC, &[("outcome", "seen")], nsfw_author_seen);
            }
            if nsfw_author_dropped > 0 {
                receiver.incr(
                    NSFW_AUTHOR_METRIC,
                    &[("outcome", "dropped")],
                    nsfw_author_dropped,
                );
            }
        }

        results
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.brand_safety_verdict = hydrated.brand_safety_verdict;
        candidate.safety_labels = hydrated.safety_labels;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_safety_label_store::types::SafetyLabelMap;
    use xai_visibility_filtering::tweet_safety_label::{SafetyLabelFailure, SafetyLabelsBatch};
    use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};

    struct FakeVfClient {
        batch: SafetyLabelsBatch,
    }

    #[async_trait]
    impl TweetSafetyLabelClient for FakeVfClient {
        async fn get_safety_labels(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> Result<SafetyLabelsBatch, tonic::Status> {
            Ok(self.batch.clone())
        }
    }

    #[tokio::test]
    async fn safe_verdict_with_grok_sfa() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    #[tokio::test]
    async fn medium_risk_when_not_scored() {
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, HashMap::new())]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn vf_failure_returns_error() {
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::new(),
                failures: HashMap::from([(1, SafetyLabelFailure::LookupFailed)]),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert!(results[0].is_err());
    }

    #[tokio::test]
    async fn quoted_tweet_failure_gives_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::from([(2, SafetyLabelFailure::LookupFailed)]),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            quoted_tweet_id: Some(2),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn retweet_uses_retweeted_id() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(100, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            retweeted_tweet_id: Some(100),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    #[tokio::test]
    async fn safe_when_candidate_and_ancestors_safe() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels.clone()), (10, labels.clone()), (11, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10, 11],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    #[tokio::test]
    async fn ancestor_medium_risk_escalates_verdict() {
        let mut safe_labels: SafetyLabelMap = HashMap::new();
        safe_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let mut risky_labels: SafetyLabelMap = HashMap::new();
        risky_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        risky_labels.insert(SafetyLabelType::NSFW_HIGH_PRECISION, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, safe_labels), (10, risky_labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
        assert!(hydrated
            .safety_labels
            .iter()
            .any(|l| l.label_type == SafetyLabelType::NSFW_HIGH_PRECISION));
    }

    #[tokio::test]
    async fn ancestor_low_risk_escalates_safe_candidate() {
        let mut safe_labels: SafetyLabelMap = HashMap::new();
        safe_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let mut low_risk_labels: SafetyLabelMap = HashMap::new();
        low_risk_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        low_risk_labels.insert(
            SafetyLabelType::NSFA_LIMITED_INVENTORY,
            SafetyLabel::default(),
        );
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, safe_labels), (10, low_risk_labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::LowRisk)
        );
    }

    #[tokio::test]
    async fn ancestor_failure_gives_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::from([(10, SafetyLabelFailure::LookupFailed)]),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn unscored_ancestor_gives_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels), (10, HashMap::new())]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn batch_rpc_failure_returns_errors_for_all() {
        struct FailingClient;
        #[async_trait]
        impl TweetSafetyLabelClient for FailingClient {
            async fn get_safety_labels(
                &self,
                _: Vec<u64>,
            ) -> Result<SafetyLabelsBatch, tonic::Status> {
                Err(tonic::Status::unavailable("vf down"))
            }
        }

        let hydrator = AdsBrandSafetyVfHydrator {
            client: Arc::new(FailingClient),
        };
        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                ..Default::default()
            },
        ];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert!(results[0].is_err());
        assert!(results[1].is_err());
    }

    #[tokio::test]
    async fn nsfw_author_escalates_to_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            nsfw_author_ads: Some(true),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn non_nsfw_author_keeps_safe_verdict() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator { client };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            nsfw_author_ads: Some(false),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }
}
