use crate::models::candidate::PostCandidate;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{
    EnablePhoenixRetrievalStatsExperimentBucket, EnablePhoenixScoreStatsExperimentBucket,
    PhoenixRetrievalInferenceClusterId, PhoenixRetrievalMOEInferenceClusterId, TRACE_USER_IDS,
};
use crate::scorers::value_model;
use crate::sources::phoenix_source::PhoenixSource;

use rand::random;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::ExperimentBucket;
use xai_home_mixer_proto::ServedType;
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets, StatsReceiverExt};
use xai_value_model::{unoffset_score, NEGATIVE_SCORES_OFFSET};

const METRIC_PREFIX: &str = "ScoredStats";

const PRESENT_SCOPE: [(&str, &str); 1] = [("score_status", "present")];
const MISSING_SCOPE: [(&str, &str); 1] = [("score_status", "missing")];

const HEAVY_RANKER_TOP_K: &[usize] = &[1, 10, 35];

const PHOENIX_RETRIEVAL_TOP_K: &[u32] = &[10, 100, 200, 500, 1000];

const SIMCLUSTERS_TOP_K: &[u32] = &[50, 100, 200, 400, 600, 800];

const DEFAULT_SAMPLING_RATE: f64 = 0.05;

pub struct ScoredStatsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for ScoredStatsSideEffect {
    fn enable(&self, _query: Arc<ScoredPostsQuery>) -> bool {
        true
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };

        record_trace_author_served_by_type(
            receiver.as_ref(),
            &input.selected_candidates,
            input.query.request_type,
        );

        record_trace_author_score_distributions(
            receiver.as_ref(),
            &input.selected_candidates,
            &input.non_selected_candidates,
        );

        let candidates = &input.selected_candidates;
        if candidates.is_empty() {
            return Ok(());
        }

        record_served_by_source(receiver.as_ref(), candidates);

        if !input.query.has_cached_posts {
            let retrieval_cluster: String =
                input.query.params.get(PhoenixRetrievalInferenceClusterId);
            let moe_cluster: String = input
                .query
                .params
                .get(PhoenixRetrievalMOEInferenceClusterId);
            let experiment_buckets = input
                .query
                .params
                .experiment_buckets(EnablePhoenixRetrievalStatsExperimentBucket);
            let score_buckets = input
                .query
                .params
                .experiment_buckets(EnablePhoenixScoreStatsExperimentBucket);

            let sampled = random::<f64>() < DEFAULT_SAMPLING_RATE;
            if !score_buckets.is_empty() || sampled {
                record_score_distributions(
                    receiver.as_ref(),
                    "score",
                    candidates.iter(),
                    &score_buckets,
                );
            }
            if input.query.request_type != RequestType::PhoenixScores
                && !input.query.in_network_only
            {
                record_net_negative_stats(
                    receiver.as_ref(),
                    &input.query,
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                );
            }
            if !experiment_buckets.is_empty() || sampled {
                record_phoenix_retrieval_stats(
                    receiver.as_ref(),
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                    &retrieval_cluster,
                    &experiment_buckets,
                );
                record_phoenix_retrieval_moe_stats(
                    receiver.as_ref(),
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                    &moe_cluster,
                    &experiment_buckets,
                );
                record_phoenix_retrieval_cold_stats(
                    receiver.as_ref(),
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                    &retrieval_cluster,
                    &experiment_buckets,
                );
                record_phoenix_retrieval_zero_fav_stats(
                    receiver.as_ref(),
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                    &retrieval_cluster,
                    &experiment_buckets,
                );
                if input.query.request_type != RequestType::PhoenixScores
                    && !input.query.in_network_only
                {
                    record_retrieval_source_contribution(
                        receiver.as_ref(),
                        &input.selected_candidates,
                        &input.non_selected_candidates,
                        &format!("{:?}", PhoenixSource::resolve_cluster(&input.query)),
                        &experiment_buckets,
                    );
                }
            }
        } else {
            if random::<f64>() < DEFAULT_SAMPLING_RATE {
                record_score_distributions(receiver.as_ref(), "score", candidates.iter(), &[]);
            }
        }

        Ok(())
    }
}

fn record_net_negative_stats(
    receiver: &dyn StatsReceiverExt,
    query: &ScoredPostsQuery,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
) {
    let weights = value_model::weights_for(query);
    let mut counts: HashMap<(&'static str, &'static str), u64> = HashMap::new();
    let (mut scored, mut negative) = (0u64, 0u64);
    for c in selected_candidates.iter().chain(non_selected_candidates) {
        let sign = match c.weighted_score {
            None => "missing",
            Some(s) if s >= NEGATIVE_SCORES_OFFSET => "positive",
            Some(s) => {
                let ni_term =
                    c.phoenix_scores.not_interested_score.unwrap_or(0.0) * weights.not_interested;
                if unoffset_score(s, &weights) - ni_term >= 0.0 {
                    "negative_ni"
                } else {
                    "negative_other"
                }
            }
        };
        if sign != "missing" {
            scored += 1;
            negative += u64::from(sign.starts_with("negative"));
        }
        let served_type = c.served_type.unwrap_or(ServedType::Undefined).as_str_name();
        *counts.entry((served_type, sign)).or_insert(0) += 1;
    }
    let candidates_key = format!("{METRIC_PREFIX}.NetNegative.Candidates");
    for ((served_type, sign), count) in counts {
        receiver.incr(
            &candidates_key,
            &[("served_type", served_type), ("sign", sign)],
            count,
        );
    }
    if scored > 0 {
        receiver.observe(
            &format!("{METRIC_PREFIX}.NetNegative.RequestShare"),
            &[],
            negative as f64 / scored as f64,
            HistogramBuckets::Bucket0To1,
        );
    }
}

fn record_served_by_source(receiver: &dyn StatsReceiverExt, selected_candidates: &[PostCandidate]) {
    let mut counts: HashMap<&'static str, u64> = HashMap::new();
    for candidate in selected_candidates {
        let served_type = candidate
            .served_type
            .unwrap_or(ServedType::Undefined)
            .as_str_name();
        *counts.entry(served_type).or_insert(0) += 1;
    }
    let key = format!("{METRIC_PREFIX}.ServedBySource");
    for (served_type, count) in counts {
        receiver.incr(
            &key,
            &[("type", "sum"), ("served_type", served_type)],
            count,
        );
    }
    receiver.incr(&key, &[("type", "requests")], 1);
}

fn record_retrieval_source_contribution(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    let empty_bucket = ExperimentBucket::new("", "");
    let buckets: &[&ExperimentBucket] = if experiment_buckets.is_empty() {
        &[&empty_bucket]
    } else {
        experiment_buckets
    };
    let selected_key = format!("{METRIC_PREFIX}.RetrievalContribution.Selected");
    let total_key = format!("{METRIC_PREFIX}.RetrievalContribution.SelectedTotal");
    let scored_top_k_key = format!("{METRIC_PREFIX}.RetrievalContribution.ScoredTopK");
    let top_k_keys = [
        (
            ServedType::ForYouPhoenixRetrieval,
            PHOENIX_RETRIEVAL_TOP_K,
            format!("{METRIC_PREFIX}.RetrievalContribution.PhoenixTopK"),
        ),
        (
            ServedType::ForYouSimclusters,
            SIMCLUSTERS_TOP_K,
            format!("{METRIC_PREFIX}.RetrievalContribution.SimclustersTopK"),
        ),
    ];

    let mut sources = HashSet::new();
    for c in selected_candidates {
        sources.extend(c.served_type);
        sources.extend(c.retrieval_sources.iter().map(|s| s.served_type));
    }
    let count = |source: ServedType, k: Option<u32>| -> [(&'static str, u64); 2] {
        let (mut credited, mut inclusive) = (0, 0);
        for c in selected_candidates {
            let is_credited = c.served_type == Some(source);
            if within_top_k(c, source, k) || (is_credited && k.is_none()) {
                inclusive += 1;
                credited += u64::from(is_credited);
            }
        }
        [("credited", credited), ("inclusive", inclusive)]
    };
    let scored_within = |source: ServedType, k: u32| -> u64 {
        selected_candidates
            .iter()
            .chain(non_selected_candidates)
            .filter(|c| within_top_k(c, source, Some(k)))
            .count() as u64
    };

    for b in buckets {
        let common = [
            ("retrieval_cluster", retrieval_cluster),
            ("ddg", b.experiment.as_str()),
            ("bucket", b.bucket.as_str()),
        ];
        let incr = |key: &str, scopes: &[(&str, &str)], value: u64| {
            receiver.incr(key, &[scopes, &common].concat(), value)
        };

        incr(
            &total_key,
            &[("type", "sum")],
            selected_candidates.len() as u64,
        );
        incr(&total_key, &[("type", "requests")], 1);

        for source in &sources {
            for (mode, value) in count(*source, None) {
                let scopes = [("source", source.as_str_name()), ("mode", mode)];
                incr(&selected_key, &scopes, value);
            }
        }

        for (source, top_k, top_k_key) in &top_k_keys {
            for k in *top_k {
                let k_str = k.to_string();
                for (mode, value) in count(*source, Some(*k)) {
                    incr(top_k_key, &[("k", &k_str), ("mode", mode)], value);
                }
                incr(
                    &scored_top_k_key,
                    &[("source", source.as_str_name()), ("k", &k_str)],
                    scored_within(*source, *k),
                );
            }
        }
    }
}

fn within_top_k(candidate: &PostCandidate, source: ServedType, k: Option<u32>) -> bool {
    candidate
        .retrieval_sources
        .iter()
        .any(|s| s.served_type == source && k.is_none_or(|k| s.position.is_some_and(|p| p <= k)))
}

fn record_head(
    receiver: &dyn StatsReceiverExt,
    metric: &str,
    name: &str,
    scores: impl Iterator<Item = Option<f64>>,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_head_with_buckets(
        receiver,
        metric,
        name,
        scores,
        HistogramBuckets::Bucket0To1,
        experiment_buckets,
    );
}

fn record_head_with_buckets(
    receiver: &dyn StatsReceiverExt,
    metric: &str,
    name: &str,
    scores: impl Iterator<Item = Option<f64>>,
    buckets: HistogramBuckets,
    experiment_buckets: &[&ExperimentBucket],
) {
    let distribution_key = format!("{METRIC_PREFIX}.{metric}Distribution.{name}");
    let missing_key = format!("{METRIC_PREFIX}.{metric}Missing.{name}");
    let by_bucket_key = (!experiment_buckets.is_empty())
        .then(|| format!("{METRIC_PREFIX}.{metric}DistributionByBucket.{name}"));
    let mut present = 0u64;
    let mut missing = 0u64;
    for score in scores {
        match score {
            Some(value) => {
                present += 1;
                receiver.observe(&distribution_key, &[], value, buckets);
                if let Some(key) = &by_bucket_key {
                    for b in experiment_buckets {
                        receiver.observe(
                            key,
                            &[("ddg", &b.experiment), ("bucket", &b.bucket)],
                            value,
                            buckets,
                        );
                    }
                }
            }
            None => {
                missing += 1;
            }
        }
    }
    receiver.incr(&missing_key, &PRESENT_SCOPE, present);
    receiver.incr(&missing_key, &MISSING_SCOPE, missing);
}

fn record_score_distributions<'a>(
    receiver: &dyn StatsReceiverExt,
    metric: &str,
    candidates: impl Iterator<Item = &'a PostCandidate> + Clone,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_head(
        receiver,
        metric,
        "favorite",
        candidates.clone().map(|c| c.phoenix_scores.favorite_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "reply",
        candidates.clone().map(|c| c.phoenix_scores.reply_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "retweet",
        candidates.clone().map(|c| c.phoenix_scores.retweet_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "click",
        candidates.clone().map(|c| c.phoenix_scores.click_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "vqv",
        candidates.clone().map(|c| c.phoenix_scores.vqv_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "share",
        candidates.clone().map(|c| c.phoenix_scores.share_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "not_interested",
        candidates
            .clone()
            .map(|c| c.phoenix_scores.not_interested_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "not_dwelled",
        candidates
            .clone()
            .map(|c| c.phoenix_scores.not_dwelled_score),
        experiment_buckets,
    );
    record_head_with_buckets(
        receiver,
        metric,
        "dwellTime",
        candidates.clone().map(|c| c.phoenix_scores.dwell_time),
        HistogramBuckets::Bucket0To50,
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "weightedScore",
        candidates.clone().map(|c| c.weighted_score),
        experiment_buckets,
    );
    record_head(
        receiver,
        metric,
        "finalScore",
        candidates.map(|c| c.score),
        experiment_buckets,
    );
}

fn record_trace_author_score_distributions(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
) {
    let Some(&author_id) = TRACE_USER_IDS.first() else {
        return;
    };

    let is_original =
        |c: &&PostCandidate| c.retweeted_tweet_id.is_none() && c.in_reply_to_tweet_id.is_none();

    let author_present = selected_candidates
        .iter()
        .chain(non_selected_candidates.iter())
        .filter(is_original)
        .any(|c| c.author_id == author_id);

    if !author_present {
        return;
    }

    if random::<f64>() >= DEFAULT_SAMPLING_RATE {
        return;
    }

    let (author, others): (Vec<&PostCandidate>, Vec<&PostCandidate>) = selected_candidates
        .iter()
        .chain(non_selected_candidates.iter())
        .filter(is_original)
        .partition(|c| c.author_id == author_id);

    record_score_distributions(receiver, "traceAuthorScore", author.iter().copied(), &[]);
    record_score_distributions(receiver, "traceOtherScore", others.iter().copied(), &[]);
}

fn post_type(candidate: &PostCandidate) -> &'static str {
    if candidate.retweeted_tweet_id.is_some() {
        "repost"
    } else if candidate.in_reply_to_tweet_id.is_some() {
        "reply"
    } else {
        "original"
    }
}

fn record_trace_author_served_by_type(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    request_type: RequestType,
) {
    let Some(&author_id) = TRACE_USER_IDS.first() else {
        return;
    };

    let author_present = selected_candidates.iter().any(|c| c.author_id == author_id);
    if !author_present {
        return;
    }

    if random::<f64>() >= DEFAULT_SAMPLING_RATE {
        return;
    }

    let mut counts: HashMap<(Option<ServedType>, &'static str), u64> = HashMap::new();
    for candidate in selected_candidates {
        if candidate.author_id == author_id {
            *counts
                .entry((candidate.served_type, post_type(candidate)))
                .or_insert(0) += 1;
        }
    }

    let request_type = request_type.to_string();
    let key = format!("{METRIC_PREFIX}.TraceAuthorServed");
    for ((served_type, post_type), count) in counts {
        let served_type = served_type.unwrap_or(ServedType::Undefined).as_str_name();
        receiver.incr(
            &key,
            &[
                ("request_type", request_type.as_str()),
                ("served_type", served_type),
                ("post_type", post_type),
            ],
            count,
        );
    }
}

fn record_phoenix_retrieval_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_retrieval_source_stats(
        receiver,
        selected_candidates,
        non_selected_candidates,
        retrieval_cluster,
        experiment_buckets,
        ServedType::ForYouPhoenixRetrieval,
        "PhoenixRetrievalTweets",
    );
}

fn record_phoenix_retrieval_moe_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_retrieval_source_stats(
        receiver,
        selected_candidates,
        non_selected_candidates,
        retrieval_cluster,
        experiment_buckets,
        ServedType::ForYouPhoenixRetrievalMoe,
        "PhoenixRetrievalMoeTweets",
    );
}

fn record_phoenix_retrieval_cold_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_retrieval_source_stats(
        receiver,
        selected_candidates,
        non_selected_candidates,
        retrieval_cluster,
        experiment_buckets,
        ServedType::ForYouPhoenixRetrievalCold,
        "PhoenixRetrievalColdTweets",
    );
}

fn record_phoenix_retrieval_zero_fav_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_retrieval_source_stats(
        receiver,
        selected_candidates,
        non_selected_candidates,
        retrieval_cluster,
        experiment_buckets,
        ServedType::ForYouPhoenixRetrievalZeroFav,
        "PhoenixRetrievalZeroFavTweets",
    );
}

fn record_retrieval_source_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
    served_type: ServedType,
    metric_name: &str,
) {
    let mut all_candidates: Vec<&PostCandidate> = selected_candidates
        .iter()
        .chain(non_selected_candidates.iter())
        .collect();
    all_candidates.sort_by(|a, b| {
        let score_a = a.weighted_score.unwrap_or(f64::NEG_INFINITY);
        let score_b = b.weighted_score.unwrap_or(f64::NEG_INFINITY);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let empty_bucket = ExperimentBucket::new("", "");
    let buckets: &[&ExperimentBucket] = if experiment_buckets.is_empty() {
        &[&empty_bucket]
    } else {
        experiment_buckets
    };

    let topk_key = format!("{METRIC_PREFIX}.{metric_name}.RankedTopK");
    for &k in HEAVY_RANKER_TOP_K {
        let count = all_candidates
            .iter()
            .take(k)
            .filter(|c| c.served_type == Some(served_type))
            .count();
        let k_str = match k {
            1 => "1",
            10 => "10",
            35 => "35",
            _ => "unknown",
        };

        for b in buckets {
            let scopes: [(&str, &str); 5] = [
                ("type", "sum"),
                ("retrieval_cluster", retrieval_cluster),
                ("k", k_str),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ];
            receiver.incr(&topk_key, &scopes, count as u64);
            let req_scopes: [(&str, &str); 5] = [
                ("type", "requests"),
                ("retrieval_cluster", retrieval_cluster),
                ("k", k_str),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ];
            receiver.incr(&topk_key, &req_scopes, 1);
        }
    }

    let served_key = format!("{METRIC_PREFIX}.{metric_name}.Served");
    let count = selected_candidates
        .iter()
        .filter(|c| c.served_type == Some(served_type))
        .count();

    for b in buckets {
        receiver.incr(
            &served_key,
            &[
                ("type", "sum"),
                ("retrieval_cluster", retrieval_cluster),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ],
            count as u64,
        );
        receiver.incr(
            &served_key,
            &[
                ("type", "requests"),
                ("retrieval_cluster", retrieval_cluster),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ],
            1,
        );
    }
}
