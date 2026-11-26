use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableSerendipityStatsExperimentBucket;
use rand::random;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::ExperimentBucket;
use xai_recsys_proto::{language_code_string_to_enum, LanguageCode};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets, StatsReceiverExt};
use xai_vm_ranker_proto::experiment::HistoryTopics;

const METRIC_PREFIX: &str = "Serendipity";
const DEFAULT_SAMPLING_RATE: f64 = 0.05;
const TOP_POSITIONS: usize = 10;
const MS_PER_DAY: f64 = 86_400_000.0;
const MIN_USER_LANGUAGE_SHARE: f64 = 0.05;
const NON_LINGUISTIC: [LanguageCode; 8] = [
    LanguageCode::LangUnknown,
    LanguageCode::LangUnd,
    LanguageCode::LangZxx,
    LanguageCode::LangQme,
    LanguageCode::LangQst,
    LanguageCode::LangQht,
    LanguageCode::LangQct,
    LanguageCode::LangQam,
];

pub struct SerendipityStatsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for SerendipityStatsSideEffect {
    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };
        let Some(history) = input.query.stats_history_topics.as_ref() else {
            return Ok(());
        };
        let selected = &input.selected_candidates;
        if selected.is_empty() {
            return Ok(());
        }

        let experiment_buckets = input
            .query
            .params
            .experiment_buckets(EnableSerendipityStatsExperimentBucket);
        if experiment_buckets.is_empty() && random::<f64>() >= DEFAULT_SAMPLING_RATE {
            return Ok(());
        }
        let unbucketed = ExperimentBucket::new("", "");
        let buckets: Vec<&ExperimentBucket> = if experiment_buckets.is_empty() {
            vec![&unbucketed]
        } else {
            experiment_buckets
        };

        let mut final_order: Vec<&PostCandidate> = selected.iter().collect();
        final_order.sort_by(|a, b| {
            b.score
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&a.score.unwrap_or(f64::NEG_INFINITY))
                .unwrap_or(Ordering::Equal)
        });
        let observations = observations(
            history,
            &final_order,
            input.query.request_time_ms,
            &input.query.language_code,
        );
        for bucket in &buckets {
            record(receiver.as_ref(), &observations, bucket);
        }
        Ok(())
    }
}

enum Scale {
    Share,
    Small,
    Count,
}

struct HistorySets {
    l1: HashSet<u32>,
    l2: HashSet<(u32, u32)>,
}

impl HistorySets {
    fn new(history: &HistoryTopics) -> Self {
        Self {
            l1: history.l1_counts.keys().copied().collect(),
            l2: history.l2_counts.iter().map(|t| (t.l1, t.l2)).collect(),
        }
    }
}

fn topic(candidate: &PostCandidate) -> Option<(u32, Option<u32>)> {
    let l1 = u32::try_from(*candidate.semantic_id_prefix(1)?.first()?).ok()?;
    let l2 = candidate
        .semantic_id_prefix(2)
        .and_then(|codes| u32::try_from(codes[1]).ok());
    Some((l1, l2))
}

fn new_l1_share(posts: &[&PostCandidate], sets: &HistorySets) -> Option<f64> {
    let l1s: Vec<u32> = posts.iter().filter_map(|c| topic(c)).map(|t| t.0).collect();
    share(
        l1s.iter().filter(|l1| !sets.l1.contains(l1)).count(),
        l1s.len(),
    )
}

fn new_l2_share(posts: &[&PostCandidate], sets: &HistorySets) -> Option<f64> {
    let l2s: Vec<(u32, u32)> = posts
        .iter()
        .filter_map(|c| topic(c))
        .filter_map(|(l1, l2)| Some((l1, l2?)))
        .collect();
    share(
        l2s.iter().filter(|t| !sets.l2.contains(t)).count(),
        l2s.len(),
    )
}

fn share(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn is_linguistic(code: i32) -> bool {
    !NON_LINGUISTIC.iter().any(|c| *c as i32 == code)
}

fn post_language(candidate: &PostCandidate) -> Option<i32> {
    let code = language_code_string_to_enum(candidate.language_code.as_deref()?) as i32;
    is_linguistic(code).then_some(code)
}

fn user_languages(history: &HistoryTopics, app_language: &str) -> HashSet<i32> {
    let known: u32 = history
        .language_counts
        .iter()
        .filter(|(code, _)| is_linguistic(**code))
        .map(|(_, n)| *n)
        .sum();
    let mut languages: HashSet<i32> = history
        .language_counts
        .iter()
        .filter(|(code, n)| {
            is_linguistic(**code) && f64::from(**n) >= MIN_USER_LANGUAGE_SHARE * f64::from(known)
        })
        .map(|(code, _)| *code)
        .collect();
    let app = language_code_string_to_enum(app_language) as i32;
    if is_linguistic(app) {
        languages.insert(app);
    }
    languages
}

fn cross_language_share(posts: &[&PostCandidate], user_languages: &HashSet<i32>) -> Option<f64> {
    let languages: Vec<i32> = posts.iter().filter_map(|c| post_language(c)).collect();
    share(
        languages
            .iter()
            .filter(|l| !user_languages.contains(l))
            .count(),
        languages.len(),
    )
}

fn observations(
    history: &HistoryTopics,
    page: &[&PostCandidate],
    request_time_ms: i64,
    app_language: &str,
) -> Vec<(&'static str, f64, Scale)> {
    let topics: Vec<(u32, Option<u32>)> = page.iter().filter_map(|c| topic(c)).collect();
    let served_l1: HashSet<u32> = topics.iter().map(|t| t.0).collect();
    let served_l2: HashSet<(u32, u32)> = topics
        .iter()
        .filter_map(|&(l1, l2)| Some((l1, l2?)))
        .collect();

    let mut out = vec![
        (
            "HistoryEngagedRows",
            f64::from(history.engaged_rows),
            Scale::Count,
        ),
        (
            "HistoryTopicsL1",
            history.l1_counts.len() as f64,
            Scale::Count,
        ),
        (
            "HistoryTopicsL2",
            history.l2_counts.len() as f64,
            Scale::Count,
        ),
        ("ServedTopicsL1", served_l1.len() as f64, Scale::Small),
        ("ServedTopicsL2", served_l2.len() as f64, Scale::Small),
        (
            "CandidateSidCoverage",
            topics.len() as f64 / page.len() as f64,
            Scale::Share,
        ),
    ];
    if history.engaged_rows > 0 {
        out.push((
            "HistorySidCoverage",
            f64::from(history.sid_rows) / f64::from(history.engaged_rows),
            Scale::Share,
        ));
    }
    if let Some(oldest_ms) = history.oldest_row_ms {
        out.push((
            "HistorySpanDays",
            (request_time_ms - oldest_ms).max(0) as f64 / MS_PER_DAY,
            Scale::Small,
        ));
    }
    let languages = user_languages(history, app_language);
    let linguistic_rows: u32 = history
        .language_counts
        .iter()
        .filter(|(code, _)| is_linguistic(**code))
        .map(|(_, n)| *n)
        .sum();
    out.push(("UserLanguages", languages.len() as f64, Scale::Small));
    out.push((
        "CandidateLanguageCoverage",
        page.iter().filter(|c| post_language(c).is_some()).count() as f64 / page.len() as f64,
        Scale::Share,
    ));
    if history.engaged_rows > 0 {
        out.push((
            "HistoryLanguageCoverage",
            f64::from(linguistic_rows) / f64::from(history.engaged_rows),
            Scale::Share,
        ));
    }
    if !languages.is_empty() {
        let top = &page[..TOP_POSITIONS.min(page.len())];
        for (name, value) in [
            ("CrossLanguageShare", cross_language_share(page, &languages)),
            (
                "CrossLanguageShareTop10",
                cross_language_share(top, &languages),
            ),
        ] {
            if let Some(value) = value {
                out.push((name, value, Scale::Share));
            }
        }
    }
    if history.sid_rows > 0 {
        let sets = HistorySets::new(history);
        let top = &page[..TOP_POSITIONS.min(page.len())];
        for (name, value) in [
            ("NewTopicShareL1", new_l1_share(page, &sets)),
            ("NewTopicShareL2", new_l2_share(page, &sets)),
            ("NewTopicShareL1Top10", new_l1_share(top, &sets)),
            ("NewTopicShareL2Top10", new_l2_share(top, &sets)),
        ] {
            if let Some(value) = value {
                out.push((name, value, Scale::Share));
            }
        }
    }
    out
}

fn record(
    receiver: &dyn StatsReceiverExt,
    observations: &[(&'static str, f64, Scale)],
    bucket: &ExperimentBucket,
) {
    let scopes: [(&str, &str); 2] = [("ddg", &bucket.experiment), ("bucket", &bucket.bucket)];
    for (name, value, scale) in observations {
        let metric = format!("{METRIC_PREFIX}.{name}");
        match scale {
            Scale::Share => {
                receiver.observe(&metric, &scopes, *value, HistogramBuckets::Bucket0To1)
            }
            Scale::Small => {
                receiver.observe(&metric, &scopes, *value, HistogramBuckets::Bucket0To50)
            }
            Scale::Count => receiver.observe_expo(&metric, &scopes, *value),
        }
    }
}
