use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableNewUserMinEngagementFilter, NewUserMinEngagementFilterMaxAccountAgeSecs,
    NewUserMinEngagementFilterMaxResurrectionAgeSecs, NewUserMinEngagementFilterMetric,
    NewUserMinEngagementFilterUseRatio, NewUserMinEngagementThreshold,
};
use std::time::Duration;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct NewUserMinEngagementFilter;

#[derive(Clone, Copy)]
enum Metric {
    Fav,
    Engagement,
    View,
}

impl Metric {
    fn parse(s: &str) -> Self {
        match s {
            "engagement" => Metric::Engagement,
            "view" => Metric::View,
            _ => Metric::Fav,
        }
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for NewUserMinEngagementFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableNewUserMinEngagementFilter)
            && (within_account_age(query) || within_resurrection_window(query))
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let metric = Metric::parse(&query.params.get(NewUserMinEngagementFilterMetric));
        let use_ratio = query.params.get(NewUserMinEngagementFilterUseRatio);

        if candidates
            .iter()
            .all(|c| metric_value(c, metric, use_ratio).is_none())
        {
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }

        let threshold = query.params.get(NewUserMinEngagementThreshold);
        FilterResult::partition(candidates, |c| should_keep(c, metric, use_ratio, threshold))
    }
}

fn should_keep(candidate: &PostCandidate, metric: Metric, use_ratio: bool, threshold: f64) -> bool {
    if candidate.in_network != Some(false) {
        return true;
    }
    match metric_value(candidate, metric, use_ratio) {
        Some(value) => value >= threshold,
        None => true,
    }
}

fn within_account_age(query: &ScoredPostsQuery) -> bool {
    let max_age_secs = query
        .params
        .get(NewUserMinEngagementFilterMaxAccountAgeSecs);
    duration_since_creation_opt(query.user_id)
        .map(|age| age < Duration::from_secs(max_age_secs))
        .unwrap_or(false)
}

fn within_resurrection_window(query: &ScoredPostsQuery) -> bool {
    let max_age_secs = query
        .params
        .get(NewUserMinEngagementFilterMaxResurrectionAgeSecs);
    match query.resurrection_time_ms {
        Some(ts_ms) => {
            let age_ms = query.request_time_ms - ts_ms;
            age_ms >= 0 && age_ms < (max_age_secs as i64) * 1000
        }
        None => false,
    }
}

fn metric_value(candidate: &PostCandidate, metric: Metric, use_ratio: bool) -> Option<f64> {
    let numerator = match metric {
        Metric::Fav => candidate.fav_count?,
        Metric::Engagement => engagement_sum(candidate)?,
        Metric::View => return Some(candidate.view_count? as f64),
    } as f64;
    if use_ratio {
        let views = candidate.view_count.filter(|&v| v > 0)? as f64;
        Some(numerator / views)
    } else {
        Some(numerator)
    }
}

fn engagement_sum(candidate: &PostCandidate) -> Option<i64> {
    let fav = candidate.fav_count?;
    Some(
        fav + candidate.reply_count.unwrap_or(0)
            + candidate.repost_count.unwrap_or(0)
            + candidate.quote_count.unwrap_or(0),
    )
}
