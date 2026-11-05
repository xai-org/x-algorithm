use crate::models::viewer::Client;
use crate::result_label::{debug_label, state_label, Label, NONE};
use anyhow::{anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;
use tonic::metadata::MetadataMap;
use tonic::transport::Channel;
use xai_safety_label_store::types::{label_meta, label_type_from_ident};
use xai_visibility_filtering::evaluated::EvaluationResult;
use xai_visibility_filtering::vf_client::XaiVfClient;
use xai_visibility_filtering_proto as vf_pb;
use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;
use xai_x_thrift::action::{Action, InterstitialAction, MediaInterstitial};
use xai_x_thrift::get_tweet_fields::TweetFieldsResultState;
use xai_x_thrift::safety_level::SafetyLevel;
use xai_x_thrift::safety_result::FilteredReason;
use xai_x_thrift::tweet_safety_label::SafetyLabelType;

pub const GRPC_PORT: u16 = 50051;

pub fn tls_domain(dc: &str) -> String {
    format!("visibility.visibility-filtering-service.staging.{dc}.s2s.twttr.net")
}

const CALL_TIMEOUT_MS: u64 = 5_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseFile {
    fixtures: BTreeMap<String, u64>,
    cases: Vec<CaseSpec>,
    #[serde(default)]
    safety_labels: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseSpec {
    #[serde(default)]
    viewer: Option<String>,
    post: String,
    #[serde(default = "us")]
    country: String,
    #[serde(default)]
    client: Client,
    #[serde(default)]
    quoted_in: Option<String>,
    expect: VerdictFields,
    #[serde(default)]
    filter_tweets: BTreeMap<FilterLevel, VerdictFields>,
}

const US: &str = "us";

fn us() -> String {
    US.to_string()
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, strum::IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
enum FilterLevel {
    TimelineHome,
    TimelineHomeRecommendations,
    ImmersiveExpandedRecommendations,
}

impl FilterLevel {
    fn proto(self) -> vf_pb::SafetyLevel {
        match self {
            Self::TimelineHome => vf_pb::SafetyLevel::TimelineHome,
            Self::TimelineHomeRecommendations => vf_pb::SafetyLevel::TimelineHomeRecommendations,
            Self::ImmersiveExpandedRecommendations => {
                vf_pb::SafetyLevel::ImmersiveExpandedRecommendations
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerdictFields {
    class: String,
    reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum Expect {
    Verdict(VerdictFields),
    SafetyLabels(BTreeSet<String>),
}

pub struct Case {
    name: String,
    viewer_id: Option<u64>,
    post_id: u64,
    country: String,
    client: Client,
    quoted_in: Option<u64>,
    filter_tweets_at: Option<FilterLevel>,
    expect: Expect,
}

pub fn parse_cases(json: &str) -> anyhow::Result<Vec<Case>> {
    let file: CaseFile = serde_json::from_str(json)?;
    let id = |alias: &str| {
        file.fixtures
            .get(alias)
            .copied()
            .ok_or_else(|| anyhow!("no fixture {alias}"))
    };
    let mut calls = Vec::new();
    for case in &file.cases {
        let viewer = case.viewer.as_deref().unwrap_or("logged_out");
        let country = case.country.to_ascii_lowercase();
        let quote_suffix = case
            .quoted_in
            .as_ref()
            .map(|outer| format!(" quoted in {outer}"))
            .unwrap_or_default();
        let origin_suffix = if country == US && case.client == Client::Web {
            String::new()
        } else {
            format!(" from {country} on {:?}", case.client)
        };
        let name = format!("{viewer} x {}{quote_suffix}{origin_suffix}", case.post);
        if case.quoted_in.is_some() && !case.filter_tweets.is_empty() {
            bail!("{name}: FilterTweets carries no quote");
        }
        let call = |name: String, filter_tweets_at, expect: &VerdictFields| {
            anyhow::Ok(Case {
                name,
                viewer_id: case.viewer.as_deref().map(id).transpose()?,
                post_id: id(&case.post)?,
                country: country.clone(),
                client: case.client,
                quoted_in: case.quoted_in.as_deref().map(id).transpose()?,
                filter_tweets_at,
                expect: Expect::Verdict(expect.clone()),
            })
        };
        calls.push(call(name.clone(), None, &case.expect)?);
        for (level, expect) in &case.filter_tweets {
            let on = format!("{name} on filter_tweets/{}", <&str>::from(level));
            calls.push(call(on, Some(*level), expect)?);
        }
    }
    for (post, labels) in &file.safety_labels {
        if let Some(unknown) = labels.iter().find(|l| label_type_from_ident(l).is_none()) {
            bail!("{post}: no label type {unknown}");
        }
        calls.push(Case {
            name: format!("{post} on get_safety_labels"),
            viewer_id: None,
            post_id: id(post)?,
            country: us(),
            client: Client::Web,
            quoted_in: None,
            filter_tweets_at: None,
            expect: Expect::SafetyLabels(labels.clone()),
        });
    }
    Ok(calls)
}

pub enum Answer {
    EvaluateTweets(anyhow::Result<Vec<EvaluationResult>>),
    FilterTweets(anyhow::Result<vf_pb::VisibilityFilterResponse>),
    GetSafetyLabels(anyhow::Result<vf_pb::GetSafetyLabelsResponse>),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
enum CaseResult {
    Pass { got: Expect },
    Wrong { got: Expect },
    CallFailed { error: String },
}

#[derive(Serialize)]
pub struct CaseLine<'a> {
    case: &'a str,
    expect: &'a Expect,
    #[serde(flatten)]
    result: CaseResult,
}

impl Case {
    pub async fn ask(&self, channel: &Channel) -> Answer {
        if let Expect::SafetyLabels(_) = self.expect {
            return Answer::GetSafetyLabels(
                async {
                    let mut request = tonic::Request::new(vf_pb::GetSafetyLabelsRequest {
                        tweet_ids: vec![self.post_id],
                    });
                    request.set_timeout(Duration::from_millis(CALL_TIMEOUT_MS));
                    let response = VisibilityFilteringServiceClient::new(channel.clone())
                        .get_safety_labels(request)
                        .await?;
                    anyhow::Ok(response.into_inner())
                }
                .await,
            );
        }
        let Some(level) = self.filter_tweets_at else {
            return Answer::EvaluateTweets(
                async {
                    XaiVfClient::from_channel(channel.clone())
                        .with_timeout_ms(CALL_TIMEOUT_MS)
                        .evaluate_tweets(self.request(), &self.metadata()?)
                        .await
                }
                .await,
            );
        };
        Answer::FilterTweets(
            async {
                let mut request = tonic::Request::new(self.filter_request(level));
                *request.metadata_mut() = self.metadata()?;
                request.set_timeout(Duration::from_millis(CALL_TIMEOUT_MS));
                let response = VisibilityFilteringServiceClient::new(channel.clone())
                    .filter_tweets(request)
                    .await?;
                anyhow::Ok(response.into_inner())
            }
            .await,
        )
    }

    fn request(&self) -> vf_pb::EvaluateTweetsRequest {
        let post = vf_pb::TweetData {
            tweet_id: self.post_id,
            quote_context: self
                .quoted_in
                .map(|outer_tweet_id| vf_pb::QuoteContext { outer_tweet_id }),
        };
        let outer = self.quoted_in.map(|tweet_id| vf_pb::TweetData {
            tweet_id,
            quote_context: None,
        });
        vf_pb::EvaluateTweetsRequest {
            safety_level: SafetyLevel::TIMELINE_HOME_HYDRATION.0,
            tweets: std::iter::once(post).chain(outer).collect(),
            viewer_id: self.viewer_id,
        }
    }

    fn filter_request(&self, level: FilterLevel) -> vf_pb::VisibilityFilterRequest {
        vf_pb::VisibilityFilterRequest {
            safety_level: level.proto().into(),
            tweets: vec![vf_pb::TweetInput {
                tweet_id: self.post_id,
                author_id: None,
            }],
            viewer_id: self.viewer_id,
            country_code: Some(self.country.to_ascii_uppercase()),
        }
    }

    fn metadata(&self) -> anyhow::Result<MetadataMap> {
        let context = self.client.context(self.viewer_id, Some(&self.country));
        let mut metadata = MetadataMap::new();
        metadata.insert(
            "twittercontext",
            xai_twittercontext::hydrate_twitter_context(&context)?,
        );
        Ok(metadata)
    }

    pub fn judge(&self, answer: &Answer) -> CaseLine<'_> {
        let judged = |got: Expect| {
            if got == self.expect {
                CaseResult::Pass { got }
            } else {
                CaseResult::Wrong { got }
            }
        };
        let verdict =
            |(class, reason): Label| judged(Expect::Verdict(VerdictFields { class, reason }));
        let failed = |error: String| CaseResult::CallFailed { error };
        let result = match answer {
            Answer::EvaluateTweets(answer) => {
                match answer.as_ref().map(|results| results.first()) {
                    Ok(Some(EvaluationResult::Evaluated(state))) => verdict(gate_label(state)),
                    Ok(Some(EvaluationResult::NotEvaluated)) => {
                        verdict(("not_evaluated".to_string(), NONE.to_string()))
                    }
                    Ok(Some(EvaluationResult::Failed)) => failed("VF answered failed".to_string()),
                    Ok(None) => failed("no result".to_string()),
                    Err(err) => failed(format!("{err:#}")),
                }
            }
            Answer::FilterTweets(answer) => match answer.as_ref().map(|r| r.results.first()) {
                Ok(Some(result)) => verdict(filter_label(result)),
                Ok(None) => failed("no result".to_string()),
                Err(err) => failed(format!("{err:#}")),
            },
            Answer::GetSafetyLabels(answer) => match answer {
                Ok(response) if response.failed_ids.contains(&self.post_id) => {
                    failed("VF answered failed".to_string())
                }
                Ok(response) => match response.results.get(&self.post_id) {
                    Some(labels) => judged(Expect::SafetyLabels(label_types(labels))),
                    None => failed("no result".to_string()),
                },
                Err(err) => failed(format!("{err:#}")),
            },
        };
        CaseLine {
            case: &self.name,
            expect: &self.expect,
            result,
        }
    }
}

fn gate_label(state: &TweetFieldsResultState) -> Label {
    let (class, mut reason) = state_label(state);
    let mut append = |suffix: &str| {
        reason = if reason == NONE {
            suffix.to_string()
        } else {
            format!("{reason}/{suffix}")
        };
    };
    if let TweetFieldsResultState::NotFound(not_found) = state {
        for (bit, name) in [
            (not_found.deleted, "deleted"),
            (not_found.bounce_deleted, "bounce_deleted"),
        ] {
            if bit == Some(true) {
                append(name);
            }
        }
    }
    if offers_age_verification(state) {
        reason = format!("{reason}/age_prompt");
    }
    (class, reason)
}

fn offers_age_verification(state: &TweetFieldsResultState) -> bool {
    let TweetFieldsResultState::Found(found) = state else {
        return false;
    };
    let Some(FilteredReason::SafetyResult(result)) = &found.suppress_reason else {
        return false;
    };
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "Action is a generated Thrift union; only these two carry a media blur"
    )]
    let media = match &result.action {
        Action::ComposedMediaVisibilityResults(media) => Some(media),
        Action::TweetInterstitial(interstitial) => {
            interstitial.all_media_visibility_results.as_ref()
        }
        _ => None,
    };
    media
        .and_then(|media| media.media_interstitial.as_deref())
        .is_some_and(|MediaInterstitial::BlurredImageInterstitial(blur)| {
            blur.interstitial_action == Some(InterstitialAction::AGE_VERIFICATION_PROMPT)
        })
}

fn filter_label(result: &vf_pb::TweetVisibilityResult) -> Label {
    let kind = result.action.and_then(|action| action.kind);
    let reason = result
        .filtered_reason
        .as_ref()
        .and_then(|r| r.reason.as_ref());
    (
        kind.map_or_else(|| NONE.to_string(), |kind| debug_label(&kind).0),
        reason.map_or_else(|| NONE.to_string(), |reason| debug_label(reason).0),
    )
}

fn label_types(labels: &vf_pb::SafetyLabelMap) -> BTreeSet<String> {
    labels
        .labels
        .keys()
        .filter_map(|&value| label_meta(SafetyLabelType::from(value)))
        .map(|meta| meta.ident.to_owned())
        .collect()
}

impl CaseLine<'_> {
    pub fn passed(&self) -> bool {
        matches!(self.result, CaseResult::Pass { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_x_thrift::tweet_service::TweetFieldsResultFiltered;

    fn read(path: &str) -> String {
        std::fs::read_to_string(crate::pinned::crate_path(path)).unwrap()
    }

    #[derive(Deserialize)]
    struct Fixtures {
        users: Vec<Fixture>,
        tweets: Vec<Fixture>,
    }

    #[derive(Deserialize)]
    struct Fixture {
        id: u64,
        alias: Option<String>,
    }

    fn fields(class: &str, reason: &str) -> Expect {
        Expect::Verdict(VerdictFields {
            class: class.to_string(),
            reason: reason.to_string(),
        })
    }

    fn case(quoted_in: Option<u64>) -> Case {
        Case {
            name: "viewer x post".to_string(),
            viewer_id: Some(1),
            post_id: 2,
            country: us(),
            client: Client::Web,
            quoted_in,
            filter_tweets_at: None,
            expect: fields("drop", "author_is_suspended"),
        }
    }

    fn evaluated(results: Vec<EvaluationResult>) -> Answer {
        Answer::EvaluateTweets(Ok(results))
    }

    fn filtered(reason: FilteredReason) -> EvaluationResult {
        EvaluationResult::Evaluated(Box::new(TweetFieldsResultState::Filtered(
            TweetFieldsResultFiltered::new(reason),
        )))
    }

    #[test]
    fn each_case_names_fixtures_with_their_fixtures_json_ids() {
        let text = read("tests/e2e/verdict-gate.json");
        let cases: CaseFile = serde_json::from_str(&text).unwrap();
        let fixtures: Fixtures = serde_json::from_str(&read("tests/e2e/fixtures.json")).unwrap();
        let ids: BTreeMap<String, u64> = fixtures
            .users
            .into_iter()
            .chain(fixtures.tweets)
            .filter_map(|fixture| Some((fixture.alias?, fixture.id)))
            .collect();
        for (alias, id) in &cases.fixtures {
            assert_eq!(ids.get(alias), Some(id), "{alias}");
        }
        parse_cases(&text).unwrap();
    }

    #[test]
    fn parse_refuses_an_unknown_alias_and_filter_tweets_on_a_quote() {
        let file = |case: &str| format!(r#"{{"fixtures": {{"v": 1, "p": 2}}, "cases": [{case}]}}"#);
        let expect = r#""expect": {"class": "allow", "reason": "none"}"#;
        let filter = r#""filter_tweets": {"timeline_home": {"class": "allow", "reason": "none"}}"#;
        let known = format!(r#"{{"viewer": "v", "post": "p", {expect}}}"#);
        assert_eq!(parse_cases(&file(&known)).unwrap()[0].name, "v x p");
        for case in [
            format!(r#"{{"viewer": "v", "post": "q", {expect}}}"#),
            format!(r#"{{"viewer": "v", "post": "p", "quoted_in": "q", {expect}}}"#),
            format!(r#"{{"viewer": "v", "post": "p", "quoted_in": "p", {expect}, {filter}}}"#),
            format!(
                r#"{{"viewer": "v", "post": "p", {expect}, "filter_tweets": {{"timeline_home_hydration": {{"class": "allow", "reason": "none"}}}}}}"#
            ),
        ] {
            assert!(parse_cases(&file(&case)).is_err(), "{case}");
        }
    }

    #[test]
    fn parse_adds_a_filter_tweets_call_per_listed_level() {
        let text = r#"{"fixtures": {"v": 1, "p": 2}, "cases": [{"viewer": "v", "post": "p",
            "expect": {"class": "allow", "reason": "none"},
            "filter_tweets": {"timeline_home_recommendations": {"class": "drop", "reason": "contain_nsfw_media"},
                              "timeline_home": {"class": "allow", "reason": "none"}}}]}"#;
        let calls = parse_cases(text).unwrap();
        let named: Vec<_> = calls
            .iter()
            .map(|call| (call.name.as_str(), call.expect.clone()))
            .collect();
        let allow = fields("allow", "none");
        assert_eq!(
            named,
            [
                ("v x p", allow.clone()),
                ("v x p on filter_tweets/timeline_home", allow),
                (
                    "v x p on filter_tweets/timeline_home_recommendations",
                    fields("drop", "contain_nsfw_media")
                ),
            ]
        );
        let request = calls[2].filter_request(calls[2].filter_tweets_at.unwrap());
        assert_eq!(
            (
                request.safety_level,
                request.viewer_id,
                request.tweets[0].tweet_id
            ),
            (
                vf_pb::SafetyLevel::TimelineHomeRecommendations.into(),
                Some(1),
                2
            )
        );
    }

    #[test]
    fn a_case_sends_its_country_and_client_and_no_viewer_when_it_names_none() {
        let text = r#"{"fixtures": {"v": 1, "p": 2}, "cases": [
            {"viewer": "v", "post": "p", "expect": {"class": "allow", "reason": "none"},
             "filter_tweets": {"timeline_home": {"class": "allow", "reason": "none"}}},
            {"post": "p", "country": "GB", "client": "ios_outdated",
             "expect": {"class": "allow", "reason": "none"},
             "filter_tweets": {"timeline_home": {"class": "allow", "reason": "none"}}}]}"#;
        let calls = parse_cases(text).unwrap();
        let sent: Vec<_> = calls
            .iter()
            .map(|call| {
                let filter_country = call
                    .filter_tweets_at
                    .and_then(|level| call.filter_request(level).country_code);
                (
                    call.name.as_str(),
                    call.request().viewer_id,
                    filter_country,
                    (call.country.as_str(), call.client),
                )
            })
            .collect();
        let us_web = ("us", Client::Web);
        let gb_ios = ("gb", Client::IosOutdated);
        let gb_case = "logged_out x p from gb on IosOutdated";
        let gb_filter = format!("{gb_case} on filter_tweets/timeline_home");
        assert_eq!(
            sent,
            [
                ("v x p", Some(1), None, us_web),
                (
                    "v x p on filter_tweets/timeline_home",
                    Some(1),
                    Some("US".to_string()),
                    us_web
                ),
                (gb_case, None, None, gb_ios),
                (gb_filter.as_str(), None, Some("GB".to_string()), gb_ios),
            ]
        );
    }

    #[test]
    fn request_sends_level_82_with_a_quote_s_outer_after_the_post() {
        assert_eq!(case(None).request().safety_level, 82);
        let tweets = case(Some(3)).request().tweets;
        let sent: Vec<_> = tweets
            .iter()
            .map(|tweet| {
                (
                    tweet.tweet_id,
                    tweet.quote_context.map(|q| q.outer_tweet_id),
                )
            })
            .collect();
        assert_eq!(sent, [(2, Some(3)), (3, None)]);
    }

    #[test]
    fn judge_compares_the_verdict_fields_of_the_first_result() {
        let case = case(None);
        let suspended = filtered(FilteredReason::AuthorIsSuspended(true));
        let pass = case.judge(&evaluated(vec![suspended, EvaluationResult::Failed]));
        assert_eq!(
            pass.result,
            CaseResult::Pass {
                got: case.expect.clone()
            }
        );
        for (answer, got) in [
            (
                filtered(FilteredReason::TweetIsBounced(true)),
                fields("drop", "tweet_is_bounced"),
            ),
            (
                EvaluationResult::NotEvaluated,
                fields("not_evaluated", "none"),
            ),
        ] {
            assert_eq!(
                case.judge(&evaluated(vec![answer])).result,
                CaseResult::Wrong { got }
            );
        }
    }

    #[test]
    fn a_blur_offering_age_verification_labels_its_reason_age_prompt() {
        use xai_x_thrift::action::{
            BlurredImageInterstitial, ComposedMediaVisibilityActions, InterstitialReason,
        };
        use xai_x_thrift::safety_result::SafetyResult;
        use xai_x_thrift::tweet_service::TweetFieldsResultFound;
        let blur = |interstitial_action| {
            let media = ComposedMediaVisibilityActions {
                media_interstitial: Some(Box::new(MediaInterstitial::BlurredImageInterstitial(
                    BlurredImageInterstitial {
                        reason: Some(InterstitialReason::Nudity(true)),
                        opacity: None,
                        interstitial_action,
                        available_verification_options: None,
                    },
                ))),
            };
            let result = SafetyResult {
                reason: None,
                action: Action::ComposedMediaVisibilityResults(media),
            };
            EvaluationResult::Evaluated(Box::new(TweetFieldsResultState::Found(
                TweetFieldsResultFound::new(FilteredReason::SafetyResult(result)),
            )))
        };
        let got = |answer| match case(None).judge(&evaluated(vec![answer])).result {
            CaseResult::Wrong { got } | CaseResult::Pass { got } => got,
            CaseResult::CallFailed { error } => panic!("{error}"),
        };
        assert_eq!(
            got(blur(Some(InterstitialAction::AGE_VERIFICATION_PROMPT))),
            fields("composed_media_visibility_results", "nudity/age_prompt")
        );
        assert_eq!(
            got(blur(None)),
            fields("composed_media_visibility_results", "nudity")
        );
    }

    #[test]
    fn a_not_found_labels_its_reason_with_its_set_deletion_bits() {
        use xai_x_thrift::tweet_service::TweetFieldsResultNotFound;
        let label = |deleted, bounce_deleted| {
            gate_label(&TweetFieldsResultState::NotFound(
                TweetFieldsResultNotFound::new(deleted, bounce_deleted, None),
            ))
        };
        let not_found = |reason: &str| ("not_found".to_string(), reason.to_string());
        assert_eq!(label(false, false), not_found("none"));
        assert_eq!(label(true, false), not_found("deleted"));
        assert_eq!(label(true, true), not_found("deleted/bounce_deleted"));
    }

    #[test]
    fn judge_labels_a_filter_tweets_action_and_reason() {
        let case = case(None);
        let answer = |kind, reason: Option<vf_pb::filtered_reason::Reason>| {
            Answer::FilterTweets(Ok(vf_pb::VisibilityFilterResponse {
                results: vec![vf_pb::TweetVisibilityResult {
                    tweet_id: 2,
                    action: Some(vf_pb::Action { kind: Some(kind) }),
                    filtered_reason: reason.map(|reason| vf_pb::FilteredReason {
                        reason: Some(reason),
                    }),
                    safety_labels: None,
                }],
            }))
        };
        let drop = vf_pb::action::Kind::Drop(vf_pb::DropReason {});
        let suspended = vf_pb::filtered_reason::Reason::AuthorIsSuspended(true);
        assert_eq!(
            case.judge(&answer(drop, Some(suspended))).result,
            CaseResult::Pass {
                got: case.expect.clone()
            }
        );
        assert_eq!(
            case.judge(&answer(vf_pb::action::Kind::Allow(true), None))
                .result,
            CaseResult::Wrong {
                got: fields("allow", "none")
            }
        );
    }

    #[test]
    fn judge_counts_an_error_no_result_or_a_failed_answer_as_a_failed_call() {
        let case = case(None);
        for answer in [
            Answer::EvaluateTweets(Err(anyhow!("deadline exceeded"))),
            evaluated(vec![]),
            evaluated(vec![EvaluationResult::Failed]),
            Answer::FilterTweets(Err(anyhow!("deadline exceeded"))),
            Answer::FilterTweets(Ok(vf_pb::VisibilityFilterResponse { results: vec![] })),
        ] {
            let result = case.judge(&answer).result;
            assert!(
                matches!(result, CaseResult::CallFailed { .. }),
                "{result:?}"
            );
        }
    }

    #[test]
    fn parse_adds_a_get_safety_labels_call_per_listed_post() {
        let text = r#"{"fixtures": {"p": 2, "q": 3}, "cases": [],
            "safety_labels": {"p": ["NSFW_HIGH_PRECISION", "BOUNCE"], "q": []}}"#;
        let sent: Vec<_> = parse_cases(text)
            .unwrap()
            .into_iter()
            .map(|call| (call.name, call.post_id, call.expect))
            .collect();
        let labels = |names: &[&str]| {
            Expect::SafetyLabels(names.iter().map(|name| name.to_string()).collect())
        };
        assert_eq!(
            sent,
            [
                (
                    "p on get_safety_labels".to_string(),
                    2,
                    labels(&["BOUNCE", "NSFW_HIGH_PRECISION"])
                ),
                ("q on get_safety_labels".to_string(), 3, labels(&[])),
            ]
        );
        for unknown in [
            r#"{"fixtures": {}, "cases": [], "safety_labels": {"p": []}}"#,
            r#"{"fixtures": {"p": 2}, "cases": [], "safety_labels": {"p": ["BOUNC"]}}"#,
        ] {
            assert!(parse_cases(unknown).is_err(), "{unknown}");
        }
    }

    #[test]
    fn judge_compares_the_post_s_label_type_names() {
        let case = parse_cases(
            r#"{"fixtures": {"p": 2}, "cases": [], "safety_labels": {"p": ["BOUNCE", "NSFW_HIGH_PRECISION"]}}"#,
        )
        .unwrap()
        .remove(0);
        let label_map = |types: &[SafetyLabelType]| vf_pb::SafetyLabelMap {
            labels: types
                .iter()
                .map(|&label_type| (i32::from(label_type), vf_pb::SafetyLabel::default()))
                .collect(),
        };
        let answer = |results: Vec<(u64, vf_pb::SafetyLabelMap)>, failed_ids| {
            Answer::GetSafetyLabels(Ok(vf_pb::GetSafetyLabelsResponse {
                results: results.into_iter().collect(),
                failed_ids,
            }))
        };
        let both = label_map(&[
            SafetyLabelType::NSFW_HIGH_PRECISION,
            SafetyLabelType::BOUNCE,
        ]);
        assert_eq!(
            case.judge(&answer(vec![(2, both)], vec![])).result,
            CaseResult::Pass {
                got: case.expect.clone()
            }
        );
        let unmapped = label_map(&[
            SafetyLabelType::BOUNCE,
            SafetyLabelType::HIGH_SPAMMY_TWEET_CONTENT_SCORE,
        ]);
        assert_eq!(
            case.judge(&answer(vec![(2, unmapped)], vec![])).result,
            CaseResult::Wrong {
                got: Expect::SafetyLabels(BTreeSet::from(["BOUNCE".to_string()]))
            }
        );
        for failed in [
            answer(vec![(3, label_map(&[]))], vec![]),
            answer(vec![], vec![2]),
            Answer::GetSafetyLabels(Err(anyhow!("deadline exceeded"))),
        ] {
            let result = case.judge(&failed).result;
            assert!(
                matches!(result, CaseResult::CallFailed { .. }),
                "{result:?}"
            );
        }
    }
}
