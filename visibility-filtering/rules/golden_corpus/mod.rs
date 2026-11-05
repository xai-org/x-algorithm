mod age_gating;
mod age_verification;
mod article;
mod author_state;
mod baseline;
mod community;
mod conversation_control;
mod exclusive_content;
mod interstitial;
mod label_interstitial;
mod legacy_interstitial;
mod local_regulations;
mod local_tweet;
mod notifications;
mod oon_media;
mod oon_tweet_label;
mod oon_user_label;
mod post_notifications;
mod quoted_tweet_rules;
mod reaction_video;
mod relationship;
mod takedown;
mod trusted_friends;
mod tweet_label;
mod tweet_state;

use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    EdgeSet, Evaluation, NotificationKind, Placement, PostCandidate, TweetId, UserCandidate,
    Verdict, ViewerFeatures,
};
use crate::rules::engine::Policy;
use crate::rules::fixtures::{builder_call, logged_out_viewer, viewer, VIEWER_ID};
use crate::rules::levels::{LevelLists, LevelRules};
use crate::rules::rule_spec::engine::{Engine, UserEngine};
use crate::rules::rule_spec::laws::WIRED_DIFFERENTLY;
use crate::rules::rule_spec::rule::{NamedCond, Rule, RuleCond};
use crate::rules::{placed_test_context, RuleEngine, SafetyLevel};
use crate::treatment::proto_action;
use prost::Message;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::iter::once;
use strum::VariantArray;
use SafetyLevel::{FilterAll, TimelineHomeHydration, TimelineHomeRecommendations};

#[derive(Clone)]
enum Role {
    NonFollower,
    Author,
    Follower,
    LoggedOut,
    As(&'static str, ViewerFeatures),
    InnerQuoted(Box<Role>),
    RetweetSource(Box<Role>),
    Each(Vec<Role>),
}

#[derive(Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "test-only rows, built once per test, where the size costs nothing"
)]
pub(super) enum Subject {
    Post(PostCandidate),
    User(UserCandidate),
}

#[derive(Clone)]
struct Row {
    name: &'static str,
    subject: Subject,
    expect: Vec<(Role, Verdict)>,
    at: Vec<(SafetyLevel, Role, Verdict)>,
}

pub(super) struct CorpusCase {
    name: String,
    level: SafetyLevel,
    kind: Option<NotificationKind>,
    pub(super) viewer: ViewerFeatures,
    pub(super) subject: Subject,
    pub(super) placement: Placement,
    expected: Verdict,
}

struct RowCase {
    row: &'static str,
    role: String,
    provenance: Provenance,
    case: CorpusCase,
}

enum Provenance {
    Expect,
    At,
    InheritedFrom(SafetyLevel),
}

struct PostLevel {
    level: SafetyLevel,
    wired: Vec<&'static str>,
    wider: Vec<SafetyLevel>,
}

type View = (String, ViewerFeatures, Placement, Subject);

const OUTER_TWEET_ID: TweetId = TweetId(2);

impl Subject {
    fn author_id(&self) -> u64 {
        match self {
            Subject::Post(post) => post.author.id,
            Subject::User(user) => user.user.id,
        }
    }

    fn edges_mut(&mut self) -> &mut EdgeSet {
        match self {
            Subject::Post(post) => &mut post.edges,
            Subject::User(user) => &mut user.edges,
        }
    }
}

impl Role {
    fn each<const N: usize>(roles: [Role; N]) -> Role {
        const { assert!(N > 0, "Role::each needs at least one role") };
        Role::Each(roles.into())
    }

    fn views(&self, subject: &Subject) -> Vec<View> {
        let standalone = |name: &str, viewer, subject| {
            vec![(name.to_owned(), viewer, Placement::Standalone, subject)]
        };
        let mut subject = subject.clone();
        match self {
            Role::NonFollower => standalone("non_follower", viewer(VIEWER_ID), subject),
            Role::Author => standalone("author", viewer(subject.author_id()), subject),
            Role::Follower => {
                let edges = subject.edges_mut();
                *edges = edges.with(Hydrator::Follows);
                standalone("follower", viewer(VIEWER_ID), subject)
            }
            Role::LoggedOut => {
                *subject.edges_mut() = EdgeSet::default();
                if let Subject::Post(post) = &mut subject {
                    post.post.conversation_root_author_blocks_viewer = false;
                }
                standalone("logged_out", logged_out_viewer(), subject)
            }
            Role::As(role_name, viewer) => standalone(role_name, viewer.clone(), subject),
            Role::InnerQuoted(role) => role.placed(
                &subject,
                "inner_quoted",
                Placement::InnerQuoted {
                    outer: OUTER_TWEET_ID,
                },
            ),
            Role::RetweetSource(role) => {
                role.placed(&subject, "retweet_source", Placement::RetweetSource)
            }
            Role::Each(roles) => roles.iter().flat_map(|role| role.views(&subject)).collect(),
        }
    }

    fn placed(&self, subject: &Subject, prefix: &str, placement: Placement) -> Vec<View> {
        self.views(subject)
            .into_iter()
            .map(|(role_name, viewer, _, subject)| {
                (format!("{prefix}_{role_name}"), viewer, placement, subject)
            })
            .collect()
    }
}

fn preempted(level: SafetyLevel, rule: &str) -> bool {
    PREEMPTED
        .iter()
        .any(|preempted| preempted.level == level && preempted.rule == rule)
}

fn decides(level: SafetyLevel, wired: &[&'static str], verdict: &Verdict) -> bool {
    deciders(verdict)
        .into_iter()
        .all(|rule| wired.contains(&rule) && !preempted(level, rule))
}

fn reaches(level: SafetyLevel, wired: &[&'static str], verdict: &Verdict) -> bool {
    decides(level, wired, verdict)
        && deciders(verdict)
            .iter()
            .all(|rule| !WIRED_DIFFERENTLY.contains(rule))
}

fn is_subsequence<T: PartialEq>(items: &[T], of: &[T]) -> bool {
    let mut rest = of.iter();
    items
        .iter()
        .all(|item| rest.any(|candidate| candidate == item))
}

fn inherited_where_unstated(
    name: &str,
    subject: &Subject,
    at: &[(SafetyLevel, Role, Verdict)],
    post_levels: &[PostLevel],
) -> Vec<(SafetyLevel, View, Verdict, SafetyLevel)> {
    let stated: Vec<(SafetyLevel, String)> = at
        .iter()
        .flat_map(|(level, role, _)| {
            role.views(subject)
                .into_iter()
                .map(move |(role_name, ..)| (*level, role_name))
        })
        .collect();
    let mut inherited: Vec<(SafetyLevel, View, Verdict, SafetyLevel)> = Vec::new();
    for (wider, role, verdict) in at {
        for narrower in post_levels.iter().filter(|narrower| {
            narrower.wider.contains(wider) && decides(narrower.level, &narrower.wired, verdict)
        }) {
            for view in role.views(subject) {
                if stated.contains(&(narrower.level, view.0.clone())) {
                    continue;
                }
                match inherited.iter().find(|(level, other_view, ..)| {
                    *level == narrower.level && other_view.0 == view.0
                }) {
                    None => inherited.push((narrower.level, view, verdict.clone(), *wider)),
                    Some((_, _, other_verdict, other_wider)) => assert!(
                        other_verdict == verdict,
                        "{name}: {:?} inherits {other_verdict:?} for {} from {other_wider:?} and \
                         {verdict:?} from {wider:?}; add an at entry at {:?}",
                        narrower.level,
                        view.0,
                        narrower.level,
                    ),
                }
            }
        }
    }
    inherited
}

impl Row {
    fn cases(self, post_levels: &[PostLevel]) -> Vec<RowCase> {
        let Row {
            name,
            subject,
            expect,
            at,
        } = self;
        let row_case = |level, (role, viewer, placement, subject), expected, provenance| RowCase {
            row: name,
            case: CorpusCase {
                name: format!("{name}/{level:?}/{role}"),
                level,
                kind: None,
                viewer,
                subject,
                placement,
                expected,
            },
            role,
            provenance,
        };
        let mut cases: Vec<RowCase> = at
            .iter()
            .flat_map(|(level, role, verdict)| {
                assert_eq!(
                    matches!(level.spec().rules(), LevelRules::UserOnly(_)),
                    matches!(subject, Subject::User(_)),
                    "{name}: {level:?} judges the other subject kind",
                );
                role.views(&subject)
                    .into_iter()
                    .map(move |view| row_case(*level, view, verdict.clone(), Provenance::At))
            })
            .collect();
        cases.extend(
            inherited_where_unstated(name, &subject, &at, post_levels)
                .into_iter()
                .map(|(level, view, verdict, wider)| {
                    row_case(level, view, verdict, Provenance::InheritedFrom(wider))
                }),
        );
        let stated = |cases: &[RowCase], level, role: &str| {
            cases.iter().any(|case| {
                !matches!(case.provenance, Provenance::Expect)
                    && case.case.level == level
                    && case.role == role
            })
        };
        assert!(
            expect.is_empty() || matches!(subject, Subject::Post(_)),
            "{name}: a user row states each verdict at its level"
        );
        let decided: Vec<&Verdict> = expect
            .iter()
            .map(|(_, verdict)| verdict)
            .filter(|verdict| !deciders(verdict).is_empty())
            .collect();
        for (role, verdict) in &expect {
            let mut holds_somewhere = false;
            for view in role.views(&subject) {
                for PostLevel { level, wired, .. } in post_levels {
                    let holds = match (deciders(verdict).is_empty(), decided.is_empty()) {
                        (false, _) => reaches(*level, wired, verdict),
                        (true, false) => decided.iter().any(|other| reaches(*level, wired, other)),
                        (true, true) => *level != FilterAll,
                    };
                    if holds && !stated(&cases, *level, &view.0) {
                        holds_somewhere = true;
                        cases.push(row_case(
                            *level,
                            view.clone(),
                            verdict.clone(),
                            Provenance::Expect,
                        ));
                    }
                }
            }
            assert!(holds_somewhere, "{name}: {verdict:?} holds at no level");
        }
        cases
    }
}

fn post_levels() -> Vec<PostLevel> {
    let rule_engine = RuleEngine::for_tests();
    let lists: Vec<(SafetyLevel, LevelLists)> = SafetyLevel::VARIANTS
        .iter()
        .filter_map(|&level| match level.spec().rules() {
            LevelRules::UserThenPost(rules) => Some((level, rules.lists())),
            LevelRules::UserOnly(_) => None,
        })
        .collect();
    lists
        .iter()
        .map(|(level, rules)| PostLevel {
            level: *level,
            wired: rule_engine.wired_rule_names(*level),
            wider: lists
                .iter()
                .filter(|(other, other_rules)| {
                    other != level
                        && is_subsequence(&rules.user, &other_rules.user)
                        && is_subsequence(&rules.post, &other_rules.post)
                })
                .map(|(other, _)| *other)
                .collect(),
        })
        .collect()
}

fn row_cases() -> Vec<RowCase> {
    let post_levels = post_levels();
    rows()
        .into_iter()
        .flat_map(|row| row.cases(&post_levels))
        .collect()
}

#[test]
fn no_at_entry_restates_what_its_row_derives_without_it() {
    let post_levels = post_levels();
    let cases = |row: Row| {
        let mut cases: Vec<(String, Verdict)> = row
            .cases(&post_levels)
            .into_iter()
            .map(|row_case| (row_case.case.name, row_case.case.expected))
            .collect();
        cases.sort_by(|(left, _), (right, _)| left.cmp(right));
        cases
    };
    let mut redundant = Vec::new();
    for row in rows() {
        let with = cases(row.clone());
        for index in 0..row.at.len() {
            let mut without = row.clone();
            let (level, role, _) = without.at.remove(index);
            if cases(without) == with {
                let roles: Vec<String> = role
                    .views(&row.subject)
                    .into_iter()
                    .map(|(role_name, ..)| role_name)
                    .collect();
                redundant.push(format!("{}: ({level:?}, {})", row.name, roles.join(" ")));
            }
        }
    }
    redundant.sort();
    redundant.dedup();
    assert!(
        redundant.is_empty(),
        "{} at entries restate what their row's expect or a wider level's at entry derives; \
         delete each:\n{}",
        redundant.len(),
        redundant.join("\n")
    );
}

impl CorpusCase {
    fn evaluate(&self, rule_engine: &RuleEngine) -> Evaluation {
        match (&self.subject, self.kind) {
            (Subject::Post(candidate), None) => {
                rule_engine.evaluate(self.level, &self.viewer, candidate, self.placement)
            }
            (Subject::Post(candidate), Some(kind)) => rule_engine.evaluate_post_kind(
                self.level,
                kind,
                &self.viewer,
                candidate,
                self.placement,
            ),
            (Subject::User(user), _) => {
                rule_engine.evaluate_user(self.level, &self.viewer, user, self.placement)
            }
        }
    }
}

fn deciders(verdict: &Verdict) -> Vec<&'static str> {
    verdict.decided().map(|(_, by)| by).collect()
}

#[test]
fn golden_corpus_pins_policy_verdicts() {
    let rule_engine = RuleEngine::for_tests();
    let row_cases = row_cases();
    let kind_cases = post_notifications::cases();
    let cases = row_cases
        .iter()
        .map(|row_case| (&row_case.case, Some(row_case)))
        .chain(kind_cases.iter().map(|case| (case, None)));
    let mut names = BTreeSet::new();
    let mut failures = Vec::new();
    for (case, row_case) in cases {
        assert!(
            names.insert(&case.name),
            "duplicate corpus case {}",
            case.name
        );
        let verdict = case.evaluate(&rule_engine).into_verdict();
        if verdict == case.expected {
            if matches!(&case.expected, Verdict::Shown { media: Some(_), .. }) {
                let (action, reason) = proto_action(&verdict);
                assert_eq!(action.encode_to_vec(), [0x20, 0x01], "{}", case.name);
                assert_eq!(
                    reason.unwrap().encode_to_vec(),
                    [0x08, 0x01],
                    "{}",
                    case.name
                );
            }
            continue;
        }
        let got = builder_call(&verdict);
        let level = case.level;
        let fix = match row_case {
            Some(RowCase {
                row,
                role,
                provenance,
                ..
            }) => {
                let at_entry = format!("({level:?}, <role {role}>, {got})");
                let differs =
                    format!("where {level:?} rightly differs, add to row {row}'s at: {at_entry}");
                match provenance {
                    Provenance::At => format!("\n  row {row}'s at entry now gets {at_entry}"),
                    Provenance::InheritedFrom(wider) => format!(
                        "\n  row {row}'s at entry at {wider:?}, which {level:?} inherits, gets {got}: \
                         {differs}"
                    ),
                    Provenance::Expect => format!("\n  {differs}"),
                }
            }
            None => String::new(),
        };
        failures.push(format!(
            "{} [{level:?}]:\n  expected {:?}\n  got      {verdict:?}{fix}",
            case.name, case.expected,
        ));
    }
    assert!(
        failures.is_empty(),
        "{} corpus case(s) diverged:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore]
fn bench_corpus_evaluation() {
    use std::hint::black_box;
    use std::time::Instant;

    const ROUNDS: u32 = 2_000;
    let rule_engine = RuleEngine::for_tests();
    let cases = corpus();
    let home_hydration: Vec<&CorpusCase> = cases
        .iter()
        .filter(|case| case.level == TimelineHomeHydration)
        .collect();
    for (label, cases) in [
        ("all", cases.iter().collect::<Vec<_>>()),
        ("home_hydration", home_hydration),
    ] {
        let start = Instant::now();
        for _ in 0..ROUNDS {
            for case in &cases {
                black_box(black_box(case).evaluate(&rule_engine));
            }
        }
        let evaluations = u128::from(ROUNDS) * cases.len() as u128;
        println!(
            "BENCH {label}: {} cases, {} ns/evaluation",
            cases.len(),
            start.elapsed().as_nanos() / evaluations.max(1)
        );
    }
}

#[test]
fn every_node_failing_changes_no_corpus_verdict_and_only_a_must_have_one_makes_it_partial() {
    let rule_engine = RuleEngine::for_tests();
    let best_effort = Hydrators::all().difference(Hydrators::all().must_have());
    for mut case in corpus() {
        for (failed, is_partial) in [(best_effort, false), (Hydrators::all(), true)] {
            match &mut case.subject {
                Subject::Post(post) => post.failed = failed,
                Subject::User(user) => user.failed = failed,
            }
            let evaluation = case.evaluate(&rule_engine);
            assert_eq!(
                matches!(evaluation, Evaluation::Partial { .. }),
                is_partial,
                "{} with {failed:?} failed",
                case.name
            );
            assert_eq!(evaluation.into_verdict(), case.expected, "{}", case.name);
        }
    }
}

fn lifted(condition: RuleCond<UserEngine>) -> RuleCond {
    let children = |children: &[RuleCond<UserEngine>]| -> &'static [RuleCond] {
        children
            .iter()
            .map(|&child| lifted(child))
            .collect::<Vec<_>>()
            .leak()
    };
    match condition {
        RuleCond::Leaf(leaf) => RuleCond::Leaf(UserEngine::lift(leaf)),
        RuleCond::All(all) => RuleCond::All(children(all)),
        RuleCond::Any(any) => RuleCond::Any(children(any)),
        RuleCond::Not(child) => RuleCond::Not(Box::leak(Box::new(lifted(*child)))),
        RuleCond::Named(named) => RuleCond::Named(Box::leak(Box::new(NamedCond {
            name: named.name,
            condition: lifted(named.condition),
        }))),
    }
}

fn one_fold(LevelLists { user, post }: &LevelLists) -> Policy {
    let user = user.iter().map(|rule| Rule {
        id: rule.id,
        audience: rule.audience,
        when: lifted(rule.when),
        action: rule.action.lift(),
    });
    let rules = user.chain(post.iter().map(|&rule| rule.clone()));
    Policy::new(rules.map(|rule| (&*rule.name().leak(), rule)).collect())
}

fn outcome(evaluation: &Evaluation) -> (&'static str, Vec<&'static str>) {
    let completeness = match evaluation {
        Evaluation::Complete { .. } => "Complete",
        Evaluation::Partial { .. } => "Partial",
        Evaluation::NotFound(_) | Evaluation::Deleted(_) => "NotFound",
        Evaluation::Failed(_) => "Failed",
    };
    let mut kinds: Vec<&'static str> = evaluation
        .verdict()
        .decided()
        .map(|(kind, _)| kind.into())
        .collect();
    if kinds.is_empty() {
        kinds.push("Allow");
    }
    (completeness, kinds)
}

#[test]
fn the_user_gate_keeps_the_outcome_of_one_severity_sorted_fold() {
    let rule_engine = RuleEngine::for_tests();
    let cases = corpus();
    let policies: Vec<_> = SafetyLevel::VARIANTS
        .iter()
        .flat_map(|&level| {
            let spec = level.spec();
            once((level, None, spec.rules().lists())).chain(
                level
                    .post_kinds()
                    .map(move |(kind, rules)| (level, Some(kind), rules.lists())),
            )
        })
        .collect();
    let diverged: Vec<String> = policies
        .par_iter()
        .flat_map_iter(|&(level, kind, ref rules)| {
            let mut diverged = Vec::new();
            let reference = one_fold(rules);
            let used = rules
                .user
                .iter()
                .map(|rule| rule.requirements())
                .chain(rules.post.iter().map(|rule| rule.requirements()))
                .fold(Hydrators::empty(), Hydrators::union);
            let failures: Vec<Hydrators> = once(Hydrators::empty())
                .chain(used.iter().map(Hydrators::of))
                .chain(once(used))
                .collect();
            for case in &cases {
                let Subject::Post(fetched) = &case.subject else {
                    continue;
                };
                for &failed in &failures {
                    let candidate = PostCandidate {
                        failed,
                        ..fetched.clone()
                    };
                    let gate = match kind {
                        None => {
                            rule_engine.evaluate(level, &case.viewer, &candidate, case.placement)
                        }
                        Some(kind) => rule_engine.evaluate_post_kind(
                            level,
                            kind,
                            &case.viewer,
                            &candidate,
                            case.placement,
                        ),
                    };
                    let fold = reference.evaluate(&placed_test_context(
                        &case.viewer,
                        &candidate,
                        case.placement,
                    ));
                    if outcome(&gate) != outcome(&fold) {
                        diverged.push(format!(
                            "{level:?} {kind:?} {} with {:?} failed:\n  gate {gate:?}\n  fold {fold:?}",
                            case.name,
                            failed.iter().collect::<Vec<Hydrator>>()
                        ));
                    }
                }
            }
            diverged
        })
        .collect();
    assert!(
        diverged.is_empty(),
        "{} evaluation(s) left the outcome of one severity-sorted fold:\n{}",
        diverged.len(),
        diverged.join("\n")
    );
}

struct Preempted {
    level: SafetyLevel,
    rule: &'static str,
    preempted_by: &'static [&'static str],
}

const PREEMPTED: [Preempted; 6] = [
    Preempted {
        level: TimelineHomeRecommendations,
        rule: "nsfw_high_precision/blur/nudity",
        preempted_by: &["nsfw_high_precision/drop/nsfw_media"],
    },
    Preempted {
        level: TimelineHomeRecommendations,
        rule: "nsfw_high_precision/blur/sensitive",
        preempted_by: &["nsfw_high_precision/drop/nsfw_media"],
    },
    Preempted {
        level: TimelineHomeRecommendations,
        rule: "gore_and_violence_high_precision/blur",
        preempted_by: &["gore_and_violence_high_precision/drop/nsfw_media"],
    },
    Preempted {
        level: TimelineHomeRecommendations,
        rule: "nsfw_card_image/blur/sensitive",
        preempted_by: &["nsfw_card_image/drop/nsfw_media"],
    },
    Preempted {
        level: TimelineHomeRecommendations,
        rule: "nsfw_admin/blur/sensitive",
        preempted_by: &[
            "nsfw_admin_author/drop/nsfw_media",
            "nsfw_admin_tweet_flag/drop/nsfw_media",
        ],
    },
    Preempted {
        level: TimelineHomeRecommendations,
        rule: "nsfw_user/blur/sensitive_user",
        preempted_by: &[
            "nsfw_user_author/drop/nsfw_media",
            "nsfw_user_tweet_flag/drop/nsfw_media",
        ],
    },
];

#[test]
fn every_wired_rule_decides_a_corpus_case_at_its_level() {
    let rule_engine = RuleEngine::for_tests();
    let cases = corpus();
    let decided: BTreeSet<(&'static str, Option<NotificationKind>, &'static str)> = cases
        .iter()
        .flat_map(|c| {
            let (level, kind) = (<&'static str>::from(c.level), c.kind);
            deciders(&c.expected)
                .into_iter()
                .map(move |rule| (level, kind, rule))
        })
        .collect();
    let missing: Vec<(SafetyLevel, Option<NotificationKind>, &'static str)> = SafetyLevel::VARIANTS
        .iter()
        .flat_map(|&level| {
            let rule_engine = &rule_engine;
            let post_kinds = level.post_kinds().flat_map(move |(kind, _)| {
                rule_engine
                    .post_kind_rule_names(level, kind)
                    .into_iter()
                    .map(move |rule| (level, Some(kind), rule))
            });
            rule_engine
                .wired_rule_names(level)
                .into_iter()
                .map(move |rule| (level, None, rule))
                .chain(post_kinds)
        })
        .filter(|&(level, kind, rule)| !decided.contains(&(level.into(), kind, rule)))
        .filter(|&(level, _, rule)| !preempted(level, rule))
        .collect();
    assert!(
        missing.is_empty(),
        "wired rules that decide no corpus case at their own level and post kind: {missing:?}"
    );
    for Preempted {
        level,
        rule,
        preempted_by,
    } in PREEMPTED
    {
        let at_level = <&'static str>::from(level);
        let wired = rule_engine.wired_rule_names(level);
        let position = |name| wired.iter().position(|&wired_name| wired_name == name);
        let Some(rule_at) = position(rule) else {
            panic!("{rule} is no longer wired at {level:?}");
        };
        assert!(
            !decided.contains(&(at_level, None, rule)),
            "{rule} decides a corpus case at {level:?}, so it is not preempted there"
        );
        for &by in preempted_by {
            assert!(
                position(by).is_some_and(|by_at| by_at < rule_at),
                "{by}, which preempts {rule} at {level:?}, does not come before it in fold order"
            );
            assert!(
                decided.contains(&(at_level, None, by)),
                "{by}, which preempts {rule} at {level:?}, decides no corpus case there"
            );
        }
    }
}

pub(super) fn corpus() -> Vec<CorpusCase> {
    row_cases()
        .into_iter()
        .map(|row_case| row_case.case)
        .chain(post_notifications::cases())
        .collect()
}

pub(super) fn post_cases() -> Vec<(SafetyLevel, ViewerFeatures, PostCandidate)> {
    corpus()
        .into_iter()
        .filter_map(|case| match case.subject {
            Subject::Post(candidate) => Some((case.level, case.viewer, candidate)),
            Subject::User(_) => None,
        })
        .collect()
}

fn rows() -> Vec<Row> {
    [
        age_gating::rows(),
        age_verification::rows(),
        article::rows(),
        author_state::rows(),
        baseline::rows(),
        community::rows(),
        conversation_control::rows(),
        exclusive_content::rows(),
        interstitial::rows(),
        label_interstitial::rows(),
        legacy_interstitial::rows(),
        local_regulations::rows(),
        local_tweet::rows(),
        notifications::rows(),
        oon_media::rows(),
        oon_tweet_label::rows(),
        oon_user_label::rows(),
        quoted_tweet_rules::rows(),
        reaction_video::rows(),
        relationship::rows(),
        takedown::rows(),
        trusted_friends::rows(),
        tweet_label::rows(),
        tweet_state::rows(),
    ]
    .into_iter()
    .flatten()
    .collect()
}
