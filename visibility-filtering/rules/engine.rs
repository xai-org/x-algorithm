use crate::hydration::{Hydration, HydrationPlan, Hydrators, RequestRows};
use crate::models::{
    AuthorId, Decided, Evaluation, LabelInterstitial, LimitedEngagement, MediaRestriction,
    NotificationKind, Placement, PostCandidate, TweetId, UserCandidate, Verdict, ViewerFeatures,
};
use crate::params::CountryLists;
use crate::rules::action_kind::{ActionKind, Filled};
use crate::rules::defaults::{Bucket, Defaults, Effect};
use crate::rules::levels::PostRules;
use crate::rules::rule_spec::engine::{Engine, Fold, PostEngine, UserEngine};
use crate::rules::rule_spec::inputs::EvaluateInputs;
use crate::rules::rule_spec::observed::{self, NoTrace, Observer, PolicyTrace, Total};
use crate::rules::rule_spec::rule::{self, Rule};
#[cfg(test)]
use crate::rules::rule_spec::ActionSpec;
use crate::rules::rule_spec::{rule_name, Contribution, RuleAction, RuleId, Truth};
use crate::rules::{Predicate, RequestLeaf, RuleContext, SafetyLevel};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::sync::Arc;
use strum::VariantArray;
use tracing::warn;

pub(super) struct Policy<E: Engine = PostEngine> {
    rules: Vec<(&'static str, Rule<E>)>,
    hydrators: Hydrators,
}

impl<E: Fold> Policy<E> {
    #[expect(
        clippy::panic,
        reason = "invalid authored rules fail startup, before the service takes traffic"
    )]
    pub(super) fn new(mut rules: Vec<(&'static str, Rule<E>)>) -> Self {
        if let Err(error) = rule::validate(rules.iter().map(|(_, rule)| rule)) {
            panic!("invalid rules: {error:?}");
        }
        rules.sort_by_key(|(_, rule)| Reverse(rule.action.spec().kind().severity()));
        let mut runs: Vec<&'static str> = Vec::new();
        let mut run: Vec<&Rule<E>> = Vec::new();
        for (name, rule) in &rules {
            if runs.last() != Some(name) {
                assert!(!runs.contains(name), "{name} is wired twice");
                runs.push(name);
                run.clear();
            }
            assert!(!run.contains(&rule), "{name} is wired twice");
            run.push(rule);
        }
        let hydrators = rules
            .iter()
            .fold(Hydrators::empty(), |hydrators, (_, rule)| {
                hydrators.union(rule.requirements())
            });
        Self { rules, hydrators }
    }

    fn reads(&self, leaf: Predicate) -> bool {
        let mut read = Vec::new();
        for (_, rule) in &self.rules {
            rule.leaves(&mut read);
        }
        read.contains(&leaf)
    }

    #[cfg(test)]
    fn rule_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.rules.iter().map(|&(name, _)| name).collect();
        names.dedup();
        names
    }

    pub(super) fn evaluate(&self, context: &RuleContext<'_, E::Candidate>) -> Evaluation {
        fold(&self.rules, context, &mut NoTrace)
    }
}

#[derive(Default)]
pub(super) struct Folded {
    decision: Decision,
    terminal: Option<(usize, Verdict)>,
}

fn fold<E: Fold, O: Observer<E>>(
    rules: &[(&'static str, Rule<E>)],
    context: &RuleContext<'_, E::Candidate>,
    observer: &mut O,
) -> Evaluation {
    let folded = fold_from(rules, context, observer, Folded::default(), 0);
    finish(folded, context.failed(), observer)
}

#[inline]
fn fold_from<E: Fold, O: Observer<E>>(
    rules: &[(&'static str, Rule<E>)],
    context: &RuleContext<'_, E::Candidate>,
    observer: &mut O,
    Folded {
        mut decision,
        mut terminal,
    }: Folded,
    first_occurrence: usize,
) -> Folded {
    for (occurrence, (by, rule)) in (first_occurrence..).zip(rules) {
        let after = terminal.as_ref().map(|&(decided, ..)| decided);
        if after.is_some() && !O::TOTAL {
            break;
        }
        let (truth, root) = observed::evaluate_rule(rule, context, observer, after);
        if !O::TOTAL && truth == Truth::False {
            continue;
        }
        let action = rule.action.spec();
        let completed = (truth.resolves_true()
            && (O::TOTAL
                || (after.is_none() && action.kind().kept_by(decision.filled()).is_none())))
        .then(|| action.interpret(&EvaluateInputs(context)));
        let diagnostic = if O::TOTAL {
            completed
                .as_ref()
                .map(|completed| (completed.value.clone(), completed.failed))
        } else {
            None
        };
        let (effect, defaults_admitted) = if after.is_some() {
            (
                Effect::AfterTerminal,
                (Bucket::Withholding, Hydrators::empty()),
            )
        } else {
            let contribution = completed.and_then(|completed| completed.value);
            let (selected, effect, admitted) =
                decision.step(rule.id, by, action.kind(), truth, contribution);
            if let Some(verdict) = selected {
                terminal = Some((occurrence, verdict));
            }
            (effect, admitted)
        };
        observer.row(by, root, truth, effect, defaults_admitted, diagnostic);
    }
    Folded { decision, terminal }
}

#[inline]
fn finish<E: Engine, O: Observer<E>>(
    Folded { decision, terminal }: Folded,
    failed: Hydrators,
    observer: &mut O,
) -> Evaluation {
    let admitted = decision.defaults;
    let verdict = match terminal {
        Some((_, verdict)) => verdict,
        None => decision.finish(),
    };
    let defaults = admitted.relied_on_by(&verdict);
    observer.retained(defaults);
    let fail_open_defaults = defaults.all();
    debug_assert!(
        !failed.is_empty() || fail_open_defaults.is_empty(),
        "a verdict over no failed node relies on no default"
    );
    if failed.has_must_have() {
        Evaluation::Partial {
            verdict,
            fail_open_defaults,
        }
    } else {
        debug_assert!(
            !fail_open_defaults.has_must_have(),
            "a complete verdict relies only on best-effort defaults"
        );
        Evaluation::Complete {
            verdict,
            fail_open_defaults,
        }
    }
}

#[derive(Default)]
struct Decision {
    label_interstitial: Option<Decided<LabelInterstitial>>,
    media: Option<Decided<MediaRestriction>>,
    engagement: Option<Decided<LimitedEngagement>>,
    defaults: Defaults,
}

impl Decision {
    fn filled(&self) -> Filled {
        Filled {
            label_interstitial: self.label_interstitial.is_some(),
            media: self.media.is_some(),
            engagement: self.engagement.is_some(),
        }
    }

    #[inline]
    fn step(
        &mut self,
        id: RuleId,
        by: &'static str,
        kind: ActionKind,
        truth: Truth,
        contribution: Option<Contribution>,
    ) -> (Option<Verdict>, Effect, (Bucket, Hydrators)) {
        let filled = self.filled();
        let reads = match truth {
            Truth::Unknown { failed, .. } if kind.admits_defaults(filled) => failed,
            Truth::Unknown { .. } | Truth::True | Truth::False => Hydrators::empty(),
        };
        let admitted = (kind.bucket(), reads);
        self.defaults.admit(kind.bucket(), reads);
        if !truth.resolves_true() {
            return (None, Effect::DidNotApply, admitted);
        }
        if let Some(effect) = kind.kept_by(filled) {
            return (None, effect, admitted);
        }
        let (terminal, effect) = match contribution {
            None => (None, Effect::DidNotApply),
            Some(Contribution::Withholding(value)) => (
                Some(Verdict::Withheld(Decided { value, id, by })),
                Effect::Decided,
            ),
            Some(Contribution::Notice(value)) => (
                Some(Verdict::Fosnr(Decided { value, id, by })),
                Effect::Decided,
            ),
            Some(Contribution::LabelInterstitial(value)) => {
                self.label_interstitial = Some(Decided { value, id, by });
                (None, Effect::FilledLabelInterstitial)
            }
            Some(Contribution::Media(value)) => {
                self.media = Some(Decided { value, id, by });
                (None, Effect::FilledMedia)
            }
            Some(Contribution::Engagement(reason)) => {
                let effect = match &mut self.engagement {
                    None => {
                        self.engagement = Some(Decided {
                            value: LimitedEngagement::new(reason),
                            id,
                            by,
                        });
                        Effect::FilledEngagement
                    }
                    Some(limit) => {
                        if limit.value.add(reason) {
                            Effect::AddedEngagementReason
                        } else {
                            Effect::DuplicateEngagementReason
                        }
                    }
                };
                (None, effect)
            }
        };
        (terminal, effect, admitted)
    }

    fn finish(self) -> Verdict {
        Verdict::Shown {
            label_interstitial: self.label_interstitial,
            media: self.media,
            engagement: self.engagement,
        }
    }
}

pub(super) struct PostPolicy {
    user: Policy<UserEngine>,
    post: Policy,
    hydrators: Hydrators,
}

impl PostPolicy {
    #[expect(
        clippy::panic,
        reason = "invalid authored rules fail startup, before the service takes traffic"
    )]
    fn new(user: Vec<(&'static str, Rule<UserEngine>)>, post: Vec<(&'static str, Rule)>) -> Self {
        if let Some((name, _)) = user
            .iter()
            .find(|(name, _)| post.iter().any(|(other, _)| other == name))
        {
            panic!("{name} is wired in both the user and the post list");
        }
        if let Err(error) = rule::validate_across(
            user.iter().map(|(_, rule)| rule),
            post.iter().map(|(_, rule)| rule),
        ) {
            panic!("invalid rules: {error:?}");
        }
        let (user, post) = (Policy::new(user), Policy::new(post));
        let hydrators = user.hydrators.union(post.hydrators);
        Self {
            user,
            post,
            hydrators,
        }
    }

    #[inline]
    fn fold_user<O: Observer<UserEngine>>(
        &self,
        context: &RuleContext<'_, UserCandidate>,
        observer: &mut O,
    ) -> Folded {
        fold_from(&self.user.rules, context, observer, Folded::default(), 0)
    }

    #[inline]
    fn gate<O: Observer<PostEngine>>(
        &self,
        user: Folded,
        context: &RuleContext<'_>,
        observer: &mut O,
    ) -> Evaluation {
        let folded = if user.terminal.is_some() && !O::TOTAL {
            user
        } else {
            fold_from(
                &self.post.rules,
                context,
                observer,
                user,
                self.user.rules.len(),
            )
        };
        finish(folded, context.failed(), observer)
    }

    #[cfg(test)]
    fn rule_names(&self) -> Vec<&'static str> {
        let mut names = self.user.rule_names();
        names.extend(self.post.rule_names());
        names
    }
}

struct Level {
    policy: PostPolicy,
    plan: HydrationPlan,
    reads_soft_nsfw_new_user_drop: bool,
    post_kinds: Option<PostKinds>,
}

struct PostKinds {
    policies: Vec<(NotificationKind, PostPolicy)>,
    plan: HydrationPlan,
}

impl Level {
    fn new(level: SafetyLevel, policy: PostPolicy) -> Self {
        let plan = HydrationPlan::new(level, policy.hydrators);
        let soft_nsfw_new_user_drop =
            Predicate::Request(RequestLeaf::RequestHasSoftNsfwNewUserDrop);
        Self {
            reads_soft_nsfw_new_user_drop: policy.user.reads(soft_nsfw_new_user_drop)
                || policy.post.reads(soft_nsfw_new_user_drop),
            policy,
            plan,
            post_kinds: None,
        }
    }
}

impl PostKinds {
    fn new(
        level: SafetyLevel,
        kinds: impl Iterator<Item = (NotificationKind, PostRules)>,
        names: &mut Vec<(RuleId, RuleAction, &'static str)>,
    ) -> Self {
        let policies: Vec<(NotificationKind, PostPolicy)> = kinds
            .map(|(kind, rules)| {
                let rules = rules.lists();
                let user = named(names, &rules.user);
                (kind, PostPolicy::new(user, named(names, &rules.post)))
            })
            .collect();
        let hydrators = policies
            .iter()
            .fold(Hydrators::empty(), |hydrators, (_, policy)| {
                hydrators.union(policy.hydrators)
            });
        Self {
            plan: HydrationPlan::with_rows(level, RequestRows::ActorPosts, hydrators),
            policies,
        }
    }
}

pub struct RuleEngine {
    country_lists: Arc<CountryLists>,
    levels: Vec<Level>,
}

#[expect(
    clippy::panic,
    reason = "invalid authored rules fail startup, before the service takes traffic"
)]
fn interned_name(
    names: &mut Vec<(RuleId, RuleAction, &'static str)>,
    id: RuleId,
    action: &RuleAction,
) -> &'static str {
    if let Some(&(_, _, name)) = names
        .iter()
        .find(|(other_id, other_action, _)| *other_id == id && other_action == action)
    {
        return name;
    }
    let name: &'static str = rule_name(id, action).leak();
    if let Some((other_id, other_action, _)) = names.iter().find(|&&(_, _, other)| other == name) {
        panic!("{other_id:?} {other_action:?} and {id:?} {action:?} derive {name}");
    }
    names.push((id, action.clone(), name));
    name
}

fn plans_from_its_rows(plan: &HydrationPlan) -> bool {
    match plan.rows() {
        RequestRows::Posts | RequestRows::ActorPosts => !plan.keys_a_node_only_user_rows_give(),
        RequestRows::Users => plan.keys_only_user_rows_or_viewer(),
    }
}

fn named<E: Engine>(
    names: &mut Vec<(RuleId, RuleAction, &'static str)>,
    rules: &[&Rule<E>],
) -> Vec<(&'static str, Rule<E>)> {
    rules
        .iter()
        .map(|&rule| {
            (
                interned_name(names, rule.id, &rule.action.lift()),
                rule.clone(),
            )
        })
        .collect()
}

pub(crate) struct Admitted<'a> {
    level: AdmittedLevel<'a>,
    hydration: &'a Hydration,
}

#[derive(Clone, Copy)]
struct AdmittedLevel<'a> {
    policy: &'a PostPolicy,
    viewer: &'a ViewerFeatures,
    country_lists: &'a CountryLists,
}

impl<'a> Admitted<'a> {
    pub(crate) fn post(
        &self,
        tweet_id: TweetId,
        copied: bool,
        placement: Placement,
    ) -> Option<AdmittedPost<'a>> {
        let candidate = self.hydration.candidate(tweet_id, copied)?;
        let author = self.author(tweet_id, placement)?;
        Some(AdmittedPost { author, candidate })
    }

    pub(crate) fn author(
        &self,
        tweet_id: TweetId,
        placement: Placement,
    ) -> Option<AdmittedUser<'a>> {
        let author =
            self.hydration
                .author(tweet_id, placement, self.level.policy.user.hydrators)?;
        Some(self.level.admit_user(author, placement))
    }

    pub(crate) fn user(&self, id: AuthorId) -> Option<AdmittedUser<'a>> {
        let user = self
            .hydration
            .user_candidate(id, self.level.policy.user.hydrators)?;
        Some(self.level.admit_user(user, Placement::Standalone))
    }
}

impl<'a> AdmittedLevel<'a> {
    fn admit_user(self, user: UserCandidate, placement: Placement) -> AdmittedUser<'a> {
        AdmittedUser {
            level: self,
            user,
            placement,
        }
    }
}

pub(crate) struct AdmittedPost<'a> {
    author: AdmittedUser<'a>,
    candidate: Cow<'a, PostCandidate>,
}

impl AdmittedPost<'_> {
    pub(crate) fn evaluate(&self) -> Evaluation {
        self.author.fold(&self.candidate, &mut NoTrace)
    }

    pub(crate) fn trace(&self) -> PolicyTrace {
        let mut total = Total::default();
        let evaluation = self.author.fold(&self.candidate, &mut total);
        total.into_trace(evaluation)
    }
}

pub(crate) struct AdmittedUser<'a> {
    level: AdmittedLevel<'a>,
    user: UserCandidate,
    placement: Placement,
}

impl AdmittedUser<'_> {
    pub(crate) fn user_candidate(&self) -> &UserCandidate {
        &self.user
    }

    pub(crate) fn evaluate(&self) -> Evaluation {
        self.level.policy.user.evaluate(&self.user_context())
    }

    fn context<'c>(&'c self, candidate: &'c PostCandidate) -> RuleContext<'c> {
        let AdmittedLevel {
            viewer,
            country_lists,
            ..
        } = self.level;
        let context = RuleContext::new(viewer, candidate, self.placement, country_lists);
        #[cfg(test)]
        let context = context.hydrated_by(self.level.policy.post.hydrators);
        context
    }

    fn user_context(&self) -> RuleContext<'_, UserCandidate> {
        let AdmittedLevel {
            viewer,
            country_lists,
            ..
        } = self.level;
        let context = RuleContext::new(viewer, &self.user, self.placement, country_lists);
        #[cfg(test)]
        let context = context.hydrated_by(self.level.policy.user.hydrators);
        context
    }

    #[inline]
    fn fold<O: Observer<UserEngine> + Observer<PostEngine>>(
        &self,
        candidate: &PostCandidate,
        observer: &mut O,
    ) -> Evaluation {
        let folded = self.level.policy.fold_user(&self.user_context(), observer);
        self.level
            .policy
            .gate(folded, &self.context(candidate), observer)
    }
}

impl RuleEngine {
    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Self::with_country_lists(Arc::new(CountryLists::starting_at_default()))
    }

    pub fn with_country_lists(country_lists: Arc<CountryLists>) -> Self {
        let mut names = Vec::new();
        let levels = SafetyLevel::VARIANTS
            .iter()
            .map(|&level| {
                let spec = level.spec();
                let rules = spec.rules().lists();
                let policy = PostPolicy::new(
                    named(&mut names, &rules.user),
                    named(&mut names, &rules.post),
                );
                let mut entry = Level::new(level, policy);
                assert!(
                    plans_from_its_rows(&entry.plan),
                    "{level:?} plans keys its request rows cannot give"
                );
                let mut kinds = level.post_kinds().peekable();
                if kinds.peek().is_some() {
                    let post_kinds = PostKinds::new(level, kinds, &mut names);
                    assert!(
                        plans_from_its_rows(&post_kinds.plan),
                        "{level:?} plans keys its actor post rows cannot give"
                    );
                    entry.post_kinds = Some(post_kinds);
                }
                entry
            })
            .collect();
        Self {
            country_lists,
            levels,
        }
    }

    pub(crate) fn admit<'a>(
        &'a self,
        level: SafetyLevel,
        hydration: &'a Hydration,
    ) -> Option<Admitted<'a>> {
        self.admit_under(self.policy(level), level, hydration)
    }

    pub(crate) fn admit_post_kind<'a>(
        &'a self,
        level: SafetyLevel,
        kind: NotificationKind,
        hydration: &'a Hydration,
    ) -> Option<Admitted<'a>> {
        self.admit_under(self.post_kind(level, kind)?, level, hydration)
    }

    fn post_kind(&self, level: SafetyLevel, kind: NotificationKind) -> Option<&PostPolicy> {
        let post_kinds = self.level(level).post_kinds.as_ref()?;
        post_kinds
            .policies
            .iter()
            .find_map(|(judged, policy)| (*judged == kind).then_some(policy))
    }

    fn admit_under<'a>(
        &'a self,
        policy: &'a PostPolicy,
        level: SafetyLevel,
        hydration: &'a Hydration,
    ) -> Option<Admitted<'a>> {
        let missing = Self::unplanned_inputs(policy, hydration);
        if !missing.is_empty() {
            warn!(
                ?missing,
                ?level,
                "hydration not admitted by its level's policy; failing its candidates"
            );
            return None;
        }
        Some(Admitted {
            level: AdmittedLevel {
                policy,
                viewer: hydration.viewer(),
                country_lists: &self.country_lists,
            },
            hydration,
        })
    }

    fn unplanned_inputs(policy: &PostPolicy, hydration: &Hydration) -> Hydrators {
        let logged_in = hydration.viewer().viewer.user_id().is_some();
        policy
            .hydrators
            .iter()
            .filter(|&node| logged_in || !node.needs_viewer())
            .fold(Hydrators::empty(), Hydrators::with)
            .difference(hydration.planned())
    }

    #[cfg(test)]
    fn admit_user<'a>(
        &'a self,
        policy: &'a PostPolicy,
        viewer: &'a ViewerFeatures,
        user: UserCandidate,
        placement: Placement,
    ) -> AdmittedUser<'a> {
        AdmittedLevel {
            policy,
            viewer,
            country_lists: &self.country_lists,
        }
        .admit_user(user, placement)
    }

    #[cfg(test)]
    pub(crate) fn evaluate(
        &self,
        level: SafetyLevel,
        viewer: &ViewerFeatures,
        candidate: &PostCandidate,
        placement: Placement,
    ) -> Evaluation {
        self.fold(
            self.policy(level),
            viewer,
            candidate,
            placement,
            &mut NoTrace,
        )
    }

    #[cfg(test)]
    pub(crate) fn evaluate_post_kind(
        &self,
        level: SafetyLevel,
        kind: NotificationKind,
        viewer: &ViewerFeatures,
        candidate: &PostCandidate,
        placement: Placement,
    ) -> Evaluation {
        let policy = self.post_kind_policy(level, kind);
        self.fold(policy, viewer, candidate, placement, &mut NoTrace)
    }

    #[cfg(test)]
    fn post_kind_policy(&self, level: SafetyLevel, kind: NotificationKind) -> &PostPolicy {
        self.post_kind(level, kind)
            .unwrap_or_else(|| panic!("{level:?} judges no {kind:?}"))
    }

    #[cfg(test)]
    fn fold<O: Observer<UserEngine> + Observer<PostEngine>>(
        &self,
        policy: &PostPolicy,
        viewer: &ViewerFeatures,
        candidate: &PostCandidate,
        placement: Placement,
        observer: &mut O,
    ) -> Evaluation {
        let author = UserCandidate {
            failed: candidate.failed.intersection(policy.user.hydrators),
            ..UserCandidate::author_of(candidate)
        };
        self.admit_user(policy, viewer, author, placement)
            .fold(candidate, observer)
    }

    #[cfg(test)]
    pub(crate) fn evaluate_user(
        &self,
        level: SafetyLevel,
        viewer: &ViewerFeatures,
        user: &UserCandidate,
        placement: Placement,
    ) -> Evaluation {
        self.admit_user(self.policy(level), viewer, user.clone(), placement)
            .evaluate()
    }

    fn level(&self, level: SafetyLevel) -> &Level {
        #[expect(
            clippy::indexing_slicing,
            reason = "`levels` maps `SafetyLevel::VARIANTS`, which is in declaration order"
        )]
        let entry = &self.levels[level as usize];
        debug_assert_eq!(entry.plan.level(), level);
        entry
    }

    fn policy(&self, level: SafetyLevel) -> &PostPolicy {
        &self.level(level).policy
    }

    pub(crate) fn plan(&self, level: SafetyLevel) -> &HydrationPlan {
        &self.level(level).plan
    }

    pub(crate) fn post_kinds_plan(&self, level: SafetyLevel) -> Option<&HydrationPlan> {
        Some(&self.level(level).post_kinds.as_ref()?.plan)
    }

    pub(crate) fn reads_soft_nsfw_new_user_drop(&self, level: SafetyLevel) -> bool {
        self.level(level).reads_soft_nsfw_new_user_drop
    }

    #[cfg(test)]
    pub(crate) fn wired_rule_names(&self, level: SafetyLevel) -> Vec<&'static str> {
        self.policy(level).rule_names()
    }

    #[cfg(test)]
    pub(crate) fn post_kind_rule_names(
        &self,
        level: SafetyLevel,
        kind: NotificationKind,
    ) -> Vec<&'static str> {
        self.post_kind_policy(level, kind).rule_names()
    }
}

#[cfg(test)]
impl Policy {
    pub(super) fn trace(&self, context: &RuleContext<'_>) -> PolicyTrace {
        let mut total = Total::default();
        let evaluation = fold(&self.rules, context, &mut total);
        total.into_trace(evaluation)
    }
}

#[cfg(test)]
pub(super) struct WiredRule<'a, E: Engine = PostEngine> {
    pub(super) name: &'static str,
    pub(super) action: &'a ActionSpec<E>,
    pub(super) when: rule::RuleCond<E>,
}

#[cfg(test)]
pub(super) type FoldOrder<'a> = (Vec<WiredRule<'a, UserEngine>>, Vec<WiredRule<'a>>);

#[cfg(test)]
fn fold_order(policy: &PostPolicy) -> FoldOrder<'_> {
    fn wired<'a, E: Engine>(rules: &'a [(&'static str, Rule<E>)]) -> Vec<WiredRule<'a, E>> {
        rules
            .iter()
            .map(|&(name, ref rule)| WiredRule {
                name,
                action: rule.action.spec(),
                when: rule.when,
            })
            .collect()
    }
    (wired(&policy.user.rules), wired(&policy.post.rules))
}

#[cfg(test)]
impl RuleEngine {
    pub(super) fn trace(
        &self,
        level: SafetyLevel,
        viewer: &ViewerFeatures,
        candidate: &PostCandidate,
    ) -> PolicyTrace {
        let mut total = Total::default();
        let evaluation = self.fold(
            self.policy(level),
            viewer,
            candidate,
            Placement::Standalone,
            &mut total,
        );
        total.into_trace(evaluation)
    }

    pub(super) fn fold_order(&self, level: SafetyLevel) -> FoldOrder<'_> {
        fold_order(self.policy(level))
    }

    pub(super) fn post_kind_fold_orders(
        &self,
        level: SafetyLevel,
    ) -> Vec<(NotificationKind, FoldOrder<'_>)> {
        self.level(level)
            .post_kinds
            .iter()
            .flat_map(|post_kinds| &post_kinds.policies)
            .map(|(kind, policy)| (*kind, fold_order(policy)))
            .collect()
    }

    pub(super) fn with_level_for_tests(
        level: SafetyLevel,
        user: Vec<(&'static str, Rule<UserEngine>)>,
        post: Vec<(&'static str, Rule)>,
    ) -> Self {
        let mut engine = Self::for_tests();
        engine.levels[level as usize] = Level::new(level, PostPolicy::new(user, post));
        engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::Hydrator;
    use crate::models::Withholding;
    use crate::models::{
        AuthorFeatures, ClientCapability, ClientSwitch, DropReason, NsfwViewerDropReason,
        RequestFacts, SafetyLabelType, VerifyBlurSupport, ViewerAge, ViewerProfile,
    };
    use crate::rules::fixtures::{
        candidate, viewer, viewer_with_profile, CandidateBuilder, VIEWER_ID,
    };
    use crate::rules::test_context;

    #[test]
    fn refreshed_config_country_reaches_the_wired_rule() {
        let country_lists = Arc::new(CountryLists::starting_at_default());
        let rule_engine = RuleEngine::with_country_lists(Arc::clone(&country_lists));
        let candidate = candidate()
            .with_label(crate::models::SafetyLabelType::NSFW_HIGH_PRECISION)
            .with_media()
            .build();
        let viewer = ViewerFeatures {
            facts: RequestFacts::default().with_country("us"),
            ..viewer_with_profile(ViewerProfile {
                viewer_age: ViewerAge::NotStated,
                ..ViewerProfile::default()
            })
        };

        let verdict = rule_engine
            .evaluate(
                SafetyLevel::TimelineHome,
                &viewer,
                &candidate,
                Placement::Standalone,
            )
            .into_verdict();
        assert!(!matches!(verdict, Verdict::Withheld(_)));

        country_lists.refresh(
            &xai_feature_switches::FeatureSwitches::load_string(
                r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_countries:
      type: array
      default:
      - "us"
"#,
            )
            .unwrap(),
        );
        let verdict = rule_engine
            .evaluate(
                SafetyLevel::TimelineHome,
                &viewer,
                &candidate,
                Placement::Standalone,
            )
            .into_verdict();
        assert!(matches!(
            verdict,
            Verdict::Withheld(Decided {
                value: Withholding::Drop(DropReason::NsfwViewer(
                    NsfwViewerDropReason::HasNoStatedAge
                )),
                id: RuleId::SensitiveViewerNoStatedAge,
                by: "sensitive_viewer_no_stated_age/drop",
            })
        ));
    }

    #[test]
    fn a_failed_article_lookup_is_a_partial_drop() {
        use crate::models::TweetFeatures;
        use crate::rules::fixtures::dropped;
        use std::num::NonZeroU64;
        use xai_visibility_filtering::models::FilteredReason;

        let mut candidate = candidate()
            .with_tweet_features(TweetFeatures {
                article_id: NonZeroU64::new(1),
                ..TweetFeatures::default()
            })
            .build();
        candidate.failed = Hydrators::of(Hydrator::ArticleLifecycle);
        assert_eq!(
            RuleEngine::for_tests().evaluate(
                SafetyLevel::TimelineHomeHydration,
                &viewer(VIEWER_ID),
                &candidate,
                Placement::Standalone,
            ),
            Evaluation::Partial {
                verdict: dropped(
                    FilteredReason::UnspecifiedReason,
                    "article_tweet_content/drop/unspecified",
                ),
                fail_open_defaults: Hydrators::of(Hydrator::ArticleLifecycle),
            }
        );
    }

    #[test]
    fn age_verification_tombstones_read_their_family_country_list() {
        let country_lists = Arc::new(CountryLists::starting_at_default());
        country_lists.refresh(
            &xai_feature_switches::FeatureSwitches::load_string(
                r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_tombstone_countries:
      type: array
      default:
      - "us"
"#,
            )
            .unwrap(),
        );
        let rule_engine = RuleEngine::with_country_lists(country_lists);
        let viewer = ViewerFeatures {
            facts: RequestFacts::default()
                .with_country("us")
                .with_client(ClientCapability {
                    verify_blur_support: Some(VerifyBlurSupport::IosNeedsUpdate),
                    ..ClientCapability::DEFAULT
                        .with(ClientSwitch::ModernBlur)
                        .with(ClientSwitch::CommunityViewerRemovedLimits)
                        .with(ClientSwitch::GoreBlurIgnoresSettings)
                }),
            ..viewer(VIEWER_ID)
        };
        let nsfw_admin = AuthorFeatures {
            is_nsfw_admin: true,
            ..AuthorFeatures::default()
        };
        let withheld_by = |candidate: CandidateBuilder| match rule_engine
            .evaluate(
                SafetyLevel::TimelineHomeHydration,
                &viewer,
                &candidate.with_media().build(),
                Placement::Standalone,
            )
            .into_verdict()
        {
            Verdict::Withheld(Decided { by, .. }) => Some(by),
            Verdict::Fosnr(_) | Verdict::Shown { .. } => None,
        };
        let labeled = |label| candidate().with_label(label);
        assert_eq!(
            withheld_by(labeled(SafetyLabelType::NSFW_HIGH_PRECISION)),
            Some("nsfw_high_precision/tombstone/update_app_ios")
        );
        assert_eq!(
            withheld_by(labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)),
            Some("gore_and_violence_high_precision/tombstone/update_app_ios")
        );
        assert_eq!(
            withheld_by(candidate().with_author_features(nsfw_admin)),
            None
        );
        assert_eq!(withheld_by(labeled(SafetyLabelType::NSFW_CARD_IMAGE)), None);
    }

    #[test]
    fn each_level_plans_only_the_keys_its_request_rows_give() {
        use crate::hydration::RequestRows;
        let user_rows = |nodes| {
            HydrationPlan::with_rows(SafetyLevel::QuotedTweetRules, RequestRows::Users, nodes)
        };
        let user_and_viewer_keyed = Hydrators::of(Hydrator::ViewerProfile)
            .with(Hydrator::AuthorSafety)
            .with(Hydrator::Blocks);
        assert!(plans_from_its_rows(&user_rows(user_and_viewer_keyed)));
        for post_keyed in [
            Hydrator::Tweet,
            Hydrator::MuteRetweets,
            Hydrator::AuthorBlocksOuterAuthor,
        ] {
            let plan = user_rows(user_and_viewer_keyed.with(post_keyed));
            assert!(!plans_from_its_rows(&plan), "{post_keyed:?}");
        }
        let post_rows = |nodes| {
            HydrationPlan::with_rows(SafetyLevel::QuotedTweetRules, RequestRows::Posts, nodes)
        };
        assert!(plans_from_its_rows(&post_rows(user_and_viewer_keyed)));
        let actor_keyed = Hydrators::of(Hydrator::AuthorProfile)
            .with(Hydrator::AuthorConfirmedEmail)
            .with(Hydrator::AuthorVerifiedPhone);
        assert!(!plans_from_its_rows(&post_rows(
            user_and_viewer_keyed.with(Hydrator::AuthorProfile)
        )));
        assert!(plans_from_its_rows(&HydrationPlan::with_rows(
            SafetyLevel::NotificationsWriterV2,
            RequestRows::ActorPosts,
            user_and_viewer_keyed
                .union(actor_keyed)
                .with(Hydrator::Tweet),
        )));
    }

    #[test]
    #[should_panic(expected = "a rule reads Follows")]
    fn a_rule_reading_an_underived_hydrator_panics_in_tests() {
        let viewer = viewer(VIEWER_ID);
        let candidate = candidate().build();
        let context = test_context(&viewer, &candidate)
            .hydrated_by(Hydrators::all().without(Hydrator::Follows));
        context.acquire_edge(Hydrator::Follows);
    }

    mod engine {
        use super::super::*;
        use crate::hydration::Hydrator;
        use crate::models::Withholding;
        use crate::models::{
            AuthorFeatures, DropReason, LimitedEngagementReason, MediaInterstitial,
            MediaRestriction, PostCandidate, SafetyLabelType, TombstoneReason, UserCandidate,
            ViewerFeatures,
        };
        use crate::rules::fixtures::{allow, author_viewer, blurred, candidate, viewer, VIEWER_ID};
        use crate::rules::rule_spec::leaves::*;
        use crate::rules::rule_spec::observed::Skip;
        use crate::rules::rule_spec::rule::Audience::{Everyone, NonFollowers};
        use crate::rules::rule_spec::rule::{all, any, NamedCond, RuleCond, UserRule};
        use crate::rules::rule_spec::{drop_post, limit, soft_intervention, ActionSpec, DropAs};
        use crate::rules::test_context;
        use xai_visibility_filtering::models::FilteredReason;
        use xai_x_thrift::action::{AppealablePolicy, InterstitialReason};

        fn rule(when: RuleCond, action: ActionSpec) -> Rule {
            Rule {
                id: RuleId::FilterAll,
                audience: Everyone,
                when,
                action: action.into(),
            }
        }

        fn always(name: &'static str, action: ActionSpec) -> (&'static str, Rule) {
            (name, rule(all![], action))
        }

        const NEVER: RuleCond = TweetCreatedAfter(u64::MAX).leaf();

        const DROP_SUSPENDED: ActionSpec =
            ActionSpec::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended));
        const TOMBSTONE: ActionSpec = ActionSpec::Tombstone(TombstoneReason::LocalRegulations);
        const INTERSTITIAL_NSFW: ActionSpec =
            ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(MediaInterstitial {
                legacy: FilteredReason::ContainNsfwMedia,
                reason: InterstitialReason::Sensitive(true),
                prompt: None,
            }));
        const INTERSTITIAL_UNSPECIFIED: ActionSpec =
            ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(MediaInterstitial {
                legacy: FilteredReason::UnspecifiedReason,
                reason: InterstitialReason::Nudity(true),
                prompt: None,
            }));
        const LIMIT: ActionSpec =
            ActionSpec::LimitedEngagement(LimitedEngagementReason::ConversationControl);

        fn context_inputs() -> (ViewerFeatures, PostCandidate) {
            (ViewerFeatures::default(), PostCandidate::default())
        }

        fn withheld(value: Withholding, by: &'static str) -> Verdict {
            Verdict::Withheld(Decided {
                value,
                id: RuleId::FilterAll,
                by,
            })
        }

        fn tombstone_first() -> Policy {
            Policy::new(vec![
                always("tombstone", TOMBSTONE),
                ("never", rule(NEVER, DROP_SUSPENDED)),
                always("drop", DROP_SUSPENDED),
                (
                    "after_drop",
                    rule(ViewerFollowsAuthor.leaf(), DROP_SUSPENDED),
                ),
            ])
        }

        #[test]
        fn the_most_severe_terminal_returns_before_later_rules() {
            let (viewer, candidate) = context_inputs();
            let context = test_context(&viewer, &candidate)
                .hydrated_by(Hydrators::all().without(Hydrator::Follows));

            assert_eq!(
                tombstone_first().evaluate(&context).into_verdict(),
                withheld(
                    Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
                    "drop"
                )
            );
        }

        #[test]
        #[should_panic(expected = "is wired twice")]
        fn an_identity_recurring_after_another_fails_construction() {
            Policy::new(vec![
                always("blur", INTERSTITIAL_NSFW),
                always("other_blur", INTERSTITIAL_UNSPECIFIED),
                always("blur", INTERSTITIAL_NSFW),
            ]);
        }

        #[test]
        #[should_panic(expected = "derive filter_all/blur/sensitive")]
        fn two_identities_deriving_one_name_fail_construction() {
            let blur = |legacy| {
                ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(
                    MediaInterstitial {
                        legacy,
                        reason: InterstitialReason::Sensitive(true),
                        prompt: None,
                    },
                ))
            };
            let mut names = Vec::new();
            for reason in [
                FilteredReason::ContainNsfwMedia,
                FilteredReason::UnspecifiedReason,
            ] {
                let rule = rule(all![], blur(reason));
                interned_name(&mut names, rule.id, &rule.action);
            }
        }

        #[test]
        #[should_panic(expected = "is wired twice")]
        fn a_rule_repeated_within_its_run_fails_construction() {
            Policy::new(vec![
                always("blur", INTERSTITIAL_NSFW),
                always("blur", INTERSTITIAL_NSFW),
            ]);
        }

        fn follows_and_blocks_failed() -> (ViewerFeatures, PostCandidate) {
            let candidate = PostCandidate {
                failed: Hydrators::of(Hydrator::Follows).with(Hydrator::Blocks),
                ..Default::default()
            };
            (ViewerFeatures::default(), candidate)
        }

        #[test]
        fn conditions_combine_in_three_values_and_unknown_resolves_to_the_default() {
            let (viewer, candidate) = follows_and_blocks_failed();
            let follows = Hydrators::of(Hydrator::Follows);
            let dropped = withheld(
                Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
                "rule",
            );
            let rows: [(RuleCond, Verdict, Hydrators); 6] = [
                (ViewerFollowsAuthor.leaf(), allow(), follows),
                (
                    const { all![ViewerFollowsAuthor, NEVER] },
                    allow(),
                    Hydrators::empty(),
                ),
                (
                    const { RuleCond::Not(&ViewerFollowsAuthor.leaf()) },
                    dropped.clone(),
                    follows,
                ),
                (
                    const { any![ViewerFollowsAuthor, ViewerLoggedOut] },
                    dropped,
                    Hydrators::empty(),
                ),
                (const { any![ViewerFollowsAuthor, NEVER] }, allow(), follows),
                (
                    const {
                        all![
                            any![ViewerBlocksAuthor, ViewerLoggedOut],
                            ViewerFollowsAuthor
                        ]
                    },
                    allow(),
                    follows,
                ),
            ];
            for (index, (when, verdict, fail_open_defaults)) in rows.into_iter().enumerate() {
                assert_eq!(
                    Policy::new(vec![("rule", rule(when, DROP_SUSPENDED))])
                        .evaluate(&test_context(&viewer, &candidate)),
                    Evaluation::Partial {
                        verdict,
                        fail_open_defaults
                    },
                    "row {index}"
                );
            }
        }

        #[test]
        fn fail_open_defaults_are_the_unknown_rules_that_could_have_changed_the_verdict() {
            let (viewer, candidate) = follows_and_blocks_failed();
            let follows = Hydrators::of(Hydrator::Follows);
            let unknown = |action| ("unknown", rule(ViewerFollowsAuthor.leaf(), action));
            let blur = blurred(InterstitialReason::Sensitive(true), "blur");
            let rows = [
                (
                    [unknown(DROP_SUSPENDED), always("drop", DROP_SUSPENDED)],
                    withheld(
                        Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
                        "drop",
                    ),
                    follows,
                ),
                (
                    [unknown(INTERSTITIAL_NSFW), always("drop", DROP_SUSPENDED)],
                    withheld(
                        Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
                        "drop",
                    ),
                    Hydrators::empty(),
                ),
                (
                    [
                        always("blur", INTERSTITIAL_NSFW),
                        unknown(INTERSTITIAL_UNSPECIFIED),
                    ],
                    blur.clone(),
                    Hydrators::empty(),
                ),
                (
                    [always("blur", INTERSTITIAL_NSFW), unknown(LIMIT)],
                    blur,
                    follows,
                ),
            ];
            for (index, (rules, verdict, fail_open_defaults)) in rows.into_iter().enumerate() {
                assert_eq!(
                    Policy::new(rules.into()).evaluate(&test_context(&viewer, &candidate)),
                    Evaluation::Partial {
                        verdict,
                        fail_open_defaults
                    },
                    "row {index}"
                );
            }
        }

        #[test]
        fn a_user_rule_judges_the_author_with_no_post() {
            let follows = Hydrators::of(Hydrator::Follows);
            let dropped = withheld(
                Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
                "rule",
            );
            let user = |post: PostCandidate| UserCandidate::author_of(&post);
            let suspended = || {
                candidate().with_author_features(AuthorFeatures {
                    is_suspended: true,
                    ..AuthorFeatures::default()
                })
            };
            let mut follows_failed = user(suspended().build());
            follows_failed.failed = follows;
            let rows = [
                (
                    "non-follower",
                    viewer(VIEWER_ID),
                    user(suspended().build()),
                    Evaluation::complete(dropped.clone()),
                ),
                (
                    "follower",
                    viewer(VIEWER_ID),
                    user(suspended().with_edge(Hydrator::Follows).build()),
                    Evaluation::complete(allow()),
                ),
                (
                    "author",
                    author_viewer(),
                    user(suspended().build()),
                    Evaluation::complete(allow()),
                ),
                (
                    "active author",
                    viewer(VIEWER_ID),
                    user(candidate().build()),
                    Evaluation::complete(allow()),
                ),
                (
                    "follows failed",
                    viewer(VIEWER_ID),
                    follows_failed,
                    Evaluation::Complete {
                        verdict: dropped,
                        fail_open_defaults: follows,
                    },
                ),
            ];
            let level = SafetyLevel::QuotedTweetRules;
            let rule_engine = RuleEngine::with_level_for_tests(
                level,
                vec![(
                    "rule",
                    UserRule {
                        id: RuleId::FilterAll,
                        audience: NonFollowers,
                        when: const { all![AuthorIsSuspended] },
                        action: drop_post(DropAs::AuthorIsSuspended),
                    },
                )],
                vec![],
            );
            for (who, viewer, candidate, evaluation) in rows {
                assert_eq!(
                    rule_engine.evaluate_user(level, &viewer, &candidate, Placement::Standalone),
                    evaluation,
                    "{who}"
                );
            }
        }

        fn user_rule(when: RuleCond<UserEngine>, action: RuleAction<UserEngine>) -> UserRule {
            UserRule {
                id: RuleId::FilterAll,
                audience: Everyone,
                when,
                action,
            }
        }

        #[test]
        fn a_terminal_user_row_decides_and_every_post_row_traces_after_it() {
            let level = SafetyLevel::FilterAll;
            let rule_engine = RuleEngine::with_level_for_tests(
                level,
                vec![(
                    "user_drop",
                    user_rule(all![], drop_post(DropAs::AuthorIsSuspended)),
                )],
                vec![
                    ("post_blur", rule(all![], INTERSTITIAL_NSFW)),
                    ("post_tombstone", rule(all![], TOMBSTONE)),
                ],
            );
            let (viewer, candidate) = (viewer(VIEWER_ID), candidate().build());
            let trace = rule_engine.trace(level, &viewer, &candidate);
            assert_eq!(
                trace.evaluation,
                Evaluation::complete(withheld(
                    Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
                    "user_drop"
                ))
            );
            assert_eq!(
                rule_engine.evaluate(level, &viewer, &candidate, Placement::Standalone),
                trace.evaluation
            );
            assert_eq!(
                trace
                    .rows
                    .iter()
                    .map(|row| (row.by, row.effect, trace.nodes[row.root].skip))
                    .collect::<Vec<_>>(),
                [
                    ("user_drop", Effect::Decided, None),
                    (
                        "post_tombstone",
                        Effect::AfterTerminal,
                        Some(Skip::Terminal(0))
                    ),
                    ("post_blur", Effect::AfterTerminal, Some(Skip::Terminal(0))),
                ]
            );
        }

        #[test]
        fn a_post_notice_relies_on_no_default_of_the_user_limit_it_discards() {
            let level = SafetyLevel::FilterAll;
            let rule_engine = RuleEngine::with_level_for_tests(
                level,
                vec![(
                    "user_limit",
                    user_rule(
                        ViewerFollowsAuthor.leaf(),
                        limit(LimitedEngagementReason::ConversationControl),
                    ),
                )],
                vec![(
                    "notice",
                    Rule {
                        id: RuleId::FilterAll,
                        audience: Everyone,
                        when: all![],
                        action: soft_intervention(
                            SafetyLabelType::FOSNR_ABUSE,
                            AppealablePolicy::ABUSE,
                            1,
                        ),
                    },
                )],
            );
            let follows = Hydrators::of(Hydrator::Follows);
            let mut candidate = candidate().build();
            candidate.failed = follows;
            let viewer = viewer(VIEWER_ID);
            let trace = rule_engine.trace(level, &viewer, &candidate);
            assert_eq!(
                rule_engine.evaluate(level, &viewer, &candidate, Placement::Standalone),
                trace.evaluation
            );
            let Evaluation::Complete {
                verdict: Verdict::Fosnr(Decided { by: "notice", .. }),
                fail_open_defaults,
            } = trace.evaluation
            else {
                panic!("the notice is the whole verdict: {:?}", trace.evaluation);
            };
            assert_eq!(fail_open_defaults, Hydrators::empty());
            assert_eq!(
                trace
                    .rows
                    .iter()
                    .map(|row| (row.by, row.defaults_admitted, row.defaults_retained))
                    .collect::<Vec<_>>(),
                [
                    ("user_limit", follows, Hydrators::empty()),
                    ("notice", Hydrators::empty(), Hydrators::empty()),
                ]
            );
        }

        #[test]
        #[should_panic(expected = "rule is wired in both the user and the post list")]
        fn a_name_in_both_lists_fails_construction() {
            PostPolicy::new(
                vec![(
                    "rule",
                    user_rule(all![], drop_post(DropAs::AuthorIsSuspended)),
                )],
                vec![("rule", rule(all![], DROP_SUSPENDED))],
            );
        }

        #[test]
        #[should_panic(expected = "invalid rules: ConflictingName(\"shared\")")]
        fn one_condition_name_defined_two_ways_across_the_lists_fails_construction() {
            static USER_FOLLOWED: NamedCond<UserEngine> = NamedCond {
                name: "shared",
                condition: ViewerFollowsAuthor.leaf(),
            };
            static POST_FOLLOWED: NamedCond = NamedCond {
                name: "shared",
                condition: ViewerFollowsAuthor.leaf(),
            };
            static POST_BLOCKED: NamedCond = NamedCond {
                name: "shared",
                condition: ViewerBlocksAuthor.leaf(),
            };
            let lists = |post: &'static NamedCond| {
                PostPolicy::new(
                    vec![(
                        "user",
                        user_rule(
                            RuleCond::Named(&USER_FOLLOWED),
                            drop_post(DropAs::AuthorIsSuspended),
                        ),
                    )],
                    vec![("post", rule(RuleCond::Named(post), TOMBSTONE))],
                )
            };
            lists(&POST_FOLLOWED);
            lists(&POST_BLOCKED);
        }
    }

    mod uniform {
        use super::super::*;
        use crate::clients::socialgraph_client::{EdgeQuery, Graph};
        use crate::hydration::fallback_cache::FallbackCaches;
        use crate::hydration::sources::Source;
        use crate::hydration::sources::{suspended, InMemorySources};
        use crate::hydration::{HydrationRequest, Hydrator};
        use crate::metrics::{Endpoint, Metrics, RequestLevel};
        use crate::models::{
            ClientCapability, RawCandidate, RequestFacts, RequestViewer, RetweetMerge,
            SafetyLabelType, TweetId,
        };
        use crate::models::{LimitedEngagementReason, Notice};
        use crate::rules::fixtures::{allow, candidate, dropped, viewer, VIEWER_ID};
        use crate::rules::golden_corpus::post_cases;
        use crate::rules::rule_spec::conditions::VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES;
        use crate::rules::rule_spec::leaves::*;
        use crate::rules::rule_spec::observed::{NodeKind, NodeTrace, RowTrace, Skip, Witness};
        use crate::rules::rule_spec::rule::{
            all, any, Audience, DescriptionError, NamedCond, RuleCond,
        };
        use crate::rules::rule_spec::Predicate;
        use crate::rules::rule_spec::{
            appealable, blur, drop_post, limit, soft_intervention, DropAs,
        };
        use crate::rules::test_context;
        use crate::rules::tweet_rules;
        use crate::scenario::{raw, Scenario};
        use rayon::prelude::*;
        use std::iter::once;
        use xai_core_entities::entities::{GizmoduckUser, Label, Labels};
        use xai_visibility_filtering::models::FilteredReason;
        use xai_x_thrift::action::{AppealablePolicy, InterstitialReason};
        use xai_x_thrift::user_labels::LabelValue;
        use Audience::{Everyone, NonAuthors, OnlyAuthor};

        fn rule(audience: Audience, when: RuleCond, action: RuleAction) -> Rule {
            Rule {
                id: RuleId::FilterAll,
                audience,
                when,
                action,
            }
        }

        fn children<'t>(trace: &'t PolicyTrace, parent: usize) -> Vec<&'t NodeTrace> {
            trace
                .nodes
                .iter()
                .filter(|node| node.parent == Some(parent))
                .collect()
        }

        fn witness(node: &NodeTrace, truth: Truth, negated: bool) -> Witness {
            Witness {
                node: node.node,
                truth,
                negated,
            }
        }

        #[test]
        fn total_observation_keeps_every_evaluation_with_none_one_or_all_nodes_failed() {
            let rule_engine = RuleEngine::for_tests();
            let cases = post_cases();
            let runs: Vec<_> = SafetyLevel::VARIANTS
                .iter()
                .flat_map(|&level| cases.iter().map(move |case| (level, case)))
                .collect();
            let seen = runs
                .par_iter()
                .map(|&(level, (own_level, viewer, fetched))| {
                    let mut seen = (0, 0);
                    let failed_alone = if level == *own_level {
                        rule_engine.policy(level).hydrators
                    } else {
                        Hydrators::empty()
                    };
                    for failed in once(Hydrators::empty())
                        .chain(failed_alone.iter().map(Hydrators::of))
                        .chain(once(Hydrators::all()))
                    {
                        let mut candidate = fetched.clone();
                        candidate.failed = candidate.failed.union(failed);
                        let served =
                            rule_engine.evaluate(level, viewer, &candidate, Placement::Standalone);
                        let observed = rule_engine.trace(level, viewer, &candidate);
                        assert_eq!(
                            observed.evaluation, served,
                            "{level:?} {viewer:?} {candidate:?}"
                        );
                        match served {
                            Evaluation::Partial {
                                fail_open_defaults, ..
                            } if !fail_open_defaults.is_empty() => {
                                seen.0 += 1;
                            }
                            Evaluation::Complete {
                                verdict:
                                    Verdict::Withheld(_) | Verdict::Shown { media: Some(_), .. },
                                ..
                            } => seen.1 += 1,
                            Evaluation::Complete { .. }
                            | Evaluation::Partial { .. }
                            | Evaluation::NotFound(_)
                            | Evaluation::Deleted(_)
                            | Evaluation::Failed(_) => {}
                        }
                    }
                    seen
                })
                .reduce(
                    || (0, 0),
                    |total, seen| (total.0 + seen.0, total.1 + seen.1),
                );
            assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
        }

        #[test]
        fn a_nested_not_records_its_truths_witnesses_and_skip_causes() {
            static MUTED: NamedCond = NamedCond {
                name: "muted",
                condition: ViewerMutesAuthor.leaf(),
            };
            let nested = rule(
                NonAuthors,
                const {
                    RuleCond::Not(&any![
                        all![ViewerFollowsAuthor, ViewerBlocksAuthor],
                        RuleCond::Named(&MUTED)
                    ])
                },
                drop_post(DropAs::UnspecifiedReason),
            );
            let policy = Policy::new(vec![("nested", nested)]);
            let viewer = viewer(VIEWER_ID);
            let observe = |candidate: &PostCandidate| {
                let context = test_context(&viewer, candidate).hydrated_by(policy.hydrators);
                let trace = policy.trace(&context);
                assert_eq!(trace.evaluation, policy.evaluate(&context));
                trace
            };
            let shape = |trace: &PolicyTrace| {
                let root = trace.rows[0].root;
                let [audience, when] = children(trace, root)[..] else {
                    panic!("a rule observes its audience and its condition")
                };
                let [any] = children(trace, when.node)[..] else {
                    panic!("not has one child")
                };
                let [all, named] = children(trace, any.node)[..] else {
                    panic!("any has two children")
                };
                let [follows, blocks] = children(trace, all.node)[..] else {
                    panic!("all has two children")
                };
                let [mutes] = children(trace, named.node)[..] else {
                    panic!("a named use observes its definition")
                };
                assert_eq!(audience.truth, Truth::True);
                assert_eq!(
                    [when, any, all, named, follows, blocks, mutes].map(|node| node.kind),
                    [
                        NodeKind::Not,
                        NodeKind::Any,
                        NodeKind::All,
                        NodeKind::Named("muted"),
                        NodeKind::Leaf(Predicate::Relationship(ViewerFollowsAuthor)),
                        NodeKind::Leaf(Predicate::Relationship(ViewerBlocksAuthor)),
                        NodeKind::Leaf(Predicate::Relationship(ViewerMutesAuthor)),
                    ]
                );
                [when, any, all, named, follows, blocks, mutes].map(Clone::clone)
            };

            let followed = candidate()
                .with_edge(Hydrator::Follows)
                .with_edge(Hydrator::Blocks)
                .build();
            let trace = observe(&followed);
            let [not, any, all, named, follows, blocks, mutes] = shape(&trace);
            assert_eq!(trace.rows[0].truth, Truth::False);
            assert_eq!(not.truth, Truth::False);
            assert_eq!(not.witnesses, [witness(&any, Truth::True, true)]);
            assert_eq!(any.witnesses, [witness(&all, Truth::True, false)]);
            assert_eq!(
                all.witnesses,
                [
                    witness(&follows, Truth::True, false),
                    witness(&blocks, Truth::True, false)
                ]
            );
            assert_eq!(named.skip, Some(Skip::TrueDisjunct(all.node)));
            assert_eq!(mutes.skip, Some(Skip::TrueDisjunct(all.node)));
            assert_eq!(mutes.truth, Truth::False);
            assert_eq!(trace.evaluation, Evaluation::complete(allow()));

            let unfollowed = candidate().build();
            let trace = observe(&unfollowed);
            let [not, any, all, named, follows, blocks, _] = shape(&trace);
            assert_eq!(not.truth, Truth::True);
            assert_eq!(not.witnesses, [witness(&any, Truth::False, true)]);
            assert_eq!(
                any.witnesses,
                [
                    witness(&all, Truth::False, false),
                    witness(&named, Truth::False, false)
                ]
            );
            assert_eq!(
                all.witnesses,
                [
                    witness(&follows, Truth::False, false),
                    witness(&blocks, Truth::False, false)
                ]
            );
            assert_eq!(follows.skip, None);
            assert_eq!(blocks.skip, Some(Skip::FalseConjunct(follows.node)));
            assert_eq!(named.skip, None);
            assert_eq!(trace.rows[0].effect, Effect::Decided);

            let mut unknown = candidate().with_edge(Hydrator::Blocks).build();
            unknown.failed = Hydrators::of(Hydrator::Follows);
            let follows_failed = |default| Truth::Unknown {
                default,
                failed: unknown.failed,
            };
            let trace = observe(&unknown);
            let [not, any, all, named, follows, blocks, _] = shape(&trace);
            assert_eq!(not.truth, follows_failed(true));
            assert_eq!(not.witnesses, [witness(&any, follows_failed(false), true)]);
            assert_eq!(not.default_witnesses, [witness(&any, Truth::False, true)]);
            assert_eq!(any.witnesses, [witness(&all, follows_failed(false), false)]);
            assert_eq!(
                any.default_witnesses,
                [
                    witness(&all, Truth::False, false),
                    witness(&named, Truth::False, false)
                ]
            );
            assert_eq!(
                all.witnesses,
                [witness(&follows, follows_failed(false), false)]
            );
            assert_eq!(
                all.default_witnesses,
                [witness(&follows, Truth::False, false)]
            );
            assert_eq!(blocks.truth, Truth::True);
            assert_eq!(blocks.skip, None);
            assert_eq!(named.skip, None);
            assert_eq!(trace.rows[0].defaults_retained, unknown.failed);
            assert_eq!(
                trace.evaluation,
                Evaluation::Complete {
                    verdict: dropped(FilteredReason::UnspecifiedReason, "nested"),
                    fail_open_defaults: unknown.failed,
                }
            );

            let double = rule(
                NonAuthors,
                const {
                    RuleCond::Not(&RuleCond::Not(&any![
                        all![ViewerMutesAuthor, ViewerBlocksAuthor],
                        ViewerFollowsAuthor
                    ]))
                },
                drop_post(DropAs::UnspecifiedReason),
            );
            let double = Policy::new(vec![("double", double)]);
            let context = test_context(&viewer, &unknown).hydrated_by(double.hydrators);
            let trace = double.trace(&context);
            assert_eq!(trace.evaluation, double.evaluate(&context));
            let [_, outer] = children(&trace, trace.rows[0].root)[..] else {
                panic!("a rule observes its audience and its condition")
            };
            let [inner] = children(&trace, outer.node)[..] else {
                panic!("not has one child")
            };
            let [any] = children(&trace, inner.node)[..] else {
                panic!("not has one child")
            };
            let [all, follows] = children(&trace, any.node)[..] else {
                panic!("any has two children")
            };
            let [mutes, blocks] = children(&trace, all.node)[..] else {
                panic!("all has two children")
            };
            assert_eq!(outer.truth, follows_failed(false));
            assert_eq!(
                outer.witnesses,
                [witness(inner, follows_failed(true), true)]
            );
            assert_eq!(outer.default_witnesses, [witness(inner, Truth::True, true)]);
            assert_eq!(inner.witnesses, [witness(any, follows_failed(false), true)]);
            assert_eq!(inner.default_witnesses, [witness(any, Truth::False, true)]);
            assert_eq!(
                any.witnesses,
                [witness(follows, follows_failed(false), false)]
            );
            assert_eq!(
                any.default_witnesses,
                [
                    witness(all, Truth::False, false),
                    witness(follows, Truth::False, false)
                ]
            );
            assert_eq!(all.witnesses, [witness(mutes, Truth::False, false)]);
            assert_eq!(
                [outer, inner, any, all, follows, mutes].map(|node| node.skip),
                [None; 6]
            );
            assert_eq!(blocks.skip, Some(Skip::FalseConjunct(mutes.node)));
            assert_eq!(trace.rows[0].effect, Effect::DidNotApply);
            assert_eq!(trace.rows[0].defaults_retained, unknown.failed);
        }

        #[tokio::test]
        async fn empty_identities_audiences_and_repeated_named_uses_run_through_filter_tweets() {
            static FOLLOWED: NamedCond = NamedCond {
                name: "followed",
                condition: ViewerFollowsAuthor.leaf(),
            };
            let rules = vec![
                (
                    "never",
                    rule(Everyone, any![], drop_post(DropAs::UnspecifiedReason)),
                ),
                (
                    "followed",
                    rule(
                        NonAuthors,
                        const {
                            all![
                                RuleCond::Named(&FOLLOWED),
                                RuleCond::Named(&FOLLOWED),
                                RuleCond::Not(&RuleCond::Not(&all![]))
                            ]
                        },
                        drop_post(DropAs::AuthorIsProtected),
                    ),
                ),
                (
                    "age_gated_author",
                    rule(
                        OnlyAuthor,
                        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES,
                        drop_post(DropAs::ExclusiveTweet),
                    ),
                ),
                (
                    "age_gated",
                    rule(
                        NonAuthors,
                        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES,
                        drop_post(DropAs::ContainNsfwMedia),
                    ),
                ),
                (
                    "always",
                    rule(Everyone, all![], drop_post(DropAs::ViewerBlocksAuthor)),
                ),
            ];
            let age_verified = GizmoduckUser {
                labels: Labels {
                    labels: vec![Label {
                        label_value: LabelValue::AGE_VERIFIED_18.0,
                        created_at_msec: 0,
                    }],
                },
                ..Default::default()
            };
            let sources = || {
                InMemorySources::default()
                    .tweet(1, 10)
                    .authors(&[10])
                    .edge(Graph::Follows, 50, 10)
                    .viewer(70, age_verified.clone())
            };
            for (viewer_id, decided) in [
                (50, ("followed", FilteredReason::AuthorIsProtected)),
                (60, ("age_gated", FilteredReason::ContainNsfwMedia)),
                (10, ("age_gated_author", FilteredReason::ExclusiveTweet)),
                (70, ("always", FilteredReason::ViewerBlocksAuthor)),
            ] {
                let scenario = Scenario::new(sources(), SafetyLevel::FilterAll)
                    .viewer(Some(viewer_id))
                    .country("fr");
                let engine = RuleEngine::with_level_for_tests(
                    SafetyLevel::FilterAll,
                    Vec::new(),
                    rules.clone(),
                );
                assert_eq!(
                    scenario
                        .evaluate_with(engine, &[raw(1, None)])
                        .await
                        .evaluations(),
                    [Evaluation::complete(dropped(decided.1, decided.0))],
                    "viewer {viewer_id}"
                );
            }

            let policy = Policy::new(rules);
            let mut followed = candidate().with_edge(Hydrator::Follows).build();
            followed.author.id = 10;
            let viewer = viewer(VIEWER_ID);
            let trace =
                policy.trace(&test_context(&viewer, &followed).hydrated_by(policy.hydrators));
            let uses: Vec<&NodeTrace> = trace
                .nodes
                .iter()
                .filter(|node| node.kind == NodeKind::Named("followed"))
                .collect();
            assert_eq!(uses.len(), 2);
            assert_ne!(uses[0].node, uses[1].node);
            assert!(uses
                .iter()
                .all(|node| node.truth == Truth::True && node.skip.is_none()));
            assert_eq!(
                trace.rows.iter().map(|row| row.truth).collect::<Vec<_>>(),
                [
                    Truth::False,
                    Truth::True,
                    Truth::False,
                    Truth::False,
                    Truth::True
                ]
            );
        }

        #[tokio::test]
        async fn a_rule_input_adds_its_source_query_and_deleting_it_removes_the_query() {
            let run = |when: RuleCond, action: RuleAction| async move {
                let sources = InMemorySources::default().tweet(1, 10).authors(&[10]);
                let scenario = Scenario::new(sources, SafetyLevel::FilterAll).viewer(Some(50));
                let engine = RuleEngine::with_level_for_tests(
                    SafetyLevel::FilterAll,
                    Vec::new(),
                    vec![("rule", rule(Everyone, when, action))],
                );
                scenario.evaluate_with(engine, &[raw(1, None)]).await;
                scenario
            };
            let selects = |when: RuleCond| async move {
                run(when, drop_post(DropAs::UnspecifiedReason))
                    .await
                    .sources()
                    .selects()
            };
            let reads_labels = |action: RuleAction| async move {
                run(all![], action)
                    .await
                    .sources()
                    .calls()
                    .contains(&Source::SafetyLabels)
            };
            assert_eq!(
                selects(ViewerFollowsAuthor.leaf()).await,
                [vec![EdgeQuery::forward(Graph::Follows, vec![10])]]
            );
            assert_eq!(
                selects(const { all![ViewerFollowsAuthor, ViewerBlocksAuthor] }).await,
                [vec![
                    EdgeQuery::forward(Graph::Follows, vec![10]),
                    EdgeQuery::forward(Graph::Blocks, vec![10]),
                ]]
            );
            assert_eq!(selects(all![]).await, Vec::<Vec<EdgeQuery>>::new());
            assert!(
                reads_labels(soft_intervention(
                    SafetyLabelType::FOSNR_ABUSE,
                    AppealablePolicy::ABUSE,
                    1
                ))
                .await
            );
            assert!(!reads_labels(drop_post(DropAs::UnspecifiedReason)).await);
        }

        async fn hydrate(plan: &HydrationPlan) -> Hydration {
            let sources = InMemorySources::default().tweet(1, 10).authors(&[10]);
            let raw = vec![RawCandidate {
                tweet_id: TweetId(1),
                request_author_id: None,
            }];
            let caches = FallbackCaches::default();
            let metrics =
                Metrics::none().request(Endpoint::FilterTweets, RequestLevel::Named(plan.level()));
            let request = HydrationRequest::new(
                RequestFacts::new(
                    RequestViewer::new(Some(50), Some("us".into())),
                    ClientCapability::DEFAULT,
                ),
                raw,
                RetweetMerge::Unmerged,
                &caches,
                &metrics,
            );
            plan.hydrate(&sources, request).await
        }

        #[tokio::test]
        async fn admission_rejects_a_hydration_without_a_required_provider_before_either_mode() {
            let rule_engine = RuleEngine::for_tests();
            let unplanned = hydrate(&HydrationPlan::new(
                SafetyLevel::TimelineHome,
                Hydrators::empty(),
            ))
            .await;
            assert!(unplanned
                .tweet(TweetId(1))
                .and_then(|tweet| tweet.candidate())
                .is_some());
            assert!(rule_engine
                .admit(SafetyLevel::TimelineHome, &unplanned)
                .is_none());
            let missing = RuleEngine::unplanned_inputs(
                rule_engine.policy(SafetyLevel::TimelineHome),
                &unplanned,
            );
            assert!(
                missing.contains(Hydrator::Follows) && missing.contains(Hydrator::AuthorSafety),
                "{missing:?}"
            );

            let planned = hydrate(rule_engine.plan(SafetyLevel::TimelineHome)).await;
            assert!(rule_engine
                .admit(SafetyLevel::TimelineHome, &planned)
                .is_some());
        }

        #[tokio::test]
        async fn admission_covers_the_inputs_of_rules_after_the_terminal_one() {
            let rules = vec![
                (
                    "terminal",
                    rule(Everyone, all![], drop_post(DropAs::UnspecifiedReason)),
                ),
                (
                    "later",
                    rule(
                        Everyone,
                        ViewerFollowsAuthor.leaf(),
                        drop_post(DropAs::AuthorIsProtected),
                    ),
                ),
            ];
            let rule_engine =
                RuleEngine::with_level_for_tests(SafetyLevel::FilterAll, Vec::new(), rules);
            let unplanned = hydrate(&HydrationPlan::new(
                SafetyLevel::FilterAll,
                Hydrators::empty(),
            ))
            .await;
            assert!(rule_engine
                .admit(SafetyLevel::FilterAll, &unplanned)
                .is_none());
            assert_eq!(
                RuleEngine::unplanned_inputs(
                    rule_engine.policy(SafetyLevel::FilterAll),
                    &unplanned
                ),
                Hydrators::of(Hydrator::Follows)
            );
            let planned = hydrate(rule_engine.plan(SafetyLevel::FilterAll)).await;
            assert!(rule_engine
                .admit(SafetyLevel::FilterAll, &planned)
                .is_some());
        }

        #[tokio::test]
        async fn a_user_row_is_judged_from_the_admitted_hydration() {
            let level = SafetyLevel::QuotedTweetRules;
            let rule_engine = RuleEngine::with_level_for_tests(
                level,
                vec![(
                    "suspended",
                    Rule {
                        id: RuleId::FilterAll,
                        audience: Everyone,
                        when: AuthorIsSuspended.leaf(),
                        action: drop_post(DropAs::AuthorIsSuspended),
                    },
                )],
                Vec::new(),
            );
            let sources = InMemorySources::default()
                .user(10, suspended())
                .authors(&[20]);
            let caches = FallbackCaches::default();
            let hydration = HydrationPlan::with_rows(
                level,
                RequestRows::Users,
                Hydrators::of(Hydrator::AuthorSafety),
            )
            .hydrate(
                &sources,
                HydrationRequest::for_users(
                    RequestFacts::logged_in(50),
                    vec![AuthorId(10), AuthorId(20)],
                    &caches,
                    &Metrics::none()
                        .request(Endpoint::FilterNotifications, RequestLevel::Named(level)),
                ),
            )
            .await;
            let users = rule_engine.admit(level, &hydration).unwrap();
            let verdict = |id| {
                users
                    .user(AuthorId(id))
                    .map(|user| user.evaluate().into_verdict())
            };
            assert_eq!(
                verdict(10),
                Some(dropped(FilteredReason::AuthorIsSuspended, "suspended"))
            );
            assert_eq!(verdict(20), Some(allow()));
        }

        #[test]
        fn total_observation_completes_suppressed_restrictions_and_tells_engagement_reasons_apart()
        {
            let always = |name, action| (name, rule(Everyone, all![], action));
            let nudity = blur(InterstitialReason::Nudity(true));
            let notice =
                soft_intervention(SafetyLabelType::FOSNR_ABUSE, AppealablePolicy::ABUSE, 1);
            let policy = Policy::new(vec![
                always("first", blur(InterstitialReason::Sensitive(true))),
                always("suppressed", nudity.clone()),
                always("notice", notice.clone()),
                always(
                    "filled",
                    limit(LimitedEngagementReason::ConversationControl),
                ),
                always("added", limit(LimitedEngagementReason::BlockedViewer)),
                always(
                    "duplicate",
                    limit(LimitedEngagementReason::ConversationControl),
                ),
            ]);
            let viewer = viewer(VIEWER_ID);
            let candidate = candidate().build();
            let context = test_context(&viewer, &candidate).hydrated_by(policy.hydrators);
            let trace = policy.trace(&context);
            let served = policy.evaluate(&context);
            assert_eq!(trace.evaluation, served);
            assert_eq!(
                trace.rows.iter().map(|row| row.effect).collect::<Vec<_>>(),
                [
                    Effect::FilledMedia,
                    Effect::KeptEarlierMedia,
                    Effect::KeptEarlierMediaInsteadOfFosnr,
                    Effect::FilledEngagement,
                    Effect::AddedEngagementReason,
                    Effect::DuplicateEngagementReason,
                ]
            );
            let completed = |action: &RuleAction| {
                Some((
                    action.spec().interpret(&EvaluateInputs(&context)).value,
                    Hydrators::empty(),
                ))
            };
            assert_eq!(trace.rows[1].completed, completed(&nudity));
            let Some((Some(Contribution::Notice(Notice::SoftIntervention(_))), _)) =
                trace.rows[2].completed
            else {
                panic!("a suppressed notice is completed: {:?}", trace.rows[2]);
            };
            let Verdict::Shown {
                label_interstitial: None,
                media: Some(media),
                engagement: Some(engagement),
            } = served.into_verdict()
            else {
                panic!("the blur and the limits are shown without the notice");
            };
            assert_eq!((media.by, engagement.by), ("first", "filled"));
            assert_eq!(
                engagement.value.reasons().collect::<Vec<_>>(),
                [
                    LimitedEngagementReason::ConversationControl,
                    LimitedEngagementReason::BlockedViewer,
                ]
            );
        }

        #[test]
        fn total_observation_admits_and_retains_the_defaults_each_opportunity_allows() {
            let viewer = viewer(VIEWER_ID);
            let mut unknown = candidate().build();
            unknown.failed = Hydrators::of(Hydrator::Follows);
            let follows = unknown.failed;
            let none = Hydrators::empty();
            let always = |name, action| (name, rule(Everyone, all![], action));
            let when_followed =
                |name, action| (name, rule(Everyone, ViewerFollowsAuthor.leaf(), action));
            let sensitive = || blur(InterstitialReason::Sensitive(true));
            let conversation = || limit(LimitedEngagementReason::ConversationControl);
            let notice =
                || soft_intervention(SafetyLabelType::FOSNR_ABUSE, AppealablePolicy::ABUSE, 1);
            let drop = || drop_post(DropAs::UnspecifiedReason);
            let unsorted = |rules: Vec<(&'static str, Rule)>| Policy {
                hydrators: Policy::new(rules.clone()).hydrators,
                rules,
            };
            for (case, policy, admitted, retained) in [
                (
                    "a withholding before the decision",
                    Policy::new(vec![
                        when_followed("unknown", drop()),
                        always("decides", drop()),
                    ]),
                    [follows, none],
                    [follows, none],
                ),
                (
                    "an empty slot in a shown verdict",
                    Policy::new(vec![
                        when_followed("unknown", sensitive()),
                        always("limit", conversation()),
                    ]),
                    [follows, none],
                    [follows, none],
                ),
                (
                    "an empty slot under a notice",
                    Policy::new(vec![
                        when_followed("unknown", sensitive()),
                        always("notice", notice()),
                    ]),
                    [follows, none],
                    [follows, none],
                ),
                (
                    "a filled media slot",
                    Policy::new(vec![
                        always("blur", sensitive()),
                        when_followed("unknown", sensitive()),
                    ]),
                    [none, none],
                    [none, none],
                ),
                (
                    "a filled media slot before a notice",
                    Policy::new(vec![
                        always("blur", sensitive()),
                        when_followed("unknown", notice()),
                    ]),
                    [none, none],
                    [none, none],
                ),
                (
                    "a filled engagement slot",
                    Policy::new(vec![
                        always("limit", conversation()),
                        when_followed("unknown", conversation()),
                    ]),
                    [none, none],
                    [none, none],
                ),
                (
                    "an empty slot ahead of a withholding",
                    unsorted(vec![
                        when_followed("unknown", sensitive()),
                        always("decides", drop()),
                    ]),
                    [follows, none],
                    [none, none],
                ),
                (
                    "an empty engagement slot ahead of a notice",
                    unsorted(vec![
                        when_followed("unknown", conversation()),
                        always("notice", notice()),
                    ]),
                    [follows, none],
                    [none, none],
                ),
            ] {
                let trace =
                    policy.trace(&test_context(&viewer, &unknown).hydrated_by(policy.hydrators));
                let by_row =
                    |row: &RowTrace| (row.by, row.defaults_admitted, row.defaults_retained);
                let mut rows: Vec<_> = trace.rows.iter().map(by_row).collect();
                rows.sort_by_key(|&(by, ..)| by != "unknown");
                assert_eq!(
                    rows.iter()
                        .map(|&(_, admitted, _)| admitted)
                        .collect::<Vec<_>>(),
                    admitted,
                    "{case}"
                );
                assert_eq!(
                    rows.iter()
                        .map(|&(.., retained)| retained)
                        .collect::<Vec<_>>(),
                    retained,
                    "{case}"
                );
                assert_eq!(trace.evaluation.fail_open_defaults(), retained[0], "{case}");
            }
        }

        #[test]
        fn total_observation_keeps_absorbed_unknowns_out_of_the_defaults() {
            let viewer = viewer(VIEWER_ID);
            let mut unknown = candidate().build();
            unknown.failed = Hydrators::of(Hydrator::Follows);
            let policy = Policy::new(vec![
                (
                    "absorbed_by_false",
                    rule(
                        Everyone,
                        const { all![any![], ViewerFollowsAuthor] },
                        drop_post(DropAs::AuthorIsProtected),
                    ),
                ),
                (
                    "absorbed_by_true",
                    rule(
                        Everyone,
                        const { any![all![], ViewerFollowsAuthor] },
                        blur(InterstitialReason::Sensitive(true)),
                    ),
                ),
            ]);
            let context = test_context(&viewer, &unknown).hydrated_by(policy.hydrators);
            let trace = policy.trace(&context);
            assert_eq!(trace.evaluation, policy.evaluate(&context));
            assert_eq!(
                trace
                    .rows
                    .iter()
                    .map(|row| (row.truth, row.defaults_admitted))
                    .collect::<Vec<_>>(),
                [
                    (Truth::False, Hydrators::empty()),
                    (Truth::True, Hydrators::empty())
                ]
            );
            let follows: Vec<&NodeTrace> = trace
                .nodes
                .iter()
                .filter(|node| {
                    node.kind == NodeKind::Leaf(Predicate::Relationship(ViewerFollowsAuthor))
                })
                .collect();
            let [after_false, after_true] = follows[..] else {
                panic!("each rule observes its follows leaf");
            };
            for (leaf, skip) in [
                (after_false, Skip::FalseConjunct(after_false.node - 1)),
                (after_true, Skip::TrueDisjunct(after_true.node - 1)),
            ] {
                assert_eq!(
                    leaf.truth,
                    Truth::Unknown {
                        default: false,
                        failed: unknown.failed
                    }
                );
                assert_eq!(leaf.skip, Some(skip));
            }
            assert_eq!(trace.evaluation.fail_open_defaults(), Hydrators::empty());
        }

        #[test]
        fn total_observation_marks_an_appeal_without_an_eligible_label() {
            let policy = Policy::new(vec![
                (
                    "appeal",
                    rule(
                        Everyone,
                        all![],
                        appealable(&tweet_rules::FOSNR_AUTHOR_VIOLATIONS),
                    ),
                ),
                (
                    "decides",
                    rule(Everyone, all![], drop_post(DropAs::UnspecifiedReason)),
                ),
            ]);
            let viewer = viewer(VIEWER_ID);
            let candidate = candidate().build();
            let context = test_context(&viewer, &candidate).hydrated_by(policy.hydrators);
            let trace = policy.trace(&context);
            assert_eq!(trace.evaluation, policy.evaluate(&context));
            let appeal = &trace.rows[0];
            assert_eq!(
                (appeal.truth, appeal.effect, appeal.completed.clone()),
                (
                    Truth::True,
                    Effect::DidNotApply,
                    Some((None, Hydrators::empty()))
                )
            );
            assert_eq!(trace.rows[1].effect, Effect::Decided);
            assert_eq!(
                trace.evaluation,
                Evaluation::complete(dropped(FilteredReason::UnspecifiedReason, "decides"))
            );
        }

        #[test]
        fn rules_compare_and_print_their_structure_payload_and_names() {
            use crate::rules::rule_spec::leaves::TweetHasLabel;
            static FOLLOWED: NamedCond = NamedCond {
                name: "followed",
                condition: ViewerFollowsAuthor.leaf(),
            };
            static RENAMED: NamedCond = NamedCond {
                name: "renamed",
                condition: ViewerFollowsAuthor.leaf(),
            };
            let base = rule(
                NonAuthors,
                const {
                    all![
                        RuleCond::Named(&FOLLOWED),
                        TweetHasLabel(SafetyLabelType::SPAM)
                    ]
                },
                drop_post(DropAs::UnspecifiedReason),
            );
            assert_eq!(
                base.authored(),
                "filter_all/drop/unspecified | audience=NonAuthors \
                 | when=all[followed=viewer_follows_author & tweet_has_label(SPAM)] \
                 | Drop(Legacy(UnspecifiedReason))"
            );
            let changes = [
                (
                    "structure",
                    const {
                        any![
                            RuleCond::Named(&FOLLOWED),
                            TweetHasLabel(SafetyLabelType::SPAM)
                        ]
                    },
                ),
                (
                    "leaf kind",
                    const { all![RuleCond::Named(&FOLLOWED), ViewerMutesAuthor] },
                ),
                (
                    "payload",
                    const {
                        all![
                            RuleCond::Named(&FOLLOWED),
                            TweetHasLabel(SafetyLabelType::BOUNCE)
                        ]
                    },
                ),
                (
                    "name",
                    const {
                        all![
                            RuleCond::Named(&RENAMED),
                            TweetHasLabel(SafetyLabelType::SPAM)
                        ]
                    },
                ),
            ];
            assert_eq!(base, base.clone());
            for (change, when) in changes {
                let changed = Rule {
                    when,
                    ..base.clone()
                };
                assert_ne!(changed, base, "{change}");
                assert_ne!(format!("{changed:?}"), format!("{base:?}"), "{change}");
                assert_ne!(changed.authored(), base.authored(), "{change}");
            }
        }

        #[test]
        fn two_definitions_of_one_name_fail_validation_and_shared_uses_pass() {
            static SHARED_ALL: NamedCond = NamedCond {
                name: "shared",
                condition: all![],
            };
            static SHARED_ANY: NamedCond = NamedCond {
                name: "shared",
                condition: any![],
            };
            let conflicting = [
                rule(
                    Everyone,
                    RuleCond::Named(&SHARED_ALL),
                    drop_post(DropAs::UnspecifiedReason),
                ),
                rule(
                    Everyone,
                    RuleCond::Named(&SHARED_ANY),
                    drop_post(DropAs::AuthorIsProtected),
                ),
            ];
            assert_eq!(
                rule::validate(&conflicting),
                Err(DescriptionError::ConflictingName("shared"))
            );
            let shared = [
                rule(
                    Everyone,
                    RuleCond::Named(&SHARED_ALL),
                    drop_post(DropAs::UnspecifiedReason),
                ),
                rule(
                    Everyone,
                    const { RuleCond::Not(&RuleCond::Named(&SHARED_ALL)) },
                    drop_post(DropAs::AuthorIsProtected),
                ),
            ];
            assert_eq!(rule::validate(&shared), Ok(()));
        }

        #[test]
        #[should_panic(expected = "invalid rules: Cycle")]
        fn a_cyclic_policy_fails_construction() {
            static CYCLE: RuleCond = RuleCond::Not(&CYCLE);
            Policy::new(vec![(
                "cyclic",
                rule(Everyone, CYCLE, drop_post(DropAs::UnspecifiedReason)),
            )]);
        }
    }
}
