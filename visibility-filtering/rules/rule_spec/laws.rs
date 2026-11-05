use super::conditions::{
    VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
    VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES,
    VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES,
};
use super::engine::{Engine, Fold, UserEngine};
use super::features::{CompoundLeaf, ReplyControlLeaf, RequestLeaf};
use super::rule::Audience::{self, Everyone, NonAuthors, NonFollowers, OnlyAuthor};
use super::rule::{Rule, RuleCond};
use super::*;
use crate::hydration::inventory::Tier;
use crate::hydration::Hydrator;
use crate::models::{
    ClientCapability, ClientSwitch, PostCandidate, RequestFacts, SafetyLabelMap, SafetyLabelType,
    UserCandidate, VerifyBlurSupport, ViewerFeatures,
};
use crate::params::CountryLists;
use crate::rules::expansion::Expansion;
use crate::rules::fixtures::{candidate, viewer, VIEWER_ID};
use crate::rules::golden_corpus::{corpus, Subject};
use crate::rules::levels::LevelLists;
use crate::rules::SafetyLevel;
use crate::rules::{placed_test_context, test_context, tweet_rules};
use std::iter::once;
use strum::VariantArray;

fn truths() -> Vec<Truth> {
    let a = Hydrators::of(Hydrator::Follows);
    let b = Hydrators::of(Hydrator::Blocks);
    let mut truths = vec![Truth::True, Truth::False];
    for default in [false, true] {
        for failed in [a, b, a.union(b)] {
            truths.push(Truth::Unknown { default, failed });
        }
    }
    truths
}

fn wired<E: Engine>(list: fn(LevelLists) -> Vec<&'static Rule<E>>) -> Vec<&'static Rule<E>> {
    let mut rules: Vec<&'static Rule<E>> = Vec::new();
    for &level in SafetyLevel::VARIANTS {
        for rule in level.rule_lists().flat_map(list) {
            if !rules.contains(&rule) {
                rules.push(rule);
            }
        }
    }
    rules
}

fn wired_user_rules() -> Vec<&'static Rule<UserEngine>> {
    wired(|rules| rules.user)
}

fn wired_rules() -> Vec<&'static Rule> {
    wired(|rules| rules.post)
}

fn wired_leaves() -> Vec<Predicate> {
    let mut read = Vec::new();
    for rule in wired_user_rules() {
        rule.leaves(&mut read);
    }
    for rule in wired_rules() {
        rule.leaves(&mut read);
    }
    let mut leaves = Vec::new();
    for leaf in read {
        if !leaves.contains(&leaf) {
            leaves.push(leaf);
        }
    }
    leaves
}

#[test]
fn kleene_algebra() {
    let truths = truths();
    for &a in &truths {
        assert_eq!(a.and(Truth::True), a, "{a:?}");
        assert_eq!(a.or(Truth::False), a, "{a:?}");
        assert_eq!(a.and(Truth::False), Truth::False, "{a:?}");
        assert_eq!(a.or(Truth::True), Truth::True, "{a:?}");
        assert_eq!(!!a, a, "{a:?}");
        assert_eq!((!a).resolves_true(), !a.resolves_true(), "{a:?}");
        for &b in &truths {
            assert_eq!(a.and(b), b.and(a), "{a:?} {b:?}");
            assert_eq!(a.or(b), b.or(a), "{a:?} {b:?}");
            assert_eq!(!a.and(b), (!a).or(!b), "{a:?} {b:?}");
            assert_eq!(!a.or(b), (!a).and(!b), "{a:?} {b:?}");
            assert_eq!(
                a.and(b).resolves_true(),
                a.resolves_true() && b.resolves_true(),
                "{a:?} {b:?}"
            );
            assert_eq!(
                a.or(b).resolves_true(),
                a.resolves_true() || b.resolves_true(),
                "{a:?} {b:?}"
            );
            for &c in &truths {
                assert_eq!(a.and(b).and(c), a.and(b.and(c)), "{a:?} {b:?} {c:?}");
                assert_eq!(a.or(b).or(c), a.or(b.or(c)), "{a:?} {b:?} {c:?}");
            }
        }
    }
}

#[test]
fn a_failed_read_is_local() {
    let viewer = viewer(VIEWER_ID);
    let fetched = candidate().build();
    for leaf in wired_leaves() {
        let clean = leaf.truth(&test_context(&viewer, &fetched));
        for &hydrator in Hydrator::VARIANTS {
            let mut failed = fetched.clone();
            failed.failed = Hydrators::of(hydrator);
            let expected = if leaf.requirements().contains(hydrator) {
                Truth::Unknown {
                    default: clean.resolves_true(),
                    failed: Hydrators::of(hydrator),
                }
            } else {
                clean
            };
            assert_eq!(
                leaf.truth(&test_context(&viewer, &failed)),
                expected,
                "{leaf:?} with {hydrator:?} failed"
            );
        }
    }
}

const FAILS_THE_NOTIFICATION: &str =
    "FilterNotifications fails a notification whose verdict relied on a failed node";

const ACCEPTED_BEST_EFFORT_READS: &[(&str, Hydrator, &str)] = &[
    (
        "protected_author/drop",
        Hydrator::Follows,
        "the default drops the post for real followers",
    ),
    (
        "abusive_high_recall_user_label/drop/unspecified",
        Hydrator::Follows,
        "the default drops the post for real followers",
    ),
    (
        "do_not_amplify_user_label/drop/unspecified",
        Hydrator::Follows,
        "the default drops the post for real followers",
    ),
    (
        "quoted_viewer_mutes_author_not_following/drop",
        Hydrator::Follows,
        "the default drops the quote of a muted author the viewer really follows",
    ),
    (
        "protected_quoted_author/drop",
        Hydrator::OuterAuthorFollowsAuthor,
        "the default drops the quote of a protected author the quoting author really follows",
    ),
    (
        "fosnr_abuse_insults_non_follower/drop/undesirable",
        Hydrator::Follows,
        "the default drops the post for real followers",
    ),
    (
        "fosnr_abuse_insults_follower/soft_intervention/abuse",
        Hydrator::Follows,
        "the non-follower drop on the same label fires instead",
    ),
    (
        "author_blocks_viewer_exclusive_content/drop/unspecified",
        Hydrator::BlockedByAuthor,
        "only a subscriber the author blocked sees the exclusive post",
    ),
    (
        "viewer_mutes_and_does_not_follow_author/drop/viewer_mutes_author",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "no_confirmed_email/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "no_confirmed_phone/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "no_default_profile_image/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "no_new_users/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "no_not_followed_by/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "only_people_i_follow/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "only_people_i_follow_outside_community/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "not_graduated_user_label/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "agatha_spam_top_user_user_label/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "engagement_spammer_user_label/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "blink_worst_user_label/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "downrank_spam_reply_user_label/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "duplicate_mention_uqf/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "abusive_uqf/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "untrusted_url_uqf/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
    (
        "downrank_spam_reply_uqf/drop/unspecified",
        Hydrator::Follows,
        FAILS_THE_NOTIFICATION,
    ),
];

#[test]
fn a_rule_reading_a_best_effort_node_limits_engagement_or_is_accepted() {
    let mut unaccepted = Vec::new();
    let mut accepted = Vec::new();
    let user = wired_user_rules().into_iter().map(|rule| {
        let limits = matches!(rule.action.spec(), ActionSpec::LimitedEngagement(_));
        (rule.name(), limits, rule.requirements(), Hydrators::empty())
    });
    let post = wired_rules().into_iter().map(|rule| {
        let limits = matches!(rule.action.spec(), ActionSpec::LimitedEngagement(_));
        let shown = rule.requirements().iter().filter(|node| {
            node.relation()
                .and_then(Expansion::of)
                .is_some_and(|expansion| {
                    let spec = expansion.spec();
                    spec.on_failure.tier() == Tier::BestEffort && spec.rules.contains(&rule)
                })
        });
        let shown = shown.fold(Hydrators::empty(), Hydrators::with);
        (rule.name(), limits, rule.requirements(), shown)
    });
    for (name, limits, requirements, shown) in user.chain(post) {
        for node in requirements.iter().filter(|node| node.is_best_effort()) {
            match ACCEPTED_BEST_EFFORT_READS
                .iter()
                .find(|&&(by, accepted, _)| by == name && accepted == node)
            {
                Some(&(by, node, _)) => accepted.push((by, node)),
                None if limits || shown.contains(node) => {}
                None => unaccepted.push((name.clone(), node)),
            }
        }
    }
    assert!(unaccepted.is_empty(), "{unaccepted:?}");
    for &(by, node, _) in ACCEPTED_BEST_EFFORT_READS {
        assert!(
            accepted.contains(&(by, node)),
            "{by} no longer reads {node:?}"
        );
    }
}

fn holds<E: Fold>(condition: RuleCond<E>, context: &RuleContext<'_, E::Candidate>) -> bool {
    match condition {
        RuleCond::Leaf(leaf) => truth_of(leaf, context).resolves_true(),
        RuleCond::All(children) => children.iter().all(|&child| holds(child, context)),
        RuleCond::Any(children) => children.iter().any(|&child| holds(child, context)),
        RuleCond::Not(child) => !holds(*child, context),
        RuleCond::Named(named) => holds(named.condition, context),
    }
}

fn faithful<E: Fold>(
    rules: &[&Rule<E>],
    context: &RuleContext<'_, E::Candidate>,
    failed: Hydrators,
) {
    for &rule in rules {
        if rule.requirements().intersection(failed) != failed {
            continue;
        }
        let (truth, ()) = observed::evaluate_rule(rule, context, &mut observed::NoTrace, None);
        let (total, _) =
            observed::evaluate_rule(rule, context, &mut observed::Total::default(), None);
        assert_eq!(truth, total, "{} with {failed:?} failed", rule.name());
        assert_eq!(
            truth.resolves_true(),
            holds(rule.audience.condition::<E>(), context) && holds(rule.when, context),
            "{} with {failed:?} failed",
            rule.name()
        );
    }
}

#[test]
fn the_rule_evaluator_is_faithful() {
    let (user_rules, rules) = (wired_user_rules(), wired_rules());
    let country_lists = CountryLists::starting_at_default();
    let failures: Vec<Hydrators> = once(Hydrators::empty())
        .chain(
            Hydrator::VARIANTS
                .iter()
                .map(|&hydrator| Hydrators::of(hydrator)),
        )
        .collect();
    for case in corpus() {
        let (user, post) = match case.subject {
            Subject::Post(post) => (UserCandidate::author_of(&post), Some(post)),
            Subject::User(user) => (user, None),
        };
        for &failed in &failures {
            let user = UserCandidate {
                failed,
                ..user.clone()
            };
            faithful(
                &user_rules,
                &RuleContext::new(&case.viewer, &user, case.placement, &country_lists),
                failed,
            );
            if let Some(post) = &post {
                let candidate = PostCandidate {
                    failed,
                    ..post.clone()
                };
                faithful(
                    &rules,
                    &placed_test_context(&case.viewer, &candidate, case.placement),
                    failed,
                );
            }
        }
    }
}

fn reads_one_definition<E: Fold>(
    rule: &Rule<E>,
    definitions: &[RuleCond<E>],
    leaves: &[Predicate],
) -> bool {
    fn walk<E: Engine>(
        condition: RuleCond<E>,
        definitions: &[RuleCond<E>],
        negated: bool,
        outside: &mut Vec<Predicate>,
        uses: &mut Vec<bool>,
    ) {
        match condition {
            condition if definitions.contains(&condition) => uses.push(negated),
            RuleCond::Leaf(leaf) => outside.push(E::lift(leaf)),
            RuleCond::All(children) | RuleCond::Any(children) => {
                for &child in children {
                    walk(child, definitions, negated, outside, uses);
                }
            }
            RuleCond::Not(child) => walk(*child, definitions, !negated, outside, uses),
            RuleCond::Named(named) => walk(named.condition, definitions, negated, outside, uses),
        }
    }
    let (mut outside, mut uses) = (Vec::new(), Vec::new());
    for root in [rule.audience.condition::<E>(), rule.when] {
        walk(root, definitions, false, &mut outside, &mut uses);
    }
    for leaf in leaves {
        assert!(
            !outside.contains(leaf),
            "{} reads {leaf:?} outside its named definition",
            rule.name()
        );
    }
    if !uses.is_empty() {
        assert_eq!(
            uses,
            [false],
            "{} must read one definition once, positively",
            rule.name()
        );
    }
    !uses.is_empty()
}

fn readers_through(definitions: &[RuleCond], leaves: &[Predicate]) -> usize {
    let user = wired_user_rules()
        .into_iter()
        .filter(|rule| reads_one_definition(rule, &[], leaves));
    let post = wired_rules()
        .into_iter()
        .filter(|rule| reads_one_definition(rule, definitions, leaves));
    user.count() + post.count()
}

#[test]
fn the_age_gate_is_whole() {
    let in_list = |list| Predicate::Request(RequestLeaf::RequestCountryIn(list));
    assert_ne!(
        readers_through(
            &[VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES],
            &[in_list(CountryList::AgeVerification)]
        ),
        0
    );
    assert_ne!(
        readers_through(
            &[VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES],
            &[in_list(CountryList::Tombstone)]
        ),
        0
    );
    assert_ne!(
        readers_through(
            &[
                VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES,
                VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES
            ],
            &[Predicate::Compound(CompoundLeaf::ViewerAgeVerified)]
        ),
        0
    );
}

#[test]
fn the_conversation_root_exemption_is_whole() {
    assert_ne!(
        readers_through(
            &[VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED],
            &[
                Predicate::ReplyControl(ReplyControlLeaf::ViewerIsConversationRootAuthor),
                Predicate::ReplyControl(ReplyControlLeaf::ViewerIsInvitedToConversation),
            ],
        ),
        0
    );
}

#[test]
fn every_client_capability_and_rule_switch_is_live() {
    let base = ClientCapability::default();
    let with_client = |client| RequestFacts::default().with_client(client);
    let blur_supports = once(None)
        .chain(VerifyBlurSupport::VARIANTS.iter().copied().map(Some))
        .filter(|&support| support != base.verify_blur_support)
        .map(|support| {
            with_client(ClientCapability {
                verify_blur_support: support,
                ..base
            })
        })
        .collect();
    let toggle = |switch| {
        with_client(if base.has(switch) {
            base.without(switch)
        } else {
            base.with(switch)
        })
    };
    let toggled: Vec<(&str, Vec<RequestFacts>)> = [
        ("verify_blur_support", blur_supports),
        (
            "soft_nsfw_new_user_drop",
            vec![RequestFacts::default().with_soft_nsfw_new_user_drop(true)],
        ),
    ]
    .into_iter()
    .chain(
        ClientSwitch::VARIANTS
            .iter()
            .map(|&switch| (<&str>::from(switch), vec![toggle(switch)])),
    )
    .collect();
    let leaves = wired_leaves();
    let fetched = candidate().build();
    let holds = |facts: &RequestFacts, leaf: Predicate| {
        let features = ViewerFeatures {
            facts: facts.clone(),
            ..viewer(VIEWER_ID)
        };
        leaf.holds(&test_context(&features, &fetched))
    };
    let default = RequestFacts::default();
    for (field, variants) in toggled {
        let live = variants.iter().any(|facts| {
            leaves
                .iter()
                .any(|&leaf| holds(facts, leaf) != holds(&default, leaf))
        });
        assert!(
            live,
            "no wired leaf reads the request's {field}: wire a rule that reads it, or delete it"
        );
    }
}

#[test]
fn only_the_quote_pass_reads_the_quote() {
    let mut readers = Vec::new();
    for &level in SafetyLevel::VARIANTS {
        let LevelLists { user, post } = level.spec().rules().lists();
        let mut read = Vec::new();
        for rule in user {
            rule.when.leaves(&mut read);
        }
        for rule in post {
            rule.when.leaves(&mut read);
        }
        let reads_quote = read.iter().any(|leaf| {
            matches!(
                leaf,
                Predicate::PairEdge(_) | Predicate::AuthorFacts(AuthorFactsLeaf::IsSelfQuote)
            )
        });
        if reads_quote && level != SafetyLevel::QuotedTweetRules {
            readers.push(level);
        }
    }
    assert_eq!(
        readers,
        Vec::<SafetyLevel>::new(),
        "a quote is hydrated only for the quote pass's inner-quoted authors"
    );
}

const POST_RULES_READING_NO_POST: &[(&str, &str)] = &[
    (
        "filter_all/drop/unspecified",
        "FilterAll's one rule drops every post, from either list",
    ),
    (
        "read_only_viewer/limited_engagement",
        "the user list would put it first among the engagement limits, so it would decide `by`",
    ),
];

#[test]
fn every_post_rule_reads_the_post() {
    let reads_the_post = |leaf: &Predicate| match leaf {
        Predicate::Request(_)
        | Predicate::ViewerAccount(_)
        | Predicate::ViewerLabel(_)
        | Predicate::NotificationFilter(_)
        | Predicate::AuthorFacts(_)
        | Predicate::AuthorSafety(_)
        | Predicate::AuthorLabels(_)
        | Predicate::AuthorProfile(_)
        | Predicate::AuthorEmail(_)
        | Predicate::AuthorPhone(_)
        | Predicate::Relationship(_)
        | Predicate::PairEdge(_) => false,
        Predicate::PostFacts(_)
        | Predicate::SourceTweetId(_)
        | Predicate::Tweet(_)
        | Predicate::TweetSafetyLabels(_)
        | Predicate::ReplyControl(_)
        | Predicate::PostModeration(_)
        | Predicate::ArticleStage(_)
        | Predicate::CommunityModerator(_)
        | Predicate::CommunityViewerRemoved(_)
        | Predicate::ReactedToPostSafetyLabels(_)
        | Predicate::ReactedToAuthorSafety(_)
        | Predicate::Edge(_)
        | Predicate::Compound(_) => true,
    };
    let reading_no_post: Vec<String> = wired_rules()
        .into_iter()
        .filter(|rule| {
            let mut read = Vec::new();
            rule.when.leaves(&mut read);
            !rule.action.spec().kind().lands_in_fosnr() && !read.iter().any(reads_the_post)
        })
        .map(|rule| rule.name())
        .collect();
    assert_eq!(
        reading_no_post,
        POST_RULES_READING_NO_POST
            .iter()
            .map(|&(name, _)| name)
            .collect::<Vec<_>>(),
        "a post rule that reads no post is a `UserRule` in its levels' user lists, or a row of \
         POST_RULES_READING_NO_POST"
    );
}

#[test]
fn a_less_severe_action_kind_never_relies_on_an_earlier_defaults_bucket() {
    let mut kinds = ActionKind::VARIANTS.to_vec();
    kinds.sort_by_key(|kind| std::cmp::Reverse(kind.severity()));
    for pair in kinds.windows(2) {
        assert!(pair[0].bucket() <= pair[1].bucket(), "{pair:?}");
    }
}

#[test]
fn a_user_rule_that_ends_the_fold_never_shadows_a_more_severe_post_rule() {
    let disjoint = |user: Audience, post: Audience| match (user, post) {
        (OnlyAuthor, NonAuthors | NonFollowers) | (NonAuthors | NonFollowers, OnlyAuthor) => true,
        (Everyone, _)
        | (_, Everyone)
        | (OnlyAuthor, OnlyAuthor)
        | (NonAuthors | NonFollowers, NonAuthors | NonFollowers) => false,
    };
    let mut shadowing = Vec::new();
    for &level in SafetyLevel::VARIANTS {
        let LevelLists { user, post } = level.spec().rules().lists();
        for &user_rule in user
            .iter()
            .filter(|rule| rule.action.spec().kind().ends_the_fold())
        {
            let severity = user_rule.action.spec().kind().severity();
            for &post_rule in &post {
                if post_rule.action.spec().kind().severity() > severity
                    && !disjoint(user_rule.audience, post_rule.audience)
                {
                    shadowing.push(format!(
                        "{level:?}: user {} shadows post {}",
                        user_rule.name(),
                        post_rule.name()
                    ));
                }
            }
        }
    }
    assert_eq!(
        shadowing,
        Vec::<String>::new(),
        "the user list folds before every post rule, so a user rule that ends the fold needs an \
         audience disjoint from every more severe post rule"
    );
}

pub(in crate::rules) const WIRED_DIFFERENTLY: [&str; 6] = [
    "nsfw_high_precision/blur/nudity",
    "nsfw_high_precision/blur/sensitive",
    "gore_and_violence_high_precision/blur",
    "nsfw_card_image/blur/sensitive",
    "nsfw_admin/blur/sensitive",
    "nsfw_user/blur/sensitive_user",
];

#[test]
fn each_rule_name_wires_the_same_rules_at_every_level() {
    let mut first: Vec<(String, SafetyLevel, Vec<String>)> = Vec::new();
    for &level in SafetyLevel::VARIANTS {
        let rules = level.spec().rules().lists();
        let authored = rules
            .user
            .iter()
            .map(|rule| (rule.name(), rule.authored()))
            .chain(rules.post.iter().map(|rule| (rule.name(), rule.authored())));
        let mut by_name: Vec<(String, Vec<String>)> = Vec::new();
        for (name, rule) in authored {
            match by_name.iter_mut().find(|(other, _)| *other == name) {
                Some((_, rules)) => rules.push(rule),
                None => by_name.push((name, vec![rule])),
            }
        }
        for (name, rules) in by_name {
            match first.iter().find(|(other, ..)| *other == name) {
                None => first.push((name, level, rules)),
                Some((_, first_level, wired)) => assert!(
                    *wired == rules || WIRED_DIFFERENTLY.contains(&name.as_str()),
                    "{name} wires other rules at {level:?} than at {first_level:?}: wire the \
                     same rules at every level, or give the {level:?} rules their own RuleId"
                ),
            }
        }
    }
}

#[test]
fn each_rule_of_a_rule_id_names_the_same_labels() {
    fn labels<E: Engine>(rule: &Rule<E>) -> (RuleId, String, Vec<String>) {
        let mut labels: Vec<String> = rule
            .when
            .conjuncts()
            .iter()
            .filter_map(|condition| condition.own_tweet_label())
            .collect();
        labels.sort_unstable();
        (rule.id, rule.name(), labels)
    }
    let mut first: Vec<(RuleId, String, Vec<String>)> = Vec::new();
    let user = wired_user_rules().into_iter().map(labels);
    for (id, name, labels) in user.chain(wired_rules().into_iter().map(labels)) {
        match first.iter().find(|(other, ..)| *other == id) {
            None => first.push((id, name, labels)),
            Some((_, other, read)) => assert!(
                *read == labels,
                "{name} names the labels {labels:?} but {other}, a rule of the same RuleId, \
                 names {read:?}: copy the label conditions of {other}, or give it its own RuleId"
            ),
        }
    }
}

#[test]
fn each_fosnr_violation_has_a_label_drop_and_each_fosnr_label_drop_a_violation() {
    let mut dropped = Vec::new();
    for rule in wired_rules() {
        let id: &str = rule.id.into();
        let RuleCond::All([only]) = rule.when else {
            continue;
        };
        let [label] = only.positive_tweet_labels()[..] else {
            continue;
        };
        if !id.starts_with("fosnr") {
            continue;
        }
        assert!(
            tweet_rules::FOSNR_AUTHOR_VIOLATIONS
                .iter()
                .any(|violation| violation.label == label),
            "{} drops a FOSNR label FOSNR_AUTHOR_VIOLATIONS does not name: add its row to \
             FOSNR_AUTHOR_VIOLATIONS in src/rules/tweet_rules.rs",
            rule.name()
        );
        dropped.push(label);
    }
    for violation in &tweet_rules::FOSNR_AUTHOR_VIOLATIONS {
        assert!(
            dropped.contains(&violation.label),
            "FOSNR_AUTHOR_VIOLATIONS names {:?} but no level drops it: add a `const \
             FOSNR_<LABEL>_DROP_UNDESIRABLE: Rule` reading it and list it in the levels",
            label_meta(violation.label).map(|meta| meta.ident),
        );
    }
}

#[test]
fn a_notice_reads_the_labels_it_names() {
    for rule in wired_rules() {
        let name = rule.name();
        let mut read = rule.when.positive_tweet_labels();
        let mut named: Vec<SafetyLabelType> = match rule.action.spec() {
            ActionSpec::SoftIntervention(FosnrNotice { label, .. }) => vec![*label],
            ActionSpec::Appealable(violations) => {
                violations.iter().map(|violation| violation.label).collect()
            }
            ActionSpec::LabelInterstitial(spec) => {
                for label in spec.labels() {
                    assert!(
                        SafetyLabelMap::SOURCE_READ_LABELS.contains(label),
                        "{name} reads the source of {} but SafetyLabelMap drops it: add the \
                         label to SafetyLabelMap::SOURCE_READ_LABELS",
                        spelled_label(*label),
                    );
                }
                spec.labels().to_vec()
            }
            ActionSpec::Drop(_)
            | ActionSpec::Tombstone(_)
            | ActionSpec::MediaRestriction(_)
            | ActionSpec::LimitedEngagement(_) => continue,
        };
        read.sort_by_key(|label| label.0);
        named.sort_by_key(|label| label.0);
        assert!(
            read == named,
            "{name} names {:?} but its conditions read {:?}: make `when` read exactly the labels \
             its action names",
            named.into_iter().map(spelled_label).collect::<Vec<_>>(),
            read.into_iter().map(spelled_label).collect::<Vec<_>>(),
        );
    }
}
