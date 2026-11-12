#[cfg(test)]
use super::leaves::TweetHasLabel;
use super::leaves::ViewerIsAuthor;
#[cfg(test)]
use super::{rule_name, spelled_label};
use super::{Predicate, RuleAction, RuleId};
use crate::hydration::Hydrators;
#[cfg(test)]
use crate::models::SafetyLabelType;
use crate::rules::rule_spec::engine::{Engine, Fold, PostEngine, UserEngine};
use std::ptr;
#[cfg(test)]
use std::slice;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::rules) struct Rule<E: Engine = PostEngine> {
    pub(in crate::rules) id: RuleId,
    pub(in crate::rules) audience: Audience,
    pub(in crate::rules) when: RuleCond<E>,
    pub(in crate::rules) action: RuleAction<E>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rules) enum RuleCond<E: Engine = PostEngine> {
    Leaf(Predicate<E>),
    All(&'static [RuleCond<E>]),
    Any(&'static [RuleCond<E>]),
    Not(&'static RuleCond<E>),
    Named(&'static NamedCond<E>),
}

macro_rules! all {
    ($($operand:tt)*) => {
        $crate::rules::rule_spec::rule::RuleCond::All(
            &$crate::rules::rule_spec::rule::conds!(@ [] $($operand)*),
        )
    };
}

macro_rules! any {
    ($($operand:tt)*) => {
        $crate::rules::rule_spec::rule::RuleCond::Any(
            &$crate::rules::rule_spec::rule::conds!(@ [] $($operand)*),
        )
    };
}

macro_rules! conds {
    (@ [$($done:expr),*]) => { [$($done),*] };
    (@ [$($done:expr),*] all ! [$($list:tt)*] $(, $($rest:tt)*)?) => {
        $crate::rules::rule_spec::rule::conds!(
            @ [$($done,)* $crate::rules::rule_spec::rule::all![$($list)*]]
            $($($rest)*)?
        )
    };
    (@ [$($done:expr),*] any ! [$($list:tt)*] $(, $($rest:tt)*)?) => {
        $crate::rules::rule_spec::rule::conds!(
            @ [$($done,)* $crate::rules::rule_spec::rule::any![$($list)*]]
            $($($rest)*)?
        )
    };
    (@ [$($done:expr),*] ! all ! [$($list:tt)*] $(, $($rest:tt)*)?) => {
        $crate::rules::rule_spec::rule::conds!(
            @ [$($done,)* $crate::rules::rule_spec::rule::RuleCond::Not(
                &$crate::rules::rule_spec::rule::all![$($list)*],
            )]
            $($($rest)*)?
        )
    };
    (@ [$($done:expr),*] ! any ! [$($list:tt)*] $(, $($rest:tt)*)?) => {
        $crate::rules::rule_spec::rule::conds!(
            @ [$($done,)* $crate::rules::rule_spec::rule::RuleCond::Not(
                &$crate::rules::rule_spec::rule::any![$($list)*],
            )]
            $($($rest)*)?
        )
    };
    (@ [$($done:expr),*] ! $operand:expr $(, $($rest:tt)*)?) => {
        $crate::rules::rule_spec::rule::conds!(
            @ [$($done,)* $crate::rules::rule_spec::rule::RuleCond::Not(&$operand.leaf())]
            $($($rest)*)?
        )
    };
    (@ [$($done:expr),*] $operand:expr $(, $($rest:tt)*)?) => {
        $crate::rules::rule_spec::rule::conds!(@ [$($done,)* $operand.leaf()] $($($rest)*)?)
    };
}

pub(in crate::rules) use {all, any, conds};

pub(in crate::rules) type UserRule = Rule<UserEngine>;

#[derive(Debug, PartialEq, Eq)]
pub(in crate::rules) struct NamedCond<E: Engine = PostEngine> {
    pub(in crate::rules) name: &'static str,
    pub(in crate::rules) condition: RuleCond<E>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rules) enum Audience {
    Everyone,
    NonAuthors,
    NonFollowers,
    OnlyAuthor,
}

impl Audience {
    pub(in crate::rules) fn condition<E: Fold>(self) -> RuleCond<E> {
        match self {
            Self::Everyone => all![],
            Self::NonAuthors => E::NON_AUTHORS,
            Self::NonFollowers => E::NON_FOLLOWERS,
            Self::OnlyAuthor => ViewerIsAuthor.leaf(),
        }
    }
}

impl<E: Fold> Rule<E> {
    pub(in crate::rules) fn requirements(&self) -> Hydrators {
        let mut leaves = Vec::new();
        self.leaves(&mut leaves);
        leaves
            .into_iter()
            .fold(self.action.spec().requirements(), |hydrators, leaf| {
                hydrators.union(leaf.requirements())
            })
    }

    pub(in crate::rules) fn leaves(&self, leaves: &mut Vec<Predicate>) {
        self.audience.condition::<E>().leaves(leaves);
        self.when.leaves(leaves);
    }
}

impl<E: Engine> RuleCond<E> {
    pub(in crate::rules) const fn leaf(self) -> Self {
        self
    }

    pub(in crate::rules) fn leaves(self, leaves: &mut Vec<Predicate>) {
        match self {
            Self::Leaf(leaf) => leaves.push(leaf.lift()),
            Self::All(children) | Self::Any(children) => {
                for child in children {
                    child.leaves(leaves);
                }
            }
            Self::Not(child) => child.leaves(leaves),
            Self::Named(named) => named.condition.leaves(leaves),
        }
    }

    fn same_as<F: Engine>(self, other: RuleCond<F>) -> bool {
        let all_same = |children: &[Self], others: &[RuleCond<F>]| {
            children.len() == others.len()
                && children
                    .iter()
                    .zip(others)
                    .all(|(child, other)| child.same_as(*other))
        };
        match (self, other) {
            (Self::Leaf(leaf), RuleCond::Leaf(other)) => leaf.lift() == other.lift(),
            (Self::All(children), RuleCond::All(others))
            | (Self::Any(children), RuleCond::Any(others)) => all_same(children, others),
            (Self::Not(child), RuleCond::Not(other)) => child.same_as(*other),
            (Self::Named(definition), RuleCond::Named(other)) => {
                definition.name == other.name && definition.condition.same_as(other.condition)
            }
            (Self::Leaf(_) | Self::All(_) | Self::Any(_) | Self::Not(_) | Self::Named(_), _) => {
                false
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rules) enum DescriptionError {
    Cycle,
    ConflictingName(&'static str),
}

pub(in crate::rules) fn validate<'a, E: Fold>(
    rules: impl IntoIterator<Item = &'a Rule<E>>,
) -> Result<(), DescriptionError> {
    definitions(rules).map(drop)
}

pub(in crate::rules) fn validate_across<'a>(
    user: impl IntoIterator<Item = &'a Rule<UserEngine>>,
    post: impl IntoIterator<Item = &'a Rule>,
) -> Result<(), DescriptionError> {
    let post = definitions(post)?;
    for definition in definitions(user)? {
        if post.iter().any(|other| {
            other.name == definition.name && !definition.condition.same_as(other.condition)
        }) {
            return Err(DescriptionError::ConflictingName(definition.name));
        }
    }
    Ok(())
}

fn definitions<'a, E: Fold>(
    rules: impl IntoIterator<Item = &'a Rule<E>>,
) -> Result<Vec<&'static NamedCond<E>>, DescriptionError> {
    enum Step<E: Engine> {
        Enter(&'static RuleCond<E>),
        Leave,
    }
    let mut named: Vec<&'static NamedCond<E>> = Vec::new();
    let mut steps: Vec<Step<E>> = Vec::new();
    let mut children = |condition: RuleCond<E>, steps: &mut Vec<Step<E>>| match condition {
        RuleCond::Leaf(_) => {}
        RuleCond::All(children) | RuleCond::Any(children) => {
            steps.extend(children.iter().rev().map(Step::Enter));
        }
        RuleCond::Not(child) => steps.push(Step::Enter(child)),
        RuleCond::Named(definition) => {
            named.push(definition);
            steps.push(Step::Enter(&definition.condition));
        }
    };
    let mut active: Vec<*const RuleCond<E>> = Vec::new();
    let mut finished: Vec<*const RuleCond<E>> = Vec::new();
    for rule in rules {
        for root in [rule.audience.condition::<E>(), rule.when] {
            children(root, &mut steps);
            while let Some(step) = steps.pop() {
                let Step::Enter(condition) = step else {
                    finished.extend(active.pop());
                    continue;
                };
                let at: *const RuleCond<E> = condition;
                if active.contains(&at) {
                    return Err(DescriptionError::Cycle);
                }
                if !finished.contains(&at) {
                    active.push(at);
                    steps.push(Step::Leave);
                    children(*condition, &mut steps);
                }
            }
        }
    }
    for (index, definition) in named.iter().enumerate() {
        if named.iter().take(index).any(|other| {
            other.name == definition.name
                && !ptr::eq(*other, *definition)
                && other.condition != definition.condition
        }) {
            return Err(DescriptionError::ConflictingName(definition.name));
        }
    }
    Ok(named)
}

#[cfg(test)]
impl<E: Engine> RuleCond<E> {
    pub(in crate::rules) fn conjuncts(&self) -> &[Self] {
        match self {
            Self::All(children) => children,
            Self::Leaf(_) | Self::Any(_) | Self::Not(_) | Self::Named(_) => slice::from_ref(self),
        }
    }

    pub(in crate::rules) fn named(self, named: &mut Vec<&'static NamedCond<E>>) {
        match self {
            Self::Leaf(_) => {}
            Self::All(children) | Self::Any(children) => {
                for child in children {
                    child.named(named);
                }
            }
            Self::Not(child) => child.named(named),
            Self::Named(definition) => {
                if !named.contains(&definition) {
                    named.push(definition);
                }
                definition.condition.named(named);
            }
        }
    }

    pub(in crate::rules) fn positive_tweet_labels(self) -> Vec<SafetyLabelType> {
        match self {
            Self::Leaf(leaf) => label_of(leaf.lift()).into_iter().collect(),
            Self::Not(_) => Vec::new(),
            Self::All(children) | Self::Any(children) => children
                .iter()
                .flat_map(|child| child.positive_tweet_labels())
                .collect(),
            Self::Named(named) => named.condition.positive_tweet_labels(),
        }
    }

    pub(in crate::rules) fn own_tweet_label(self) -> Option<String> {
        let label = |leaf: Predicate<E>| label_of(leaf.lift()).map(spelled_label);
        match self {
            Self::Leaf(leaf) => label(leaf),
            Self::Not(Self::Leaf(leaf)) => label(*leaf).map(|label| format!("not {label}")),
            Self::All(_) | Self::Any(_) | Self::Not(_) | Self::Named(_) => None,
        }
    }

    pub(in crate::rules) fn spelled(self) -> String {
        self.spelled_with(|named| format!("{}={}", named.name, named.condition.spelled()))
    }

    pub(in crate::rules) fn spelled_with(self, named: fn(&NamedCond<E>) -> String) -> String {
        let joined = |children: &[Self], separator: &str| {
            children
                .iter()
                .map(|child| child.spelled_with(named))
                .collect::<Vec<_>>()
                .join(separator)
        };
        match self {
            Self::Leaf(leaf) => leaf.lift().spelled(),
            Self::All(children) => format!("all[{}]", joined(children, " & ")),
            Self::Any(children) => format!("any[{}]", joined(children, " | ")),
            Self::Not(child) => format!("not({})", child.spelled_with(named)),
            Self::Named(definition) => named(definition),
        }
    }
}

#[cfg(test)]
fn label_of(leaf: Predicate) -> Option<SafetyLabelType> {
    let Predicate::TweetSafetyLabels(TweetHasLabel(label)) = leaf else {
        return None;
    };
    Some(label)
}

#[cfg(test)]
impl<E: Engine> Rule<E> {
    pub(in crate::rules) fn name(&self) -> String {
        rule_name(self.id, &self.action)
    }

    pub(in crate::rules) fn authored(&self) -> String {
        format!(
            "{} | audience={:?} | when={} | {:?}",
            self.name(),
            self.audience,
            self.when.spelled(),
            self.action
        )
    }
}
