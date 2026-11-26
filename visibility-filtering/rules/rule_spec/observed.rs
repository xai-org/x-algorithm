use super::rule::{Rule, RuleCond};
use super::{truth_of, Contribution, Predicate, Truth};
use crate::hydration::Hydrators;
use crate::models::Evaluation;
use crate::rules::action_kind::ActionKind;
use crate::rules::defaults::{Bucket, Defaults, Effect};
use crate::rules::rule_spec::engine::{Engine, Fold};
use crate::rules::RuleContext;

type Completed = (Option<Contribution>, Hydrators);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Skip<H> {
    FalseConjunct(H),
    TrueDisjunct(H),
    Terminal,
}

pub(in crate::rules) trait Observer<E: Engine> {
    const TOTAL: bool;
    type Handle: Copy;

    fn enter(
        &mut self,
        node: Option<RuleCond<E>>,
        parent: Option<Self::Handle>,
        skip: Option<Skip<Self::Handle>>,
    ) -> Self::Handle;

    fn finish(&mut self, handle: Self::Handle, truth: Truth);

    fn row(
        &mut self,
        by: &'static str,
        root: Self::Handle,
        truth: Truth,
        effect: Effect,
        kind: ActionKind,
        defaults_admitted: Hydrators,
        completed: Option<Completed>,
    );

    fn retained(&mut self, defaults: Defaults);
}

pub(crate) struct NoTrace;

impl<E: Engine> Observer<E> for NoTrace {
    const TOTAL: bool = false;
    type Handle = ();

    #[inline]
    fn enter(&mut self, _: Option<RuleCond<E>>, _: Option<()>, _: Option<Skip<()>>) {}

    #[inline]
    fn finish(&mut self, (): (), _: Truth) {}

    #[inline]
    fn row(
        &mut self,
        _: &'static str,
        (): (),
        _: Truth,
        _: Effect,
        _: ActionKind,
        _: Hydrators,
        _: Option<Completed>,
    ) {
    }

    #[inline]
    fn retained(&mut self, _: Defaults) {}
}

#[inline]
pub(in crate::rules) fn evaluate<E: Fold, O: Observer<E>>(
    condition: RuleCond<E>,
    context: &RuleContext<'_, E::Candidate>,
    observer: &mut O,
    parent: Option<O::Handle>,
    skip: Option<Skip<O::Handle>>,
) -> (Truth, O::Handle) {
    let handle = observer.enter(Some(condition), parent, skip);
    let truth = match condition {
        RuleCond::Leaf(leaf) => truth_of(leaf, context),
        RuleCond::All(children) => group(
            children,
            context,
            observer,
            (handle, skip),
            (Truth::True, Truth::False),
            Truth::and,
            Skip::FalseConjunct,
        ),
        RuleCond::Any(children) => group(
            children,
            context,
            observer,
            (handle, skip),
            (Truth::False, Truth::True),
            Truth::or,
            Skip::TrueDisjunct,
        ),
        RuleCond::Not(child) => !evaluate(*child, context, observer, Some(handle), skip).0,
        RuleCond::Named(named) => {
            evaluate(named.condition, context, observer, Some(handle), skip).0
        }
    };
    observer.finish(handle, truth);
    (truth, handle)
}

#[inline]
fn group<E: Fold, O: Observer<E>>(
    children: &[RuleCond<E>],
    context: &RuleContext<'_, E::Candidate>,
    observer: &mut O,
    (handle, skip): (O::Handle, Option<Skip<O::Handle>>),
    (identity, absorbing): (Truth, Truth),
    combine: fn(Truth, Truth) -> Truth,
    after: fn(O::Handle) -> Skip<O::Handle>,
) -> Truth {
    let mut truth = identity;
    let mut decided = None;
    for &child in children {
        let answer = match child {
            RuleCond::Leaf(leaf) if !O::TOTAL => truth_of(leaf, context),
            RuleCond::Not(RuleCond::Leaf(leaf)) if !O::TOTAL => !truth_of(*leaf, context),
            RuleCond::Leaf(_)
            | RuleCond::All(_)
            | RuleCond::Any(_)
            | RuleCond::Not(_)
            | RuleCond::Named(_) => {
                let (answer, child_handle) = evaluate(
                    child,
                    context,
                    observer,
                    Some(handle),
                    skip.or(decided.map(after)),
                );
                if decided.is_none() && answer == absorbing {
                    decided = Some(child_handle);
                }
                answer
            }
        };
        if answer == absorbing && !O::TOTAL {
            return answer;
        }
        truth = combine(truth, answer);
    }
    truth
}

#[inline]
pub(in crate::rules) fn evaluate_rule<E: Fold, O: Observer<E>>(
    rule: &Rule<E>,
    context: &RuleContext<'_, E::Candidate>,
    observer: &mut O,
    after_terminal: bool,
) -> (Truth, O::Handle) {
    let skip = after_terminal.then_some(Skip::Terminal);
    let root = observer.enter(None, None, skip);
    let truth = if O::TOTAL {
        let (audience, audience_handle) = evaluate(
            rule.audience.condition::<E>(),
            context,
            observer,
            Some(root),
            skip,
        );
        let skip =
            skip.or((audience == Truth::False).then_some(Skip::FalseConjunct(audience_handle)));
        audience.and(evaluate(rule.when, context, observer, Some(root), skip).0)
    } else {
        let audience = match rule.audience.condition::<E>() {
            RuleCond::All([]) => Truth::True,
            RuleCond::Leaf(leaf) => truth_of(leaf, context),
            RuleCond::Not(RuleCond::Leaf(leaf)) => !truth_of(*leaf, context),
            condition @ (RuleCond::All(_)
            | RuleCond::Any(_)
            | RuleCond::Not(_)
            | RuleCond::Named(_)) => evaluate(condition, context, observer, Some(root), skip).0,
        };
        if audience == Truth::False {
            audience
        } else {
            audience.and(evaluate(rule.when, context, observer, Some(root), skip).0)
        }
    };
    observer.finish(root, truth);
    (truth, root)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Rule,
    Leaf(Predicate),
    All,
    Any,
    Not,
    Named(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness {
    pub node: usize,
    pub truth: Truth,
    pub negated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeTrace {
    pub node: usize,
    pub parent: Option<usize>,
    pub kind: NodeKind,
    pub truth: Truth,
    pub skip: Option<Skip<usize>>,
    pub witnesses: Vec<Witness>,
    pub default_witnesses: Vec<Witness>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyTrace {
    pub evaluation: Evaluation,
    pub nodes: Vec<NodeTrace>,
    pub rows: Vec<RowTrace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RowTrace {
    pub by: &'static str,
    pub root: usize,
    pub truth: Truth,
    pub effect: Effect,
    pub defaults_admitted: Hydrators,
    pub defaults_retained: Hydrators,
    pub completed: Option<Completed>,
}

#[derive(Default)]
pub(in crate::rules) struct Total {
    nodes: Vec<NodeTrace>,
    rows: Vec<(Bucket, RowTrace)>,
}

impl Total {
    pub(in crate::rules) fn into_trace(self, evaluation: Evaluation) -> PolicyTrace {
        PolicyTrace {
            evaluation,
            nodes: self.nodes,
            rows: self.rows.into_iter().map(|(_, row)| row).collect(),
        }
    }
}

impl<E: Engine> Observer<E> for Total {
    const TOTAL: bool = true;
    type Handle = usize;

    fn enter(
        &mut self,
        node: Option<RuleCond<E>>,
        parent: Option<usize>,
        skip: Option<Skip<usize>>,
    ) -> usize {
        let kind = match node {
            None => NodeKind::Rule,
            Some(RuleCond::Leaf(leaf)) => NodeKind::Leaf(leaf.lift()),
            Some(RuleCond::All(_)) => NodeKind::All,
            Some(RuleCond::Any(_)) => NodeKind::Any,
            Some(RuleCond::Not(_)) => NodeKind::Not,
            Some(RuleCond::Named(named)) => NodeKind::Named(named.name),
        };
        let node = self.nodes.len();
        self.nodes.push(NodeTrace {
            node,
            parent,
            kind,
            truth: Truth::False,
            skip,
            witnesses: Vec::new(),
            default_witnesses: Vec::new(),
        });
        node
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "`enter` returned `handle` as the index of the node it pushed"
    )]
    fn finish(&mut self, handle: usize, truth: Truth) {
        let kind = self.nodes[handle].kind;
        let children: Vec<(usize, Truth)> = self
            .nodes
            .iter()
            .filter(|child| child.parent == Some(handle))
            .map(|child| (child.node, child.truth))
            .collect();
        let negated = kind == NodeKind::Not;
        let conjunction = matches!(kind, NodeKind::All | NodeKind::Rule);
        let decides = |child: Truth| match (kind, truth) {
            (NodeKind::Leaf(_), _) => false,
            (NodeKind::Not | NodeKind::Named(_), _) => true,
            (_, Truth::Unknown { .. }) => matches!(child, Truth::Unknown { .. }),
            (_, Truth::False) if conjunction => child == Truth::False,
            (_, Truth::True) if !conjunction => child == Truth::True,
            (_, Truth::True | Truth::False) => true,
        };
        let witnesses = children
            .iter()
            .filter(|&&(_, child)| decides(child))
            .map(|&(node, child)| Witness {
                node,
                truth: child,
                negated,
            })
            .collect();
        let default_witnesses = match truth {
            Truth::Unknown { default, .. } => children
                .iter()
                .filter(|&&(_, child)| match kind {
                    NodeKind::Leaf(_) => false,
                    NodeKind::Not | NodeKind::Named(_) => true,
                    NodeKind::All | NodeKind::Rule | NodeKind::Any => {
                        child.resolves_true() == default
                    }
                })
                .map(|&(node, child)| Witness {
                    node,
                    truth: if child.resolves_true() {
                        Truth::True
                    } else {
                        Truth::False
                    },
                    negated,
                })
                .collect(),
            Truth::True | Truth::False => Vec::new(),
        };
        let node = &mut self.nodes[handle];
        node.truth = truth;
        node.witnesses = witnesses;
        node.default_witnesses = default_witnesses;
    }

    fn row(
        &mut self,
        by: &'static str,
        root: usize,
        truth: Truth,
        effect: Effect,
        kind: ActionKind,
        defaults_admitted: Hydrators,
        completed: Option<Completed>,
    ) {
        self.rows.push((
            kind.bucket(),
            RowTrace {
                by,
                root,
                truth,
                effect,
                defaults_admitted,
                defaults_retained: Hydrators::empty(),
                completed,
            },
        ));
    }

    fn retained(&mut self, defaults: Defaults) {
        for (bucket, row) in &mut self.rows {
            row.defaults_retained = row.defaults_admitted.intersection(defaults.get(*bucket));
        }
    }
}
