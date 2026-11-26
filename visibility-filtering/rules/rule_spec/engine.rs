use crate::hydration::Hydrators;
use crate::models::{EdgeSet, PostCandidate, Quote, UserCandidate, UserEntity};
use crate::rules::context::RuleContext;
use crate::rules::rule_spec::features::leaves::{ViewerFollowsAuthor, ViewerIsAuthor};
use crate::rules::rule_spec::inputs::{Inputs, Interpreter};
use crate::rules::rule_spec::rule::{all, RuleCond};
use std::convert::Infallible;
use std::fmt::Debug;
use std::marker::PhantomData;

pub trait Engine: Copy + Debug + Eq + 'static {
    type PostOnly<L: Copy + Debug + Eq + 'static>: Copy + Debug + Eq + 'static;
    type Candidate: Candidate;

    fn open<L: Copy + Debug + Eq + 'static>(only: Self::PostOnly<L>) -> L;
}

pub(in crate::rules) trait Fold: Engine {
    const NON_AUTHORS: RuleCond<Self>;
    const NON_FOLLOWERS: RuleCond<Self>;

    fn post<'c, 'a, L: Copy + Debug + Eq + 'static>(
        context: &'c RuleContext<'a, Self::Candidate>,
        only: Self::PostOnly<L>,
    ) -> (L, &'c RuleContext<'a>);

    fn run_post<
        'a,
        I: Interpreter<'a, Self::Candidate>,
        L: Copy + Debug + Eq + 'static,
        A: Inputs<'a>,
        T,
    >(
        interpreter: &I,
        only: Self::PostOnly<L>,
        kernel: impl FnOnce(L, A) -> T,
    ) -> I::Output<T>;
}

pub trait Candidate {
    fn author(&self) -> &UserEntity;
    fn edges(&self) -> EdgeSet;
    fn quote(&self) -> Option<Quote>;
    fn failed(&self) -> Hydrators;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostEngine {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserEngine {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Absent<L>(Infallible, PhantomData<L>);

impl Engine for PostEngine {
    type PostOnly<L: Copy + Debug + Eq + 'static> = L;
    type Candidate = PostCandidate;

    #[inline]
    fn open<L: Copy + Debug + Eq + 'static>(only: L) -> L {
        only
    }
}

impl Fold for PostEngine {
    const NON_AUTHORS: RuleCond<Self> = RuleCond::Not(&ViewerIsAuthor.leaf());
    const NON_FOLLOWERS: RuleCond<Self> = all![!ViewerIsAuthor, !ViewerFollowsAuthor];

    #[inline]
    fn post<'c, 'a, L: Copy + Debug + Eq + 'static>(
        context: &'c RuleContext<'a>,
        only: L,
    ) -> (L, &'c RuleContext<'a>) {
        (only, context)
    }

    #[inline]
    fn run_post<'a, I: Interpreter<'a>, L: Copy + Debug + Eq + 'static, A: Inputs<'a>, T>(
        interpreter: &I,
        only: L,
        kernel: impl FnOnce(L, A) -> T,
    ) -> I::Output<T> {
        interpreter.run(|inputs| kernel(only, inputs))
    }
}

impl Engine for UserEngine {
    type PostOnly<L: Copy + Debug + Eq + 'static> = Absent<L>;
    type Candidate = UserCandidate;

    fn open<L: Copy + Debug + Eq + 'static>(only: Absent<L>) -> L {
        match only.0 {}
    }
}

impl Fold for UserEngine {
    const NON_AUTHORS: RuleCond<Self> = RuleCond::Not(&ViewerIsAuthor.leaf());
    const NON_FOLLOWERS: RuleCond<Self> = all![!ViewerIsAuthor, !ViewerFollowsAuthor];

    fn post<'c, 'a, L: Copy + Debug + Eq + 'static>(
        _: &'c RuleContext<'a, UserCandidate>,
        only: Absent<L>,
    ) -> (L, &'c RuleContext<'a>) {
        match only.0 {}
    }

    fn run_post<
        'a,
        I: Interpreter<'a, UserCandidate>,
        L: Copy + Debug + Eq + 'static,
        A: Inputs<'a>,
        T,
    >(
        _: &I,
        only: Absent<L>,
        _: impl FnOnce(L, A) -> T,
    ) -> I::Output<T> {
        match only.0 {}
    }
}

impl Candidate for PostCandidate {
    #[inline]
    fn author(&self) -> &UserEntity {
        &self.author
    }

    #[inline]
    fn edges(&self) -> EdgeSet {
        self.edges
    }

    #[inline]
    fn quote(&self) -> Option<Quote> {
        None
    }

    #[inline]
    fn failed(&self) -> Hydrators {
        self.failed
    }
}

impl Candidate for UserCandidate {
    #[inline]
    fn author(&self) -> &UserEntity {
        &self.user
    }

    #[inline]
    fn edges(&self) -> EdgeSet {
        self.edges
    }

    #[inline]
    fn quote(&self) -> Option<Quote> {
        self.quote
    }

    #[inline]
    fn failed(&self) -> Hydrators {
        self.failed
    }
}
