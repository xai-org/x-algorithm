use crate::hydration::Hydrators;
use crate::models::PostCandidate;
use crate::rules::rule_spec::engine::Candidate;
use crate::rules::RuleContext;

pub(in crate::rules) trait Inputs<'a, C = PostCandidate>: Sized {
    const REQUIRED: Hydrators;
    fn acquire(context: &RuleContext<'a, C>) -> Self;
}

pub(in crate::rules) trait Interpreter<'a, C = PostCandidate> {
    type Output<T>;
    fn run<A: Inputs<'a, C>, T>(&self, kernel: impl FnOnce(A) -> T) -> Self::Output<T>;
}

pub(in crate::rules) struct Required;

impl<'a, C> Interpreter<'a, C> for Required {
    type Output<T> = Hydrators;

    fn run<A: Inputs<'a, C>, T>(&self, _: impl FnOnce(A) -> T) -> Hydrators {
        A::REQUIRED
    }
}

pub(in crate::rules) struct EvaluateInputs<'c, 'a, C = PostCandidate>(
    pub(in crate::rules) &'c RuleContext<'a, C>,
);

pub(in crate::rules) struct Acquired<T> {
    pub(in crate::rules) value: T,
    pub(in crate::rules) failed: Hydrators,
}

impl<'a, C: Candidate> Interpreter<'a, C> for EvaluateInputs<'_, 'a, C> {
    type Output<T> = Acquired<T>;

    #[inline]
    fn run<A: Inputs<'a, C>, T>(&self, kernel: impl FnOnce(A) -> T) -> Acquired<T> {
        let inputs = A::acquire(self.0);
        Acquired {
            value: kernel(inputs),
            failed: self.0.failed().intersection(A::REQUIRED),
        }
    }
}

impl<'a, C> Inputs<'a, C> for () {
    const REQUIRED: Hydrators = Hydrators::empty();
    fn acquire(_: &RuleContext<'a, C>) -> Self {}
}

impl<'a, A: Inputs<'a>, B: Inputs<'a>> Inputs<'a> for (A, B) {
    const REQUIRED: Hydrators = A::REQUIRED.union(B::REQUIRED);
    #[inline]
    fn acquire(context: &RuleContext<'a>) -> Self {
        (A::acquire(context), B::acquire(context))
    }
}

impl<'a, A: Inputs<'a>, B: Inputs<'a>, C: Inputs<'a>> Inputs<'a> for (A, B, C) {
    const REQUIRED: Hydrators = A::REQUIRED.union(B::REQUIRED).union(C::REQUIRED);
    #[inline]
    fn acquire(context: &RuleContext<'a>) -> Self {
        (
            A::acquire(context),
            B::acquire(context),
            C::acquire(context),
        )
    }
}
