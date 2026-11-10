use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use strum::VariantArray;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub(crate) enum Experiment {
    #[cfg(test)]
    TestLaunch,
    #[cfg(test)]
    TestHoldback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no experiment is wired yet; tests construct both shapes"
    )
)]
pub(crate) enum Shape {
    Launch,
    Holdback,
}

impl Experiment {
    pub(crate) const fn key(self) -> &'static str {
        match self {
            #[cfg(test)]
            Self::TestLaunch => "rust_vf_test_launch_rule_enabled",
            #[cfg(test)]
            Self::TestHoldback => "rust_vf_test_holdback_rule_enabled",
        }
    }

    pub(crate) const fn shape(self) -> Shape {
        match self {
            #[cfg(test)]
            Self::TestLaunch => Shape::Launch,
            #[cfg(test)]
            Self::TestHoldback => Shape::Holdback,
        }
    }

    pub(crate) fn is_on(self, value: Option<bool>) -> bool {
        match self.shape() {
            Shape::Launch => value == Some(true),
            Shape::Holdback => value == Some(false),
        }
    }

    const fn applies(self, on: bool) -> bool {
        match self.shape() {
            Shape::Launch => on,
            Shape::Holdback => !on,
        }
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExperimentSet(u8);

const _: () = assert!(Experiment::VARIANTS.len() <= u8::BITS as usize);

impl ExperimentSet {
    pub(crate) const NONE: Self = Self(0);

    #[must_use]
    pub(crate) const fn with(self, experiment: Experiment) -> Self {
        Self(self.0 | experiment.bit())
    }

    #[must_use]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(crate) const fn contains(self, experiment: Experiment) -> bool {
        self.0 & experiment.bit() != 0
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = Experiment> {
        Experiment::VARIANTS
            .iter()
            .copied()
            .filter(move |&experiment| self.contains(experiment))
    }
}

impl FromIterator<Experiment> for ExperimentSet {
    fn from_iter<I: IntoIterator<Item = Experiment>>(experiments: I) -> Self {
        experiments.into_iter().fold(Self::NONE, Self::with)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RequestExperiments {
    on: ExperimentSet,
    exposed: Option<Arc<AtomicU8>>,
}

impl RequestExperiments {
    pub(crate) fn resolved(on: ExperimentSet) -> Self {
        Self {
            on,
            exposed: Some(Arc::default()),
        }
    }

    pub(crate) fn applies(&self, experiment: Experiment) -> bool {
        experiment.applies(self.on.contains(experiment))
    }

    pub(crate) fn expose(&self, experiment: Experiment) {
        if let Some(exposed) = &self.exposed {
            exposed.fetch_or(experiment.bit(), Ordering::Relaxed);
        }
    }

    pub(crate) fn exposed(&self) -> ExperimentSet {
        ExperimentSet(
            self.exposed
                .as_ref()
                .map_or(0, |exposed| exposed.load(Ordering::Relaxed)),
        )
    }
}
