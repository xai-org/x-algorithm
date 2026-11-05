use crate::hydration::inventory::Stake;
use crate::hydration::{Relation, BEST_EFFORT_HYDRATION_TIMEOUT};
use crate::rules::levels;
use crate::rules::rule_spec::rule::Rule;
use std::time::Duration;
use strum::VariantArray;

#[derive(Clone, Copy, Debug, strum::VariantArray)]
pub(crate) enum Expansion {
    ReactedTo,
}

pub(crate) struct ExpansionSpec {
    pub(crate) relation: Relation,
    pub(super) rules: &'static [&'static Rule],
    pub(crate) deadline: Duration,
    pub(crate) on_failure: Stake,
}

impl Expansion {
    pub(crate) const fn spec(self) -> ExpansionSpec {
        match self {
            Self::ReactedTo => ExpansionSpec {
                relation: Relation::ReactedTo,
                rules: levels::REACTED_TO_RULES,
                deadline: BEST_EFFORT_HYDRATION_TIMEOUT,
                on_failure: Stake::BestEffort,
            },
        }
    }

    pub(crate) const fn of(relation: Relation) -> Option<Self> {
        let mut rest = Self::VARIANTS;
        while let [expansion, tail @ ..] = rest {
            if expansion.spec().relation as u8 == relation as u8 {
                return Some(*expansion);
            }
            rest = tail;
        }
        None
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "the assert bounds every index: `rules` is non-empty and `N` is both lengths summed"
    )]
    pub(super) const fn placed<const N: usize>(
        self,
        parent: &'static [&'static [&'static Rule]],
    ) -> [&'static Rule; N] {
        let rules = self.spec().rules;
        assert!(!rules.is_empty() && levels::rule_count(parent) + rules.len() == N);
        let mut placed = [rules[0]; N];
        let mut at = 0;
        let mut i = 0;
        while i < rules.len() {
            if !withholds(rules[i]) {
                placed[at] = rules[i];
                at += 1;
            }
            i += 1;
        }
        let mut rest = parent;
        while let [family, tail @ ..] = rest {
            i = 0;
            while i < family.len() {
                placed[at] = family[i];
                at += 1;
                i += 1;
            }
            rest = tail;
        }
        i = 0;
        while i < rules.len() {
            if withholds(rules[i]) {
                placed[at] = rules[i];
                at += 1;
            }
            i += 1;
        }
        placed
    }
}

const fn withholds(rule: &Rule) -> bool {
    rule.action.spec().kind().withholds()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::Hydrator;
    use crate::rules::SafetyLevel;

    #[test]
    fn a_level_lists_all_of_an_expansions_rules_exactly_when_it_reads_its_relation() {
        let reads = |rule: &Rule, relation| {
            rule.requirements()
                .iter()
                .any(|node| node.relation() == Some(relation))
        };
        for &expansion in Expansion::VARIANTS {
            let spec = expansion.spec();
            for rule in spec.rules {
                assert!(
                    reads(rule, spec.relation),
                    "{} reads no {expansion:?} node",
                    rule.name()
                );
            }
            for &level in SafetyLevel::VARIANTS {
                let post = level.spec().rules().lists().post;
                let listed = spec.rules.iter().filter(|rule| post.contains(rule)).count();
                assert!(
                    listed == 0 || listed == spec.rules.len(),
                    "{level:?} lists part of {expansion:?}"
                );
                for rule in post.iter().filter(|rule| reads(rule, spec.relation)) {
                    assert!(
                        spec.rules.contains(rule),
                        "{level:?} lists {} outside {expansion:?}",
                        rule.name()
                    );
                }
            }
        }
    }

    #[test]
    fn a_node_reading_an_expansions_relation_fails_as_the_expansion_says() {
        for &node in Hydrator::VARIANTS {
            let Some(expansion) = node.relation().and_then(Expansion::of) else {
                continue;
            };
            assert_eq!(
                node.declared_stake(),
                expansion.spec().on_failure,
                "{node:?}"
            );
        }
    }
}
