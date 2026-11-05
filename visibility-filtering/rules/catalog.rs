use crate::rules::engine::WiredRule;
use crate::rules::rule_spec::engine::Engine;
use crate::rules::rule_spec::rule::{NamedCond, Rule, RuleCond};
use crate::rules::rule_spec::rule_name;
use crate::rules::{RuleEngine, SafetyLevel};
use std::cmp::Reverse;
use std::fmt::Write;
use strum::VariantArray;

fn spelled<E: Engine>(condition: RuleCond<E>) -> String {
    condition.spelled_with(|named| named.name.to_owned())
}

#[derive(PartialEq)]
struct CatalogRule {
    severity: u8,
    kind_and_slot: (&'static str, &'static str),
    line: String,
    named: Vec<(&'static str, String)>,
}

fn catalog_rules<E: Engine>(rules: &[&'static Rule<E>]) -> impl Iterator<Item = CatalogRule> {
    rules.iter().map(|rule| {
        let action = rule.action.spec();
        let mut named: Vec<&'static NamedCond<E>> = Vec::new();
        rule.when.named(&mut named);
        CatalogRule {
            severity: action.kind().severity(),
            kind_and_slot: action.kind().catalog(),
            line: format!(
                "`{}`: audience `{:?}`, when `{}`, `{}`",
                rule_name(rule.id, &rule.action),
                rule.audience,
                spelled(rule.when),
                action.spelled()
            ),
            named: named
                .into_iter()
                .map(|definition| (definition.name, spelled(definition.condition)))
                .collect(),
        }
    })
}

pub(super) fn catalog() -> String {
    let mut rules: Vec<(CatalogRule, Vec<&'static str>)> = Vec::new();
    for &level in SafetyLevel::VARIANTS {
        let level_name: &'static str = level.into();
        for listed in level.rule_lists() {
            for entry in catalog_rules(&listed.user).chain(catalog_rules(&listed.post)) {
                match rules.iter_mut().find(|(other, _)| *other == entry) {
                    Some((_, levels)) if levels.last() == Some(&level_name) => {}
                    Some((_, levels)) => levels.push(level_name),
                    None => rules.push((entry, vec![level_name])),
                }
            }
        }
    }
    rules.sort_by_key(|(entry, _)| Reverse(entry.severity));
    let mut named: Vec<&(&'static str, String)> = Vec::new();
    for (entry, _) in &rules {
        for definition in &entry.named {
            if !named.contains(&definition) {
                named.push(definition);
            }
        }
    }

    let mut doc = String::from(
        "# Policy catalog\n\n\
         Generated from the rules in `src/rules/` by `scripts/bless`, which\n\
         `rules::catalog::tests::the_policy_catalog_is_printed_from_the_rules` checks. Edit the\n\
         rules, not this file.\n\n\
         ## Precedence\n\n\
         A level folds its user list, then its post list. Each list folds in this order of\n\
         action and keeps its listed order within one action, so every user rule comes before\n\
         every post rule. The first withholding decides; the slots fill as `GLOSSARY.md` (Slot)\n\
         says.\n\n\
         | Action | Slot |\n|---|---|\n",
    );
    let mut kinds: Vec<(&str, &str)> = rules.iter().map(|(entry, _)| entry.kind_and_slot).collect();
    kinds.dedup();
    for (kind, slot) in kinds {
        writeln!(doc, "| {kind} | {slot} |").unwrap();
    }
    doc.push_str("\n## Named conditions\n\n");
    for &definition in &named {
        let uses = rules
            .iter()
            .filter(|(entry, _)| entry.named.contains(definition))
            .count();
        let (name, condition) = definition;
        writeln!(doc, "- `{name}` = `{condition}`. Uses: {uses}.").unwrap();
    }
    let mut band = None;
    for (entry, levels) in &rules {
        let (kind, _) = entry.kind_and_slot;
        if band != Some(kind) {
            band = Some(kind);
            writeln!(doc, "\n## {kind} rules\n").unwrap();
        }
        writeln!(doc, "- {}. Levels: {}.", entry.line, levels.join(", ")).unwrap();
    }
    doc.push_str(
        "\n## Levels\n\n\
         Each level's rules in fold order: the user list, then the post list, each under its\n\
         actions in precedence order. A notification level's post kinds follow it, each as\n\
         `<level>_<kind>`.\n",
    );
    let rule_engine = RuleEngine::for_tests();
    for &level in SafetyLevel::VARIANTS {
        let level_name: &str = level.into();
        let post_kinds = rule_engine
            .post_kind_fold_orders(level)
            .into_iter()
            .map(|(kind, fold_order)| (format!("{level_name}_{}", <&str>::from(kind)), fold_order));
        for (heading, (user, post)) in
            std::iter::once((level_name.to_owned(), rule_engine.fold_order(level)))
                .chain(post_kinds)
        {
            writeln!(doc, "\n### {heading}\n").unwrap();
            write_wired(&mut doc, "user", &user);
            write_wired(&mut doc, "post", &post);
        }
    }
    doc
}

fn write_wired<E: Engine>(doc: &mut String, list: &str, rules: &[WiredRule<'_, E>]) {
    let mut band = None;
    for rule in rules {
        let (kind, slot) = rule.action.kind().catalog();
        if band != Some(kind) {
            band = Some(kind);
            match slot {
                "withholding" | "fosnr" => writeln!(doc, "- {list}, {kind}, {slot}:"),
                _ => writeln!(doc, "- {list}, {kind}, {slot} slot:"),
            }
            .unwrap();
        }
        if rules.iter().filter(|other| other.name == rule.name).count() > 1 {
            writeln!(doc, "  - `{}` when `{}`", rule.name, spelled(rule.when)).unwrap();
        } else {
            writeln!(doc, "  - `{}`", rule.name).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DropReason;
    use crate::rules::rule_spec::ActionSpec;
    use std::collections::BTreeSet;

    #[test]
    fn the_policy_catalog_is_printed_from_the_rules() {
        crate::pinned::assert_pinned("docs/policy-catalog.md", &catalog());
    }

    fn viewer_gated_drop_rules<E: Engine>(
        rules: &[&'static Rule<E>],
    ) -> impl Iterator<Item = String> {
        rules
            .iter()
            .filter(|rule| {
                matches!(
                    rule.action.spec(),
                    ActionSpec::Drop(DropReason::NsfwViewer(_))
                )
            })
            .map(|rule| rule_name(rule.id, &rule.action))
    }

    #[test]
    fn dashboard_lists_every_viewer_gated_drop_rule() {
        let gated: BTreeSet<String> = SafetyLevel::VARIANTS
            .iter()
            .flat_map(|level| {
                let listed = level.spec().rules().lists();
                viewer_gated_drop_rules(&listed.user)
                    .chain(viewer_gated_drop_rules(&listed.post))
                    .collect::<Vec<_>>()
            })
            .collect();
        let dashboard = crate::pinned::dashboard_tuple("VIEWER_GATED_DROP_RULES");
        assert_eq!(dashboard, gated);
    }
}
