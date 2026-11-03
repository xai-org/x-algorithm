use crate::clients::socialgraph_client::{EdgeDirection, Graph};
use crate::hydration::fetcher::{AskedFields, FetcherId};
use crate::hydration::hydrator::{Edge, Part, RequestRows};
use crate::hydration::relation::{Relation, Role};
use crate::hydration::{Hydrator, Hydrators, BEST_EFFORT_HYDRATION_TIMEOUT, HYDRATION_TIMEOUT};
use crate::rules::{Expansion, Rpc, SafetyLevel};
use std::fmt;
use std::iter;
use std::time::Duration;
use strum::VariantArray;
use xai_core_entities::gizmoduck_client::QueryFields;

pub(crate) struct HydrationPlan {
    level: SafetyLevel,
    rows: RequestRows,
    groups: Vec<Group>,
    edge_ceilings: Vec<(FetcherId, Edge, Duration)>,
    nodes: Hydrators,
    logged_out_nodes: Hydrators,
}

pub(super) struct Group {
    pub(super) position: usize,
    pub(super) fetcher: FetcherId,
    pub(super) input: Option<Hydrator>,
    pub(super) nodes: Hydrators,
    label: (&'static str, &'static str),
    edges: Vec<(Edge, Graph, EdgeDirection, Hydrators)>,
    fields: Vec<QueryFields>,
    readers: Vec<usize>,
}

impl HydrationPlan {
    pub(crate) fn new(level: SafetyLevel, hydrators: Hydrators) -> Self {
        let rows = level.spec().rows();
        let asks_tweet_state = level.is_carried_by(Rpc::EvaluateTweets)
            && hydrators.closed(rows).contains(Hydrator::Tweet);
        let hydrators = if asks_tweet_state {
            hydrators.with(Hydrator::TweetState)
        } else {
            hydrators
        };
        Self::with_rows(level, rows, hydrators)
    }

    pub(crate) fn with_rows(level: SafetyLevel, rows: RequestRows, hydrators: Hydrators) -> Self {
        let nodes = hydrators.closed(rows);
        let mut groups: Vec<Group> = Vec::new();
        for node in nodes.iter() {
            let spec = node.spec_at(rows);
            let input = node.input_at(rows);
            match groups
                .iter_mut()
                .find(|group| group.fetcher == spec.fetcher() && group.input == input)
            {
                Some(group) => {
                    group.nodes = group.nodes.with(node);
                }
                None => groups.push(Group {
                    position: groups.len(),
                    fetcher: spec.fetcher(),
                    input,
                    nodes: Hydrators::of(node),
                    label: spec.label,
                    edges: Vec::new(),
                    fields: Vec::new(),
                    readers: Vec::new(),
                }),
            }
        }
        let reads: Vec<Hydrators> = groups
            .iter()
            .map(|group| {
                group.nodes.iter().fold(Hydrators::empty(), |reads, node| {
                    reads.union(node.reads_at(rows))
                })
            })
            .collect();
        let readers: Vec<Vec<usize>> = groups
            .iter()
            .map(|landed| {
                reads
                    .iter()
                    .enumerate()
                    .filter(|(_, reads)| !reads.intersection(landed.nodes).is_empty())
                    .map(|(position, _)| position)
                    .collect()
            })
            .collect();
        let mut edge_ceilings: Vec<(FetcherId, Edge, Duration)> = Vec::new();
        for group in &groups {
            for edge in group.nodes.iter().filter_map(Hydrator::edge) {
                let ceiling = group.ceiling();
                match edge_ceilings
                    .iter_mut()
                    .find(|(fetcher, read, _)| *fetcher == group.fetcher && *read == edge)
                {
                    Some((.., longest)) => *longest = (*longest).max(ceiling),
                    None => edge_ceilings.push((group.fetcher, edge, ceiling)),
                }
            }
        }
        for (group, readers) in groups.iter_mut().zip(readers) {
            group.edges = edges(group.nodes);
            group.fields = fields(group.fetcher, group.nodes, rows);
            group.readers = readers;
        }
        Self {
            level,
            rows,
            groups,
            edge_ceilings,
            nodes,
            logged_out_nodes: nodes
                .iter()
                .filter(|node| !node.needs_viewer())
                .fold(Hydrators::empty(), Hydrators::with),
        }
    }

    pub(crate) fn level(&self) -> SafetyLevel {
        self.level
    }

    pub(crate) fn rows(&self) -> RequestRows {
        self.rows
    }

    pub(crate) fn keys_only_user_rows_or_viewer(&self) -> bool {
        self.nodes
            .iter()
            .all(|node| node.key_at(self.rows).keys_user_rows())
    }

    pub(crate) fn keys_a_node_only_user_rows_give(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| node.key_at(self.rows).keys_only_user_rows())
    }

    pub(super) fn reads_viewer_profile(&self) -> bool {
        self.nodes.contains(Hydrator::ViewerProfile)
    }

    pub(super) fn reads_viewer_labels(&self) -> bool {
        self.nodes.contains(Hydrator::ViewerLabels)
    }

    pub(super) fn callable(&self, viewer_id: Option<u64>) -> Hydrators {
        match viewer_id {
            Some(_) => self.nodes,
            None => self.logged_out_nodes,
        }
    }

    pub(super) fn groups(&self) -> impl Iterator<Item = &Group> {
        self.groups.iter()
    }

    fn relations(&self) -> impl Iterator<Item = Relation> + '_ {
        Relation::VARIANTS
            .iter()
            .copied()
            .filter(|&relation| match relation.spec().role {
                Role::Judged => {
                    self.level.may_merge_retweets()
                        && !self
                            .nodes
                            .intersection(relation.spec().trigger.reads())
                            .is_empty()
                }
                Role::Read => self
                    .nodes
                    .iter()
                    .any(|node| node.spec().relation == Some(relation)),
            })
    }

    pub(super) fn readers<'a>(&'a self, landed: &'a Group) -> impl Iterator<Item = &'a Group> {
        landed
            .readers
            .iter()
            .filter_map(|&position| self.groups.get(position))
    }

    pub(super) fn call_deadline(
        &self,
        group: &Group,
        claimed_edges: impl Iterator<Item = Edge>,
    ) -> Duration {
        claimed_edges
            .filter_map(|edge| {
                self.edge_ceilings
                    .iter()
                    .find(|&&(fetcher, read, _)| fetcher == group.fetcher && read == edge)
                    .map(|&(.., ceiling)| ceiling)
            })
            .fold(group.ceiling(), Duration::max)
    }
}

impl Group {
    pub(super) fn ceiling(&self) -> Duration {
        if self.nodes.has_must_have() {
            return HYDRATION_TIMEOUT;
        }
        self.nodes
            .iter()
            .find_map(|node| Expansion::of(node.relation()?))
            .map_or(BEST_EFFORT_HYDRATION_TIMEOUT, |expansion| {
                expansion.spec().deadline
            })
    }

    pub(super) fn label(&self) -> (&'static str, &'static str) {
        self.label
    }

    pub(super) fn fields(&self) -> &[QueryFields] {
        &self.fields
    }

    pub(super) fn edges(&self) -> &[(Edge, Graph, EdgeDirection, Hydrators)] {
        &self.edges
    }
}

fn fields(fetcher: FetcherId, nodes: Hydrators, rows: RequestRows) -> Vec<QueryFields> {
    let nodes = match fetcher.asked_fields() {
        AskedFields::EveryLevel => Hydrator::VARIANTS
            .iter()
            .copied()
            .filter(|node| node.spec_at(rows).fetcher() == fetcher)
            .fold(Hydrators::empty(), Hydrators::with),
        AskedFields::Planned => nodes,
    };
    let mut fields = Vec::new();
    for node in nodes.iter() {
        if let Part::Fields(node_fields) = node.spec().part {
            for field in node_fields {
                if !fields.contains(field) {
                    fields.push(*field);
                }
            }
        }
    }
    fields
}

fn edges(nodes: Hydrators) -> Vec<(Edge, Graph, EdgeDirection, Hydrators)> {
    let mut edges: Vec<(Edge, Graph, EdgeDirection, Hydrators)> = Vec::new();
    for node in nodes.iter() {
        let Some(edge) = node.edge() else {
            continue;
        };
        let Some((graph, direction)) = edge.flock() else {
            continue;
        };
        match edges.iter_mut().find(|(queried, ..)| *queried == edge) {
            Some((.., nodes)) => *nodes = nodes.with(node),
            None => edges.push((edge, graph, direction, Hydrators::of(node))),
        }
    }
    edges
}

fn direction_label(direction: EdgeDirection) -> &'static str {
    match direction {
        EdgeDirection::Forward => "fwd",
        EdgeDirection::Reverse => "rev",
    }
}

impl fmt::Display for HydrationPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = match self.rows {
            RequestRows::ActorPosts => " actor post rows",
            RequestRows::Posts | RequestRows::Users => "",
        };
        writeln!(
            f,
            "{}{rows}: {} calls",
            <&str>::from(self.level),
            self.groups.len()
        )?;
        for group in &self.groups {
            let (backend, call) = group.label();
            let after = group.input.map_or("-", <&str>::from);
            let nodes: Vec<&str> = group.nodes.iter().map(<&str>::from).collect();
            write!(
                f,
                "{backend}/{call} after: {after} nodes: {}",
                nodes.join(",")
            )?;
            let fields = group.fields();
            if !fields.is_empty() {
                let fields: Vec<String> = fields.iter().map(|field| format!("{field:?}")).collect();
                write!(f, " fields: {}", fields.join("|"))?;
            }
            for &(_, graph, direction, nodes) in group.edges() {
                let direction = direction_label(direction);
                let keys: Vec<String> = nodes
                    .iter()
                    .map(|node| node.key_at(self.rows).to_string())
                    .collect();
                write!(
                    f,
                    " {}-{direction}[{}]",
                    <&str>::from(graph),
                    keys.join(",")
                )?;
            }
            if group.nodes.iter().all(Hydrator::needs_viewer) {
                f.write_str(" (skipped logged out)")?;
            }
            if group.ceiling() != HYDRATION_TIMEOUT {
                write!(f, " timeout: {:?}", group.ceiling())?;
            }
            for &(edge, graph, direction, _) in group.edges() {
                let deadline = self.call_deadline(group, iter::once(edge));
                if deadline > group.ceiling() {
                    let graph: &str = graph.into();
                    write!(
                        f,
                        " call: {deadline:?} on {graph}-{}",
                        direction_label(direction)
                    )?;
                }
            }
            writeln!(f)?;
        }
        for relation in self.relations() {
            writeln!(f, "{relation}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Group, HydrationPlan, RequestRows};
    use crate::hydration::{Hydrator, Hydrators};
    use crate::rules::{RuleEngine, SafetyLevel};
    use strum::VariantArray;

    #[test]
    fn each_level_plans_these_calls() {
        let engine = RuleEngine::for_tests();
        let plans: String = SafetyLevel::VARIANTS
            .iter()
            .flat_map(|&level| {
                std::iter::once(engine.plan(level)).chain(engine.post_kinds_plan(level))
            })
            .map(ToString::to_string)
            .collect();
        crate::pinned::assert_pinned("tests/fixtures/plans.txt", &plans);
    }

    #[test]
    fn every_node_of_a_call_carries_the_calls_label() {
        for rows in [
            RequestRows::Posts,
            RequestRows::Users,
            RequestRows::ActorPosts,
        ] {
            let plan = HydrationPlan::with_rows(SafetyLevel::FilterAll, rows, Hydrators::all());
            for group in plan.groups() {
                for node in group.nodes.iter() {
                    assert_eq!(node.spec_at(rows).label, group.label(), "{rows:?} {node:?}");
                }
            }
        }
    }

    #[test]
    fn no_two_calls_of_a_level_share_a_label() {
        const SHARED: &[(SafetyLevel, (&str, &str), &str)] = &[(
            SafetyLevel::QuotedTweetRules,
            ("socialgraph", "quote_author_edges"),
            "the must-have block select and the best-effort follow select ask one quote pair",
        )];
        let engine = RuleEngine::for_tests();
        let mut shared = Vec::new();
        for &level in SafetyLevel::VARIANTS {
            for plan in std::iter::once(engine.plan(level)).chain(engine.post_kinds_plan(level)) {
                let labels: Vec<(&str, &str)> = plan.groups().map(Group::label).collect();
                for (position, &label) in labels.iter().enumerate() {
                    if labels[..position].contains(&label) {
                        shared.push((level, label));
                    }
                }
            }
        }
        let listed: Vec<(SafetyLevel, (&str, &str))> = SHARED
            .iter()
            .map(|&(level, label, _)| (level, label))
            .collect();
        assert_eq!(shared, listed);
    }

    #[test]
    fn groups_of_one_fetcher_claim_each_others_keys_only_through_an_edge() {
        let engine = RuleEngine::for_tests();
        for &level in SafetyLevel::VARIANTS {
            for plan in std::iter::once(engine.plan(level)).chain(engine.post_kinds_plan(level)) {
                let groups: Vec<_> = plan.groups().collect();
                for (position, group) in groups.iter().enumerate() {
                    for other in groups.iter().skip(position + 1) {
                        if other.fetcher != group.fetcher {
                            continue;
                        }
                        let nodes = group.nodes.union(other.nodes);
                        assert!(
                            nodes.iter().all(|node| node.edge().is_some()),
                            "{level:?} {:?}: two groups claim one fetcher's keys",
                            group.fetcher
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_level_of_user_rows_plans_no_pure_core_and_asks_the_actor_source_by_user() {
        let nodes = Hydrators::of(Hydrator::ViewerProfile)
            .with(Hydrator::AuthorSafety)
            .with(Hydrator::Follows)
            .with(Hydrator::Blocks);
        let plan =
            HydrationPlan::with_rows(SafetyLevel::QuotedTweetRules, RequestRows::Users, nodes);
        assert_eq!(
            plan.to_string(),
            "\
quoted_tweet_rules: 3 calls
gizmoduck/viewer_profile after: - nodes: viewer_profile fields: ACCOUNT|EXTENDED_PROFILE|SAFETY (skipped logged out)
gizmoduck/actor_profile after: - nodes: author_safety fields: SAFETY|LABELS|PROFILE
socialgraph/viewer_author_edges after: - nodes: follows,blocks follows-fwd[users] blocks-fwd[users] (skipped logged out)
"
        );
        assert!(plan.keys_only_user_rows_or_viewer());
    }
}
