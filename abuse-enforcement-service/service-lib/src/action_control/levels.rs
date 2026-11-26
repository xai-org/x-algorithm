use super::*;
use crate::confidence::{self, Claim, RowOutcome};

pub fn level_lkey(
    entity: ControlEntity,
    head: &str,
    classifier: Option<&str>,
    row: LevelRow,
) -> String {
    let e = entity.as_str();
    let r = row.as_str();
    match classifier {
        None => format!("{e}/l/{head}/{r}"),
        Some(c) => format!("{e}/cl/{head}/{c}/{r}"),
    }
}

pub(super) struct RowScope<'a> {
    pub record: Option<(&'a str, &'a ActionControlRecord)>,
    pub head: &'a str,
    pub classifier: Option<&'a str>,
    pub unreadable: bool,
}

impl Snapshot {
    fn first<'a>(&'a self, keys: impl Iterator<Item = String>) -> Option<Hit<'a>> {
        keys.into_iter().find_map(|k| self.at(&k))
    }

    pub(super) fn row_scope<'a>(
        &'a self,
        entity: ControlEntity,
        names: &[&'a str],
        classifier: Option<&'a str>,
        factory: bool,
    ) -> Option<RowScope<'a>> {
        if !factory {
            return self
                .mover(entity, names, classifier)
                .map(|(k, r)| RowScope {
                    record: Some((k, r)),
                    head: &r.head,
                    classifier: r.classifier.as_deref(),
                    unreadable: false,
                });
        }
        let head = names[0];
        let own = RowScope {
            record: None,
            head,
            classifier: None,
            unreadable: false,
        };
        let Some(c) = classifier else {
            return Some(own);
        };
        Some(match self.at(&lkey_for(entity, head, Some(c), None)) {
            Some(Hit::Record(k, r)) if r.mode == ControlMode::Rows => RowScope {
                record: Some((k, r)),
                head,
                classifier: Some(c),
                unreadable: false,
            },
            Some(Hit::Unreadable) => RowScope {
                unreadable: true,
                ..own
            },
            _ => own,
        })
    }

    pub(super) fn mover(
        &self,
        entity: ControlEntity,
        names: &[&str],
        classifier: Option<&str>,
    ) -> Option<(&str, &ActionControlRecord)> {
        if let Some(c) = classifier {
            match self.first(names.iter().map(|n| lkey_for(entity, n, Some(c), None))) {
                Some(Hit::Record(k, r)) if r.mode == ControlMode::Rows => return Some((k, r)),
                Some(Hit::Record(_, r)) if r.mode == ControlMode::Rules => {}
                Some(_) => return None,
                None => {}
            }
        }
        match self.first(names.iter().map(|n| lkey_for(entity, n, None, None))) {
            Some(Hit::Record(k, r)) if r.mode == ControlMode::Rows => Some((k, r)),
            _ => None,
        }
    }

    pub(super) fn row_decider(
        &self,
        entity: ControlEntity,
        names: &[&str],
        classifier: Option<&str>,
        row: LevelRow,
        recording: &ActionControlRecord,
        factory: bool,
    ) -> Option<ActionControlRecord> {
        if let Some(scope) = self.row_scope(entity, names, classifier, factory) {
            if scope.unreadable {
                return Some(recording.clone());
            }
            let lkey = level_lkey(entity, scope.head, scope.classifier, row);
            return Some(match self.at(&lkey) {
                Some(Hit::Record(_, r))
                    if r.mode != ControlMode::Rows
                        && !(factory && r.mode == ControlMode::Rules) =>
                {
                    r.clone()
                }
                _ => recording.clone(),
            });
        }
        match self.plain_decider(
            entity,
            names[0],
            names.get(1).copied(),
            classifier.unwrap_or(""),
            &[],
        ) {
            Some(Hit::Record(_, r)) if r.mode != ControlMode::Rows => Some(r.clone()),
            Some(_) => Some(recording.clone()),
            None => None,
        }
    }

    pub(super) fn under_scope(
        &self,
        entity: ControlEntity,
        names: &[&str],
        classifier: Option<&str>,
    ) -> Vec<(String, Option<ControlMode>)> {
        let e = entity.as_str();
        let mut prefixes = Vec::new();
        for n in names {
            match classifier {
                Some(c) => {
                    prefixes.push(format!("{e}/b/{n}/"));
                    prefixes.push(format!("{e}/cb/{n}/{c}/"));
                }
                None => {
                    prefixes.push(format!("{e}/b/{n}/"));
                    prefixes.push(format!("{e}/cb/{n}/"));
                }
            }
        }
        let classifiers: Vec<String> = match classifier {
            None => names.iter().map(|n| format!("{e}/c/{n}/")).collect(),
            Some(_) => Vec::new(),
        };
        let mut out: Vec<(String, Option<ControlMode>)> = self
            .records
            .iter()
            .filter(|(k, r)| {
                prefixes.iter().any(|p| k.starts_with(p))
                    || (r.mode == ControlMode::Rules
                        && classifiers.iter().any(|p| k.starts_with(p)))
            })
            .map(|(k, r)| (k.clone(), Some(r.mode)))
            .chain(
                self.unreadable
                    .keys()
                    .filter(|k| prefixes.iter().any(|p| k.starts_with(p)))
                    .map(|k| (k.clone(), None)),
            )
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

pub(super) struct RowContext<'a> {
    pub entity: EntityType,
    pub head: &'a str,
    pub alias: Option<&'a str>,
    pub classifier: &'a str,
    pub factory: bool,
    pub stale: bool,
    pub author_id: i64,
    pub rollout_salt: &'a str,
}

fn logged(
    matched: Option<Matched>,
    actions: Vec<PageAction>,
    forced: Option<&'static str>,
    outcome: RowOutcome,
) -> Resolved {
    Resolved {
        matched,
        mode: ControlMode::Recording,
        page_actions: actions,
        forced,
        row: Some(outcome),
        by_row: true,
        ..Default::default()
    }
}

pub(super) fn resolve_rows(
    snap: &Snapshot,
    m: &RowContext<'_>,
    claim: &Claim,
    cfg: &ControlConfig,
    safety_ok: &mut dyn FnMut(EntityType) -> bool,
) -> Option<Resolved> {
    let entity = entity_of(m.entity);
    let names: Vec<&str> = std::iter::once(m.head).chain(m.alias).collect();
    let classifier = (!m.classifier.is_empty() && valid_name(m.classifier)).then_some(m.classifier);
    let scope = snap.row_scope(entity, &names, classifier, m.factory)?;
    let mut outcome = RowOutcome::new(claim);
    if scope.unreadable {
        return Some(logged(None, Vec::new(), Some("unreadable"), outcome));
    }
    let row_lkey = level_lkey(entity, scope.head, scope.classifier, claim.row);
    let scope_matched = || {
        scope.record.map(|(k, r)| Matched {
            lkey: k.to_owned(),
            record: r.clone(),
        })
    };
    let record = match snap.at(&row_lkey) {
        None => return Some(logged(scope_matched(), Vec::new(), None, outcome)),
        Some(Hit::Record(_, r)) if r.mode != ControlMode::Rows => r,
        Some(_) => {
            return Some(logged(
                scope_matched(),
                Vec::new(),
                Some("unreadable"),
                outcome,
            ));
        }
    };
    let matched = Some(Matched {
        lkey: row_lkey,
        record: record.clone(),
    });
    let entity_ok =
        record.entity_type == entity && record.actions.iter().all(|a| a.entity() == entity);
    match record.mode {
        ControlMode::Rules if m.factory => {
            return Some(logged(
                matched,
                Vec::new(),
                Some("factory_rules_row"),
                outcome,
            ));
        }
        ControlMode::Rules => {
            return Some(Resolved {
                matched,
                row: Some(outcome),
                ..Default::default()
            });
        }
        ControlMode::Recording => {
            let forced = if !record.actions.is_empty() && !entity_ok {
                Some("wrong_entity")
            } else {
                None
            };
            let actions = match forced {
                None => record.actions.clone(),
                Some(_) => Vec::new(),
            };
            return Some(logged(matched, actions, forced, outcome));
        }
        _ => {}
    }
    if record.actions.is_empty() || !entity_ok {
        return Some(logged(matched, Vec::new(), Some("wrong_entity"), outcome));
    }
    let actions = record.actions.clone();
    let forced = if !cfg.enabled {
        Some("kill_switch")
    } else if m.stale {
        Some("stale_snapshot")
    } else if !safety_ok(m.entity) {
        Some("safety_checks_changed")
    } else {
        None
    };
    if forced.is_some() {
        return Some(logged(matched, actions, forced, outcome));
    }
    judge_evidence(Some(record), claim, cfg, m.factory, &mut outcome);
    if let Some(bp) = record.rollout_bp {
        let inside = record.rollout_takes(m.rollout_salt, m.author_id);
        outcome.rollout = Some((
            bp,
            rollout_bucket(m.rollout_salt, &record.head, m.author_id),
        ));
        metrics::ACTION_CONTROL_ROLLOUT_TOTAL
            .with_label_values(&[
                record.head.as_str(),
                claim.row.as_str(),
                if inside { "inside" } else { "outside" },
            ])
            .inc();
        if !inside {
            return Some(logged(
                matched,
                actions,
                Some(confidence::OUTSIDE_ROLLOUT),
                outcome,
            ));
        }
    }
    Some(Resolved {
        matched,
        mode: ControlMode::Live,
        page_actions: actions,
        row: Some(outcome),
        by_row: true,
        ..Default::default()
    })
}

fn judge_evidence(
    row: Option<&ActionControlRecord>,
    claim: &Claim,
    cfg: &ControlConfig,
    factory: bool,
    outcome: &mut RowOutcome,
) {
    if claim.row.level().is_none() {
        return;
    }
    let (evidence, max_age_ms) = if factory {
        (
            claim.evidence.clone(),
            cfg.confidence.factory_evidence_max_age_ms,
        )
    } else {
        (
            row.and_then(|r| r.evidence.as_deref().cloned())
                .or_else(|| claim.evidence.clone()),
            cfg.confidence.evidence_max_age_ms,
        )
    };
    outcome.evidence_supports = Some(
        evidence
            .as_ref()
            .and_then(|e| confidence::supported_level(e, claim.now_ms, max_age_ms)),
    );
}

pub(super) fn resolve_run(
    snap: &Snapshot,
    m: &RowContext<'_>,
    claim: &Claim,
    cfg: &ControlConfig,
    safety_ok: &mut dyn FnMut(EntityType) -> bool,
    actions: &[PageAction],
) -> Resolved {
    let entity = entity_of(m.entity);
    let names: Vec<&str> = std::iter::once(m.head).chain(m.alias).collect();
    let classifier = (!m.classifier.is_empty() && valid_name(m.classifier)).then_some(m.classifier);
    let scope = snap.row_scope(entity, &names, classifier, m.factory);
    let on_rows = scope.is_some();
    let unreadable = scope.as_ref().is_some_and(|s| s.unreadable);
    let mut outcome = RowOutcome::new(claim);
    let row = scope.and_then(|scope| {
        match snap.at(&level_lkey(entity, scope.head, scope.classifier, claim.row)) {
            Some(Hit::Record(_, r)) if r.mode != ControlMode::Rows => Some(r),
            _ => None,
        }
    });
    let forced = if unreadable {
        Some("unreadable")
    } else if actions.is_empty() || actions.iter().any(|a| a.entity() != entity) {
        Some("wrong_entity")
    } else if !cfg.enabled {
        Some("kill_switch")
    } else if m.stale {
        Some("stale_snapshot")
    } else if !safety_ok(m.entity) {
        Some("safety_checks_changed")
    } else {
        None
    };
    if forced.is_none() && on_rows {
        judge_evidence(row, claim, cfg, m.factory, &mut outcome);
    }
    Resolved {
        matched: None,
        mode: if forced.is_some() {
            ControlMode::Recording
        } else {
            ControlMode::Live
        },
        page_actions: actions.to_vec(),
        forced,
        dedup_class: Some(page_class(actions)),
        producer_dry: false,
        row: on_rows.then_some(outcome),
        by_row: on_rows,
        factory: m.factory,
        unlisted: false,
        unknown: false,
        author_skips: snap.author_skips(entity, &names, m.factory),
    }
}

pub(super) fn for_compare(rec: &ActionControlRecord) -> ActionControlRecord {
    match rec.mode {
        ControlMode::Rows => ActionControlRecord {
            mode: ControlMode::Recording,
            actions: Vec::new(),
            ..rec.clone()
        },
        _ => rec.clone(),
    }
}

pub(super) fn as_running(
    rec: ActionControlRecord,
    cfg: &ControlConfig,
    stale: bool,
    safety_ok: bool,
) -> ActionControlRecord {
    if rec.mode != ControlMode::Live {
        return rec;
    }
    let entity = match rec.entity_type {
        ControlEntity::User => EntityType::User,
        ControlEntity::Post => EntityType::Post,
    };
    let (mode, _, _) = effective(&rec, entity, cfg, stale, safety_ok);
    ActionControlRecord { mode, ..rec }
}

fn acts_on_every_account(rec: &ActionControlRecord) -> bool {
    rec.mode == ControlMode::Live && !rec.actions.is_empty() && rec.rollout_bp.is_none()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EveryAccountEnded {
    At(i64),
    Unknown(i64),
}

impl EveryAccountEnded {
    fn at_ms(self) -> i64 {
        match self {
            Self::At(t) | Self::Unknown(t) => t,
        }
    }
}

pub(crate) fn every_account_ended(
    lkey: &str,
    current: &ActionControlRecord,
    history: Option<&[ActionControlHistoryEntry]>,
    now_ms: i64,
) -> Option<EveryAccountEnded> {
    const MAX_STEPS: usize = 64;
    let since = now_ms - ROLLOUT_WAIT_MS;
    let unknown = Some(EveryAccountEnded::Unknown(current.updated_at_ms));
    let mut rec = current;
    for _ in 0..MAX_STEPS {
        if rec.updated_at_ms <= since {
            return None;
        }
        let entry = history.unwrap_or_default().iter().find(|e| {
            e.lkey == lkey && e.version == rec.version && e.after.write_nonce == rec.write_nonce
        });
        let Some(entry) = entry else {
            return unknown;
        };
        match &entry.before {
            None => return None,
            Some(p) if acts_on_every_account(p) => {
                return (entry.at_ms > since).then_some(EveryAccountEnded::At(entry.at_ms));
            }
            Some(p) => rec = p,
        }
    }
    unknown
}

fn rollout_wait(row: LevelRow, ended: EveryAccountEnded, now_ms: i64) -> SetError {
    let r = row.as_str();
    let ago = (now_ms - ended.at_ms()).max(0) / 1_000;
    let left = ((ended.at_ms() + ROLLOUT_WAIT_MS - now_ms).max(0) + 999) / 1_000;
    let what = match ended {
        EveryAccountEnded::At(_) => {
            format!("the {r} row stopped acting on every account {ago} s ago")
        }
        EveryAccountEnded::Unknown(_) => format!(
            "the {r} row was saved {ago} s ago and its history doesn't show whether it acted on every account before"
        ),
    };
    SetError::Invalid(
        "rollout_wait",
        format!(
            "{what}: set its rollout in {left} s, once every pod has read that save \
             (a pod on an image without rollouts keeps acting on the last row it could read)"
        ),
    )
}

fn row_problem(
    entity: ControlEntity,
    row: LevelRow,
    mode: ControlMode,
    actions: &[PageAction],
    evidence: Option<&ConfidenceEvidence>,
    rollout_bp: Option<u32>,
    now_ms: i64,
) -> Result<(), SetError> {
    let r = row.as_str();
    let bad = |code: &'static str, msg: String| Err(SetError::Invalid(code, msg));
    match mode {
        ControlMode::Rows => {
            return bad(
                "invalid_row",
                format!("the {r} row's mode is rules, recording or live"),
            );
        }
        ControlMode::Rules if !actions.is_empty() => {
            return bad(
                "rules_takes_no_actions",
                format!("the {r} row follows the GrowthBook rules; send actions: []"),
            );
        }
        ControlMode::Live if actions.is_empty() => {
            return bad(
                "live_needs_actions",
                format!("the {r} row is live and needs at least one action"),
            );
        }
        _ => {}
    }
    let mut kinds = HashSet::new();
    for a in actions {
        if !kinds.insert(a.kind()) {
            return bad(
                "duplicate_action",
                format!("{} appears more than once on the {r} row", a.kind()),
            );
        }
        if a.entity() != entity {
            return bad(
                "wrong_entity_actions",
                format!(
                    "a {} row takes only {} actions",
                    entity.as_str(),
                    entity.as_str()
                ),
            );
        }
        if !PageAction::menu(entity).contains(a) {
            return bad(
                "invalid_lifetime",
                format!(
                    "{} is not offered with that lifetime; a post label lasts 30 days or permanently",
                    a.kind()
                ),
            );
        }
    }
    if let Some(ev) = evidence {
        if row.level().is_none() {
            return bad(
                "invalid_evidence",
                "the missing row has no level, so it takes no evidence".into(),
            );
        }
        if let Some(p) = confidence::evidence_problem(ev, now_ms) {
            return bad("invalid_evidence", p);
        }
    }
    if rollout_bp.is_some_and(|bp| bp > ROLLOUT_ALL_BP) {
        return bad(
            "invalid_rollout",
            format!("the {r} row's rollout is 0 to {ROLLOUT_ALL_BP} basis points (0% to 100%)"),
        );
    }
    Ok(())
}

pub(super) fn validate_confidence(
    req: &ActionControlPutRequest,
    now_ms: i64,
) -> Result<(), SetError> {
    let bad = |code: &'static str, msg: String| Err(SetError::Invalid(code, msg));
    let row_overrides = req
        .rows
        .iter()
        .flat_map(|rows| {
            LevelRow::ALL
                .into_iter()
                .filter_map(move |row| rows.get(row))
        })
        .any(|s| s.evidence_override.is_some());
    if req.evidence_override.is_some() || row_overrides {
        return bad(
            "invalid_override",
            "evidence never keeps a row from acting, so there is no evidence override; send none"
                .into(),
        );
    }
    if req.confidence.is_some() && req.band.is_some() {
        return bad(
            "invalid_scope",
            "a record is keyed by band or by row, not both".into(),
        );
    }
    if req.rows.is_some() && req.mode != ControlMode::Rows {
        return bad(
            "rows_only_with_mode_rows",
            "rows are sent with mode rows, the move onto rows".into(),
        );
    }
    if req.mode == ControlMode::Rows {
        if req.band.is_some() || req.confidence.is_some() {
            return bad(
                "invalid_scope",
                "a head or head + classifier moves onto rows, not a band or a row".into(),
            );
        }
        if !req.actions.is_empty() {
            return bad(
                "rows_take_no_actions",
                "mode rows acts with each row's own setting; send actions: [] and the rows in rows"
                    .into(),
            );
        }
        if req.evidence.is_some() {
            return bad("invalid_evidence", "evidence belongs on a row".into());
        }
        if req.rollout_bp.is_some() {
            return bad(
                "invalid_rollout",
                "a rollout belongs on a row; send it in rows".into(),
            );
        }
        if let Some(rows) = &req.rows {
            for row in LevelRow::ALL {
                if let Some(s) = rows.get(row) {
                    row_problem(
                        req.entity_type,
                        row,
                        s.mode,
                        &s.actions,
                        s.evidence.as_ref(),
                        s.rollout_bp,
                        now_ms,
                    )?;
                }
            }
        }
        return Ok(());
    }
    match req.confidence {
        Some(row) => row_problem(
            req.entity_type,
            row,
            req.mode,
            &req.actions,
            req.evidence.as_ref(),
            req.rollout_bp,
            now_ms,
        ),
        None if req.rollout_bp.is_some() => bad(
            "invalid_rollout",
            "a rollout belongs on a low, mid, high or missing row".into(),
        ),
        None if req.evidence.is_some() => bad(
            "invalid_evidence",
            "evidence belongs on a low, mid or high row".into(),
        ),
        None => Ok(()),
    }
}

pub(super) fn record_problem(rec: &ActionControlRecord, now_ms: i64) -> Option<&'static str> {
    let ahead = now_ms.saturating_add(confidence::CLOCK_SKEW_MS);
    if rec.updated_at_ms > ahead {
        return Some("written later than this pod's clock");
    }
    if rec
        .evidence
        .as_ref()
        .is_some_and(|ev| ev.read_at_ms > ahead)
    {
        return Some("evidence read later than this pod's clock");
    }
    if let Some(ov) = &rec.evidence_override {
        if ov.by.trim().is_empty() || ov.reason.trim().is_empty() {
            return Some("an evidence override must name a person and a reason");
        }
        if ov.expires_at_ms <= rec.updated_at_ms {
            return Some("an evidence override can't expire before its own write");
        }
        if ov.expires_at_ms
            > rec
                .updated_at_ms
                .saturating_add(confidence::EVIDENCE_OVERRIDE_MAX_MS)
        {
            return Some("an evidence override can't last past 7 days from its write");
        }
    }
    if rec.confidence.is_some() && rec.band.is_some() {
        return Some("a record can't be keyed by both band and row");
    }
    if rec.mode == ControlMode::Rows
        && (rec.band.is_some() || rec.confidence.is_some() || !rec.actions.is_empty())
    {
        return Some("mode rows is set on a head or head + classifier, with no actions");
    }
    let level = rec.confidence.and_then(LevelRow::level);
    if rec.evidence.is_some() && level.is_none() {
        return Some("evidence on a record that isn't a low, mid or high row");
    }
    if rec.evidence_override.is_some() && level.is_none() {
        return Some("an evidence override on a record that isn't a low, mid or high row");
    }
    if let Some(ev) = &rec.evidence {
        let lb = ev.precision_lower_bound;
        if !(lb.is_finite() && (0.0..=1.0).contains(&lb)) {
            return Some("evidence lower bound is not a number from 0 to 1");
        }
    }
    if rec.rollout_bp.is_some() && rec.confidence.is_none() {
        return Some("a rollout on a record that isn't a row");
    }
    if rec.rollout_bp.is_some_and(|bp| bp > ROLLOUT_ALL_BP) {
        return Some("a rollout over 10,000 basis points");
    }
    None
}

pub(super) fn view(cfg: &ControlConfig) -> ConfidenceView {
    ConfidenceView {
        rows: LevelRow::ALL
            .into_iter()
            .map(|row| RowView {
                row,
                floor: row.level().map(ConfidenceLevel::floor),
                suggested_actions: AllowedActions {
                    user: PageAction::suggested(ControlEntity::User, row),
                    post: PageAction::suggested(ControlEntity::Post, row),
                },
            })
            .collect(),
        evidence_max_age_ms: cfg.confidence.evidence_max_age_ms,
        factory_evidence_max_age_ms: cfg.confidence.factory_evidence_max_age_ms,
        config_error: cfg.confidence.unreadable.clone(),
    }
}

struct RowWrite {
    row: LevelRow,
    lkey: String,
    after: ActionControlRecord,
    prior: Option<Vec<u8>>,
}

struct Change {
    lkey: String,
    after: Option<ActionControlRecord>,
    before: Option<ActionControlRecord>,
    prior: Option<Vec<u8>>,
}

impl ActionControl {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn set_rows(
        &self,
        req: ActionControlPutRequest,
        who: &str,
        key_id: &str,
        now_ms: i64,
        cfg: &ControlConfig,
        safety: &SafetyChecksByEntity,
        seen: &[ActionControlSeen],
        deadline: Instant,
    ) -> Result<SetOutcome, SetError> {
        let entity = req.entity_type;
        let classifier = req.classifier.as_deref();
        let scope_lkey = lkey_for(entity, &req.head, classifier, None);
        let alias = cfg.aliases.get(&req.head).map(String::as_str);
        let names: Vec<&str> = std::iter::once(req.head.as_str()).chain(alias).collect();
        let rows = self.scan_rows().await?;
        let snap = Snapshot::of_rows(&rows, &[]);
        let scope_prior = self.get_raw(&scope_lkey).await?;
        let scope_version = scope_prior.as_deref().map(raw_version);
        let scope_before = scope_prior.as_deref().and_then(|b| parse_record(b).ok());
        let factory = seen
            .iter()
            .any(|s| s.factory && s.head == req.head && s.entity_type == entity);
        let unknown = !factory && !snap.names_head(entity, &names);
        first_save_of_unknown_head(&req, unknown)?;
        let moving = req.mode == ControlMode::Rows;
        let leaving = !moving && req.confidence.is_none();
        let on_rows = scope_before
            .as_ref()
            .is_some_and(|r| r.mode == ControlMode::Rows);
        if ((moving || leaving) && scope_version != req.expected_version) || (leaving && !on_rows) {
            return Err(SetError::Conflict(
                scope_before.map(Box::new),
                scope_version,
            ));
        }
        if moving {
            let not_rules: Vec<String> = snap
                .under_scope(entity, &names, classifier)
                .into_iter()
                .filter(|(_, mode)| !factory && *mode != Some(ControlMode::Rules))
                .map(|(k, _)| k)
                .collect();
            if !not_rules.is_empty() {
                return Err(SetError::Invalid(
                    "band_settings_not_rules",
                    format!(
                        "set these to rules before moving onto rows: {}",
                        not_rules.join(", ")
                    ),
                ));
            }
        } else if !(on_rows || (factory && classifier.is_none())) {
            return Err(SetError::Invalid(
                "scope_not_on_rows",
                format!("{scope_lkey} is not on rows; move it with mode rows and its four rows"),
            ));
        }
        let record = |row: Option<LevelRow>,
                      mode: ControlMode,
                      actions: Vec<PageAction>,
                      evidence: Option<ConfidenceEvidence>,
                      rollout_bp: Option<u32>,
                      version: u64| {
            let mut rec = ActionControlRecord {
                v: ACTION_CONTROL_RECORD_VERSION,
                entity_type: entity,
                head: req.head.clone(),
                classifier: req.classifier.clone(),
                band: None,
                mode,
                actions,
                reason: req.reason.trim().to_owned(),
                updated_by: who.to_owned(),
                key_id: key_id.to_owned(),
                updated_at_ms: now_ms,
                version,
                write_nonce: new_nonce(),
                confidence: row,
                evidence: evidence.map(Box::new),
                evidence_override: None,
                admit_unlisted_labels: false,
                rollout_bp: rollout_bp.filter(|&bp| bp < ROLLOUT_ALL_BP),
                follower_threshold: None,
                skip_high_credibility: None,
            };
            rec.v = rec.schema_version();
            rec
        };
        let targets: Vec<LevelRow> = match req.confidence {
            Some(row) => vec![row],
            None if leaving => Vec::new(),
            None => LevelRow::ALL.to_vec(),
        };
        let mut writes: Vec<RowWrite> = Vec::with_capacity(targets.len());
        let mut befores: Vec<Option<ActionControlRecord>> = Vec::with_capacity(targets.len());
        for row in targets {
            let lkey = level_lkey(entity, &req.head, classifier, row);
            let prior = self.get_raw(&lkey).await?;
            let version = prior.as_deref().map(raw_version);
            let before = prior.as_deref().and_then(|b| parse_record(b).ok());
            let (setting, expected) = if moving {
                let s = req.rows.as_ref().and_then(|rows| rows.get(row));
                (
                    s.map(|s| (s.mode, s.actions.clone(), s.evidence.clone(), s.rollout_bp)),
                    s.and_then(|s| s.expected_version),
                )
            } else {
                (
                    Some((
                        req.mode,
                        req.actions.clone(),
                        req.evidence.clone(),
                        req.rollout_bp,
                    )),
                    req.expected_version,
                )
            };
            if expected != version {
                return Err(SetError::Conflict(before.map(Box::new), version));
            }
            let (mode, actions, evidence, rollout_bp) =
                setting.unwrap_or((ControlMode::Recording, Vec::new(), None, None));
            writes.push(RowWrite {
                row,
                lkey,
                after: record(
                    Some(row),
                    mode,
                    actions,
                    evidence,
                    rollout_bp,
                    version.unwrap_or(0) + 1,
                ),
                prior,
            });
            befores.push(before);
        }
        if let Some(w) = writes
            .iter()
            .zip(&befores)
            .find(|(w, before)| {
                w.after.rollout_bp.is_some() && before.as_ref().is_some_and(acts_on_every_account)
            })
            .map(|(w, _)| w)
        {
            let r = w.row.as_str();
            return Err(SetError::Invalid(
                "rollout_from_every_account",
                format!(
                    "the {r} row acts on every account now: set it to log only and save, then set its rollout \
                     (a pod on an image without rollouts keeps acting on the last row it could read)"
                ),
            ));
        }
        let recent: Vec<(&RowWrite, &ActionControlRecord)> = writes
            .iter()
            .zip(&befores)
            .filter_map(|(w, b)| Some((w, b.as_ref()?)))
            .filter(|(w, b)| {
                w.after.rollout_bp.is_some() && b.updated_at_ms > now_ms - ROLLOUT_WAIT_MS
            })
            .collect();
        if !recent.is_empty() {
            let history = self.row_history(&req.head).await;
            for (w, before) in recent {
                if let Some(ended) =
                    every_account_ended(&w.lkey, before, history.as_deref(), now_ms)
                {
                    return Err(rollout_wait(w.row, ended, now_ms));
                }
            }
        }

        let recording = record(None, ControlMode::Recording, Vec::new(), None, None, 0);
        let has_set_aside = !factory && !snap.under_scope(entity, &names, classifier).is_empty();
        let set_aside = has_set_aside && snap.mover(entity, &names, classifier).is_none();
        let scope_write = (moving || leaving).then(|| {
            let (mode, actions) = if moving {
                (ControlMode::Rows, Vec::new())
            } else {
                (req.mode, req.actions.clone())
            };
            let skips = author_skips_after(&req, scope_before.as_ref());
            let mut rec = ActionControlRecord {
                admit_unlisted_labels: admits_unlisted_after(&req, scope_before.as_ref(), unknown),
                follower_threshold: skips.follower_threshold,
                skip_high_credibility: skips.skip_high_credibility,
                ..record(
                    None,
                    mode,
                    actions,
                    None,
                    None,
                    scope_version.unwrap_or(0) + 1,
                )
            };
            rec.v = rec.schema_version();
            rec
        });
        let rules_after = record(None, ControlMode::Rules, Vec::new(), None, None, 0);
        let mut compare: Vec<(LevelRow, ActionControlRecord)> =
            writes.iter().map(|w| (w.row, w.after.clone())).collect();
        if let (true, Some(after)) = (leaving, &scope_write) {
            let gone: Vec<String> = LevelRow::ALL
                .into_iter()
                .map(|row| level_lkey(entity, &req.head, classifier, row))
                .chain([scope_lkey.clone()])
                .collect();
            let gone: Vec<&str> = gone.iter().map(String::as_str).collect();
            let without = Snapshot::of_rows(&rows, &gone);
            for row in LevelRow::ALL {
                let next = if after.mode == ControlMode::Rules || factory {
                    without
                        .row_decider(entity, &names, classifier, row, &recording, factory)
                        .unwrap_or_else(|| rules_after.clone())
                } else {
                    after.clone()
                };
                compare.push((row, next));
                if has_set_aside && after.mode != ControlMode::Rules {
                    compare.push((row, rules_after.clone()));
                }
            }
        }
        let safety_ok = match entity {
            ControlEntity::User => safety.user.ok,
            ControlEntity::Post => safety.post.ok,
        };
        let stale = self.is_stale();
        let rules_specs: Option<Vec<Value>> = {
            let scope_rows: Vec<&ActionControlSeen> = seen
                .iter()
                .filter(|s| s.head == req.head && s.entity_type == entity)
                .filter(|s| classifier.is_none_or(|c| s.classifier == c))
                .collect();
            (!scope_rows.is_empty()).then(|| {
                scope_rows
                    .iter()
                    .flat_map(|s| s.action_specs.as_array().cloned().unwrap_or_default())
                    .collect()
            })
        };
        let mut harsher: Vec<String> = Vec::new();
        for (row, after) in compare {
            let mut priors = vec![snap
                .row_decider(entity, &names, classifier, row, &recording, factory)
                .or_else(|| unknown.then(|| recording.clone()))];
            if set_aside {
                priors.push(None);
            }
            let judged_priors = priors
                .into_iter()
                .map(|prior| prior.map(|p| as_running(p, cfg, stale, safety_ok)));
            for judged in judged_priors {
                let why = harsher_because(judged.as_ref(), &after, rules_specs.as_deref());
                for reason in why {
                    let reason = format!("{} row: {reason}", row.as_str());
                    if !harsher.contains(&reason) {
                        harsher.push(reason);
                    }
                }
            }
        }
        if let Some(why) = scope_write
            .as_ref()
            .and_then(|after| admits_unlisted_harsher(scope_before.as_ref(), after, unknown))
        {
            harsher.push(why.to_owned());
        }
        if let Some(after) = &scope_write {
            for why in author_skips_harsher(scope_before.as_ref(), after) {
                if !harsher.contains(&why) {
                    harsher.push(why);
                }
            }
        }
        if !harsher.is_empty() && req.confirm.as_deref() != Some(CONFIRM_WORD) {
            return Err(SetError::Invalid(
                "confirm_required",
                format!(
                    "this change enforces more ({}); send confirm: \"{CONFIRM_WORD}\"",
                    harsher.join("; ")
                ),
            ));
        }

        let removals = LevelRow::ALL
            .into_iter()
            .filter(|_| leaving)
            .filter_map(|row| {
                let lkey = level_lkey(entity, &req.head, classifier, row);
                let (_, prior) = rows.iter().find(|(k, _)| *k == lkey)?;
                Some(Change {
                    before: parse_record(prior).ok(),
                    prior: Some(prior.clone()),
                    after: None,
                    lkey,
                })
            });
        let changes: Vec<Change> = writes
            .iter()
            .zip(&befores)
            .map(|(w, before)| Change {
                lkey: w.lkey.clone(),
                after: Some(w.after.clone()),
                before: before.clone(),
                prior: w.prior.clone(),
            })
            .chain(scope_write.as_ref().map(|rec| Change {
                lkey: scope_lkey.clone(),
                after: Some(rec.clone()),
                before: scope_before.clone(),
                prior: scope_prior.clone(),
            }))
            .chain(removals)
            .collect();
        let values = changes
            .iter()
            .map(|c| c.after.as_ref().map(serde_json::to_vec).transpose())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| SetError::Store(e.to_string()))?;
        if Instant::now() + self.scan_timeout * (2 * changes.len()) as u32 > deadline {
            return Err(SetError::Store(
                "the save took longer than its hold on the head; nothing was written, reload and save again"
                    .into(),
            ));
        }
        for (i, (c, value)) in changes.iter().zip(values).enumerate() {
            if let Err(e) = self.write_raw(&c.lkey, value, deadline).await {
                return Err(self.put_back(&changes[..=i], e, deadline).await);
            }
        }
        let mut history_written = true;
        for c in &changes {
            if let Some(rec) = &c.after {
                history_written &= self
                    .write_history(&c.lkey, rec, c.before.as_ref(), key_id, now_ms)
                    .await;
            }
        }
        for c in &changes {
            match &c.after {
                Some(rec) => self.read_back(&c.lkey, rec).await?,
                None => self.read_back_removed(&c.lkey).await?,
            }
        }
        for c in changes {
            match c.after {
                Some(rec) => self.apply_local(c.lkey, rec),
                None => self.forget_local(&c.lkey),
            }
        }
        info!(
            lkey = scope_lkey,
            moved = moving,
            rows = writes.len(),
            updated_by = %who,
            key_id,
            "action control rows written"
        );
        let (lkey, rec, before) = match scope_write {
            Some(rec) => (scope_lkey, rec, scope_before),
            None => {
                let w = writes.remove(0);
                (w.lkey, w.after, befores.remove(0))
            }
        };
        let view = self.view(&lkey, &rec, cfg, stale, safety);
        Ok(SetOutcome {
            view,
            before,
            harsher_because: harsher,
            history_written,
        })
    }

    async fn put_back(&self, written: &[Change], err: SetError, deadline: Instant) -> SetError {
        let SetError::Store(msg) = err else {
            return err;
        };
        for c in written.iter().rev() {
            if self
                .write_raw(&c.lkey, c.prior.clone(), deadline)
                .await
                .is_err()
            {
                return SetError::Store(format!(
                    "{msg}; putting {} back failed, so the save may have applied: the records \
                     before it keep what this save confirmed; reload to see what the store holds",
                    c.lkey
                ));
            }
        }
        SetError::Store(format!(
            "{msg}; every record this save wrote was put back as it was, so nothing changed"
        ))
    }

    async fn row_history(&self, head: &str) -> Option<Vec<ActionControlHistoryEntry>> {
        let pkey = format!("{HISTORY_PKEY_PREFIX}{head}");
        match tokio::time::timeout(self.scan_timeout, self.store.scan(&pkey)).await {
            Ok(Ok(rows)) => Some(
                rows.iter()
                    .filter_map(|(_, v)| serde_json::from_slice(v).ok())
                    .collect(),
            ),
            Ok(Err(e)) => {
                warn!(error = %e, head, "action control: history read failed");
                None
            }
            Err(_) => {
                warn!(head, "action control: history read timed out");
                None
            }
        }
    }

    async fn write_history(
        &self,
        lkey: &str,
        rec: &ActionControlRecord,
        before: Option<&ActionControlRecord>,
        key_id: &str,
        now_ms: i64,
    ) -> bool {
        let entry = ActionControlHistoryEntry {
            at_ms: now_ms,
            version: rec.version,
            lkey: lkey.to_owned(),
            scope: rec.scope().to_owned(),
            updated_by: rec.updated_by.clone(),
            key_id: key_id.to_owned(),
            reason: rec.reason.clone(),
            env: self.env.clone(),
            before: before.cloned(),
            after: rec.clone(),
        };
        let Ok(v) = serde_json::to_vec(&entry) else {
            return false;
        };
        let pkey = format!("{HISTORY_PKEY_PREFIX}{}", rec.head);
        let hkey = format!("{now_ms:013}-{:010}-{lkey}", rec.version);
        let put = self.store.put(&pkey, &hkey, v, HISTORY_TTL_SECS);
        match tokio::time::timeout(self.scan_timeout, put).await {
            Ok(Ok(())) => true,
            Ok(Err(e)) => {
                warn!(error = %e, lkey, "action control: history write failed");
                false
            }
            Err(_) => {
                warn!(lkey, "action control: history write timed out");
                false
            }
        }
    }

    async fn read_back_removed(&self, lkey: &str) -> Result<(), SetError> {
        match self.get_record(lkey).await {
            Ok(None) => Ok(()),
            Ok(Some((r, _))) => Err(SetError::LostRace(r.map(Box::new))),
            Err(_) => Err(SetError::Store(
                "the write landed but reading it back failed; pods pick up whatever the store holds within 5 s, so reload to see it"
                    .into(),
            )),
        }
    }

    async fn read_back(&self, lkey: &str, rec: &ActionControlRecord) -> Result<(), SetError> {
        match self.get_record(lkey).await {
            Ok(Some((Some(r), _))) if r.write_nonce == rec.write_nonce => Ok(()),
            Ok(Some((Some(r), _))) => Err(SetError::LostRace(Some(Box::new(r)))),
            Ok(_) | Err(_) => Err(SetError::Store(
                "the write landed but reading it back failed; pods pick up whatever the store holds within 5 s, so reload to see it"
                    .into(),
            )),
        }
    }
}
