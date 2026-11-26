use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{ControlMode, PageAction};
use crate::replay_tests::{
    base_config, message, Account, BedConfig, TestBed, BAKED_POST_RULES, BAKED_USER_RULES,
};

async fn post_bed(action: PageAction) -> TestBed {
    TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: base_config(false, None),
        off_switches: vec![],
        control_records: vec![record(
            "SpamEmbeddingPtosDistilled",
            None,
            None,
            ControlMode::Live,
            vec![action],
        )],
    })
    .await
}

#[tokio::test]
async fn post_page_actions_never_waive_the_follower_ceiling() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut got = Vec::new();
    for action in [
        PageAction::SUSPEND_AUTHOR,
        PageAction::PostSpamHighRecall {
            lifetime: action_control::Lifetime::Days30,
        },
    ] {
        let bed = post_bed(action).await;
        let mut score = message(
            EntityType::Post,
            "SpamEmbeddingPtosDistilled",
            "clf",
            &["anchor_campaign_post_label"],
            Account::AboveTopLimit,
        );
        score.skip_author_credibility_prechecks = true;
        let trace = bed.run_fresh(&score).await;
        got.push((trace.status, trace.ais.len(), trace.tickets));
    }
    assert_eq!(
        got,
        vec![
            ("very_high_follower_count".into(), 0, 0),
            ("very_high_follower_count".into(), 0, 0),
        ]
    );
}

#[tokio::test]
async fn absent_action_control_config_keeps_page_live_records_recording_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let bed = TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: base_config(false, None),
        off_switches: vec![],
        control_records: vec![record(
            "UserEmbeddingPostSpamBot",
            None,
            None,
            ControlMode::Live,
            vec![PageAction::SUSPEND],
        )],
    })
    .await;
    let score = message(
        EntityType::User,
        "UserEmbeddingPostSpamBot",
        "clf",
        &["anchor_campaign_suspend"],
        Account::Clean,
    );
    let trace = bed.run_fresh(&score).await;
    assert_eq!(
        (trace.status, trace.ais.len(), trace.tickets),
        ("dry_run".into(), 0, 0)
    );
}
