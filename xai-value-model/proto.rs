use crate::inputs::CandidateScoringInputs;
use crate::phoenix_scores::PhoenixScores;
use xai_vm_ranker_proto as pb;

impl From<&pb::PhoenixScores> for PhoenixScores {
    fn from(s: &pb::PhoenixScores) -> Self {
        Self {
            favorite_score: s.favorite_score,
            reply_score: s.reply_score,
            retweet_score: s.retweet_score,
            photo_expand_score: s.photo_expand_score,
            video_open_score: s.video_open_score,
            click_score: s.click_score,
            open_link_score: s.open_link_score,
            profile_click_score: s.profile_click_score,
            vqv_score: s.vqv_score,
            share_score: s.share_score,
            share_via_dm_score: s.share_via_dm_score,
            share_via_copy_link_score: s.share_via_copy_link_score,
            dwell_score: s.dwell_score,
            quote_score: s.quote_score,
            quoted_click_score: s.quoted_click_score,
            quoted_vqv_score: s.quoted_vqv_score,
            follow_author_score: s.follow_author_score,
            not_interested_score: s.not_interested_score,
            block_author_score: s.block_author_score,
            mute_author_score: s.mute_author_score,
            report_score: s.report_score,
            not_dwelled_score: s.not_dwelled_score,
            post_unexplored_score: s.post_unexplored_score,
            post_video_score: s.post_video_score,
            grok_analyze_score: s.grok_analyze_score,
            dwell_time: s.dwell_time,
            click_dwell_time: s.click_dwell_time,
            home_video_continuation_secs: s.home_video_continuation_secs,
            home_profile_visit_secs: s.home_profile_visit_secs,
        }
    }
}

impl From<&PhoenixScores> for pb::PhoenixScores {
    fn from(s: &PhoenixScores) -> Self {
        Self {
            favorite_score: s.favorite_score,
            reply_score: s.reply_score,
            retweet_score: s.retweet_score,
            photo_expand_score: s.photo_expand_score,
            video_open_score: s.video_open_score,
            click_score: s.click_score,
            open_link_score: s.open_link_score,
            profile_click_score: s.profile_click_score,
            vqv_score: s.vqv_score,
            share_score: s.share_score,
            share_via_dm_score: s.share_via_dm_score,
            share_via_copy_link_score: s.share_via_copy_link_score,
            dwell_score: s.dwell_score,
            quote_score: s.quote_score,
            quoted_click_score: s.quoted_click_score,
            quoted_vqv_score: s.quoted_vqv_score,
            follow_author_score: s.follow_author_score,
            not_interested_score: s.not_interested_score,
            block_author_score: s.block_author_score,
            mute_author_score: s.mute_author_score,
            report_score: s.report_score,
            not_dwelled_score: s.not_dwelled_score,
            post_unexplored_score: s.post_unexplored_score,
            post_video_score: s.post_video_score,
            grok_analyze_score: s.grok_analyze_score,
            dwell_time: s.dwell_time,
            click_dwell_time: s.click_dwell_time,
            home_video_continuation_secs: s.home_video_continuation_secs,
            home_profile_visit_secs: s.home_profile_visit_secs,
        }
    }
}

impl CandidateScoringInputs {
    pub fn from_rank_candidate(c: &pb::RankCandidate, min_video_duration_ms: i32) -> Self {
        Self {
            phoenix_scores: c
                .phoenix_scores
                .as_ref()
                .map(PhoenixScores::from)
                .unwrap_or_default(),
            author_id: c.author_id,
            in_network: Some(c.in_network),
            is_reply: c.is_reply,
            is_retweet: c.is_retweet,
            is_mutual_follow_author: c.is_mutual_follow_author,
            vqv_eligible: c
                .min_video_duration_ms
                .is_some_and(|ms| ms > min_video_duration_ms),
            quoted_vqv_eligible: true,
            cold_start_lift_to_rank: c.cold_start_lift_to_rank,
            weighted_score: c.weighted_score,
            author_exploration_bonus: 0.0,
            user_video_continuation_secs: None,
        }
    }
}
