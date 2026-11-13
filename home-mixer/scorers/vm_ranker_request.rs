use crate::models::candidate::{PhoenixScores, PostCandidate};
use crate::models::fs_recipient::FsRecipientInputs;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::*;
use crate::scorers::vm_ranker_debias_payload::{candidate_payload, request_payload};
use rustc_hash::FxHashSet;
use std::cmp::Reverse;
use xai_vm_ranker_proto as pb;
use xai_vm_ranker_proto::experiment::PostTopics;
use xai_x_thrift::served_history::{EntityIdType, ServedHistory};

const DPP_VALUE_MODEL_ID: &str = "dpp";

pub(crate) struct RequestShape {
    debias: bool,
    pacing: bool,
    history_topics: bool,
    post_topics: bool,
    dpp_history: bool,
}

impl RequestShape {
    pub(crate) fn from_query(query: &ScoredPostsQuery) -> Self {
        Self {
            debias: (!query.has_cached_posts || query.is_under_the_hood_request)
                && query.params.get(VMRankerSendDebiasInputs),
            pacing: query.params.get(VMRankerSendPacingInputs),
            history_topics: query.params.get(VMRankerSendHistoryTopics)
                && query.history_topics.is_some(),
            post_topics: query.params.get(VMRankerSendPostTopics) && !query.recent_views.is_empty(),
            dpp_history: query.request_type == RequestType::ForYou
                && query.params.get(VMRankerSendDppHistory),
        }
    }

    pub(crate) fn sends_post_topics(&self) -> bool {
        self.post_topics
    }

    pub(crate) fn sends_dpp_history(&self) -> bool {
        self.dpp_history
    }

    pub(crate) fn build(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        local: &[PostCandidate],
        topics: &[Option<PostTopics>],
    ) -> pb::RankRequest {
        let proto_candidates = candidates
            .iter()
            .zip(local)
            .enumerate()
            .map(|(i, (c, local))| self.candidate(c, local, topics.get(i).and_then(Option::as_ref)))
            .collect();
        let mut request = pb::RankRequest {
            viewer_id: query.user_id,
            request_timestamp_ms: u64::try_from(query.request_time_ms).unwrap_or(0),
            candidates: proto_candidates,
            value_model_id: DPP_VALUE_MODEL_ID.to_string(),
            compute_value_model: true,
            viewer: query.fs_recipient_inputs.as_ref().map(viewer_context_proto),
            topic_request: query.is_topic_request(),
            include_top_heads: query.is_under_the_hood_request,
            dpp_history: if self.dpp_history {
                dpp_history(query)
            } else {
                Vec::new()
            },
            ..Default::default()
        };
        if self.debias || self.history_topics || self.post_topics {
            request.experiment_payload = request_payload(
                query,
                candidates,
                self.debias,
                self.history_topics,
                self.post_topics,
            )
            .into();
        }
        request
    }

    fn candidate(
        &self,
        c: &PostCandidate,
        local: &PostCandidate,
        topics: Option<&PostTopics>,
    ) -> pb::RankCandidate {
        let mut out = pb::RankCandidate {
            tweet_id: c.tweet_id,
            author_id: c.author_id,
            retweeted_tweet_id: c.retweeted_tweet_id.unwrap_or(0),
            score: local.score,
            in_network: c.in_network.unwrap_or(false),
            is_retweet: c.retweeted_tweet_id.is_some(),
            is_reply: c.in_reply_to_tweet_id.is_some(),
            is_mutual_follow_author: c.is_mutual_follow_author == Some(true),
            cold_start_lift_to_rank: local.cold_start_lift_to_rank,
            min_video_duration_ms: c.min_video_duration_ms,
            phoenix_scores: Some(phoenix_scores_proto(&c.phoenix_scores)),
            weighted_score: c.weighted_score,
            semantic_ids: c.semantic_ids.clone().unwrap_or_default(),
            ..Default::default()
        };
        if self.debias || self.pacing || topics.is_some() {
            out.experiment_payload = candidate_payload(c, self.debias, self.pacing, topics).into();
        }
        out
    }
}

fn dpp_history(query: &ScoredPostsQuery) -> Vec<pb::DppHistoryItem> {
    let mut responses: Vec<(i64, &ServedHistory)> = query
        .served_history
        .iter()
        .filter_map(|r| Some((r.served_time_ms?, r)))
        .collect();
    responses.sort_by_key(|&(served_time_ms, _)| Reverse(served_time_ms));

    let max_items = query.params.get(VMRankerDppHistoryMaxItems);
    let mut seen = FxHashSet::default();
    let mut history = Vec::new();
    for (served_time_ms, response) in responses {
        for entry in &response.entries {
            let carousel = match entry.entity_type {
                EntityIdType::TWEET => false,
                EntityIdType::VIDEO_CAROUSEL_MODULE => true,
                _ => continue,
            };
            for ids in entry.item_ids.iter().flatten() {
                if history.len() >= max_items {
                    return history;
                }
                let Some(tweet_id) = ids
                    .source_tweet_id
                    .or(ids.tweet_id)
                    .and_then(|id| u64::try_from(id).ok())
                else {
                    continue;
                };
                if seen.insert(tweet_id) {
                    history.push(pb::DppHistoryItem {
                        tweet_id,
                        served_time_ms,
                        carousel,
                    });
                }
            }
        }
    }
    history
}

fn viewer_context_proto(inputs: &FsRecipientInputs) -> pb::ViewerContext {
    pb::ViewerContext {
        user_id: inputs.user_id,
        country_code: inputs.country_code.clone(),
        language_code: inputs.language_code.clone(),
        client_app_id: inputs.client_app_id,
        client_version: inputs.client_version.clone(),
        user_roles: inputs.user_roles.clone(),
        datacenter: inputs.datacenter.clone(),
        has_phone_number: inputs.has_phone_number,
        resurrection_time_ms: inputs.resurrection_time_ms,
        product: inputs.product.clone(),
        now_ms: inputs.now_ms,
        fs_overrides: inputs.fs_overrides.clone(),
    }
}

fn phoenix_scores_proto(s: &PhoenixScores) -> pb::PhoenixScores {
    pb::PhoenixScores {
        favorite_score: s.favorite_score,
        reply_score: s.reply_score,
        retweet_score: s.retweet_score,
        photo_expand_score: s.photo_expand_score,
        click_score: s.click_score,
        profile_click_score: s.profile_click_score,
        vqv_score: s.vqv_score,
        share_score: s.share_score,
        share_via_dm_score: s.share_via_dm_score,
        share_via_copy_link_score: s.share_via_copy_link_score,
        dwell_score: s.dwell_score,
        quote_score: s.quote_score,
        quoted_click_score: s.quoted_click_score,
        follow_author_score: s.follow_author_score,
        not_interested_score: s.not_interested_score,
        block_author_score: s.block_author_score,
        mute_author_score: s.mute_author_score,
        report_score: s.report_score,
        dwell_time: s.dwell_time,
        click_dwell_time: s.click_dwell_time,
        not_dwelled_score: s.not_dwelled_score,
        video_open_score: s.video_open_score,
        open_link_score: s.open_link_score,
        quoted_vqv_score: s.quoted_vqv_score,
        post_unexplored_score: s.post_unexplored_score,
        post_video_score: s.post_video_score,
        grok_analyze_score: s.grok_analyze_score,
        home_video_continuation_secs: s.home_video_continuation_secs,
        home_profile_visit_secs: s.home_profile_visit_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};
    use xai_x_thrift::served_history::{
        EntryWithItemIds, ItemIds, RequestType as ServedRequestType,
    };

    const NOW_MS: i64 = 1_800_000_000_000;

    fn query(
        send_dpp_history: &str,
        max_items: &str,
        served_history: Vec<ServedHistory>,
    ) -> ScoredPostsQuery {
        let fs = FeatureSwitches::new(vec![]).unwrap();
        let mut results = fs.match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_vm_ranker_send_dpp_history".to_string(),
            send_dpp_history,
        );
        results.override_fs(
            "rust_home_mixer_vm_ranker_dpp_history_max_items".to_string(),
            max_items,
        );
        ScoredPostsQuery {
            request_type: RequestType::ForYou,
            request_time_ms: NOW_MS,
            served_history,
            params: results.into(),
            ..Default::default()
        }
    }

    fn build(query: &ScoredPostsQuery) -> pb::RankRequest {
        RequestShape::from_query(query).build(query, &[], &[], &[])
    }

    fn ids(tweet_id: i64, source_tweet_id: Option<i64>) -> ItemIds {
        ItemIds {
            tweet_id: Some(tweet_id),
            source_tweet_id,
            ..Default::default()
        }
    }

    fn entry(
        entity_type: EntityIdType,
        sort_index: i64,
        item_ids: Vec<ItemIds>,
    ) -> EntryWithItemIds {
        EntryWithItemIds {
            entity_type,
            sort_index: Some(sort_index),
            size: None,
            item_ids: Some(item_ids),
        }
    }

    fn tweet(sort_index: i64, tweet_id: i64) -> EntryWithItemIds {
        entry(EntityIdType::TWEET, sort_index, vec![ids(tweet_id, None)])
    }

    fn response(
        age_secs: i64,
        request_type: ServedRequestType,
        entries: Vec<EntryWithItemIds>,
    ) -> ServedHistory {
        ServedHistory {
            request_type,
            entries,
            served_id: None,
            served_time_ms: Some(NOW_MS - age_secs * 1_000),
        }
    }

    fn pin(tweet_id: u64, age_secs: i64, carousel: bool) -> pb::DppHistoryItem {
        pb::DppHistoryItem {
            tweet_id,
            served_time_ms: NOW_MS - age_secs * 1_000,
            carousel,
        }
    }

    #[test]
    fn history_is_sent_only_for_for_you_with_the_switch_on() {
        let history = vec![response(30, ServedRequestType::INITIAL, vec![tweet(0, 10)])];

        let for_you = build(&query("true", "200", history.clone()));
        assert_eq!(for_you.dpp_history, vec![pin(10, 30, false)]);
        assert_eq!(for_you.request_timestamp_ms, NOW_MS as u64);

        let ranked_following = ScoredPostsQuery {
            request_type: RequestType::RankedFollowing,
            ..query("true", "200", history.clone())
        };
        assert!(build(&ranked_following).dpp_history.is_empty());
        assert!(build(&query("false", "200", history))
            .dpp_history
            .is_empty());
    }

    #[test]
    fn every_served_post_is_sent_and_ads_and_modules_are_not() {
        let history = vec![response(
            30,
            ServedRequestType::INITIAL,
            vec![
                entry(EntityIdType::PROMOTED_TWEET, 0, vec![ids(900, None)]),
                tweet(1, 100),
                tweet(1, 101),
                tweet(1, 102),
                entry(
                    EntityIdType::WHO_TO_FOLLOW,
                    2,
                    vec![ItemIds {
                        user_id: Some(7),
                        ..Default::default()
                    }],
                ),
                entry(EntityIdType::TWEET, 3, vec![ids(11, Some(77))]),
                entry(
                    EntityIdType::VIDEO_CAROUSEL_MODULE,
                    4,
                    vec![ids(21, None), ids(22, None)],
                ),
            ],
        )];

        assert_eq!(
            build(&query("true", "200", history)).dpp_history,
            vec![
                pin(100, 30, false),
                pin(101, 30, false),
                pin(102, 30, false),
                pin(77, 30, false),
                pin(21, 30, true),
                pin(22, 30, true),
            ]
        );
    }

    #[test]
    fn polling_is_included_duplicates_keep_the_newest_serve_and_only_the_newest_are_sent() {
        let history = vec![
            response(3_600, ServedRequestType::OLDER, vec![tweet(0, 30)]),
            response(
                10,
                ServedRequestType::POLLING,
                vec![tweet(0, 500), tweet(1, 10)],
            ),
            response(
                60,
                ServedRequestType::INITIAL,
                vec![tweet(0, 10), tweet(1, 20)],
            ),
        ];

        assert_eq!(
            build(&query("true", "200", history)).dpp_history,
            vec![
                pin(500, 10, false),
                pin(10, 10, false),
                pin(20, 60, false),
                pin(30, 3_600, false),
            ]
        );

        let pages: Vec<ServedHistory> = (0..10i64)
            .rev()
            .map(|page| {
                let posts = (0..25).map(|i| tweet(i, page * 100 + i)).collect();
                response((page + 1) * 10, ServedRequestType::OLDER, posts)
            })
            .collect();
        let newest_first: Vec<pb::DppHistoryItem> = (0..10i64)
            .flat_map(|page| {
                (0..25).map(move |i| pin((page * 100 + i) as u64, (page + 1) * 10, false))
            })
            .collect();
        assert_eq!(newest_first.len(), 250);
        assert_eq!(
            build(&query("true", "200", pages)).dpp_history,
            &newest_first[..200]
        );
    }
}
