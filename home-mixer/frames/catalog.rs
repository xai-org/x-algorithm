use super::placement::Placement;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableJetfuelFrameGrokbotSuggestedTasks, EnableJetfuelFrameMlbCarousel, EnableJetfuelFrameNfl,
    EnableJetfuelFrameNflTimelinePinned, EnableJetfuelFrameNflTimelineRotation,
    EnableJetfuelFrameSoccer, EnableJetfuelFrames, JetfuelFrameGrokbotSuggestedTasksFatigueHours,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    FirstResponseOnly,
    FirstResponseExceptPolling,
    EveryResponse,
}

impl Cadence {
    fn should_emit(self, query: &ScoredPostsQuery) -> bool {
        match self {
            Cadence::FirstResponseOnly => !query.is_bottom_request,
            Cadence::FirstResponseExceptPolling => !query.is_bottom_request && !query.is_polling,
            Cadence::EveryResponse => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fetch {
    Strato,
    Jetfuel { height: i32 },
}

#[derive(Clone, Copy)]
pub struct Fatigue {
    pub served_key: &'static str,
    pub dismissal_key: &'static str,
    pub hours: fn(&ScoredPostsQuery) -> u32,
}

impl Fatigue {
    pub fn window_ms(self, query: &ScoredPostsQuery) -> Option<i64> {
        let hours = (self.hours)(query);
        (hours > 0).then_some(i64::from(hours) * 3_600_000)
    }
}

pub struct FrameDef {
    pub route: &'static str,
    pub fetch: Fetch,
    pub fatigue: Option<Fatigue>,
    pub is_dismissible: bool,
    pub placement_id: i64,
    pub placement: Placement,
    pub cadence: Cadence,
    pub enabled: fn(&ScoredPostsQuery) -> bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Topic(i64),
    ForYouHome,
}

impl Surface {
    fn matches(self, query: &ScoredPostsQuery) -> bool {
        match self {
            Surface::Topic(topic_id) => query.topic_ids.contains(&topic_id),
            Surface::ForYouHome => !query.is_topic_request() && !query.in_network_only,
        }
    }
}

pub struct Timeline {
    pub surface: Surface,
    pub enabled: fn(&ScoredPostsQuery) -> bool,
    pub frames: &'static [FrameDef],
}

pub const SOCCER_TOPIC_ID: i64 = 1925950498420469760;
pub const NFL_TOPIC_ID: i64 = 1925951614067650560;
pub const MLB_TOPIC_ID: i64 = 1925951672561319936;
pub const BASEBALL_TOPIC_ID: i64 = 1925950576740675584;

pub const GROKBOT_SUGGESTED_TASKS_ROUTE: &str = "/grokbot/suggested_tasks";
pub const GROKBOT_SUGGESTED_TASKS_SERVED_KEY: &str = "home.grokbot_suggested_tasks.served";
pub const GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY: &str = "home.grokbot_suggested_tasks";
const GROKBOT_SUGGESTED_TASKS_SLOT: usize = 3;
const GROKBOT_SUGGESTED_TASKS_HEIGHT: i32 = 433;

const NFL_ROTATION_INTERVAL: usize = 7;

pub const TIMELINES: &[Timeline] = &[
    Timeline {
        surface: Surface::ForYouHome,
        enabled: |q| q.params.get(EnableJetfuelFrameGrokbotSuggestedTasks),
        frames: &[FrameDef {
            route: GROKBOT_SUGGESTED_TASKS_ROUTE,
            fetch: Fetch::Jetfuel {
                height: GROKBOT_SUGGESTED_TASKS_HEIGHT,
            },
            fatigue: Some(Fatigue {
                served_key: GROKBOT_SUGGESTED_TASKS_SERVED_KEY,
                dismissal_key: GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY,
                hours: |q| q.params.get(JetfuelFrameGrokbotSuggestedTasksFatigueHours),
            }),
            is_dismissible: false,
            placement_id: 0,
            placement: Placement::AtSlot {
                slot: GROKBOT_SUGGESTED_TASKS_SLOT,
            },
            cadence: Cadence::FirstResponseExceptPolling,
            enabled: |_| true,
        }],
    },
    Timeline {
        surface: Surface::Topic(SOCCER_TOPIC_ID),
        enabled: |q| q.params.get(EnableJetfuelFrameSoccer),
        frames: &[FrameDef {
            route: "/cards/soccer/timelineCarousel",
            fetch: Fetch::Strato,
            fatigue: None,
            is_dismissible: false,
            placement_id: 0,
            placement: Placement::Pinned { rank: 0 },
            cadence: Cadence::FirstResponseOnly,
            enabled: |_| true,
        }],
    },
    Timeline {
        surface: Surface::Topic(NFL_TOPIC_ID),
        enabled: |q| q.params.get(EnableJetfuelFrameNfl),
        frames: &[
            FrameDef {
                route: "/cards/nfl/timeline/pinned",
                fetch: Fetch::Strato,
                fatigue: None,
                is_dismissible: false,
                placement_id: 0,
                placement: Placement::Pinned { rank: 0 },
                cadence: Cadence::FirstResponseOnly,
                enabled: |q| q.params.get(EnableJetfuelFrameNflTimelinePinned),
            },
            FrameDef {
                route: "/cards/nfl/timeline/card",
                fetch: Fetch::Strato,
                fatigue: None,
                is_dismissible: false,
                placement_id: 0,
                placement: Placement::EveryNItems {
                    first_slot: NFL_ROTATION_INTERVAL,
                    interval: NFL_ROTATION_INTERVAL,
                    rotating: true,
                },
                cadence: Cadence::EveryResponse,
                enabled: |q| q.params.get(EnableJetfuelFrameNflTimelineRotation),
            },
        ],
    },
    Timeline {
        surface: Surface::Topic(MLB_TOPIC_ID),
        enabled: |_| true,
        frames: &[FrameDef {
            route: "/cards/mlb/timelineCarousel",
            fetch: Fetch::Strato,
            fatigue: None,
            is_dismissible: false,
            placement_id: 0,
            placement: Placement::Pinned { rank: 0 },
            cadence: Cadence::FirstResponseOnly,
            enabled: |q| q.params.get(EnableJetfuelFrameMlbCarousel),
        }],
    },
    Timeline {
        surface: Surface::Topic(BASEBALL_TOPIC_ID),
        enabled: |_| true,
        frames: &[FrameDef {
            route: "/cards/mlb/timelineCarousel",
            fetch: Fetch::Strato,
            fatigue: None,
            is_dismissible: false,
            placement_id: 0,
            placement: Placement::Pinned { rank: 0 },
            cadence: Cadence::FirstResponseOnly,
            enabled: |q| q.params.get(EnableJetfuelFrameMlbCarousel),
        }],
    },
];

pub fn eligible(query: &ScoredPostsQuery) -> impl Iterator<Item = &'static FrameDef> + '_ {
    eligible_in(TIMELINES, query)
}

pub fn placement_for_route(route: &str) -> Option<Placement> {
    placement_in(TIMELINES, route)
}

pub fn fatigue_for_route(route: &str) -> Option<Fatigue> {
    TIMELINES
        .iter()
        .flat_map(|timeline| timeline.frames.iter())
        .find(|frame| frame.route == route)
        .and_then(|frame| frame.fatigue)
}

pub fn fatigue_keys(query: &ScoredPostsQuery) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = candidates_in(TIMELINES, query)
        .filter_map(|frame| frame.fatigue)
        .filter(|fatigue| fatigue.window_ms(query).is_some())
        .flat_map(|fatigue| [fatigue.served_key, fatigue.dismissal_key])
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

pub(super) fn eligible_in<'a>(
    timelines: &'static [Timeline],
    query: &'a ScoredPostsQuery,
) -> impl Iterator<Item = &'static FrameDef> + 'a {
    candidates_in(timelines, query).filter(move |frame| !fatigued(frame, query))
}

fn candidates_in<'a>(
    timelines: &'static [Timeline],
    query: &'a ScoredPostsQuery,
) -> impl Iterator<Item = &'static FrameDef> + 'a {
    let jetfuel_enabled = query.params.get(EnableJetfuelFrames);
    timelines
        .iter()
        .filter(move |timeline| {
            jetfuel_enabled && timeline.surface.matches(query) && (timeline.enabled)(query)
        })
        .flat_map(|timeline| timeline.frames.iter())
        .filter(move |frame| frame.cadence.should_emit(query) && (frame.enabled)(query))
}

fn fatigued(frame: &FrameDef, query: &ScoredPostsQuery) -> bool {
    let Some(fatigue) = frame
        .fatigue
        .filter(|fatigue| fatigue.window_ms(query).is_some())
    else {
        return false;
    };
    query.frame_dismissals.as_ref().is_none_or(|dismissals| {
        [fatigue.served_key, fatigue.dismissal_key]
            .iter()
            .filter_map(|key| dismissals.get(*key))
            .any(|&expires_at_ms| expires_at_ms > query.request_time_ms)
    })
}

pub(super) fn placement_in(timelines: &'static [Timeline], route: &str) -> Option<Placement> {
    timelines
        .iter()
        .flat_map(|timeline| timeline.frames.iter())
        .find(|frame| frame.route == route)
        .map(|frame| frame.placement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};

    const HOUR_MS: i64 = 3_600_000;
    const NOW_MS: i64 = 1_000 * HOUR_MS;

    fn home_query(grokbot_enabled: bool, fatigue_hours: u32) -> ScoredPostsQuery {
        home_query_with(true, grokbot_enabled, fatigue_hours)
    }

    fn home_query_with(
        jetfuel_frames_enabled: bool,
        grokbot_enabled: bool,
        fatigue_hours: u32,
    ) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_jetfuel_frames".to_string(),
            if jetfuel_frames_enabled {
                "true"
            } else {
                "false"
            },
        );
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_enabled".to_string(),
            if grokbot_enabled { "true" } else { "false" },
        );
        results.override_fs(
            "rust_home_mixer_jetfuel_frame_grokbot_suggested_tasks_fatigue_hours".to_string(),
            &fatigue_hours.to_string(),
        );
        ScoredPostsQuery {
            user_id: 1,
            request_time_ms: NOW_MS,
            params: results.into(),
            frame_dismissals: Some(HashMap::new()),
            ..Default::default()
        }
    }

    fn eligible_routes(query: &ScoredPostsQuery) -> Vec<&'static str> {
        eligible(query).map(|frame| frame.route).collect()
    }

    #[test]
    fn grokbot_suggested_tasks_frame_is_served_on_the_home_feed() {
        assert_eq!(
            eligible_routes(&home_query(true, 24)),
            vec![GROKBOT_SUGGESTED_TASKS_ROUTE]
        );
        assert!(eligible_routes(&home_query(false, 24)).is_empty());
    }

    #[test]
    fn the_master_switch_also_stops_the_grokbot_suggested_tasks_frame() {
        assert!(eligible_routes(&home_query_with(false, true, 24)).is_empty());
    }

    #[test]
    fn grokbot_suggested_tasks_frame_is_not_fetched_on_polls() {
        let poll = ScoredPostsQuery {
            is_polling: true,
            ..home_query(true, 24)
        };
        assert!(eligible_routes(&poll).is_empty());
    }

    #[test]
    fn grokbot_suggested_tasks_frame_stays_off_topic_and_following_feeds() {
        let topic = ScoredPostsQuery {
            topic_ids: vec![42],
            ..home_query(true, 24)
        };
        let following = ScoredPostsQuery {
            in_network_only: true,
            ..home_query(true, 24)
        };
        assert!(eligible_routes(&topic).is_empty());
        assert!(eligible_routes(&following).is_empty());
    }

    #[test]
    fn grokbot_suggested_tasks_frame_stays_hidden_while_its_served_or_dismissed_time_runs() {
        for key in [
            GROKBOT_SUGGESTED_TASKS_SERVED_KEY,
            GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY,
        ] {
            let hidden_until = |expires_at_ms: i64| ScoredPostsQuery {
                frame_dismissals: Some(HashMap::from([(key.to_string(), expires_at_ms)])),
                ..home_query(true, 6)
            };
            assert!(eligible_routes(&hidden_until(NOW_MS + 1)).is_empty());
            assert_eq!(
                eligible_routes(&hidden_until(NOW_MS)),
                vec![GROKBOT_SUGGESTED_TASKS_ROUTE]
            );
        }
    }

    #[test]
    fn grokbot_suggested_tasks_frame_is_hidden_when_its_dismissal_could_not_be_read() {
        let unread = ScoredPostsQuery {
            frame_dismissals: None,
            ..home_query(true, 6)
        };
        assert!(eligible_routes(&unread).is_empty());
        assert_eq!(
            fatigue_keys(&unread),
            vec![
                GROKBOT_SUGGESTED_TASKS_DISMISSAL_KEY,
                GROKBOT_SUGGESTED_TASKS_SERVED_KEY
            ]
        );
    }

    #[test]
    fn grokbot_suggested_tasks_frame_never_fatigues_when_its_fatigue_is_zero() {
        let never_fatigued = ScoredPostsQuery {
            frame_dismissals: None,
            ..home_query(true, 0)
        };
        assert_eq!(
            eligible_routes(&never_fatigued),
            vec![GROKBOT_SUGGESTED_TASKS_ROUTE]
        );
        assert!(fatigue_keys(&never_fatigued).is_empty());
    }

    #[test]
    fn cadence_controls_scroll_down_eligibility() {
        let first_response = ScoredPostsQuery::default();
        let scroll_down = ScoredPostsQuery {
            is_bottom_request: true,
            ..Default::default()
        };
        let poll = ScoredPostsQuery {
            is_polling: true,
            ..Default::default()
        };
        assert!(Cadence::FirstResponseOnly.should_emit(&first_response));
        assert!(!Cadence::FirstResponseOnly.should_emit(&scroll_down));
        assert!(Cadence::FirstResponseOnly.should_emit(&poll));
        assert!(Cadence::FirstResponseExceptPolling.should_emit(&first_response));
        assert!(!Cadence::FirstResponseExceptPolling.should_emit(&scroll_down));
        assert!(!Cadence::FirstResponseExceptPolling.should_emit(&poll));
        assert!(Cadence::EveryResponse.should_emit(&first_response));
        assert!(Cadence::EveryResponse.should_emit(&scroll_down));
    }
}
