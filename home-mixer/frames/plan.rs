use std::collections::{BTreeMap, HashSet};

use xai_home_mixer_proto::Frame;

use super::catalog::{placement_in, Timeline, TIMELINES};
use super::placement::Placement;
use crate::params::MAX_JETFUEL_FRAMES_PER_RESPONSE;

pub fn plan(frames: Vec<Frame>, feed_len: usize) -> Vec<(usize, Frame)> {
    plan_in(TIMELINES, frames, feed_len)
}

fn plan_in(
    timelines: &'static [Timeline],
    frames: Vec<Frame>,
    feed_len: usize,
) -> Vec<(usize, Frame)> {
    let mut seen = HashSet::new();
    let mut pinned = Vec::new();
    let mut slotted: BTreeMap<String, Vec<Frame>> = BTreeMap::new();
    for frame in frames {
        if !seen.insert((frame.route.clone(), frame.occurrence)) {
            continue;
        }
        let Some(placement) = placement_in(timelines, &frame.route) else {
            continue;
        };
        match placement {
            Placement::Pinned { rank } => pinned.push((rank, frame)),
            Placement::AtSlot { .. } | Placement::EveryNItems { .. } => {
                slotted.entry(frame.route.clone()).or_default().push(frame)
            }
        }
    }

    pinned.sort_by_key(|(rank, _)| *rank);
    let mut planned: Vec<(usize, Frame)> =
        pinned.into_iter().map(|(_, frame)| (0, frame)).collect();

    for (route, mut occurrences) in slotted {
        let Some(placement) = placement_in(timelines, &route) else {
            continue;
        };
        occurrences.sort_by_key(|frame| frame.occurrence);
        let slots = placement.slots(feed_len);
        planned.extend(slots.into_iter().zip(occurrences));
    }

    planned.sort_by_key(|(slot, frame)| (*slot, frame.occurrence));
    planned.truncate(MAX_JETFUEL_FRAMES_PER_RESPONSE);
    planned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::catalog::{
        Cadence, Fetch, FrameDef, Surface, GROKBOT_SUGGESTED_TASKS_ROUTE,
    };
    use crate::frames::placement::Placement;

    const TOPIC: i64 = 1;

    const fn pinned(route: &'static str, rank: usize) -> FrameDef {
        FrameDef {
            route,
            fetch: Fetch::Strato,
            fatigue: None,
            is_dismissible: false,
            placement_id: 0,
            placement: Placement::Pinned { rank },
            cadence: Cadence::FirstResponseOnly,
            enabled: |_| true,
        }
    }

    const fn recurring(route: &'static str, first_slot: usize, interval: usize) -> FrameDef {
        FrameDef {
            route,
            fetch: Fetch::Strato,
            fatigue: None,
            is_dismissible: false,
            placement_id: 0,
            placement: Placement::EveryNItems {
                first_slot,
                interval,
                rotating: false,
            },
            cadence: Cadence::EveryResponse,
            enabled: |_| true,
        }
    }

    const fn timeline(frames: &'static [FrameDef]) -> [Timeline; 1] {
        [Timeline {
            surface: Surface::Topic(TOPIC),
            enabled: |_| true,
            frames,
        }]
    }

    static MANY_PINNED: [Timeline; 1] =
        timeline(&[pinned("/a", 0), pinned("/b", 1), pinned("/c", 2)]);
    static ONE_RECURRING: [Timeline; 1] = timeline(&[recurring("/r", 5, 5)]);
    static EVERY_SLOT: [Timeline; 1] = timeline(&[recurring("/flood", 0, 1)]);
    static MIXED: [Timeline; 1] = timeline(&[
        pinned("/a", 0),
        pinned("/b", 1),
        recurring("/r", 6, 6),
        recurring("/s", 9, 20),
    ]);

    fn frame(route: &str) -> Frame {
        Frame {
            payload: "RENDERED".to_string(),
            height: 100,
            is_dismissible: false,
            placement_id: 0,
            route: route.to_string(),
            occurrence: 0,
            fetch_route: route.to_string(),
        }
    }

    fn occurrences_of(route: &str, count: u32) -> Vec<Frame> {
        (0..count)
            .map(|occurrence| Frame {
                occurrence,
                ..frame(route)
            })
            .collect()
    }

    fn routes(planned: &[(usize, Frame)]) -> Vec<(usize, &str)> {
        planned
            .iter()
            .map(|(slot, frame)| (*slot, frame.route.as_str()))
            .collect()
    }

    #[test]
    fn the_nfl_timeline_pins_one_card_and_spaces_the_rest() {
        let rotating = |occurrence: u32| Frame {
            occurrence,
            ..frame("/cards/nfl/timeline/card")
        };
        let planned = plan(
            vec![
                rotating(0),
                rotating(1),
                frame("/cards/nfl/timeline/pinned"),
            ],
            30,
        );
        assert_eq!(
            routes(&planned),
            vec![
                (0, "/cards/nfl/timeline/pinned"),
                (7, "/cards/nfl/timeline/card"),
                (14, "/cards/nfl/timeline/card"),
            ]
        );
    }

    #[test]
    fn the_grokbot_suggested_tasks_frame_lands_on_its_slot() {
        let planned = plan(vec![frame(GROKBOT_SUGGESTED_TASKS_ROUTE)], 30);
        assert_eq!(routes(&planned), vec![(3, GROKBOT_SUGGESTED_TASKS_ROUTE)]);
    }

    #[test]
    fn a_route_that_is_not_in_the_catalog_is_dropped() {
        assert!(plan(vec![frame("/cards/unknown/thing")], 30).is_empty());
    }

    #[test]
    fn a_duplicate_route_is_placed_once() {
        let planned = plan(
            vec![
                frame("/cards/mlb/timelineCarousel"),
                frame("/cards/mlb/timelineCarousel"),
            ],
            30,
        );
        assert_eq!(routes(&planned), vec![(0, "/cards/mlb/timelineCarousel")]);
    }

    fn placed(timelines: &'static [Timeline], routes: &[&str]) -> Vec<(usize, String)> {
        let frames = routes.iter().map(|route| frame(route)).collect();
        plan_in(timelines, frames, 30)
            .into_iter()
            .map(|(slot, frame)| (slot, frame.route))
            .collect()
    }

    #[test]
    fn each_fetched_occurrence_lands_on_its_own_slot() {
        let planned = plan_in(&ONE_RECURRING, occurrences_of("/r", 6), 30);
        assert_eq!(
            planned
                .iter()
                .map(|(slot, frame)| (*slot, frame.occurrence))
                .collect::<Vec<_>>(),
            vec![(5, 0), (10, 1), (15, 2), (20, 3), (25, 4), (30, 5)]
        );
    }

    #[test]
    fn a_failed_occurrence_closes_up_instead_of_leaving_a_double_gap() {
        let mut frames = occurrences_of("/r", 4);
        frames.remove(1);
        let planned = plan_in(&ONE_RECURRING, frames, 30);
        assert_eq!(
            planned
                .iter()
                .map(|(slot, frame)| (*slot, frame.occurrence))
                .collect::<Vec<_>>(),
            vec![(5, 0), (10, 2), (15, 3)]
        );
    }

    #[test]
    fn surplus_occurrences_are_dropped_when_the_page_is_short() {
        let planned = plan_in(&ONE_RECURRING, occurrences_of("/r", 6), 12);
        assert_eq!(planned.len(), 2);
    }

    #[test]
    fn pinned_and_recurring_frames_coexist_on_one_timeline() {
        let mut frames = vec![frame("/a"), frame("/b")];
        frames.extend(occurrences_of("/r", 5));
        frames.extend(occurrences_of("/s", 2));
        let planned: Vec<(usize, String)> = plan_in(&MIXED, frames, 30)
            .into_iter()
            .map(|(slot, frame)| (slot, frame.route))
            .collect();
        assert_eq!(
            planned,
            vec![
                (0, "/a".to_string()),
                (0, "/b".to_string()),
                (6, "/r".to_string()),
                (9, "/s".to_string()),
                (12, "/r".to_string()),
                (18, "/r".to_string()),
                (24, "/r".to_string()),
                (29, "/s".to_string()),
            ]
        );
    }

    #[test]
    fn a_lower_pinned_frame_moves_up_when_the_one_above_is_absent() {
        assert_eq!(placed(&MANY_PINNED, &["/c"]), vec![(0, "/c".to_string())]);
        assert_eq!(placed(&MANY_PINNED, &["/b"]), vec![(0, "/b".to_string())]);
        assert_eq!(
            placed(&MANY_PINNED, &["/b", "/c"]),
            vec![(0, "/b".to_string()), (0, "/c".to_string())]
        );
    }

    #[test]
    fn no_response_carries_more_frames_than_the_reserved_budget() {
        let planned = plan_in(&EVERY_SLOT, occurrences_of("/flood", 31), 30);
        assert_eq!(planned.len(), MAX_JETFUEL_FRAMES_PER_RESPONSE);
    }
}
