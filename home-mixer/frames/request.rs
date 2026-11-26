use tracing::warn;
use xai_home_mixer_proto::{feed_item, FeedItem, Frame};
use xai_stats_receiver::global_stats_receiver;

use super::catalog::{eligible, FrameDef};
use super::placement::Placement;
use super::served::resume_index;
use crate::models::query::ScoredPostsQuery;
use crate::params::{FOR_YOU_MAX_RESULT_SIZE, MAX_JETFUEL_FRAMES_PER_RESPONSE};

const MAX_ORGANIC_FEED_LEN: usize = FOR_YOU_MAX_RESULT_SIZE - MAX_JETFUEL_FRAMES_PER_RESPONSE;

pub type FetchedPayload = Result<Option<(String, i32)>, String>;

pub struct FrameRequest {
    pub frame: &'static FrameDef,
    pub occurrence: u32,
    pub fetch_route: String,
}

pub fn requests(
    query: &ScoredPostsQuery,
    include: impl Fn(&FrameDef) -> bool,
) -> Vec<FrameRequest> {
    let mut wanted: Vec<(&'static FrameDef, u32)> = Vec::new();
    for frame in eligible(query).filter(|frame| include(frame)) {
        for occurrence in 0..frame.placement.occurrences(MAX_ORGANIC_FEED_LEN) {
            wanted.push((frame, occurrence as u32));
        }
    }
    wanted.sort_by_key(|(frame, occurrence)| {
        let recurring = !matches!(frame.placement, Placement::Pinned { .. });
        (recurring, *occurrence)
    });
    wanted.truncate(MAX_JETFUEL_FRAMES_PER_RESPONSE);

    wanted
        .into_iter()
        .map(|(frame, occurrence)| {
            let fetch_route = match frame.placement {
                Placement::EveryNItems { rotating: true, .. } => {
                    let next_card = resume_index(&query.served_history, frame.route)
                        .map_or(0, |last_served| last_served.wrapping_add(1));
                    format!("{}?i={}", frame.route, next_card.wrapping_add(occurrence))
                }
                _ => frame.route.to_string(),
            };
            FrameRequest {
                frame,
                occurrence,
                fetch_route,
            }
        })
        .collect()
}

pub fn collect(
    component: &str,
    fetched: Vec<(FrameRequest, FetchedPayload)>,
) -> Result<Vec<FeedItem>, String> {
    let mut items = Vec::new();
    let mut failed_routes = Vec::new();
    for (request, payload) in fetched {
        match payload {
            Ok(Some((payload, height))) if !payload.is_empty() => items.push(FeedItem {
                position: 0,
                item: Some(feed_item::Item::Frame(Frame {
                    payload,
                    height,
                    is_dismissible: request.frame.is_dismissible,
                    placement_id: request.frame.placement_id,
                    route: request.frame.route.to_string(),
                    occurrence: request.occurrence,
                    fetch_route: request.fetch_route,
                })),
            }),
            Ok(_) => {}
            Err(_) => failed_routes.push(request.frame.route),
        }
    }

    if failed_routes.is_empty() {
        return Ok(items);
    }
    if let Some(receiver) = global_stats_receiver() {
        let metric = format!("{component}.payload_failed");
        for route in &failed_routes {
            receiver.incr(&metric, &[("route", route)], 1);
        }
    }
    if items.is_empty() {
        return Err(format!(
            "jetfuel payload fetch failed for {}",
            failed_routes.join(", ")
        ));
    }
    warn!(
        routes = failed_routes.join(", "),
        "{component}: some jetfuel payloads failed, serving the rest"
    );
    Ok(items)
}
