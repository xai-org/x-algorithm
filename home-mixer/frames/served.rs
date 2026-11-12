use xai_x_thrift::served_history::{EntityIdType, ServedHistory};

pub const FRAME_ENTITY_TYPE: EntityIdType = EntityIdType::ENTRY_POINT_PIVOT;

pub fn resume_index(history: &[ServedHistory], route: &str) -> Option<u32> {
    history
        .iter()
        .flat_map(|served| served.entries.iter())
        .filter(|entry| entry.entity_type == FRAME_ENTITY_TYPE)
        .flat_map(|entry| entry.item_ids.iter().flatten())
        .filter_map(|ids| ids.impression_id.as_deref())
        .filter_map(split_fetch_route)
        .filter(|(served_route, _)| *served_route == route)
        .map(|(_, index)| index)
        .max()
}

fn split_fetch_route(fetch_route: &str) -> Option<(&str, u32)> {
    let (route, query) = fetch_route.split_once('?')?;
    let index = query.strip_prefix("i=")?.parse().ok()?;
    Some((route, index))
}
