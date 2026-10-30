use crate::schema::{events::Event, tweet_events::TweetEvent};
use anyhow::{Context, Result};
use prost::Message;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};
use thrift::TConfiguration;
use xai_thunder_proto::InNetworkEvent;

const MAX_THRIFT_CONTAINER_ELEMENTS: usize = 100_000;

fn thrift_config(payload_len: usize) -> Result<TConfiguration> {
    TConfiguration::builder()
        .max_message_size(Some(payload_len))
        .max_frame_size(None)
        .max_container_size(Some(MAX_THRIFT_CONTAINER_ELEMENTS))
        .max_string_size(Some(payload_len))
        .build()
        .context("Failed to configure Thrift input limits")
}

pub fn deserialize_tweet_event(payload: &[u8]) -> Result<TweetEvent> {
    let mut cursor = std::io::Cursor::new(payload);
    let mut protocol =
        TBinaryInputProtocol::with_config(&mut cursor, true, thrift_config(payload.len())?);

    TweetEvent::read_from_in_protocol(&mut protocol).context("Failed to deserialize TweetEvent")
}

pub fn deserialize_event(payload: &[u8]) -> Result<Event> {
    let mut cursor = std::io::Cursor::new(payload);
    let mut protocol =
        TBinaryInputProtocol::with_config(&mut cursor, true, thrift_config(payload.len())?);

    Event::read_from_in_protocol(&mut protocol).context("Failed to deserialize Event")
}

pub fn deserialize_tweet_event_v2(payload: &[u8]) -> Result<InNetworkEvent> {
    InNetworkEvent::decode(payload).context("Failed to deserialize InNetworkEvent")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_container_length_before_preallocating_from_it() {
        let payload = [
            12, 0, 2, // TweetEvent.flags: struct
            15, 0, 1, // TweetEventFlags.unused1: list
            11, 0, 0x0f, 0x42, 0x40, // string list with 1,000,000 elements
        ];

        let error = deserialize_tweet_event(&payload).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to deserialize TweetEvent"),
            "unexpected error: {error:#}"
        );
        assert!(
            format!("{error:#}").contains("message too long"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn accepts_a_valid_string_list() {
        let payload = [
            12, 0, 2, // TweetEvent.flags: struct
            15, 0, 1, // TweetEventFlags.unused1: list
            11, 0, 0, 0, 1, // one string
            0, 0, 0, 2, b'o', b'k', // "ok"
            0,    // end TweetEventFlags
            0,    // end TweetEvent
        ];

        let event = deserialize_tweet_event(&payload).expect("valid TweetEvent");
        assert_eq!(
            event.flags.and_then(|flags| flags.unused1),
            Some(vec!["ok".to_owned()])
        );
    }
}
