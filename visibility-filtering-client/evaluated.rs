use anyhow::{bail, ensure, Result};
use thrift::protocol::{TCompactInputProtocol, TSerializable};
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::action::Action;
use xai_x_thrift::safety_result::FilteredReason;
use xai_x_thrift::tweet_service::TweetFieldsResultState;

pub const MAX_RESULT_STATE_BYTES: usize = 64 * 1024;

#[derive(Debug, PartialEq)]
pub enum EvaluationResult {
    Evaluated(Box<TweetFieldsResultState>),
    NotEvaluated,
    Failed,
}

impl EvaluationResult {
    pub(crate) fn decode(outcome: Option<vf_pb::tweet_evaluation::Outcome>) -> Result<Self> {
        use vf_pb::tweet_evaluation::Outcome;
        Ok(match outcome {
            Some(Outcome::ResultStateThriftCompact(bytes)) => {
                Self::Evaluated(Box::new(decode_result_state(&bytes)?))
            }
            Some(Outcome::NotEvaluated(_)) => Self::NotEvaluated,
            Some(Outcome::Failed(_)) => Self::Failed,
            None => bail!("missing VF outcome"),
        })
    }

    pub fn into_result_state(
        self,
        fetched: TweetFieldsResultState,
    ) -> Option<TweetFieldsResultState> {
        match fetched {
            TweetFieldsResultState::Found(_) => match self {
                Self::Evaluated(state) => Some(*state),
                Self::NotEvaluated | Self::Failed => None,
            },
            kept @ (TweetFieldsResultState::Filtered(_)
            | TweetFieldsResultState::NotFound(_)
            | TweetFieldsResultState::Failed(_)) => Some(kept),
        }
    }
}

fn decode_result_state(bytes: &[u8]) -> Result<TweetFieldsResultState> {
    ensure!(
        bytes.len() <= MAX_RESULT_STATE_BYTES,
        "result state byte limit exceeded"
    );
    let element_size = size_of::<xai_x_thrift::action::MessageLink>()
        .max(size_of::<xai_x_thrift::action::LimitedAction>())
        .max(size_of::<xai_x_thrift::action::TweetVisibilityNudgeAction>());
    let config = thrift::TConfiguration::builder()
        .max_string_size(Some(MAX_RESULT_STATE_BYTES))
        .max_message_size(Some(MAX_RESULT_STATE_BYTES))
        .max_frame_size(Some(MAX_RESULT_STATE_BYTES))
        .max_container_size(Some(MAX_RESULT_STATE_BYTES / element_size))
        .build()?;
    let mut cursor = std::io::Cursor::new(bytes);
    let state = TweetFieldsResultState::read_from_in_protocol(
        &mut TCompactInputProtocol::with_config(&mut cursor, config),
    )?;
    ensure!(
        cursor.position() == bytes.len() as u64,
        "trailing result state bytes"
    );
    let reason = match &state {
        TweetFieldsResultState::Found(found) => found.suppress_reason.as_ref(),
        TweetFieldsResultState::Filtered(filtered) => Some(&filtered.reason),
        TweetFieldsResultState::NotFound(not_found) => not_found.filtered_reason.as_ref(),
        TweetFieldsResultState::Failed(_) => bail!("VF sends no Thrift failed state"),
    };
    ensure!(
        !matches!(reason, Some(FilteredReason::SafetyResult(result)) if matches!(result.action, Action::NotEvaluated(_))),
        "not an evaluated action"
    );
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_x_thrift::action;
    use xai_x_thrift::safety_result::SafetyResult;
    use xai_x_thrift::tweet_service::{
        TweetFieldsResultFailed, TweetFieldsResultFiltered, TweetFieldsResultFound,
        TweetFieldsResultNotFound,
    };

    fn safety_result(action: Action) -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult::new(None, action))
    }

    fn filtered(reason: FilteredReason) -> TweetFieldsResultState {
        TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason))
    }

    #[test]
    fn declared_lengths_are_rejected_before_reading_payloads() {
        let safety_result_action = [0x4c, 0x1c, 0xbc, 0x2c];
        for bytes in [
            vec![0x4c, 0x1c, 0x9c, 0x18, 0xf8, 0xa0, 0x8d, 0x06],
            [
                &safety_result_action[..],
                &[0x3c, 0x29, 0xf8, 0xa0, 0x8d, 0x06],
            ]
            .concat(),
            [
                &safety_result_action[..],
                &[0x2c, 0x19, 0xf8, 0xa0, 0x8d, 0x06],
            ]
            .concat(),
        ] {
            let error = decode_result_state(&bytes).unwrap_err();
            assert!(
                matches!(
                    error.downcast_ref::<thrift::Error>(),
                    Some(thrift::Error::Protocol(thrift::ProtocolError {
                        kind: thrift::ProtocolErrorKind::SizeLimit,
                        ..
                    }))
                ),
                "expected a size limit before payload read: {error:?}"
            );
        }
    }

    #[test]
    fn decode_result_state_roundtrips_found_filtered_and_not_found_and_rejects_the_rest() {
        let drop = safety_result(Action::Drop(action::Drop::new(
            Some(action::DropReason::LegalDemandsWithheld(true)),
            Some(vec!["US".into(), "DE".into()]),
        )));
        for state in [
            TweetFieldsResultState::Found(TweetFieldsResultFound::new(None)),
            filtered(drop),
            filtered(FilteredReason::TweetIsBounced(true)),
            TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(false, false, None)),
        ] {
            let bytes = xai_x_thrift::serialize_compact(&state).unwrap();
            assert_eq!(decode_result_state(&bytes).unwrap(), state);
        }
        let encode =
            |state: &TweetFieldsResultState| xai_x_thrift::serialize_compact(state).unwrap();
        let mut trailing = encode(&filtered(FilteredReason::TweetIsBounced(true)));
        trailing.push(0);
        for bytes in [
            vec![],
            vec![0x4c],
            trailing,
            encode(&filtered(safety_result(Action::NotEvaluated(
                action::NotEvaluated::new(),
            )))),
            encode(&TweetFieldsResultState::Failed(
                TweetFieldsResultFailed::new(false, None),
            )),
            vec![0; MAX_RESULT_STATE_BYTES + 1],
        ] {
            assert!(decode_result_state(&bytes).is_err(), "accepted {bytes:?}");
        }
    }
}
