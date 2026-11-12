use serde::{Deserialize, Serialize};
use thrift::protocol::{TInputProtocol, TOutputProtocol, TType};
use xai_strato::MValCodec;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct KeywordMatch {
    pub keyword: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, strum::FromRepr)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyResultReason {
    Episodic = 1,
    AbuseEpisodicEncourageSelfHarm = 2,
    AbuseEpisodicHatefulConduct = 3,
    AbuseGlorificationOfViolence = 4,
    AbuseGratuitousGore = 5,
    AbuseMobHarassment = 6,
    AbuseMomentOfDeathOrDeceasedUser = 7,
    AbusePrivateInformation = 8,
    AbuseRightToPrivacy = 9,
    AbuseThreatToExpose = 10,
    AbuseViolentSexualConduct = 11,
    AbuseViolentThreatHatefulConduct = 12,
    AbuseViolentThreatOrBounty = 13,
    OneOff = 14,
    VotingMisinformation = 15,
    HackedMaterials = 16,
    Scams = 17,
    PlatformManipulation = 18,
    MisinfoCivic = 19,
    MisinfoCovid19 = 20,
    MisinfoGeneric = 21,
    MisinfoMedical = 22,
    MisinfoCrisis = 23,
    DevelopmentOnlyPublicInterest = 24,
    FosnrHatefulConduct = 25,
    FosnrAbuse = 26,
    FosnrUnspecified = 27,
    FosnrViolentSpeech = 28,
    FosnrCivicIntegrity = 29,
    FosnrSyntheticAndManipulatedMedia = 30,
    NsfwHighPrecision = 31,
    GoreAndViolenceHighPrecision = 32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct SafetyResult {
    pub reason: Option<SafetyResultReason>,
    pub action: Action,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum Action {
    #[default]
    NotEvaluated,
    Allow,
    Drop(DropReason),
    Interstitial,
    Downrank,
    Tombstone,
    Avoid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DropReason {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum FilteredReason {
    ContainNsfwMedia,
    AuthorBlockViewer,
    PossiblyUndesirable,
    UnspecifiedReason,
    AuthorAccountIsInactive,
    AuthorIsProtected,
    AuthorIsUnsafe,
    ReportedTweet,
    TweetMatchesViewerMutedKeyword(KeywordMatch),
    TweetIsBounced,
    SafetyResult(SafetyResult),
    AuthorIsDeactivated,
    AuthorIsSuspended,
    ViewerMutesAuthor,
    TweetIsNullcast,
    ExclusiveTweet,
    ViewerBlocksAuthor,
}

impl MValCodec for KeywordMatch {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut keyword = String::new();
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(1) => {
                    keyword = proto.read_string().unwrap();
                }
                _ => {
                    proto.skip(field.field_type).unwrap();
                }
            }
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        KeywordMatch { keyword }
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for KeywordMatch")
    }
}

impl MValCodec for SafetyResult {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut reason: Option<SafetyResultReason> = None;
        let mut action = Action::default();
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(1) => {
                    let reason_value = usize::try_from(proto.read_i32().unwrap()).ok();
                    reason = Some(
                        reason_value
                            .and_then(SafetyResultReason::from_repr)
                            .unwrap_or(SafetyResultReason::Episodic),
                    );
                }
                Some(2) => {
                    action = Action::from_thrift(proto);
                }
                _ => {
                    proto.skip(field.field_type).unwrap();
                }
            }
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        SafetyResult { reason, action }
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for SafetyResult")
    }
}

impl MValCodec for Action {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut result = Action::NotEvaluated;
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(1) => {
                    proto.skip(field.field_type).unwrap();
                    result = Action::NotEvaluated;
                }
                Some(2) => {
                    proto.skip(field.field_type).unwrap();
                    result = Action::Allow;
                }
                Some(3) => {
                    result = Action::Drop(DropReason::from_thrift(proto));
                }
                Some(15) => {
                    proto.skip(field.field_type).unwrap();
                    result = Action::Avoid;
                }
                Some(13) => {
                    proto.read_struct_begin().unwrap();
                    let mut has_avoid = false;
                    loop {
                        let sub_field = proto.read_field_begin().unwrap();
                        if sub_field.field_type == TType::Stop {
                            break;
                        }
                        if sub_field.id == Some(5) {
                            has_avoid = true;
                        }
                        proto.skip(sub_field.field_type).unwrap();
                        proto.read_field_end().unwrap();
                    }
                    proto.read_struct_end().unwrap();
                    if has_avoid {
                        result = Action::Avoid;
                    }
                }
                _ => {
                    proto.skip(field.field_type).unwrap();
                }
            }
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        result
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for Action")
    }
}

impl MValCodec for DropReason {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            proto.skip(field.field_type).unwrap();
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        DropReason {}
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for DropReason")
    }
}

impl FilteredReason {
    fn of_bool_field(id: i16) -> Option<Self> {
        Some(match id {
            1 => Self::ContainNsfwMedia,
            2 => Self::AuthorBlockViewer,
            3 => Self::PossiblyUndesirable,
            4 => Self::UnspecifiedReason,
            5 => Self::AuthorAccountIsInactive,
            6 => Self::AuthorIsProtected,
            7 => Self::AuthorIsUnsafe,
            8 => Self::ReportedTweet,
            10 => Self::TweetIsBounced,
            12 => Self::AuthorIsDeactivated,
            13 => Self::AuthorIsSuspended,
            14 => Self::ViewerMutesAuthor,
            15 => Self::TweetIsNullcast,
            16 => Self::ExclusiveTweet,
            _ => return None,
        })
    }
}

impl MValCodec for FilteredReason {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut result = FilteredReason::UnspecifiedReason;
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            result = match field.id {
                Some(9) => {
                    FilteredReason::TweetMatchesViewerMutedKeyword(KeywordMatch::from_thrift(proto))
                }
                Some(11) => FilteredReason::SafetyResult(SafetyResult::from_thrift(proto)),
                id => match id.and_then(FilteredReason::of_bool_field) {
                    Some(reason) => {
                        proto.read_bool().unwrap();
                        reason
                    }
                    None => {
                        proto.skip(field.field_type).unwrap();
                        result
                    }
                },
            };
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        result
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for FilteredReason")
    }
}

impl From<vf_pb::KeywordMatch> for KeywordMatch {
    fn from(value: vf_pb::KeywordMatch) -> Self {
        KeywordMatch {
            keyword: value.keyword,
        }
    }
}

impl From<KeywordMatch> for vf_pb::KeywordMatch {
    fn from(value: KeywordMatch) -> Self {
        vf_pb::KeywordMatch {
            keyword: value.keyword,
        }
    }
}

impl From<vf_pb::DropReason> for DropReason {
    fn from(_value: vf_pb::DropReason) -> Self {
        DropReason {}
    }
}

impl From<DropReason> for vf_pb::DropReason {
    fn from(_value: DropReason) -> Self {
        vf_pb::DropReason {}
    }
}

impl From<vf_pb::Action> for Action {
    fn from(value: vf_pb::Action) -> Self {
        match value.kind {
            Some(vf_pb::action::Kind::NotEvaluated(_)) => Action::NotEvaluated,
            Some(vf_pb::action::Kind::Allow(_)) => Action::Allow,
            Some(vf_pb::action::Kind::Drop(reason)) => Action::Drop(reason.into()),
            Some(vf_pb::action::Kind::Interstitial(_)) => Action::Interstitial,
            Some(vf_pb::action::Kind::Downrank(_)) => Action::Downrank,
            Some(vf_pb::action::Kind::Tombstone(_)) => Action::Tombstone,
            Some(vf_pb::action::Kind::Avoid(_)) => Action::Avoid,
            None => Action::NotEvaluated,
        }
    }
}

impl From<Action> for vf_pb::Action {
    fn from(value: Action) -> Self {
        let kind = match value {
            Action::NotEvaluated => Some(vf_pb::action::Kind::NotEvaluated(true)),
            Action::Allow => Some(vf_pb::action::Kind::Allow(true)),
            Action::Drop(reason) => Some(vf_pb::action::Kind::Drop(reason.into())),
            Action::Interstitial => Some(vf_pb::action::Kind::Interstitial(true)),
            Action::Downrank => Some(vf_pb::action::Kind::Downrank(true)),
            Action::Tombstone => Some(vf_pb::action::Kind::Tombstone(true)),
            Action::Avoid => Some(vf_pb::action::Kind::Avoid(true)),
        };
        vf_pb::Action { kind }
    }
}

impl From<vf_pb::SafetyResult> for SafetyResult {
    fn from(value: vf_pb::SafetyResult) -> Self {
        let action = value.action.map(Action::from).unwrap_or_default();
        SafetyResult {
            reason: None,
            action,
        }
    }
}

impl From<SafetyResult> for vf_pb::SafetyResult {
    fn from(value: SafetyResult) -> Self {
        vf_pb::SafetyResult {
            action: Some(value.action.into()),
        }
    }
}

impl From<vf_pb::FilteredReason> for FilteredReason {
    fn from(value: vf_pb::FilteredReason) -> Self {
        match value.reason {
            Some(vf_pb::filtered_reason::Reason::ContainNsfwMedia(_)) => {
                FilteredReason::ContainNsfwMedia
            }
            Some(vf_pb::filtered_reason::Reason::AuthorBlockViewer(_)) => {
                FilteredReason::AuthorBlockViewer
            }
            Some(vf_pb::filtered_reason::Reason::PossiblyUndesirable(_)) => {
                FilteredReason::PossiblyUndesirable
            }
            Some(vf_pb::filtered_reason::Reason::UnspecifiedReason(_)) => {
                FilteredReason::UnspecifiedReason
            }
            Some(vf_pb::filtered_reason::Reason::AuthorAccountIsInactive(_)) => {
                FilteredReason::AuthorAccountIsInactive
            }
            Some(vf_pb::filtered_reason::Reason::AuthorIsProtected(_)) => {
                FilteredReason::AuthorIsProtected
            }
            Some(vf_pb::filtered_reason::Reason::AuthorIsUnsafe(_)) => {
                FilteredReason::AuthorIsUnsafe
            }
            Some(vf_pb::filtered_reason::Reason::ReportedTweet(_)) => FilteredReason::ReportedTweet,
            Some(vf_pb::filtered_reason::Reason::TweetMatchesViewerMutedKeyword(keyword_match)) => {
                FilteredReason::TweetMatchesViewerMutedKeyword(keyword_match.into())
            }
            Some(vf_pb::filtered_reason::Reason::TweetIsBounced(_)) => {
                FilteredReason::TweetIsBounced
            }
            Some(vf_pb::filtered_reason::Reason::SafetyResult(safety_result)) => {
                FilteredReason::SafetyResult(safety_result.into())
            }
            Some(vf_pb::filtered_reason::Reason::AuthorIsDeactivated(_)) => {
                FilteredReason::AuthorIsDeactivated
            }
            Some(vf_pb::filtered_reason::Reason::AuthorIsSuspended(_)) => {
                FilteredReason::AuthorIsSuspended
            }
            Some(vf_pb::filtered_reason::Reason::ViewerMutesAuthor(_)) => {
                FilteredReason::ViewerMutesAuthor
            }
            Some(vf_pb::filtered_reason::Reason::TweetIsNullcast(_)) => {
                FilteredReason::TweetIsNullcast
            }
            Some(vf_pb::filtered_reason::Reason::ExclusiveTweet(_)) => {
                FilteredReason::ExclusiveTweet
            }
            Some(vf_pb::filtered_reason::Reason::ViewerBlocksAuthor(_)) => {
                FilteredReason::ViewerBlocksAuthor
            }
            None => FilteredReason::UnspecifiedReason,
        }
    }
}

impl From<FilteredReason> for vf_pb::FilteredReason {
    fn from(value: FilteredReason) -> Self {
        let reason = match value {
            FilteredReason::ContainNsfwMedia => {
                Some(vf_pb::filtered_reason::Reason::ContainNsfwMedia(true))
            }
            FilteredReason::AuthorBlockViewer => {
                Some(vf_pb::filtered_reason::Reason::AuthorBlockViewer(true))
            }
            FilteredReason::PossiblyUndesirable => {
                Some(vf_pb::filtered_reason::Reason::PossiblyUndesirable(true))
            }
            FilteredReason::UnspecifiedReason => {
                Some(vf_pb::filtered_reason::Reason::UnspecifiedReason(true))
            }
            FilteredReason::AuthorAccountIsInactive => Some(
                vf_pb::filtered_reason::Reason::AuthorAccountIsInactive(true),
            ),
            FilteredReason::AuthorIsProtected => {
                Some(vf_pb::filtered_reason::Reason::AuthorIsProtected(true))
            }
            FilteredReason::AuthorIsUnsafe => {
                Some(vf_pb::filtered_reason::Reason::AuthorIsUnsafe(true))
            }
            FilteredReason::ReportedTweet => {
                Some(vf_pb::filtered_reason::Reason::ReportedTweet(true))
            }
            FilteredReason::TweetMatchesViewerMutedKeyword(keyword_match) => Some(
                vf_pb::filtered_reason::Reason::TweetMatchesViewerMutedKeyword(
                    keyword_match.into(),
                ),
            ),
            FilteredReason::TweetIsBounced => {
                Some(vf_pb::filtered_reason::Reason::TweetIsBounced(true))
            }
            FilteredReason::SafetyResult(safety_result) => Some(
                vf_pb::filtered_reason::Reason::SafetyResult(safety_result.into()),
            ),
            FilteredReason::AuthorIsDeactivated => {
                Some(vf_pb::filtered_reason::Reason::AuthorIsDeactivated(true))
            }
            FilteredReason::AuthorIsSuspended => {
                Some(vf_pb::filtered_reason::Reason::AuthorIsSuspended(true))
            }
            FilteredReason::ViewerMutesAuthor => {
                Some(vf_pb::filtered_reason::Reason::ViewerMutesAuthor(true))
            }
            FilteredReason::TweetIsNullcast => {
                Some(vf_pb::filtered_reason::Reason::TweetIsNullcast(true))
            }
            FilteredReason::ExclusiveTweet => {
                Some(vf_pb::filtered_reason::Reason::ExclusiveTweet(true))
            }
            FilteredReason::ViewerBlocksAuthor => {
                Some(vf_pb::filtered_reason::Reason::ViewerBlocksAuthor(true))
            }
        };
        vf_pb::FilteredReason { reason }
    }
}

#[cfg(test)]
mod proto_conversion_tests {
    use super::*;

    #[test]
    fn roundtrip_keyword_match_reason() {
        let model_reason = FilteredReason::TweetMatchesViewerMutedKeyword(KeywordMatch {
            keyword: "spoilers".to_string(),
        });
        let proto_reason: vf_pb::FilteredReason = model_reason.clone().into();
        let back_to_model: FilteredReason = proto_reason.into();
        assert_eq!(model_reason, back_to_model);
    }

    #[test]
    fn map_safety_result_with_action() {
        let model_safety = SafetyResult {
            reason: Some(SafetyResultReason::MisinfoGeneric),
            action: Action::Drop(DropReason {}),
        };
        let proto: vf_pb::SafetyResult = model_safety.clone().into();
        let back: SafetyResult = proto.into();
        assert!(back.reason.is_none());
        assert!(matches!(back.action, Action::Drop(_)));
    }
}
