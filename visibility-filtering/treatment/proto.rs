use crate::models::{Decided, Verdict, Withholding};
use xai_visibility_filtering::models::FilteredReason;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) fn proto_action(verdict: &Verdict) -> (vf_pb::Action, Option<vf_pb::FilteredReason>) {
    let (kind, filtered_reason) = match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(reason.legacy().clone().into()),
        ),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(_),
            ..
        }) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(FilteredReason::UnspecifiedReason.into()),
        ),
        Verdict::Shown {
            label_interstitial: None | Some(_),
            media: Some(Decided { value, .. }),
            engagement: None | Some(_),
        } => (
            vf_pb::action::Kind::Interstitial(true),
            Some(value.legacy().clone().into()),
        ),
        Verdict::Fosnr(_)
        | Verdict::Shown {
            label_interstitial: None | Some(_),
            media: None,
            engagement: None | Some(_),
        } => (vf_pb::action::Kind::Allow(true), None),
    };
    (vf_pb::Action { kind: Some(kind) }, filtered_reason)
}
