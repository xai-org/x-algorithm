use crate::hydration::RequestRows;
use crate::models::NotificationKind;
use crate::rules::levels;
use crate::rules::levels::PostRules;

#[derive(Clone, Copy)]
pub(crate) enum Judged {
    Actor,
    ActorPost(PostRules),
}

impl Judged {
    pub(crate) const fn rows(self) -> RequestRows {
        match self {
            Self::Actor => RequestRows::Users,
            Self::ActorPost(_) => RequestRows::ActorPosts,
        }
    }
}

pub(crate) const fn judged(kind: NotificationKind) -> Judged {
    match kind {
        NotificationKind::Follow | NotificationKind::FollowRequestAccepted => Judged::Actor,
        NotificationKind::Mention => Judged::ActorPost(levels::NOTIFICATIONS_WRITER_V2_MENTION),
        NotificationKind::Reply => Judged::ActorPost(levels::NOTIFICATIONS_WRITER_V2_REPLY),
        NotificationKind::MediaTagged => {
            Judged::ActorPost(levels::NOTIFICATIONS_WRITER_V2_MEDIA_TAGGED)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::SafetyLevel;
    use strum::VariantArray;

    #[test]
    fn each_kind_names_a_served_level() {
        for &kind in NotificationKind::VARIANTS {
            let judging = SafetyLevel::VARIANTS
                .iter()
                .filter(|level| level.judges(kind));
            assert_eq!(judging.count(), 1, "{kind:?}");
        }
    }
}
