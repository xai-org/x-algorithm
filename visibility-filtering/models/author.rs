#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorFeatures {
    pub is_suspended: bool,
    pub is_deactivated: bool,
    pub is_protected: bool,
    pub is_nsfw_user: bool,
    pub is_nsfw_admin: bool,
    pub is_erased: bool,
    pub is_offboarded: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorProfile {
    pub has_default_profile_image: bool,
    pub is_new_account: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
#[repr(u8)]
pub enum AuthorLabel {
    NsfwHighRecall,
    NsfwHighPrecision,
    NsfwNearPerfect,
    NsfwAvatarImage,
    NsfwBannerImage,
    SpamHighRecall,
    AbusiveHighRecall,
    Compromised,
    ReadOnly,
    ImpersonationHighPrecision,
    DoNotAmplify,
    NotGraduated,
    AgathaSpamTopUser,
    EngagementSpammer,
    BlinkWorst,
    DownrankSpamReply,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorLabelSet(u64);

impl AuthorLabelSet {
    #[inline]
    pub fn insert(&mut self, label: AuthorLabel) {
        self.0 |= 1 << label as u8;
    }

    #[inline]
    pub fn has_label(self, label: AuthorLabel) -> bool {
        self.0 & (1 << label as u8) != 0
    }
}
