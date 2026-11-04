use crate::hydration::Hydrators;
use crate::models::{Notice, Verdict};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    DidNotApply,
    Decided,
    AfterTerminal,
    FilledMedia,
    KeptEarlierMedia,
    KeptEarlierMediaInsteadOfFosnr,
    FilledLabelInterstitial,
    KeptEarlierLabelInterstitial,
    FilledEngagement,
    AddedEngagementReason,
    DuplicateEngagementReason,
}

#[derive(Clone, Copy)]
pub(super) enum Bucket {
    Withholding,
    Media,
    Engagement,
}

impl Bucket {
    fn relied_on_by(self, verdict: &Verdict) -> bool {
        let notice = match verdict {
            Verdict::Withheld(_) => return matches!(self, Self::Withholding),
            Verdict::Fosnr(notice) => &notice.value,
            Verdict::Shown { .. } => return true,
        };
        match self {
            Self::Withholding => true,
            Self::Media => match notice {
                Notice::SoftIntervention(_) => true,
                Notice::Appealable { .. } => false,
            },
            Self::Engagement => false,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Defaults {
    withholding: Hydrators,
    media: Hydrators,
    engagement: Hydrators,
}

impl Defaults {
    pub(super) fn get(self, bucket: Bucket) -> Hydrators {
        match bucket {
            Bucket::Withholding => self.withholding,
            Bucket::Media => self.media,
            Bucket::Engagement => self.engagement,
        }
    }

    pub(super) fn admit(&mut self, bucket: Bucket, reads: Hydrators) {
        let admitted = match bucket {
            Bucket::Withholding => &mut self.withholding,
            Bucket::Media => &mut self.media,
            Bucket::Engagement => &mut self.engagement,
        };
        *admitted = admitted.union(reads);
    }

    pub(super) fn relied_on_by(self, verdict: &Verdict) -> Self {
        let relied = |bucket: Bucket| {
            if bucket.relied_on_by(verdict) {
                self.get(bucket)
            } else {
                Hydrators::empty()
            }
        };
        Self {
            withholding: relied(Bucket::Withholding),
            media: relied(Bucket::Media),
            engagement: relied(Bucket::Engagement),
        }
    }

    pub(super) fn all(self) -> Hydrators {
        self.withholding.union(self.media).union(self.engagement)
    }
}
