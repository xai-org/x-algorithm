use crate::models::VerdictAction;
use crate::rules::defaults::{Bucket, Effect};

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
pub(crate) enum ActionKind {
    Appealable,
    Drop,
    Tombstone,
    LabelInterstitial,
    MediaRestriction,
    SoftIntervention,
    LimitedEngagement,
}

#[derive(Clone, Copy)]
enum Lands {
    Withheld,
    Fosnr,
    Slot(Slot),
}

#[derive(Clone, Copy)]
enum Slot {
    LabelInterstitial,
    Media,
    Engagement,
}

#[derive(Clone, Copy)]
pub(super) struct Filled {
    pub(super) label_interstitial: bool,
    pub(super) media: bool,
    pub(super) engagement: bool,
}

impl Filled {
    const fn has(self, slot: Slot) -> bool {
        match slot {
            Slot::LabelInterstitial => self.label_interstitial,
            Slot::Media => self.media,
            Slot::Engagement => self.engagement,
        }
    }
}

struct Row {
    severity: u8,
    lands: Lands,
    kept_by: &'static [(Slot, Effect)],
    bucket: Bucket,
    verdict_action: VerdictAction,
    #[cfg(test)]
    catalog: (&'static str, &'static str),
}

impl ActionKind {
    const fn row(self) -> Row {
        match self {
            Self::Appealable => Row {
                severity: 18,
                lands: Lands::Fosnr,
                kept_by: &[],
                bucket: Bucket::Withholding,
                verdict_action: VerdictAction::Appealable,
                #[cfg(test)]
                catalog: ("appealable", "fosnr"),
            },
            Self::Drop => Row {
                severity: 17,
                lands: Lands::Withheld,
                kept_by: &[],
                bucket: Bucket::Withholding,
                verdict_action: VerdictAction::Drop,
                #[cfg(test)]
                catalog: ("drop", "withholding"),
            },
            Self::Tombstone => Row {
                severity: 16,
                lands: Lands::Withheld,
                kept_by: &[],
                bucket: Bucket::Withholding,
                verdict_action: VerdictAction::Tombstone,
                #[cfg(test)]
                catalog: ("tombstone", "withholding"),
            },
            Self::LabelInterstitial => Row {
                severity: 11,
                lands: Lands::Slot(Slot::LabelInterstitial),
                kept_by: &[(
                    Slot::LabelInterstitial,
                    Effect::KeptEarlierLabelInterstitial,
                )],
                bucket: Bucket::Media,
                verdict_action: VerdictAction::Interstitial,
                #[cfg(test)]
                catalog: ("label interstitial", "label-interstitial"),
            },
            Self::MediaRestriction => Row {
                severity: 10,
                lands: Lands::Slot(Slot::Media),
                kept_by: &[(Slot::Media, Effect::KeptEarlierMedia)],
                bucket: Bucket::Media,
                verdict_action: VerdictAction::Interstitial,
                #[cfg(test)]
                catalog: ("media restriction", "media"),
            },
            Self::SoftIntervention => Row {
                severity: 8,
                lands: Lands::Fosnr,
                kept_by: &[
                    (
                        Slot::LabelInterstitial,
                        Effect::KeptEarlierLabelInterstitial,
                    ),
                    (Slot::Media, Effect::KeptEarlierMediaInsteadOfFosnr),
                ],
                bucket: Bucket::Media,
                verdict_action: VerdictAction::SoftIntervention,
                #[cfg(test)]
                catalog: ("soft intervention", "fosnr"),
            },
            Self::LimitedEngagement => Row {
                severity: 6,
                lands: Lands::Slot(Slot::Engagement),
                kept_by: &[],
                bucket: Bucket::Engagement,
                verdict_action: VerdictAction::LimitedEngagement,
                #[cfg(test)]
                catalog: ("limited engagement", "engagement"),
            },
        }
    }

    pub(super) const fn severity(self) -> u8 {
        self.row().severity
    }

    pub(super) fn kept_by(self, filled: Filled) -> Option<Effect> {
        self.row()
            .kept_by
            .iter()
            .find(|&&(slot, _)| filled.has(slot))
            .map(|&(_, effect)| effect)
    }

    pub(super) fn admits_defaults(self, filled: Filled) -> bool {
        let own_slot_filled = match self.row().lands {
            Lands::Slot(slot) => filled.has(slot),
            Lands::Withheld | Lands::Fosnr => false,
        };
        !own_slot_filled && self.kept_by(filled).is_none()
    }

    pub(super) const fn withholds(self) -> bool {
        match self.row().lands {
            Lands::Withheld => true,
            Lands::Fosnr | Lands::Slot(_) => false,
        }
    }

    pub(super) const fn bucket(self) -> Bucket {
        self.row().bucket
    }

    pub(crate) const fn verdict_action(self) -> VerdictAction {
        self.row().verdict_action
    }

    #[cfg(test)]
    pub(super) const fn ends_the_fold(self) -> bool {
        match self.row().lands {
            Lands::Withheld | Lands::Fosnr => true,
            Lands::Slot(_) => false,
        }
    }

    #[cfg(test)]
    pub(super) const fn lands_in_fosnr(self) -> bool {
        match self.row().lands {
            Lands::Fosnr => true,
            Lands::Withheld | Lands::Slot(_) => false,
        }
    }

    #[cfg(test)]
    pub(super) const fn catalog(self) -> (&'static str, &'static str) {
        self.row().catalog
    }
}
