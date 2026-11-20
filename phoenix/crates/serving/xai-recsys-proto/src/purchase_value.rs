// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::{ContinuousActionName, PredictNextActionsRequest, PurchaseValueBaseline};

pub const ACTION_INDEX: usize = ContinuousActionName::AdsWebCtPurchaseValue as usize;

pub const MAX_EVENT_USD: f64 = 100_000.0;

pub fn is_countable_usd(usd: f64) -> bool {
    usd.is_finite() && usd > 0.0 && usd <= MAX_EVENT_USD
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueEvent {
    Purchase,
    ContentView,
    AddToCart,
}

impl ValueEvent {
    pub const ALL: [ValueEvent; 3] = [Self::Purchase, Self::ContentView, Self::AddToCart];

    pub const fn conversion_type(self) -> i32 {
        match self {
            Self::Purchase => 2,
            Self::ContentView => 19,
            Self::AddToCart => 12,
        }
    }

    pub const fn clickhouse_name(self) -> &'static str {
        match self {
            Self::Purchase => "PURCHASE",
            Self::ContentView => "CONTENT_VIEW",
            Self::AddToCart => "ADD_TO_CART",
        }
    }

    pub const fn snake_name(self) -> &'static str {
        match self {
            Self::Purchase => "purchase",
            Self::ContentView => "content_view",
            Self::AddToCart => "add_to_cart",
        }
    }

    pub const fn camel_name(self) -> &'static str {
        match self {
            Self::Purchase => "Purchase",
            Self::ContentView => "ContentView",
            Self::AddToCart => "AddToCart",
        }
    }

    pub fn from_snake_name(name: &str) -> Option<ValueEvent> {
        Self::ALL
            .into_iter()
            .find(|event| event.snake_name() == name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueLevel {
    ClickThrough,
    ViewThrough,
}

impl ValueLevel {
    pub const fn clickhouse_name(self) -> &'static str {
        match self {
            Self::ClickThrough => "CLICK_THROUGH",
            Self::ViewThrough => "VIEW_THROUGH",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AppValueEvent {
    Purchase,
    AddToCart,
    LevelAchieved,
    TutorialComplete,
    SignUp,
}

impl AppValueEvent {
    pub const fn clickhouse_name(self) -> &'static str {
        match self {
            Self::Purchase => "PURCHASE",
            Self::AddToCart => "ADD_TO_CART",
            Self::LevelAchieved => "LEVEL_ACHIEVED",
            Self::TutorialComplete => "TUTORIAL_COMPLETE",
            Self::SignUp => "SIGN_UP",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueTarget {
    Web(ValueEvent, ValueLevel),
    App(AppValueEvent),
}

impl ValueTarget {
    pub const PURCHASE_CT: ValueTarget = Self::new(ValueEvent::Purchase, ValueLevel::ClickThrough);
    pub const PURCHASE_VT: ValueTarget = Self::new(ValueEvent::Purchase, ValueLevel::ViewThrough);

    pub const fn new(event: ValueEvent, level: ValueLevel) -> Self {
        Self::Web(event, level)
    }

    pub const ALL_TARGETS: [ValueTarget; 11] = [
        Self::PURCHASE_CT,
        Self::PURCHASE_VT,
        Self::new(ValueEvent::ContentView, ValueLevel::ClickThrough),
        Self::new(ValueEvent::AddToCart, ValueLevel::ClickThrough),
        Self::new(ValueEvent::ContentView, ValueLevel::ViewThrough),
        Self::new(ValueEvent::AddToCart, ValueLevel::ViewThrough),
        Self::App(AppValueEvent::Purchase),
        Self::App(AppValueEvent::AddToCart),
        Self::App(AppValueEvent::LevelAchieved),
        Self::App(AppValueEvent::TutorialComplete),
        Self::App(AppValueEvent::SignUp),
    ];

    pub const REQUIRED: [ValueTarget; 2] = [Self::PURCHASE_CT, Self::PURCHASE_VT];

    pub const fn target_definition(self) -> &'static str {
        match self {
            Self::Web(ValueEvent::Purchase, ValueLevel::ClickThrough) => "web_ct_purchase",
            Self::Web(ValueEvent::Purchase, ValueLevel::ViewThrough) => "web_vt_purchase",
            Self::Web(ValueEvent::ContentView, ValueLevel::ClickThrough) => "web_ct_content_view",
            Self::Web(ValueEvent::ContentView, ValueLevel::ViewThrough) => "web_vt_content_view",
            Self::Web(ValueEvent::AddToCart, ValueLevel::ClickThrough) => "web_ct_add_to_cart",
            Self::Web(ValueEvent::AddToCart, ValueLevel::ViewThrough) => "web_vt_add_to_cart",
            Self::App(AppValueEvent::Purchase) => "app_purchase",
            Self::App(AppValueEvent::AddToCart) => "app_add_to_cart",
            Self::App(AppValueEvent::LevelAchieved) => "app_level_achieved",
            Self::App(AppValueEvent::TutorialComplete) => "app_tutorial_complete",
            Self::App(AppValueEvent::SignUp) => "app_sign_up",
        }
    }

    pub fn from_target_definition(name: &str) -> Option<ValueTarget> {
        Self::ALL_TARGETS
            .into_iter()
            .find(|target| target.target_definition() == name)
    }

    pub fn mean(self, baseline: &PurchaseValueBaseline) -> Option<f64> {
        usable_mean(*self.field(baseline))
    }

    pub fn set_mean(self, baseline: &mut PurchaseValueBaseline, mean: f64) {
        *self.field_mut(baseline) = mean;
    }

    fn field(self, b: &PurchaseValueBaseline) -> &f64 {
        match self {
            Self::Web(ValueEvent::Purchase, ValueLevel::ClickThrough) => &b.mean_value_usd_28d,
            Self::Web(ValueEvent::Purchase, ValueLevel::ViewThrough) => &b.mean_value_usd_28d_vt,
            Self::Web(ValueEvent::ContentView, ValueLevel::ClickThrough) => {
                &b.mean_content_view_usd_28d
            }
            Self::Web(ValueEvent::ContentView, ValueLevel::ViewThrough) => {
                &b.mean_content_view_usd_28d_vt
            }
            Self::Web(ValueEvent::AddToCart, ValueLevel::ClickThrough) => {
                &b.mean_add_to_cart_usd_28d
            }
            Self::Web(ValueEvent::AddToCart, ValueLevel::ViewThrough) => {
                &b.mean_add_to_cart_usd_28d_vt
            }
            Self::App(AppValueEvent::Purchase) => &b.mean_app_purchase_usd_28d,
            Self::App(AppValueEvent::AddToCart) => &b.mean_app_add_to_cart_usd_28d,
            Self::App(AppValueEvent::LevelAchieved) => &b.mean_app_level_achieved_usd_28d,
            Self::App(AppValueEvent::TutorialComplete) => &b.mean_app_tutorial_complete_usd_28d,
            Self::App(AppValueEvent::SignUp) => &b.mean_app_sign_up_usd_28d,
        }
    }

    fn field_mut(self, b: &mut PurchaseValueBaseline) -> &mut f64 {
        match self {
            Self::Web(ValueEvent::Purchase, ValueLevel::ClickThrough) => &mut b.mean_value_usd_28d,
            Self::Web(ValueEvent::Purchase, ValueLevel::ViewThrough) => {
                &mut b.mean_value_usd_28d_vt
            }
            Self::Web(ValueEvent::ContentView, ValueLevel::ClickThrough) => {
                &mut b.mean_content_view_usd_28d
            }
            Self::Web(ValueEvent::ContentView, ValueLevel::ViewThrough) => {
                &mut b.mean_content_view_usd_28d_vt
            }
            Self::Web(ValueEvent::AddToCart, ValueLevel::ClickThrough) => {
                &mut b.mean_add_to_cart_usd_28d
            }
            Self::Web(ValueEvent::AddToCart, ValueLevel::ViewThrough) => {
                &mut b.mean_add_to_cart_usd_28d_vt
            }
            Self::App(AppValueEvent::Purchase) => &mut b.mean_app_purchase_usd_28d,
            Self::App(AppValueEvent::AddToCart) => &mut b.mean_app_add_to_cart_usd_28d,
            Self::App(AppValueEvent::LevelAchieved) => &mut b.mean_app_level_achieved_usd_28d,
            Self::App(AppValueEvent::TutorialComplete) => &mut b.mean_app_tutorial_complete_usd_28d,
            Self::App(AppValueEvent::SignUp) => &mut b.mean_app_sign_up_usd_28d,
        }
    }
}

fn usable_mean(mean: f64) -> Option<f64> {
    (mean.is_finite() && mean > 0.0).then_some(mean)
}

pub fn ct_mean(baseline: &PurchaseValueBaseline) -> Option<f64> {
    ValueTarget::PURCHASE_CT.mean(baseline)
}

pub fn is_valid_baseline(baseline: &PurchaseValueBaseline) -> bool {
    baseline.impression_id > 0
        && baseline.advertiser_account_id > 0
        && ValueTarget::ALL_TARGETS
            .iter()
            .any(|target| target.mean(baseline).is_some())
}

pub fn find_baseline(
    request: &PredictNextActionsRequest,
    impression_id: i64,
    advertiser_account_id: i64,
) -> Option<&PurchaseValueBaseline> {
    request.purchase_value_baselines.iter().find(|b| {
        b.impression_id == impression_id
            && b.advertiser_account_id == advertiser_account_id
            && is_valid_baseline(b)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn baseline(impression_id: i64, advertiser_account_id: i64) -> PurchaseValueBaseline {
        PurchaseValueBaseline {
            impression_id,
            advertiser_account_id,
            mean_value_usd_28d: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn slot_five_and_existing_slots_are_stable() {
        assert_eq!(ACTION_INDEX, 5);
        assert_eq!(ContinuousActionName::DwellTime as u32, 1);
        assert_eq!(ContinuousActionName::ClickDwellTime as u32, 2);
        assert_eq!(ContinuousActionName::HomeVideoContinuationSecs as u32, 3);
        assert_eq!(ContinuousActionName::BridgeProbability as u32, 4);
    }

    #[test]
    fn old_requests_decode_with_no_baselines() {
        let old = PredictNextActionsRequest {
            return_logits_list: true,
            ..Default::default()
        };
        let decoded = PredictNextActionsRequest::decode(old.encode_to_vec().as_slice()).unwrap();
        assert!(decoded.return_logits_list);
        assert!(decoded.purchase_value_baselines.is_empty());
    }

    #[test]
    fn baselines_and_return_logits_list_use_independent_wire_tags() {
        let request = PredictNextActionsRequest {
            return_logits_list: true,
            purchase_value_baselines: vec![baseline(7, 9)],
            ..Default::default()
        };
        let bytes = request.encode_to_vec();
        assert!(bytes.windows(3).any(|w| w == [0xB0, 0x01, 0x01]));
        assert!(bytes.windows(2).any(|w| w == [0xBA, 0x01]));

        let decoded = PredictNextActionsRequest::decode(bytes.as_slice()).unwrap();
        assert!(decoded.return_logits_list);
        assert_eq!(decoded.purchase_value_baselines, vec![baseline(7, 9)]);
    }

    #[test]
    fn baseline_wire_tags_are_stable() {
        let bytes = baseline(1, 2).encode_to_vec();
        assert_eq!(bytes[0], 0x08);
        assert_eq!(bytes[2], 0x10);
        assert_eq!(bytes[4], 0x19);
        assert_eq!(bytes.len(), 5 + 8);
        for (i, target) in ValueTarget::ALL_TARGETS.iter().enumerate() {
            let mut b = PurchaseValueBaseline::default();
            target.set_mean(&mut b, 1.0);
            let bytes = b.encode_to_vec();
            assert_eq!(bytes.len(), 9, "{target:?}");
            assert_eq!(bytes[0], ((3 + i as u8) << 3) | 1, "{target:?} tag");
        }
        let mut full = baseline(1_900_000_000_000_000_000, 4_503_599_700_000_000);
        for target in ValueTarget::ALL_TARGETS {
            target.set_mean(&mut full, 12.5);
        }
        assert_eq!(full.encode_to_vec().len(), 118);
        let decoded = PurchaseValueBaseline::decode(full.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, full);
    }

    #[test]
    fn valid_baseline_requires_positive_ids_and_at_least_one_positive_finite_mean() {
        assert!(is_valid_baseline(&baseline(1, 2)));
        assert!(!is_valid_baseline(&baseline(0, 2)));
        assert!(!is_valid_baseline(&baseline(1, 0)));
        assert!(!is_valid_baseline(&baseline(-1, 2)));
        for mean in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut b = baseline(1, 2);
            b.mean_value_usd_28d = mean;
            assert!(
                !is_valid_baseline(&b),
                "ct mean {mean} alone must be invalid"
            );
            assert_eq!(ct_mean(&b), None);
            b.mean_content_view_usd_28d = 3.0;
            assert!(
                is_valid_baseline(&b),
                "another usable mean carries the entry"
            );
        }
        assert_eq!(ct_mean(&baseline(1, 2)), Some(10.0));
    }

    #[test]
    fn target_table_round_trips_names_and_fields() {
        for target in ValueTarget::ALL_TARGETS {
            assert_eq!(
                ValueTarget::from_target_definition(target.target_definition()),
                Some(target)
            );
            let mut b = baseline(1, 2);
            assert_eq!(
                target.mean(&b),
                (target == ValueTarget::PURCHASE_CT).then_some(10.0)
            );
            target.set_mean(&mut b, 4.0);
            assert_eq!(target.mean(&b), Some(4.0));
        }
        assert_eq!(ValueTarget::from_target_definition("web_ct_checkout"), None);
        assert!(
            ValueTarget::REQUIRED
                .iter()
                .all(|t| ValueTarget::ALL_TARGETS.contains(t))
        );
        assert_eq!(ValueEvent::Purchase.conversion_type(), 2);
        assert_eq!(ValueEvent::AddToCart.conversion_type(), 12);
        assert_eq!(ValueEvent::ContentView.conversion_type(), 19);
        for event in ValueEvent::ALL {
            assert_eq!(ValueEvent::from_snake_name(event.snake_name()), Some(event));
        }
        assert_eq!(ValueEvent::from_snake_name("checkout"), None);
        assert_eq!(MAX_EVENT_USD, 100_000.0);
        assert!(is_countable_usd(0.01) && is_countable_usd(MAX_EVENT_USD));
        for usd in [0.0, -1.0, MAX_EVENT_USD + 0.01, f64::NAN, f64::INFINITY] {
            assert!(!is_countable_usd(usd), "{usd}");
        }
    }

    #[test]
    fn a_baseline_with_only_an_app_mean_is_valid() {
        assert_eq!(ValueTarget::from_target_definition("app_install"), None);
        let mut app_only = PurchaseValueBaseline {
            impression_id: 1,
            advertiser_account_id: 2,
            ..Default::default()
        };
        assert!(!is_valid_baseline(&app_only));
        ValueTarget::App(AppValueEvent::Purchase).set_mean(&mut app_only, 7.5);
        assert!(is_valid_baseline(&app_only));
        assert_eq!(ct_mean(&app_only), None);
    }

    #[test]
    fn find_baseline_matches_identity_not_order_and_skips_invalid() {
        let mut stale = baseline(7, 9);
        stale.mean_value_usd_28d = 0.0;
        let request = PredictNextActionsRequest {
            purchase_value_baselines: vec![baseline(3, 4), stale, baseline(7, 8)],
            ..Default::default()
        };
        assert_eq!(find_baseline(&request, 7, 8), Some(&baseline(7, 8)));
        assert_eq!(find_baseline(&request, 3, 4), Some(&baseline(3, 4)));
        assert_eq!(find_baseline(&request, 7, 9), None);
        assert_eq!(find_baseline(&request, 3, 8), None);
        assert_eq!(
            find_baseline(&PredictNextActionsRequest::default(), 3, 4),
            None
        );
    }
}
