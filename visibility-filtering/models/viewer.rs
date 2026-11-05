use serde::{Deserialize, Serialize};
use std::sync::Arc;
use strum::VariantArray;
use xai_twittercontext_proto::TwitterContextViewer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Viewer {
    LoggedIn {
        id: u64,
        profile: ViewerProfile,
        has_age_verified_18_label: bool,
    },
    #[default]
    LoggedOut,
}

impl Viewer {
    pub fn user_id(&self) -> Option<u64> {
        match self {
            Viewer::LoggedIn { id, .. } => Some(*id),
            Viewer::LoggedOut => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RequestViewer {
    id: Option<u64>,
    country_code: Option<String>,
    context: Option<Arc<TwitterContextViewer>>,
}

impl RequestViewer {
    pub(crate) fn new(raw_id: Option<u64>, country_code: Option<String>) -> Self {
        Self {
            id: raw_id.filter(|&id| id.cast_signed() > 0),
            country_code: country_code.filter(|code| !code.is_empty()),
            context: None,
        }
    }

    pub(crate) fn with_context(raw_id: Option<u64>, context: Option<TwitterContextViewer>) -> Self {
        let country_code = context
            .as_ref()
            .map(|context| context.request_country_code.clone());
        Self {
            context: context.map(Arc::new),
            ..Self::new(raw_id, country_code)
        }
    }

    pub(crate) fn id(&self) -> Option<u64> {
        self.id
    }

    pub(crate) fn country_code(&self) -> Option<&str> {
        self.country_code.as_deref()
    }

    pub(crate) fn context(&self) -> Option<&TwitterContextViewer> {
        self.context.as_deref()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RequestFacts {
    viewer: RequestViewer,
    request_country: Option<String>,
    client_capability: ClientCapability,
    soft_nsfw_new_user_drop: bool,
}

impl RequestFacts {
    pub(crate) fn new(viewer: RequestViewer, client_capability: ClientCapability) -> Self {
        Self {
            request_country: viewer.country_code().map(str::to_ascii_lowercase),
            viewer,
            client_capability,
            soft_nsfw_new_user_drop: false,
        }
    }

    pub(crate) fn logged_in(id: u64) -> Self {
        Self::new(
            RequestViewer::new(Some(id), None),
            ClientCapability::DEFAULT,
        )
    }

    #[must_use]
    pub(crate) fn with_soft_nsfw_new_user_drop(self, soft_nsfw_new_user_drop: bool) -> Self {
        Self {
            soft_nsfw_new_user_drop,
            ..self
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_country(self, country_code: &str) -> Self {
        let viewer = RequestViewer {
            context: self.viewer.context,
            ..RequestViewer::new(self.viewer.id, Some(country_code.to_owned()))
        };
        Self::new(viewer, self.client_capability)
            .with_soft_nsfw_new_user_drop(self.soft_nsfw_new_user_drop)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_client(self, client_capability: ClientCapability) -> Self {
        Self {
            client_capability,
            ..self
        }
    }

    pub(crate) fn viewer(&self) -> &RequestViewer {
        &self.viewer
    }

    pub(crate) fn viewer_id(&self) -> Option<u64> {
        self.viewer.id()
    }

    pub(crate) fn request_country(&self) -> Option<&str> {
        self.request_country.as_deref()
    }

    pub(crate) fn client_capability(&self) -> ClientCapability {
        self.client_capability
    }

    pub(crate) fn soft_nsfw_new_user_drop(&self) -> bool {
        self.soft_nsfw_new_user_drop
    }
}

pub const ADULT_AGE_YEARS: i32 = 18;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewerAge {
    Known(i32),
    NotStated,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewerProfile {
    pub allows_sensitive_media: bool,
    pub viewer_age: ViewerAge,
    pub has_verified_badge: bool,
    pub has_idv_premium: bool,
    pub is_read_only: bool,
    pub account_country_code: Option<String>,
    pub is_new_account: bool,
    pub is_suspended: bool,
    pub is_deactivated: bool,
    pub has_uqf_enabled: bool,
    pub mention_filter_is_following: bool,
    pub has_blink_label: bool,
    pub notification_filters: NotificationFilters,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NotificationFilters {
    pub no_confirmed_email: bool,
    pub no_confirmed_phone: bool,
    pub default_profile_image: bool,
    pub new_users: bool,
    pub not_followed_by: bool,
}

impl ViewerProfile {
    pub fn is_underage(&self) -> bool {
        matches!(self.viewer_age, ViewerAge::Known(age) if age < ADULT_AGE_YEARS)
    }

    pub fn has_no_stated_age(&self) -> bool {
        self.viewer_age == ViewerAge::NotStated
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, strum::VariantArray)]
pub enum VerifyBlurSupport {
    #[default]
    Unsupported,
    IosNeedsUpdate,
    AndroidNeedsUpdate,
    Supported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ClientSwitch {
    ModernBlur,
    StaleTweetLimits,
    CommunityViewerRemovedLimits,
    GoreBlurIgnoresSettings,
    FosnrRules,
    FosnrFallbackDrops,
}

impl ClientSwitch {
    pub const fn default_on(self) -> bool {
        match self {
            Self::StaleTweetLimits | Self::FosnrRules => true,
            Self::ModernBlur
            | Self::CommunityViewerRemovedLimits
            | Self::GoreBlurIgnoresSettings
            | Self::FosnrFallbackDrops => false,
        }
    }

    const fn bit(self) -> u32 {
        1 << self as u8
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ClientSwitchSet(u32);

const _: () = assert!(ClientSwitch::VARIANTS.len() <= u32::BITS as usize);

impl ClientSwitchSet {
    pub const NONE: Self = Self(0);

    pub const DEFAULT_ON: Self = {
        let mut set = Self::NONE;
        let mut rest = ClientSwitch::VARIANTS;
        while let [switch, tail @ ..] = rest {
            if switch.default_on() {
                set = set.with(*switch);
            }
            rest = tail;
        }
        set
    };

    #[must_use]
    pub const fn with(self, switch: ClientSwitch) -> Self {
        Self(self.0 | switch.bit())
    }

    #[cfg(test)]
    #[must_use]
    pub const fn without(self, switch: ClientSwitch) -> Self {
        Self(self.0 & !switch.bit())
    }

    pub const fn contains(self, switch: ClientSwitch) -> bool {
        self.0 & switch.bit() != 0
    }
}

impl std::fmt::Debug for ClientSwitchSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set()
            .entries(
                ClientSwitch::VARIANTS
                    .iter()
                    .filter(|&&switch| self.contains(switch)),
            )
            .finish()
    }
}

impl FromIterator<ClientSwitch> for ClientSwitchSet {
    fn from_iter<I: IntoIterator<Item = ClientSwitch>>(switches: I) -> Self {
        switches.into_iter().fold(Self::NONE, Self::with)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientCapability {
    pub verify_blur_support: Option<VerifyBlurSupport>,
    pub switches_on: ClientSwitchSet,
}

impl ClientCapability {
    pub const DEFAULT: Self = Self {
        verify_blur_support: Some(VerifyBlurSupport::Unsupported),
        switches_on: ClientSwitchSet::DEFAULT_ON,
    };

    pub const fn has(self, switch: ClientSwitch) -> bool {
        self.switches_on.contains(switch)
    }

    #[must_use]
    pub const fn with(self, switch: ClientSwitch) -> Self {
        Self {
            switches_on: self.switches_on.with(switch),
            ..self
        }
    }

    #[cfg(test)]
    #[must_use]
    pub const fn without(self, switch: ClientSwitch) -> Self {
        Self {
            switches_on: self.switches_on.without(switch),
            ..self
        }
    }
}

impl Default for ClientCapability {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum Client {
    #[default]
    Web,
    IosCurrent,
    IosOutdated,
    AndroidCurrent,
    AndroidOutdated,
    AndroidWithoutFosnr,
    MacApp,
}

impl Client {
    const fn app(self) -> (i64, Option<&'static str>) {
        match self {
            Self::Web => (3_033_300, None),
            Self::IosCurrent => (
                129_032,
                Some("Twitter-iPhone/11.11.5 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)"),
            ),
            Self::IosOutdated => (
                129_032,
                Some("Twitter-iPhone/11.11.4 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)"),
            ),
            Self::AndroidCurrent => (
                258_901,
                Some(
                    "TwitterAndroid/11.11.0-release.00 (311110000-r-0) Pixel 7/14 \
                     (Google;panther;google;panther;0;;1;2022)",
                ),
            ),
            Self::AndroidOutdated => (
                258_901,
                Some(
                    "TwitterAndroid/11.10.9-release.00 (311109000-r-0) Pixel 7/14 \
                     (Google;panther;google;panther;0;;1;2022)",
                ),
            ),
            Self::AndroidWithoutFosnr => (
                258_901,
                Some(
                    "TwitterAndroid/9.82.0-release.00 (29820000-r-0) Pixel 7/14 \
                     (Google;panther;google;panther;0;;1;2022)",
                ),
            ),
            Self::MacApp => (
                557_701,
                Some("Twitter-Mac/11.11.5 macOS/14.0 (Apple;Mac14,2)"),
            ),
        }
    }

    pub(crate) fn context(
        self,
        viewer_id: Option<u64>,
        country_code: Option<&str>,
    ) -> TwitterContextViewer {
        let (client_application_id, user_agent) = self.app();
        TwitterContextViewer {
            user_id: viewer_id.map_or(0, u64::cast_signed),
            client_application_id,
            user_agent: user_agent.unwrap_or_default().to_string(),
            request_country_code: country_code.unwrap_or_default().to_string(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ViewerFeatures {
    pub viewer: Viewer,
    pub(crate) facts: RequestFacts,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stated_age_below_adult_is_underage_and_only_a_missing_age_is_unstated() {
        for (age, underage, no_stated_age) in [
            (ViewerAge::Known(ADULT_AGE_YEARS - 1), true, false),
            (ViewerAge::Known(ADULT_AGE_YEARS), false, false),
            (ViewerAge::NotStated, false, true),
            (ViewerAge::Unknown, false, false),
        ] {
            let profile = ViewerProfile {
                viewer_age: age,
                ..ViewerProfile::default()
            };
            assert_eq!(profile.is_underage(), underage, "{age:?}");
            assert_eq!(profile.has_no_stated_age(), no_stated_age, "{age:?}");
        }
    }
}
