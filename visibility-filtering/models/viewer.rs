#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Viewer {
    LoggedIn(u64),
    #[default]
    LoggedOut,
}

impl Viewer {
    pub fn user_id(&self) -> Option<u64> {
        match self {
            Viewer::LoggedIn(id) => Some(*id),
            Viewer::LoggedOut => None,
        }
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

#[derive(Clone, Debug, Default)]
pub struct ViewerFeatures {
    pub viewer: Viewer,
    pub allows_sensitive_media: bool,
    pub country_code: Option<String>,
    pub account_country_code: Option<String>,
    pub viewer_age: ViewerAge,
}

impl ViewerFeatures {
    pub fn viewer_is_underage(&self) -> bool {
        matches!(self.viewer_age, ViewerAge::Known(age) if age < ADULT_AGE_YEARS)
    }

    /// Logged-in viewers whose age was not loaded (gizmoduck miss/error) use the
    /// same jurisdiction-scoped NSFW gate as a confirmed missing birthday.
    /// Logged-out viewers stay on `SensitiveViewerLoggedOutDropRule`.
    pub fn viewer_has_no_stated_age(&self) -> bool {
        matches!(self.viewer, Viewer::LoggedIn(_))
            && matches!(self.viewer_age, ViewerAge::NotStated | ViewerAge::Unknown)
    }
}

impl ViewerFeatures {
    pub fn viewer_id(&self) -> Option<u64> {
        self.viewer.user_id()
    }

    pub fn viewer_is_logged_out(&self) -> bool {
        matches!(self.viewer, Viewer::LoggedOut)
    }
}
