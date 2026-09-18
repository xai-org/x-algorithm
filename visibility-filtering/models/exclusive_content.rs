#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExclusiveContentFeatures {
    pub conversation_author_id: u64,
    pub viewer_super_follows_author: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExclusiveHydration {
    Public,
    Exclusive(ExclusiveContentFeatures),
    Failed,
}

impl Default for ExclusiveHydration {
    fn default() -> Self {
        ExclusiveHydration::Public
    }
}

impl ExclusiveHydration {
    pub fn features(&self) -> Option<ExclusiveContentFeatures> {
        match self {
            ExclusiveHydration::Exclusive(features) => Some(features.clone()),
            ExclusiveHydration::Public | ExclusiveHydration::Failed => None,
        }
    }

    pub fn failed(&self) -> bool {
        matches!(self, ExclusiveHydration::Failed)
    }
}
