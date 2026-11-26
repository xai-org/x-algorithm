pub use xai_x_thrift::tweet_safety_label::SafetyLabelType;

use rustc_hash::FxHashSet;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap {
    types: FxHashSet<SafetyLabelType>,
    by_agent: FxHashSet<SafetyLabelType>,
    sources: Vec<(SafetyLabelType, String)>,
}

impl SafetyLabelMap {
            pub(crate) const SOURCE_READ_LABELS: [SafetyLabelType; 7] = [
        SafetyLabelType::LOW_QUALITY,
        SafetyLabelType::MISINFO_CIVIC,
        SafetyLabelType::MISINFO_GENERIC,
        SafetyLabelType::MISINFO_MEDICAL,
        SafetyLabelType::MISINFO_CRISIS,
        SafetyLabelType::IPI_DEVELOPMENT_ONLY,
        SafetyLabelType::FOR_EMERGENCY_USE_ONLY,
    ];

    #[cfg(test)]
    pub fn new(label_types: impl IntoIterator<Item = SafetyLabelType>) -> Self {
        Self {
            types: label_types.into_iter().collect(),
            by_agent: FxHashSet::default(),
            sources: Vec::new(),
        }
    }

    #[cfg(test)]
    pub fn assigned_by_agent(mut self, label_type: SafetyLabelType) -> Self {
        self.by_agent.insert(label_type);
        self
    }

    #[cfg(test)]
    pub fn with_source(mut self, label_type: SafetyLabelType, source: &str) -> Self {
        self.types.insert(label_type);
        self.sources.push((label_type, source.to_owned()));
        self
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        let mut map = Self::default();
        for (&label_type, label) in &proto.labels {
            let label_type = SafetyLabelType(label_type);
            map.types.insert(label_type);
            if matches!(
                label.safety_label_source,
                Some(vf_pb::safety_label::SafetyLabelSource::ToolAction(_))
            ) {
                map.by_agent.insert(label_type);
            }
            if let Some(source) = &label.source
                && Self::SOURCE_READ_LABELS.contains(&label_type)
            {
                map.sources.push((label_type, source.clone()));
            }
        }
        map
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.types.contains(&label_type)
    }

        #[inline]
    pub fn is_by_agent(&self, label_type: SafetyLabelType) -> bool {
        self.by_agent.contains(&label_type)
    }

        pub fn source(&self, label_type: SafetyLabelType) -> Option<&str> {
        self.sources
            .iter()
            .find(|(held, _)| *held == label_type)
            .map(|(_, source)| source.as_str())
    }
}
