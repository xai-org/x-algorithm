use std::collections::HashMap;
use std::time::Duration;
use xai_safety_label_store::types::SafetyLabelMap;
use xai_visibility_filtering::tweet_safety_label::TweetSafetyLabelClient;
use xai_x_thrift::tweet_safety_label::SafetyLabelType;

const TIMEOUT: Duration = Duration::from_millis(80);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EngagementIneligible {
    pub(crate) reply: bool,
    pub(crate) share: bool,
    pub(crate) click: bool,
}

impl EngagementIneligible {
    fn from_labels(labels: &SafetyLabelMap) -> Self {
        Self {
            reply: labels.contains_key(&SafetyLabelType::PROMOTES_REPLIES),
            share: labels.contains_key(&SafetyLabelType::PROMOTES_SHARES),
            click: labels.contains_key(&SafetyLabelType::PROMOTES_CLICKS),
        }
    }

    pub(crate) fn any(self) -> bool {
        self.reply || self.share || self.click
    }
}

pub(crate) async fn fetch_engagement_ineligible(
    client: &dyn TweetSafetyLabelClient,
    post_ids: Vec<u64>,
) -> HashMap<u64, EngagementIneligible> {
    match tokio::time::timeout(TIMEOUT, client.get_safety_labels(post_ids)).await {
        Ok(Ok(batch)) => batch
            .labels
            .iter()
            .map(|(id, labels)| (*id, EngagementIneligible::from_labels(labels)))
            .collect(),
        _ => HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_x_thrift::tweet_safety_label::SafetyLabel;

    #[test]
    fn each_promotes_label_marks_its_own_head() {
        let flags = |label| {
            EngagementIneligible::from_labels(&HashMap::from([(label, SafetyLabel::default())]))
        };

        assert_eq!(
            flags(SafetyLabelType::PROMOTES_REPLIES),
            EngagementIneligible {
                reply: true,
                ..Default::default()
            }
        );
        assert_eq!(
            flags(SafetyLabelType::PROMOTES_SHARES),
            EngagementIneligible {
                share: true,
                ..Default::default()
            }
        );
        assert_eq!(
            flags(SafetyLabelType::PROMOTES_CLICKS),
            EngagementIneligible {
                click: true,
                ..Default::default()
            }
        );
        assert!(!flags(SafetyLabelType::SOFT_NSFW).any());
    }
}
