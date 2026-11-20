use crate::clients::resurrection_date_client::compute_resurrection_fs_fields;
use std::collections::HashMap;
use xai_candidate_pipeline::component_library::utils::{
    creation_epoch_days, days_since_creation, duration_since_creation_opt,
};
use xai_core_entities::entities::SubscriptionLevel;
use xai_feature_switches::{RecipientBuilder, SimpleRecipient};

#[derive(Clone, Debug)]
pub struct FsRecipientInputs {
    pub user_id: u64,
    pub country_code: String,
    pub language_code: String,
    pub client_app_id: i64,
    pub client_version: Option<String>,
    pub user_roles: Vec<String>,
    pub datacenter: String,
    pub has_phone_number: bool,
    pub subscription_level: Option<SubscriptionLevel>,
    pub follower_count: Option<i64>,
    pub resurrection_time_ms: Option<i64>,
    pub product: String,
    pub now_ms: i64,
    pub fs_overrides: HashMap<String, String>,
}

impl FsRecipientInputs {
    pub fn recipient(&self) -> SimpleRecipient {
        let account_age_days = days_since_creation(self.user_id);
        let account_creation_date = creation_epoch_days(self.user_id);
        let account_age_minutes =
            duration_since_creation_opt(self.user_id).map(|age| (age.as_secs() / 60) as i64);
        let (user_resurrected_date, days_since_resurrection) =
            compute_resurrection_fs_fields(self.now_ms, self.resurrection_time_ms);
        let minutes_since_resurrection = self
            .resurrection_time_ms
            .filter(|&t| t >= 0)
            .map(|t| (self.now_ms - t) / 60_000);

        RecipientBuilder::new()
            .user_id(self.user_id)
            .country(&self.country_code)
            .language(&self.language_code)
            .client_app_id(self.client_app_id)
            .client_version_opt(self.client_version.as_deref())
            .user_roles(self.user_roles.iter().cloned())
            .custom_string("datacenter", &self.datacenter)
            .custom_i64("account_age_days", account_age_days)
            .custom_i64("account_creation_date", account_creation_date)
            .custom_bool("has_phone_number", self.has_phone_number)
            .custom_opt_string(
                "subscription_level",
                self.subscription_level.map(|level| level.as_str()),
            )
            .custom_opt_i64("follower_count", self.follower_count)
            .custom_string("product", &self.product)
            .custom_opt_i64("user_resurrected_date", user_resurrected_date)
            .custom_opt_i64("days_since_resurrection", days_since_resurrection)
            .custom_opt_i64("account_age_minutes", account_age_minutes)
            .custom_opt_i64("minutes_since_resurrection", minutes_since_resurrection)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::parse_criteria;

    fn inputs(subscription_level: Option<SubscriptionLevel>) -> FsRecipientInputs {
        FsRecipientInputs {
            user_id: 1839569058817323008,
            country_code: "US".to_string(),
            language_code: "en".to_string(),
            client_app_id: 3033300,
            client_version: None,
            user_roles: vec![],
            datacenter: "atla".to_string(),
            has_phone_number: false,
            subscription_level,
            follower_count: None,
            resurrection_time_ms: None,
            product: "for_you".to_string(),
            now_ms: 0,
            fs_overrides: HashMap::new(),
        }
    }

    #[test]
    fn subscription_level_matches_custom_field_criteria() {
        let premium_plus = parse_criteria("[$subscription_level eq premium_plus]", "test").unwrap();

        assert!(premium_plus.evaluate(&inputs(Some(SubscriptionLevel::PremiumPlus)).recipient()));
        assert!(!premium_plus.evaluate(&inputs(Some(SubscriptionLevel::Premium)).recipient()));
        assert!(!premium_plus.evaluate(&inputs(Some(SubscriptionLevel::Basic)).recipient()));
        assert!(!premium_plus.evaluate(&inputs(None).recipient()));
    }
}
