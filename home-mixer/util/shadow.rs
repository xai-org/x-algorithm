use crate::params::{
    ShadowTrafficDefaultPercent, ShadowTrafficPhoenixClusterAggregationTypes,
    ShadowTrafficPhoenixClusterRates,
};
use strum::VariantArray;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, Surface,
};
use xai_candidate_pipeline::component_library::utils::is_sampled;
use xai_feature_switches::Params;
use xai_recsys_proto::UserActionAggregationType;

pub fn shadow_rate_for_cluster(params: &Params, cluster: PhoenixCluster) -> f64 {
    let cluster_name = format!("{cluster:?}");
    let overrides: Vec<String> = params.get(ShadowTrafficPhoenixClusterRates);
    overrides
        .iter()
        .find_map(|entry| {
            let (name, pct) = entry.split_once(':')?;
            if name.trim() == cluster_name {
                pct.trim().parse::<f64>().ok()
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            if cluster.is_shadow_opt_in() {
                0.0
            } else {
                params.get(ShadowTrafficDefaultPercent)
            }
        })
}

pub fn shadow_aggregation_type_for_cluster(
    params: &Params,
    cluster: PhoenixCluster,
) -> Option<UserActionAggregationType> {
    let cluster_name = format!("{cluster:?}");
    let overrides: Vec<String> = params.get(ShadowTrafficPhoenixClusterAggregationTypes);
    overrides.iter().find_map(|entry| {
        let (name, aggregation_type) = entry.split_once(':')?;
        if name.trim() == cluster_name {
            UserActionAggregationType::from_str_name(aggregation_type.trim())
        } else {
            None
        }
    })
}

pub fn is_shadow_sampled_for_cluster(
    params: &Params,
    prediction_request_id: u64,
    cluster: PhoenixCluster,
) -> bool {
    if !cluster.is_shadow_eligible_for(Surface::Home) {
        return false;
    }
    is_sampled(
        prediction_request_id,
        shadow_rate_for_cluster(params, cluster),
    )
}

pub fn is_shadow_traffic(params: &Params, prediction_request_id: u64) -> bool {
    PhoenixCluster::VARIANTS
        .iter()
        .any(|&cluster| is_shadow_sampled_for_cluster(params, prediction_request_id, cluster))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder, Value};

    fn params_with(rates: Vec<&str>, default_pct: Option<f64>) -> Params {
        let fs = FeatureSwitches::new(vec![]).unwrap();
        let mut results = fs.match_recipient(&RecipientBuilder::new().build());
        let array: Vec<Value> = rates.into_iter().map(Value::from).collect();
        results.override_value(
            "rust_home_mixer_shadow_traffic_phoenix_cluster_rates".to_string(),
            Value::Array(array),
        );
        if let Some(pct) = default_pct {
            results.override_value(
                "rust_home_mixer_shadow_traffic_default_percent".to_string(),
                Value::from(pct),
            );
        }
        results.into()
    }

    #[test]
    fn override_applies_to_listed_cluster_only() {
        let params = params_with(vec!["Experiment6Fou:2.0"], Some(0.5));
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment6Fou),
            2.0
        );
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment1Fou),
            0.5
        );
    }

    #[test]
    fn override_can_disable_a_cluster() {
        let params = params_with(vec!["Experiment3Fou:0"], None);
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment3Fou),
            0.0
        );
        assert!(!is_shadow_sampled_for_cluster(
            &params,
            0,
            PhoenixCluster::Experiment3Fou,
        ));
    }

    #[test]
    fn malformed_entry_falls_back_to_default() {
        let params = params_with(vec!["Experiment6Fou:abc", "Experiment6Fou"], Some(1.0));
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment6Fou),
            1.0
        );
    }

    #[test]
    fn entry_whitespace_is_tolerated() {
        let params = params_with(vec![" Experiment6Fou : 2.5 "], None);
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment6Fou),
            2.5
        );
    }

    #[test]
    fn configurable_default_percent() {
        let params = params_with(vec![], Some(1.0));
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment1Fou),
            1.0
        );
    }

    #[test]
    fn opt_in_cluster_is_off_unless_overridden() {
        let params = params_with(vec![], Some(1.0));
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment18Fou),
            0.0
        );
        assert!(!is_shadow_sampled_for_cluster(
            &params,
            0,
            PhoenixCluster::Experiment18Fou,
        ));
        let params = params_with(vec!["Experiment18Fou:2.0"], Some(1.0));
        assert_eq!(
            shadow_rate_for_cluster(&params, PhoenixCluster::Experiment18Fou),
            2.0
        );
    }

    #[test]
    fn shadow_traffic_union_matches_per_cluster_gates() {
        let params = params_with(vec!["Experiment6Fou:2.0"], Some(0.5));
        let mut exp6_only = None;
        let mut all_clusters = None;
        for request_id in 0..100_000u64 {
            let exp6 =
                is_shadow_sampled_for_cluster(&params, request_id, PhoenixCluster::Experiment6Fou);
            let exp1 =
                is_shadow_sampled_for_cluster(&params, request_id, PhoenixCluster::Experiment1Fou);
            if exp6 && !exp1 {
                exp6_only = Some(request_id);
            }
            if exp6 && exp1 {
                all_clusters = Some(request_id);
            }
            if exp6_only.is_some() && all_clusters.is_some() {
                break;
            }
        }
        let exp6_only = exp6_only.expect("expected a request sampled for Experiment6Fou only");
        let all_clusters = all_clusters.expect("expected a request sampled for all clusters");
        assert!(is_shadow_traffic(&params, exp6_only));
        assert!(is_shadow_traffic(&params, all_clusters));
    }

    #[test]
    fn zero_default_and_no_overrides_disables_shadow_traffic() {
        let params = params_with(vec![], Some(0.0));
        for request_id in 0..10_000u64 {
            assert!(!is_shadow_traffic(&params, request_id));
        }
    }
}
