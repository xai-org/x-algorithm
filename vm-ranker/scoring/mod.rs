mod debias;
pub mod dpp_model;
pub mod value_model;

use std::sync::Arc;

use log::error;

use xai_feature_switches::Params;
use xai_user_debias::UserDebiasModel;
use xai_vm_ranker_proto::{DppParams, RankRequest, RankedCandidate};

use crate::dpp::DppConfig;
use crate::embedding_store::EmbeddingStore;
use crate::metrics::PARAMS_RESOLVED;
use crate::params::*;
use crate::ranking_config::RankingConfig;
use dpp_model::EmbeddingMemo;
use value_model::ValueModelOutput;

#[derive(Clone)]
pub struct DppContext {
    pub store: Arc<EmbeddingStore>,
    pub config: DppConfig,
}

#[derive(Clone, Default)]
pub struct DebiasModels {
    pub user: Option<Arc<UserDebiasModel>>,
    pub user_changed_at_ms: u64,
}

#[derive(Clone, Default)]
pub struct RankContext {
    pub dpp: Option<DppContext>,
    pub debias: DebiasModels,
    pub config: Option<Arc<RankingConfig>>,
}

pub async fn rank(req: RankRequest, ctx: RankContext) -> Result<Vec<RankedCandidate>, String> {
    let params = resolve_params(&req, ctx.config.as_deref());
    let compute_value_model =
        req.compute_value_model || params.as_ref().is_some_and(|p| p.get(ComputeValueModel));
    let dpp_enabled = dpp_enabled(&req, params.as_ref());

    if let (Some(dpp), true) = (ctx.dpp.clone(), dpp_enabled) {
        if req.value_model_id != "dpp" {
            error!(
                "DPP context provided but value_model_id='{}' is not 'dpp'",
                req.value_model_id
            );
        }
        let mut dpp = dpp;

        match (&params, &req.dpp_params) {
            (Some(p), _) => {
                dpp.config.theta = p.get(DppTheta);
                dpp.config.max_selected_rank = p.get(DppMaxSelectedRank) as usize;
            }
            (None, Some(overrides)) => apply_dpp_overrides(&mut dpp.config, overrides),
            (None, None) => {}
        }

        let config = ctx.config.clone();
        return tokio::task::spawn_blocking(move || {
            let value_model = value_model::compute(
                &req,
                params.as_ref(),
                config.as_deref().map(RankingConfig::author_rules),
                compute_value_model,
                &ctx.debias,
            );
            let served_pre_dpp = served_pre_dpp_scores(&req, value_model.as_ref());
            let mut memo = EmbeddingMemo::default();
            let served = dpp_model::rank_with_memo(&req, &served_pre_dpp, &dpp, &mut memo);
            ranked_candidates(&req, served, value_model.as_ref())
        })
        .await
        .map_err(|e| format!("DPP spawn_blocking failed: {e}"));
    }

    let value_model = value_model::compute(
        &req,
        params.as_ref(),
        ctx.config.as_deref().map(RankingConfig::author_rules),
        compute_value_model,
        &ctx.debias,
    );
    let served = served_pre_dpp_scores(&req, value_model.as_ref())
        .iter()
        .map(|s| s.unwrap_or(0.0))
        .collect();
    Ok(ranked_candidates(&req, served, value_model.as_ref()))
}

fn dpp_enabled(req: &RankRequest, params: Option<&Params>) -> bool {
    params.is_none_or(|p| p.get(DppEnabled) && (!req.topic_request || p.get(TopicDppEnabled)))
}

fn apply_dpp_overrides(config: &mut DppConfig, overrides: &DppParams) {
    if overrides.theta != 0.0 {
        config.theta = overrides.theta;
    }
    if overrides.max_selected_rank != 0 {
        config.max_selected_rank = overrides.max_selected_rank as usize;
    }
    if overrides.top_k != 0 {
        config.top_k = overrides.top_k as usize;
    }
}

fn resolve_params(req: &RankRequest, config: Option<&RankingConfig>) -> Option<Params> {
    let outcome = match (req.viewer.as_ref(), config) {
        (Some(_), Some(_)) => "resolved",
        (None, _) => "no_viewer_context",
        (_, None) => "no_config",
    };
    PARAMS_RESOLVED.with_label_values(&[outcome]).inc();
    let config = config?;
    Some(match req.viewer.as_ref() {
        Some(viewer) => config.resolve(viewer),
        None => config.resolve_anonymous(),
    })
}

fn served_pre_dpp_scores(
    req: &RankRequest,
    value_model: Option<&ValueModelOutput>,
) -> Vec<Option<f64>> {
    match value_model {
        Some(vm) => vm.scores.iter().copied().map(Some).collect(),
        None => req.candidates.iter().map(|c| c.score).collect(),
    }
}

fn ranked_candidates(
    req: &RankRequest,
    final_scores: Vec<f64>,
    value_model: Option<&ValueModelOutput>,
) -> Vec<RankedCandidate> {
    let weighted = value_model;
    req.candidates
        .iter()
        .zip(final_scores)
        .enumerate()
        .map(|(i, (c, score))| RankedCandidate {
            tweet_id: c.tweet_id,
            score,
            weighted_score: weighted.map(|vm| vm.weighted[i]),
            top_heads: weighted
                .and_then(|vm| vm.top_heads.as_ref())
                .map(|heads| heads[i].clone())
                .unwrap_or_default(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, FixedSizeListArray, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use prost::Message;
    use xai_feature_switches::{
        BucketMembership, ExperimentBucket, ExperimentBucketsChooser, FeatureSwitches,
        NullBucketImpressor, Recipient,
    };
    use xai_user_debias::{arrow as seq, debiased_probability, FEATURE_DIM};
    use xai_value_model::{
        compute_value_scores, CandidateScoringInputs, QueryScoringContext, ValueModelWeights,
        NEGATIVE_SCORES_OFFSET,
    };
    use xai_vm_ranker_proto::experiment::RequestPayload;
    use xai_vm_ranker_proto::{PhoenixScores, RankCandidate, ViewerContext};

    const FEATURES_YAML: &str = r#"
rust_home_mixer:
  description: "test"
  owner: "test@example.com"
  parameters:
    rust_home_mixer_favorite_weight:
      type: double
      default: 2.0
    rust_home_mixer_reply_weight:
      type: double
      default: 4.0
    rust_home_mixer_report_weight:
      type: double
      default: -100.0
    rust_home_mixer_oon_weight_factor:
      type: double
      default: 0.75
    rust_home_mixer_vm_ranker_user_debias_enabled:
      type: boolean
      default: false
    rust_home_mixer_vm_ranker_user_debias_alpha:
      type: double
      default: 0.0
    rust_home_mixer_vm_ranker_user_debias_clip_logit:
      type: double
      default: 2.0
  rules:
    - description: "debias arm"
      query: "[country eq DE]"
      values:
        rust_home_mixer_vm_ranker_user_debias_enabled: true
        rust_home_mixer_vm_ranker_user_debias_alpha: 1.0
"#;

    const HEAD_NAMES: [&str; 20] = [
        "fav",
        "reply",
        "quote",
        "retweet",
        "dwelled",
        "not_dwelled",
        "vqv",
        "photo_expand",
        "report",
        "not_interested",
        "follow_author",
        "block",
        "mute",
        "click",
        "profile_click",
        "bookmark",
        "share",
        "open_link",
        "video_open",
        "post_unexplored",
    ];

    fn candidate(
        tweet_id: u64,
        author_id: u64,
        fav: f64,
        reply: Option<f64>,
        in_network: bool,
        upstream_score: f64,
    ) -> RankCandidate {
        RankCandidate {
            tweet_id,
            author_id,
            in_network,
            score: Some(upstream_score),
            phoenix_scores: Some(PhoenixScores {
                favorite_score: Some(fav),
                reply_score: reply,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn weights() -> ValueModelWeights {
        let params = config().resolve(&viewer("US"));
        value_model::weights_from_params(&params, 7)
    }

    fn scoring_context() -> QueryScoringContext {
        QueryScoringContext {
            effective_oon_weight: 0.75,
        }
    }

    fn config() -> Arc<RankingConfig> {
        Arc::new(RankingConfig::from_yaml(FEATURES_YAML).unwrap())
    }

    fn context(config: Arc<RankingConfig>, debias: DebiasModels) -> RankContext {
        RankContext {
            dpp: None,
            debias,
            config: Some(config),
        }
    }

    fn viewer(country: &str) -> ViewerContext {
        ViewerContext {
            user_id: 7,
            country_code: country.to_string(),
            product: "ForYou".to_string(),
            now_ms: 1_700_000_000_000,
            ..Default::default()
        }
    }

    fn request(candidates: Vec<RankCandidate>, country: &str) -> RankRequest {
        RankRequest {
            viewer_id: 7,
            request_timestamp_ms: 1_700_000_000_000,
            candidates,
            value_model_id: "dpp".to_string(),
            viewer: Some(viewer(country)),
            compute_value_model: true,
            ..Default::default()
        }
    }

    fn inputs(candidates: &[RankCandidate]) -> Vec<CandidateScoringInputs> {
        candidates
            .iter()
            .map(|c| CandidateScoringInputs::from_rank_candidate(c, 10_000))
            .collect()
    }

    fn order_by_score_desc(scores: &[f64]) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        idx
    }

    #[tokio::test]
    async fn value_model_scores_match_shared_crate_and_passthrough_keeps_upstream() {
        let candidates = vec![
            candidate(1, 10, 0.5, None, true, 1.0),
            candidate(2, 20, 0.4, Some(0.2), false, 2.0),
            candidate(3, 10, 0.45, None, true, 3.0),
        ];
        let req = request(candidates.clone(), "US");
        let weights = weights();
        assert_eq!(weights.favorite, 2.0);
        assert_eq!(weights.retweet, 1.0);

        let ranked = rank(req.clone(), context(config(), DebiasModels::default()))
            .await
            .unwrap();

        let expected = compute_value_scores(&weights, &scoring_context(), &inputs(&candidates));
        assert_eq!(ranked.len(), 3);
        for (i, r) in ranked.iter().enumerate() {
            assert_eq!(r.tweet_id, candidates[i].tweet_id);
            assert_eq!(r.score, expected.scores[i]);
        }
        let ranked_scores: Vec<f64> = ranked.iter().map(|r| r.score).collect();
        assert_eq!(
            order_by_score_desc(&ranked_scores),
            order_by_score_desc(&expected.scores)
        );
        assert_eq!(order_by_score_desc(&expected.scores), vec![1, 0, 2]);

        let passthrough = rank(
            RankRequest {
                compute_value_model: false,
                ..req.clone()
            },
            context(config(), DebiasModels::default()),
        )
        .await
        .unwrap();
        for (r, c) in passthrough.iter().zip(&candidates) {
            assert_eq!(r.tweet_id, c.tweet_id);
            assert_eq!(r.score, c.score.unwrap());
        }
    }

    #[derive(Debug)]
    struct TreatmentAuthor(u64);

    impl ExperimentBucketsChooser for TreatmentAuthor {
        fn choose_buckets(&self, _recipient: &dyn Recipient) -> BucketMembership {
            BucketMembership::new()
        }

        fn choose_bucket_without_overrides(
            &self,
            experiment_key: &str,
            recipient: &dyn Recipient,
        ) -> Option<ExperimentBucket> {
            (recipient.user_id()? == self.0)
                .then(|| ExperimentBucket::new(experiment_key, "treatment").with_version(1))
        }
    }

    #[tokio::test]
    async fn author_exploration_bonus_boosts_treatment_authors() {
        let yaml = format!(
            "{FEATURES_YAML}\n{}",
            r#"
rust_home_mixer_author_exploration:
  parameters:
    rust_home_mixer_author_exploration_bonus:
      type: double
      default: 0.0
  rules:
    - query: "[author_exploration author_bucket_membership treatment]"
      values:
        rust_home_mixer_author_exploration_bonus: 0.2
"#
        );
        let fs = FeatureSwitches::with_options(
            xai_feature_switches::load_yaml_string(&yaml).unwrap(),
            Arc::new(TreatmentAuthor(10)),
            Arc::new(NullBucketImpressor),
            None,
            false,
        )
        .unwrap();
        let author_config = Arc::new(RankingConfig::new(Arc::new(fs)));
        let req = request(
            vec![
                candidate(1, 10, 0.5, None, true, 0.0),
                candidate(2, 20, 0.5, None, true, 0.0),
            ],
            "US",
        );

        for pre_offset in [false, true] {
            let mut req = req.clone();
            req.viewer.as_mut().unwrap().fs_overrides.insert(
                "rust_home_mixer_multiplier_pre_offset".to_string(),
                pre_offset.to_string(),
            );
            let base = rank(req.clone(), context(config(), DebiasModels::default()))
                .await
                .unwrap();
            let boosted = rank(
                req,
                context(Arc::clone(&author_config), DebiasModels::default()),
            )
            .await
            .unwrap();

            assert!((boosted[0].score - (base[0].score + 0.2)).abs() < 1e-12);
            assert_eq!(boosted[1].score, base[1].score);
            assert_eq!(boosted[0].weighted_score, base[0].weighted_score);
        }
    }

    #[test]
    fn dpp_skipped_for_topic_requests_by_default() {
        let mut req = request(vec![], "US");
        let params = resolve_params(&req, Some(&config()));
        assert!(dpp_enabled(&req, params.as_ref()));
        req.topic_request = true;
        assert!(!dpp_enabled(&req, params.as_ref()));
    }

    #[test]
    fn dpp_overrides_apply_only_non_zero_fields() {
        let mut config = DppConfig {
            top_k: 120,
            theta: 0.7,
            max_selected_rank: 150,
            debug_viewer_id: 0,
        };
        apply_dpp_overrides(
            &mut config,
            &DppParams {
                top_k: 24,
                ..Default::default()
            },
        );
        assert_eq!(config.theta, 0.7);
        assert_eq!(config.max_selected_rank, 150);
        assert_eq!(config.top_k, 24);
    }

    fn toy_model(user_fav_logit: f32, global_fav_logit_home: f32) -> Arc<UserDebiasModel> {
        let n_heads = HEAD_NAMES.len();
        let in_dim = FEATURE_DIM + 4;
        let head_names = HEAD_NAMES
            .iter()
            .map(|h| format!("\"{h}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let meta = format!(
            concat!(
                r#"{{"feature_version": "serving_v1", "feature_dim": {d}, "head_names": [{heads}], "#,
                r#""n_layers": 0, "emb_dim": 1, "trained_at_ms": {trained_at_ms}, "train_records": 1, "tensors": {{"#,
                r#""feat_mean": [{d}], "feat_std": [{d}], "ln_g": [{d}], "ln_b": [{d}], "#,
                r#""emb_surface": [3, 1], "emb_hour": [24, 1], "emb_dow": [7, 1], "emb_app": [32, 1], "#,
                r#""w_out": [{in_dim}, {out}], "b_out": [{out}], "global_logit_by_surface": [3, {n}], "#,
                r#""global_log_dwell_by_surface": [3]}}}}"#
            ),
            d = FEATURE_DIM,
            heads = head_names,
            trained_at_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            in_dim = in_dim,
            out = n_heads + 1,
            n = n_heads,
        );
        let mut b_out = vec![0.0f32; n_heads + 1];
        b_out[0] = user_fav_logit;
        let mut global_logits = vec![0.0f32; 3 * n_heads];
        global_logits[n_heads] = global_fav_logit_home;
        let tensors: Vec<Vec<f32>> = vec![
            vec![0.0; FEATURE_DIM],
            vec![1.0; FEATURE_DIM],
            vec![1.0; FEATURE_DIM],
            vec![0.0; FEATURE_DIM],
            vec![0.0; 3],
            vec![0.0; 24],
            vec![0.0; 7],
            vec![0.0; 32],
            vec![0.0; in_dim * (n_heads + 1)],
            b_out,
            global_logits,
            vec![0.0; 3],
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"UDB1");
        bytes.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        bytes.extend_from_slice(meta.as_bytes());
        for t in tensors {
            for v in t {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        Arc::new(UserDebiasModel::from_bytes(&bytes).unwrap())
    }

    fn sequence_ipc(rows: usize) -> Vec<u8> {
        let list_field = Arc::new(Field::new("item", DataType::Boolean, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new(seq::COL_IMPRESSED_TIME_MS, DataType::Int64, false),
            Field::new(seq::COL_DWELL_TIME_MS, DataType::Int64, true),
            Field::new(seq::COL_FIRST_VIDEO_DURATION_MS, DataType::Int64, true),
            Field::new(seq::COL_TIMEZONE_ID, DataType::Int32, true),
            Field::new(seq::COL_PRODUCT_SURFACE, DataType::Int32, true),
            Field::new(seq::COL_HAS_MEDIA, DataType::Boolean, true),
            Field::new(
                seq::COL_IS_AUTHOR_FOLLOWED_BY_VIEWER,
                DataType::Boolean,
                true,
            ),
            Field::new(seq::COL_TWEET_POSITION, DataType::Int32, true),
            Field::new(seq::COL_MEDIA_CONTENT_TYPE, DataType::Int32, true),
            Field::new(
                seq::COL_ACTION_NAME_MULTI_HOT,
                DataType::FixedSizeList(list_field.clone(), 64),
                true,
            ),
        ]));
        let mut hot = vec![false; rows * 64];
        for r in 0..rows {
            hot[r * 64 + 1] = r % 2 == 0;
            hot[r * 64 + 11] = true;
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..rows).map(|r| 1_699_990_000_000 + r as i64 * 60_000),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..rows).map(|r| 500 + r as i64),
                )),
                Arc::new(Int64Array::from(vec![0i64; rows])),
                Arc::new(Int32Array::from(vec![1i32; rows])),
                Arc::new(Int32Array::from(vec![1i32; rows])),
                Arc::new(BooleanArray::from(vec![false; rows])),
                Arc::new(BooleanArray::from(vec![true; rows])),
                Arc::new(Int32Array::from_iter_values((0..rows).map(|r| r as i32))),
                Arc::new(Int32Array::from(vec![0i32; rows])),
                Arc::new(FixedSizeListArray::new(
                    list_field,
                    64,
                    Arc::new(BooleanArray::from(hot)),
                    None,
                )),
            ],
        )
        .unwrap();
        let mut buf = Vec::new();
        let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        buf
    }

    #[tokio::test]
    async fn user_debias_recenters_fav_probability_before_fusion() {
        let (p, user_logit, global_logit, clip) = (0.3, 1.5f32, 0.5f32, 2.0);
        let mut req = request(vec![candidate(1, 10, p, None, true, 1.0)], "DE");
        req.experiment_payload = RequestPayload {
            user_sequence_arrow_ipc: sequence_ipc(5).into(),
            ..Default::default()
        }
        .encode_to_vec()
        .into();

        let debias = DebiasModels {
            user: Some(toy_model(user_logit, global_logit)),
            user_changed_at_ms: 0,
        };
        let ranked = rank(req, context(config(), debias)).await.unwrap();

        let debiased_p = debiased_probability(p, user_logit, global_logit, clip);
        assert!((debiased_p - p).abs() > 0.05);
        let expected = 2.0 * debiased_p + NEGATIVE_SCORES_OFFSET;
        let raw = 2.0 * p + NEGATIVE_SCORES_OFFSET;
        let got = ranked[0].score;
        assert!((got - expected).abs() < 1e-12, "got {got} want {expected}");
        assert!(got < raw, "debiased {got} should be below raw {raw}");
    }
}
