use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use prost::Message;
use xai_feature_switches::Params;
use xai_user_debias::{
    debiased_probability, featurize_sequence, parse_sequence, request_context, Prediction,
    UserDebiasModel,
};
use xai_value_model::{CandidateScoringInputs, PhoenixScores};
use xai_vm_ranker_proto::experiment::RequestPayload;
use xai_vm_ranker_proto::RankRequest;

use crate::metrics::{USER_DEBIAS_PREDICT_DURATION, USER_DEBIAS_REQUESTS};
use crate::params::*;

const DEFAULT_HEADS: &[&str] = &[
    "fav",
    "reply",
    "quote",
    "retweet",
    "dwelled",
    "not_dwelled",
    "vqv",
    "photo_expand",
    "click",
    "profile_click",
    "bookmark",
    "share",
    "open_link",
    "video_open",
];
const PROB_FLOOR: f64 = 1e-7;
const HOME_SURFACE: i32 = 1;

#[derive(Clone, Copy)]
struct Strength {
    alpha: f64,
    clip: f64,
}

struct Config {
    alpha: f64,
    clip: f64,
    heads: Vec<String>,
    head_alpha: HashMap<String, f64>,
    head_clip: HashMap<String, f64>,
    max_artifact_age_secs: f64,
}

impl Config {
    fn from_params(params: &Params) -> Result<Self, Skip> {
        let config = Self {
            alpha: params.get(UserDebiasAlpha),
            clip: params.get(UserDebiasClipLogit),
            heads: csv_list(&params.get(UserDebiasHeads)),
            head_alpha: head_values(&params.get(UserDebiasHeadAlpha))?,
            head_clip: head_values(&params.get(UserDebiasHeadClipLogit))?,
            max_artifact_age_secs: params.get(UserDebiasMaxArtifactAgeSecs),
        };
        let strengths = std::iter::once(config.default_strength())
            .chain(config.head_alpha.keys().map(|h| config.strength(h)))
            .chain(config.head_clip.keys().map(|h| config.strength(h)));
        for s in strengths {
            if !s.alpha.is_finite() || !s.clip.is_finite() || s.clip < 0.0 {
                return Err(Skip::InvalidParams);
            }
        }
        Ok(config)
    }

    fn default_strength(&self) -> Strength {
        Strength {
            alpha: self.alpha,
            clip: self.clip,
        }
    }

    fn strength(&self, head: &str) -> Strength {
        Strength {
            alpha: self.head_alpha.get(head).copied().unwrap_or(self.alpha),
            clip: self.head_clip.get(head).copied().unwrap_or(self.clip),
        }
    }

    fn heads(&self) -> Vec<&str> {
        if self.heads.is_empty() {
            DEFAULT_HEADS.to_vec()
        } else {
            self.heads.iter().map(String::as_str).collect()
        }
    }
}

#[derive(Clone, Copy)]
enum Skip {
    ModelNotLoaded,
    InvalidParams,
    StaleArtifact,
    NoSequence,
    SequenceError,
    FeatureMismatch,
}

impl Skip {
    fn label(self) -> &'static str {
        match self {
            Skip::ModelNotLoaded => "model_not_loaded",
            Skip::InvalidParams => "invalid_params",
            Skip::StaleArtifact => "stale_artifact",
            Skip::NoSequence => "no_sequence",
            Skip::SequenceError => "sequence_error",
            Skip::FeatureMismatch => "feature_mismatch",
        }
    }
}

type ScoreField = fn(&mut PhoenixScores) -> &mut Option<f64>;

struct Recentering {
    heads: Vec<(ScoreField, usize, Strength)>,
    prediction: Prediction,
}

pub fn recenter(
    req: &RankRequest,
    params: &Params,
    model: Option<&UserDebiasModel>,
    inputs: &[CandidateScoringInputs],
) -> Option<Vec<CandidateScoringInputs>> {
    if !params.get(UserDebiasEnabled) {
        return None;
    }
    match Recentering::for_request(req, params, model) {
        Ok(recentering) => Some(
            inputs
                .iter()
                .map(|c| {
                    let mut out = c.clone();
                    recentering.apply(&mut out);
                    out
                })
                .collect(),
        ),
        Err(skip) => {
            USER_DEBIAS_REQUESTS
                .with_label_values(&[skip.label()])
                .inc();
            None
        }
    }
}

impl Recentering {
    fn for_request(
        req: &RankRequest,
        params: &Params,
        model: Option<&UserDebiasModel>,
    ) -> Result<Self, Skip> {
        let model = model.ok_or(Skip::ModelNotLoaded)?;
        let config = Config::from_params(params)?;
        let artifact_age_ms = now_ms() - model.trained_at_ms();
        if config.max_artifact_age_secs > 0.0
            && artifact_age_ms as f64 > config.max_artifact_age_secs * 1000.0
        {
            return Err(Skip::StaleArtifact);
        }
        let payload = RequestPayload::decode(req.experiment_payload.as_ref()).unwrap_or_default();
        if payload.user_sequence_arrow_ipc.is_empty() {
            return Err(Skip::NoSequence);
        }

        let started = Instant::now();
        let mut rows =
            parse_sequence(&payload.user_sequence_arrow_ipc).map_err(|_| Skip::SequenceError)?;
        let request_ms = req.viewer.as_ref().map_or_else(now_ms, |v| v.now_ms);
        let (req_ms, path) = match payload.engine_time_hint_ms {
            Some(scored_at_ms) if scored_at_ms < request_ms => (scored_at_ms, "cached"),
            Some(scored_at_ms) => (scored_at_ms, "fresh"),
            None => (request_ms, "no_hint"),
        };
        if path == "cached" {
            rows.retain_until(req_ms);
        }
        if rows.is_empty() {
            return Err(Skip::NoSequence);
        }
        let features = featurize_sequence(&rows, req_ms);
        if features.len() != model.feature_dim() {
            return Err(Skip::FeatureMismatch);
        }
        let surface = if payload.product_surface == 0 {
            HOME_SURFACE
        } else {
            payload.product_surface
        };
        let tz = rows.tz.last().copied().unwrap_or(0);
        let ctx = request_context(req_ms, tz, payload.client_app_id, surface);
        let prediction = model.predict(&features, &ctx);
        USER_DEBIAS_PREDICT_DURATION.observe(started.elapsed().as_secs_f64());
        USER_DEBIAS_REQUESTS.with_label_values(&[path]).inc();

        let head_index = |name: &str| model.head_names().iter().position(|h| h == name);
        let heads = config
            .heads()
            .into_iter()
            .filter_map(|name| Some((head_field(name)?, head_index(name)?, config.strength(name))))
            .collect();
        Ok(Self { heads, prediction })
    }

    fn apply(&self, out: &mut CandidateScoringInputs) {
        for &(field, idx, strength) in &self.heads {
            let slot = field(&mut out.phoenix_scores);
            if let Some(p) = *slot {
                *slot = Some(self.recenter_probability(p, idx, strength));
            }
        }
    }

    fn recenter_probability(&self, p: f64, head_idx: usize, strength: Strength) -> f64 {
        if !(PROB_FLOOR..=1.0 - PROB_FLOOR).contains(&p) {
            return p;
        }
        let debiased = debiased_probability(
            p,
            self.prediction.user_logits[head_idx],
            self.prediction.global_logits[head_idx],
            strength.clip,
        );
        if !debiased.is_finite() {
            return p;
        }
        ((1.0 - strength.alpha) * p + strength.alpha * debiased).clamp(PROB_FLOOR, 1.0 - PROB_FLOOR)
    }
}

fn head_field(head: &str) -> Option<ScoreField> {
    let field: ScoreField = match head {
        "fav" => |s| &mut s.favorite_score,
        "reply" => |s| &mut s.reply_score,
        "quote" => |s| &mut s.quote_score,
        "retweet" => |s| &mut s.retweet_score,
        "dwelled" => |s| &mut s.dwell_score,
        "not_dwelled" => |s| &mut s.not_dwelled_score,
        "vqv" => |s| &mut s.vqv_score,
        "photo_expand" => |s| &mut s.photo_expand_score,
        "report" => |s| &mut s.report_score,
        "not_interested" => |s| &mut s.not_interested_score,
        "follow_author" => |s| &mut s.follow_author_score,
        "block" => |s| &mut s.block_author_score,
        "mute" => |s| &mut s.mute_author_score,
        "click" => |s| &mut s.click_score,
        "profile_click" => |s| &mut s.profile_click_score,
        "share" => |s| &mut s.share_score,
        "share_copylink" => |s| &mut s.share_via_copy_link_score,
        "share_dm" => |s| &mut s.share_via_dm_score,
        "open_link" => |s| &mut s.open_link_score,
        "video_open" => |s| &mut s.video_open_score,
        "post_unexplored" => |s| &mut s.post_unexplored_score,
        "quoted_click" => |s| &mut s.quoted_click_score,
        "quoted_vqv" => |s| &mut s.quoted_vqv_score,
        _ => return None,
    };
    Some(field)
}

fn head_values(spec: &str) -> Result<HashMap<String, f64>, Skip> {
    csv_list(spec)
        .iter()
        .map(|entry| {
            let (head, value) = entry.split_once(':').ok_or(Skip::InvalidParams)?;
            let value: f64 = value.trim().parse().map_err(|_| Skip::InvalidParams)?;
            Ok((head.trim().to_string(), value))
        })
        .collect()
}

fn csv_list(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(str::trim)
        .filter(|h| !h.is_empty())
        .map(String::from)
        .collect()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}
