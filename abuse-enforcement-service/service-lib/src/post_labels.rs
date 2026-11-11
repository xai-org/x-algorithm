use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use xai_abuse_thrift_codec::{Transcoder, SCHEMA_AIS};
use xai_visibility_filtering::tweet_safety_label::{
    ProdTweetSafetyLabelClient, TweetSafetyLabelClient,
};

use crate::decision::ActionSpec;

const SAFETY_LABEL_TYPE: &str = "com.twitter.spam.rtf.thriftscala.SafetyLabelType";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub type CarriedLabels = BTreeMap<String, Option<i64>>;

#[async_trait::async_trait]
pub trait PostLabelReader: Send + Sync {
    async fn labels(&self, post_id: i64, now_ms: i64) -> Result<Option<CarriedLabels>>;
}

pub struct VfPostLabels {
    client: Arc<dyn TweetSafetyLabelClient>,
    names: HashMap<i32, String>,
}

impl VfPostLabels {
    pub async fn connect(zone: &str) -> Result<Self> {
        let client = tokio::time::timeout(CONNECT_TIMEOUT, ProdTweetSafetyLabelClient::new(zone))
            .await
            .map_err(|_| anyhow!("timed out after {CONNECT_TIMEOUT:?}"))?
            .map_err(|s| anyhow!("{s}"))?;
        Self::new(Arc::new(client))
    }

    pub fn new(client: Arc<dyn TweetSafetyLabelClient>) -> Result<Self> {
        Ok(Self {
            client,
            names: label_names()?,
        })
    }
}

pub fn label_names() -> Result<HashMap<i32, String>> {
    let schema = Transcoder::from_schema_bytes(SCHEMA_AIS).context("AIS thrift schema")?;
    schema
        .enum_names(SAFETY_LABEL_TYPE)
        .cloned()
        .ok_or_else(|| anyhow!("the AIS schema has no {SAFETY_LABEL_TYPE}"))
}

#[async_trait::async_trait]
impl PostLabelReader for VfPostLabels {
    async fn labels(&self, post_id: i64, now_ms: i64) -> Result<Option<CarriedLabels>> {
        let id = u64::try_from(post_id).context("post id")?;
        let batch = self
            .client
            .get_safety_labels(vec![id])
            .await
            .map_err(|s| anyhow!("GetSafetyLabels: {s}"))?;
        if let Some(failure) = batch.failures.get(&id) {
            bail!("GetSafetyLabels could not read the post: {failure:?}");
        }
        Ok(batch.labels.get(&id).map(|map| {
            carried(
                map.iter().map(|(t, l)| StoredLabel {
                    type_id: t.0,
                    expires_at_ms: l.expires_at_msec,
                    limited: l.applicable_users.as_ref().is_some_and(|u| !u.is_empty())
                        || l.applicable_countries
                            .as_ref()
                            .is_some_and(|c| !c.is_empty())
                        || l.holdback_experiment == Some(true),
                }),
                &self.names,
                now_ms,
            )
        }))
    }
}

pub struct StoredLabel {
    pub type_id: i32,
    pub expires_at_ms: Option<i64>,
    pub limited: bool,
}

pub fn carried(
    labels: impl IntoIterator<Item = StoredLabel>,
    names: &HashMap<i32, String>,
    now_ms: i64,
) -> CarriedLabels {
    labels
        .into_iter()
        .filter(|l| !l.limited && l.expires_at_ms.is_none_or(|e| e > now_ms))
        .map(|l| {
            let name = names
                .get(&l.type_id)
                .cloned()
                .unwrap_or_else(|| format!("UnknownLabel({})", l.type_id));
            (name, l.expires_at_ms)
        })
        .collect()
}

pub fn drop_carried(
    specs: &mut Vec<ActionSpec>,
    carried: &CarriedLabels,
    now_ms: i64,
) -> Vec<String> {
    let mut dropped = Vec::new();
    for spec in specs.iter_mut() {
        if let ActionSpec::AddPostLabelsV2 { labels, ttl_msec } = spec {
            let until = ttl_msec.map(|t| now_ms.saturating_add(t));
            labels.retain(|l| {
                let lasts = carried.get(l).is_some_and(|exp| match (exp, until) {
                    (None, _) => true,
                    (Some(_), None) => false,
                    (Some(exp), Some(until)) => *exp >= until,
                });
                if lasts {
                    dropped.push(l.clone());
                }
                !lasts
            });
        }
    }
    specs.retain(|s| !matches!(s, ActionSpec::AddPostLabelsV2 { labels, .. } if labels.is_empty()));
    dropped.sort_unstable();
    dropped
}

pub fn sent_labels(specs: &[ActionSpec]) -> impl Iterator<Item = &str> {
    specs
        .iter()
        .flat_map(|s| match s {
            ActionSpec::AddPostLabelsV2 { labels, .. } => labels.as_slice(),
            _ => &[],
        })
        .map(String::as_str)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    pub(crate) struct FakePostLabels {
        pub posts: Mutex<HashMap<i64, std::result::Result<CarriedLabels, String>>>,
        pub reads: Mutex<Vec<i64>>,
    }

    impl FakePostLabels {
        pub(crate) fn carry(&self, post_id: i64, labels: &[(&str, Option<i64>)]) {
            self.posts.lock().unwrap().insert(
                post_id,
                Ok(labels.iter().map(|(l, e)| ((*l).to_owned(), *e)).collect()),
            );
        }

        pub(crate) fn fail(&self, post_id: i64) {
            self.posts
                .lock()
                .unwrap()
                .insert(post_id, Err("label store down".into()));
        }
    }

    #[async_trait::async_trait]
    impl PostLabelReader for FakePostLabels {
        async fn labels(&self, post_id: i64, _now_ms: i64) -> Result<Option<CarriedLabels>> {
            self.reads.lock().unwrap().push(post_id);
            match self.posts.lock().unwrap().get(&post_id) {
                None => Ok(None),
                Some(Ok(labels)) => Ok(Some(labels.clone())),
                Some(Err(e)) => Err(anyhow!("{e}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_000_000_000_000;
    const DAY: i64 = 86_400_000;

    fn label(labels: &[&str], ttl_msec: Option<i64>) -> ActionSpec {
        ActionSpec::AddPostLabelsV2 {
            labels: labels.iter().map(|l| (*l).to_owned()).collect(),
            ttl_msec,
        }
    }

    fn carrying(labels: &[(&str, Option<i64>)]) -> CarriedLabels {
        labels.iter().map(|(l, e)| ((*l).to_owned(), *e)).collect()
    }

    #[test]
    fn the_ais_schema_names_every_label_the_page_sends() {
        let names = label_names().unwrap();
        let by_name: HashMap<&str, i32> = names.iter().map(|(v, n)| (n.as_str(), *v)).collect();
        assert_eq!(by_name.get("NsfwHighPrecision"), Some(&3));
        assert_eq!(by_name.get("SoftNsfw"), Some(&193));
        for label in xai_abuse_enforcement_types::action_control::PAGE_POST_LABELS
            .iter()
            .chain(&["SpamHighRecall", "RiskyHighVizReply"])
        {
            assert!(by_name.contains_key(label), "{label}");
        }
    }

    #[test]
    fn only_labels_in_force_for_every_viewer_are_carried() {
        let names = HashMap::from([
            (3, "NsfwHighPrecision".to_owned()),
            (4, "NsfwHighRecall".into()),
        ]);
        let stored = |type_id, expires_at_ms, limited| StoredLabel {
            type_id,
            expires_at_ms,
            limited,
        };
        assert_eq!(
            carried(
                [stored(3, None, false), stored(4, Some(NOW + DAY), false)],
                &names,
                NOW
            ),
            carrying(&[
                ("NsfwHighPrecision", None),
                ("NsfwHighRecall", Some(NOW + DAY))
            ])
        );
        assert!(
            carried([stored(3, Some(NOW), false)], &names, NOW).is_empty(),
            "expired"
        );
        assert!(
            carried([stored(3, None, true)], &names, NOW).is_empty(),
            "some viewers only"
        );
        assert_eq!(
            carried([stored(999, None, false)], &names, NOW),
            carrying(&[("UnknownLabel(999)", None)])
        );
    }

    #[test]
    fn a_label_carried_at_least_as_long_is_dropped() {
        let permanent = carrying(&[("NsfwHighPrecision", None)]);
        let for_a_week = carrying(&[("NsfwHighPrecision", Some(NOW + 7 * DAY))]);
        for (carried, ttl, dropped) in [
            (&permanent, None, true),
            (&permanent, Some(7 * DAY), true),
            (&for_a_week, Some(7 * DAY), true),
            (&for_a_week, Some(7 * DAY - 1), true),
            (&for_a_week, Some(30 * DAY), false),
            (&for_a_week, None, false),
        ] {
            let mut specs = vec![label(&["NsfwHighPrecision"], ttl)];
            let out = drop_carried(&mut specs, carried, NOW);
            assert_eq!(out.len() == 1, dropped, "{ttl:?}");
            assert_eq!(specs.is_empty(), dropped, "{ttl:?}");
        }
    }

    #[test]
    fn only_the_carried_labels_leave_and_other_actions_stay() {
        let carried = carrying(&[("NsfwHighPrecision", None), ("SpamHighRecall", None)]);
        let mut specs = vec![
            label(&["NsfwHighPrecision", "RiskyHighVizReply"], None),
            label(&["SpamHighRecall"], Some(30 * DAY)),
            ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into(),
            },
        ];
        let dropped = drop_carried(&mut specs, &carried, NOW);
        assert_eq!(dropped, ["NsfwHighPrecision", "SpamHighRecall"]);
        assert_eq!(
            specs,
            vec![
                label(&["RiskyHighVizReply"], None),
                ActionSpec::SuspendUser {
                    perm: false,
                    policy: "PlatformManipulation".into(),
                },
            ]
        );
        assert_eq!(
            sent_labels(&specs).collect::<Vec<_>>(),
            ["RiskyHighVizReply"]
        );
        let mut none = vec![label(&["NsfwHighPrecision"], None)];
        assert!(drop_carried(&mut none, &CarriedLabels::new(), NOW).is_empty());
        assert_eq!(none.len(), 1);
    }
}
