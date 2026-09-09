pub mod batch;
pub mod exclusive_content_hydrator;
pub(crate) mod fallback_cache;
pub mod gizmoduck_hydrator;
pub mod metrics;
pub mod safety_label_hydrator;
pub mod socialgraph_hydrator;
pub mod tes_hydrator;
pub mod viewer_hydrator;

use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::clients::socialgraph_client::SocialgraphClient;
use crate::models::{
    assemble, resolve_candidates, AuthorFeatures, AuthorId, ExclusiveContentFeatures,
    HydratedTweetCandidate, RawCandidate, SafetyLabelMap, TweetCandidateInput, TweetFeatures,
    TweetId, Viewer, ViewerAuthorRelationship, ViewerFeatures,
};
use crate::rules::SafetyLevel;
use crate::safety_label_source::SafetyLabelSource;
use batch::TweetHydrationBatch;
use exclusive_content_hydrator::ExclusiveContentHydrator;
use fallback_cache::FallbackCache;
use gizmoduck_hydrator::GizmoduckAuthorHydrator;
use safety_label_hydrator::{SafetyLabelHydration, SafetyLabelHydrator};
use socialgraph_hydrator::SocialgraphHydrator;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tes_hydrator::TesHydrator;
use viewer_hydrator::ViewerHydrator;
use xai_core_entities::gizmoduck_client::GizmoduckClient;
use xai_core_entities::tweet_entity_service_client::TESClient;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) struct HydrationRequest<'a> {
    viewer_id: Option<u64>,
    country_code: Option<String>,
    raw_candidates: &'a [RawCandidate],
    safety_level: SafetyLevel,
}

impl<'a> HydrationRequest<'a> {
    pub(crate) fn new(
        viewer_id: Option<u64>,
        country_code: Option<String>,
        raw_candidates: &'a [RawCandidate],
        safety_level: SafetyLevel,
    ) -> Self {
        Self {
            viewer_id,
            country_code,
            raw_candidates,
            safety_level,
        }
    }
}

struct CandidateFeatures {
    tweet_features: HashMap<TweetId, TweetFeatures>,
    author_features: TweetHydrationBatch<AuthorFeatures>,
    safety_labels: HashMap<TweetId, SafetyLabelMap>,
    relationships: TweetHydrationBatch<ViewerAuthorRelationship>,
    tes_hydration_failed: HashSet<TweetId>,
    exclusive_content: HashMap<TweetId, Option<ExclusiveContentFeatures>>,
}

impl CandidateFeatures {
    fn assemble(self, candidates: &[TweetCandidateInput]) -> Vec<HydratedTweetCandidate> {
        candidates
            .iter()
            .map(|c| {
                let safety_hydration_failed = !self.safety_labels.contains_key(&c.tweet_id)
                    || self.tes_hydration_failed.contains(&c.tweet_id);
                assemble(
                    c,
                    self.tweet_features
                        .get(&c.tweet_id)
                        .cloned()
                        .unwrap_or_default(),
                    self.author_features.get_or_default(&c.tweet_id),
                    self.safety_labels
                        .get(&c.tweet_id)
                        .cloned()
                        .unwrap_or_default(),
                    self.relationships.get_or_default(&c.tweet_id),
                    safety_hydration_failed,
                    self.exclusive_content
                        .get(&c.tweet_id)
                        .cloned()
                        .unwrap_or_default(),
                )
            })
            .collect()
    }
}

pub(crate) fn tweets_per_author(candidates: &[TweetCandidateInput]) -> HashMap<AuthorId, usize> {
    let mut candidate_count_by_key = HashMap::with_capacity(candidates.len());
    for candidate in candidates {
        *candidate_count_by_key
            .entry(candidate.author_id)
            .or_default() += 1;
    }
    candidate_count_by_key
}

pub(crate) fn keyed_by_author<V>(
    expected: &HashMap<AuthorId, usize>,
    response: HashMap<u64, V>,
) -> HashMap<AuthorId, V> {
    let author_by_raw: HashMap<u64, AuthorId> = expected
        .keys()
        .map(|&author| (author.get(), author))
        .collect();
    response
        .into_iter()
        .filter_map(|(id, value)| author_by_raw.get(&id).map(|&author| (author, value)))
        .collect()
}

pub(crate) struct HydrationPipeline {
    viewer_hydrator: ViewerHydrator,
    tes_hydrator: TesHydrator,
    gizmoduck_author_hydrator: GizmoduckAuthorHydrator,
    socialgraph_hydrator: SocialgraphHydrator,
    safety_label_hydrator: SafetyLabelHydrator,
    exclusive_content_hydrator: ExclusiveContentHydrator,
}

pub(crate) struct HydrationOutput {
    pub(crate) viewer_features: ViewerFeatures,
    pub(crate) candidates: Vec<HydratedTweetCandidate>,
    pub(crate) safety_labels: HashMap<TweetId, Arc<vf_pb::SafetyLabelMap>>,
}

impl HydrationPipeline {
    pub(crate) fn new(
        tes_client: Arc<dyn TESClient + Send + Sync>,
        gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
        socialgraph_client: Arc<dyn SocialgraphClient + Send + Sync>,
        safety_label_source: Arc<SafetyLabelSource>,
        fallback_cache: Option<FallbackCache<AuthorId, AuthorFeatures>>,
    ) -> Self {
        Self {
            viewer_hydrator: ViewerHydrator {
                gizmoduck_client: gizmoduck_client.clone(),
            },
            tes_hydrator: TesHydrator {
                tes_client: tes_client.clone(),
            },
            gizmoduck_author_hydrator: GizmoduckAuthorHydrator::new(
                GizmoduckLookup::new(gizmoduck_client),
                fallback_cache,
            ),
            socialgraph_hydrator: SocialgraphHydrator {
                sg_client: socialgraph_client.clone(),
            },
            safety_label_hydrator: SafetyLabelHydrator {
                source: safety_label_source,
            },
            exclusive_content_hydrator: ExclusiveContentHydrator {
                tes_client,
                sg_client: socialgraph_client,
            },
        }
    }

    pub(crate) async fn hydrate(&self, request: HydrationRequest<'_>) -> HydrationOutput {
        let HydrationRequest {
            viewer_id,
            country_code,
            raw_candidates,
            safety_level,
        } = request;
        let viewer = viewer_id.map_or(Viewer::LoggedOut, Viewer::LoggedIn);
        let viewer_hydration = self
            .viewer_hydrator
            .hydrate(viewer_id, country_code, safety_level);
        let candidate_hydration = async {
            let tweet_ids: Vec<TweetId> = raw_candidates.iter().map(|c| c.tweet_id).collect();
            let independent_group = async {
                tokio::join!(
                    self.safety_label_hydrator.hydrate(&tweet_ids, safety_level),
                    self.tes_hydrator.hydrate_tweets(&tweet_ids, safety_level),
                    self.exclusive_content_hydrator
                        .hydrate(&tweet_ids, viewer, safety_level),
                )
            };

            let author_hop = async {
                let core_datas = self
                    .tes_hydrator
                    .fetch_pure_core(&tweet_ids, safety_level)
                    .await;
                let candidates = resolve_candidates(raw_candidates, &core_datas);
                let (author_features, relationships) = tokio::join!(
                    self.gizmoduck_author_hydrator
                        .hydrate(&candidates, safety_level),
                    self.socialgraph_hydrator
                        .hydrate(&candidates, viewer, safety_level),
                );
                (core_datas, candidates, author_features, relationships)
            };

            let (
                (safety_labels, tes_tweet_keyed, exclusive_content),
                (core_datas, candidates, author_features, relationships),
            ) = tokio::join!(independent_group, author_hop);

            let SafetyLabelHydration {
                label_types,
                label_response,
            } = safety_labels;

            let tweet_features = self.tes_hydrator.assemble_tweet_features(
                &candidates,
                &core_datas,
                &tes_tweet_keyed,
            );

            let tes_hydration_failed = candidates
                .iter()
                .map(|c| c.tweet_id)
                .filter(|id| tes_tweet_keyed.safety_hydration_failed(id, &core_datas))
                .collect();
            let features = CandidateFeatures {
                tweet_features,
                author_features,
                safety_labels: label_types,
                relationships,
                tes_hydration_failed,
                exclusive_content,
            };
            let hydrated_candidates = features.assemble(&candidates);

            (hydrated_candidates, label_response)
        };

        let (viewer_features, (candidates, safety_labels)) =
            tokio::join!(viewer_hydration, candidate_hydration);

        HydrationOutput {
            viewer_features,
            candidates,
            safety_labels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resolve_candidate;
    use xai_core_entities::entities::PureCoreData;

    fn core(tweet_id: u64, author_id: u64) -> HashMap<TweetId, PureCoreData> {
        HashMap::from([(
            TweetId(tweet_id),
            PureCoreData {
                author_id,
                ..Default::default()
            },
        )])
    }

    #[test]
    fn assemble_handles_mismatched_cardinality_without_mispairing() {
        let resolved = resolve_candidate(&raw(2, None), &core(2, 200)).expect("tweet 2 resolves");
        let candidates = vec![resolved];

        let results = CandidateFeatures {
            tweet_features: HashMap::from([
                (TweetId(1), TweetFeatures::default()),
                (
                    TweetId(2),
                    TweetFeatures {
                        core: crate::models::CoreFeature {
                            source_tweet_id: Some(2),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ),
            ]),
            author_features: TweetHydrationBatch::from_results(
                [TweetId(2)],
                HashMap::from([(
                    TweetId(2),
                    Ok::<_, anyhow::Error>(Some(AuthorFeatures {
                        is_suspended: true,
                        ..Default::default()
                    })),
                )]),
            ),
            safety_labels: HashMap::from([
                (TweetId(1), SafetyLabelMap::default()),
                (TweetId(2), SafetyLabelMap::default()),
            ]),
            relationships: TweetHydrationBatch::from_results(
                [TweetId(2)],
                HashMap::from([(
                    TweetId(2),
                    Ok::<_, anyhow::Error>(Some(ViewerAuthorRelationship {
                        viewer_follows_author: true,
                        ..Default::default()
                    })),
                )]),
            ),
            tes_hydration_failed: HashSet::new(),
            exclusive_content: HashMap::new(),
        };

        let assembled = results.assemble(&candidates);

        assert_eq!(assembled.len(), 1);
        let c = &assembled[0];
        assert_eq!(c.tweet_id, 2);
        assert_eq!(c.author_id, 200);
        assert_eq!(c.tweet_features.core.source_tweet_id, Some(2));
        assert!(c.author_features.is_suspended);
        assert!(c.relationship.viewer_follows_author);
        assert!(!c.safety_hydration_failed);
    }

    #[test]
    fn assemble_marks_failed_safety_label_hydration_without_dropping_candidate() {
        let candidate = resolve_candidate(&raw(1, Some(100)), &HashMap::new()).unwrap();
        let results = CandidateFeatures {
            tweet_features: HashMap::new(),
            author_features: TweetHydrationBatch::from_results(
                [TweetId(1)],
                HashMap::from([(
                    TweetId(1),
                    Ok::<_, anyhow::Error>(Some(AuthorFeatures::default())),
                )]),
            ),
            safety_labels: HashMap::new(),
            relationships: TweetHydrationBatch::from_results(
                [TweetId(1)],
                HashMap::from([(
                    TweetId(1),
                    Ok::<_, anyhow::Error>(Some(ViewerAuthorRelationship::default())),
                )]),
            ),
            tes_hydration_failed: HashSet::new(),
            exclusive_content: HashMap::new(),
        };

        let assembled = results.assemble(&[candidate]);

        assert_eq!(assembled.len(), 1);
        assert!(assembled[0].safety_hydration_failed);
    }

    #[test]
    fn assemble_marks_failed_tes_tweet_flag_hydration() {
        let candidate = resolve_candidate(&raw(1, Some(100)), &HashMap::new()).unwrap();
        let results = CandidateFeatures {
            tweet_features: HashMap::new(),
            author_features: TweetHydrationBatch::from_results(
                [TweetId(1)],
                HashMap::from([(
                    TweetId(1),
                    Ok::<_, anyhow::Error>(Some(AuthorFeatures::default())),
                )]),
            ),
            safety_labels: HashMap::from([(TweetId(1), SafetyLabelMap::default())]),
            relationships: TweetHydrationBatch::from_results(
                [TweetId(1)],
                HashMap::from([(
                    TweetId(1),
                    Ok::<_, anyhow::Error>(Some(ViewerAuthorRelationship::default())),
                )]),
            ),
            tes_hydration_failed: HashSet::from([TweetId(1)]),
            exclusive_content: HashMap::new(),
        };

        let assembled = results.assemble(&[candidate]);

        assert_eq!(assembled.len(), 1);
        assert!(assembled[0].safety_hydration_failed);
    }

    fn raw(tweet_id: u64, request_author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id,
        }
    }

    #[test]
    fn author_candidate_counts_deduplicate_shared_authors() {
        let candidates: Vec<_> = [(1, 10), (2, 10), (3, 20)]
            .into_iter()
            .map(|(tweet_id, author_id)| {
                resolve_candidate(&raw(tweet_id, Some(author_id)), &HashMap::new()).unwrap()
            })
            .collect();

        let counts: HashMap<u64, usize> = tweets_per_author(&candidates)
            .into_iter()
            .map(|(author, count)| (author.get(), count))
            .collect();
        assert_eq!(counts, HashMap::from([(10, 2), (20, 1)]));
    }
}
