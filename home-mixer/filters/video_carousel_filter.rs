use crate::models::brand_safety::BrandSafetyVerdict;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableVideoCarouselAuthorDedupe, EnableVideoCarouselEmbeddingDedupe,
    EnableVideoCarouselHighRiskFilter, EnableVideoCarouselMediumRiskFilter,
    VideoCarouselEmbeddingDedupeThreshold, VideoCarouselSemanticIdDedupeLevels, RESULT_SIZE,
};
use crate::util::video_carousel::{self, CarouselScoring};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct VideoCarouselFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for VideoCarouselFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| candidate.video_carousel_extra);
        let main_feed = &kept[..RESULT_SIZE.min(kept.len())];
        let drop_high_risk = query.params.get(EnableVideoCarouselHighRiskFilter);
        let drop_medium_risk = query.params.get(EnableVideoCarouselMediumRiskFilter);
        let _ = query.video_carousel.set(video_carousel::select_videos(
            kept.iter()
                .skip(RESULT_SIZE)
                .chain(&removed)
                .filter(|candidate| match candidate.brand_safety_verdict {
                    Some(BrandSafetyVerdict::HighRisk) => !drop_high_risk,
                    Some(BrandSafetyVerdict::MediumRisk) => !drop_medium_risk,
                    Some(_) => true,
                    None => !drop_high_risk && !drop_medium_risk,
                }),
            main_feed,
            &CarouselScoring::from_query(query),
            query.params.get(EnableVideoCarouselAuthorDedupe),
            query.params.get(VideoCarouselSemanticIdDedupeLevels) as usize,
            query
                .params
                .get(EnableVideoCarouselEmbeddingDedupe)
                .then(|| query.params.get(VideoCarouselEmbeddingDedupeThreshold)),
        ));
        FilterResult { kept, removed }
    }
}
