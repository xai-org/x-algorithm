use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{ArticleLifecycle, PostCandidate};
use crate::rules::context::At;
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

pub(in crate::rules) enum ArticleStage {}

impl Feature for ArticleStage {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ArticleLifecycle);
    type Value<'a> = Option<ArticleLifecycle>;
}

impl Reads<PostCandidate> for ArticleStage {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> Option<ArticleLifecycle> {
        at.post().article_lifecycle
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ArticleStageLeaf {
    TweetArticleIsPublished,
}

impl Leaf for ArticleStageLeaf {
    type Feature = ArticleStage;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ArticleStage>>(self, m: M) -> M::Out {
        match self {
            Self::TweetArticleIsPublished => {
                m.run(|lifecycle, _| lifecycle == Some(ArticleLifecycle::Published))
            }
        }
    }
}

impl ArticleStageLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::ArticleStage(self))
    }
}

pub(in crate::rules) enum ArticleMediaEntities {}

impl Feature for ArticleMediaEntities {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ArticleMediaEntities);
    type Value<'a> = bool;
}

impl Reads<PostCandidate> for ArticleMediaEntities {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> bool {
        at.post().article_has_media_entities
    }
}

pub(in crate::rules) enum ArticleCoverMedia {}

impl Feature for ArticleCoverMedia {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ArticleCoverMedia);
    type Value<'a> = bool;
}

impl Reads<PostCandidate> for ArticleCoverMedia {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> bool {
        at.post().article_has_cover_media
    }
}
