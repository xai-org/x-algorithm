use super::author::VIEWER_LOOKUPS;
use super::PostEvaluator;
use crate::hydration::{
    Cause, HydratedTweet, Hydration, HydrationPlan, HydrationRequest, Lookup, Unresolved,
};
use crate::metrics::RequestMetrics;
use crate::models::{
    AuthorId, Evaluation, NotificationKind, Placement, RawCandidate, RequestFacts, RetweetMerge,
    TweetId,
};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use tracing::warn;

impl PostEvaluator {
    pub(crate) async fn evaluate_post_notifications(
        &self,
        level: SafetyLevel,
        facts: RequestFacts,
        posts: &[(TweetId, NotificationKind, AuthorId)],
        metrics: &RequestMetrics,
    ) -> HashMap<(TweetId, NotificationKind, AuthorId), Option<Evaluation>> {
        let Some(plan) = self.rule_engine.post_kinds_plan(level) else {
            warn!(
                ?level,
                "post notifications evaluated at a level without post kinds; failing them"
            );
            return posts.iter().map(|&key| (key, None)).collect();
        };
        let mut ids: Vec<TweetId> = posts.iter().map(|&(post, ..)| post).collect();
        ids.sort_unstable_by_key(|post| post.0);
        ids.dedup();
        let hydration = self.hydrate_posts(plan, facts, &ids, metrics).await;
        let viewer_failed = !hydration
            .viewer_failed()
            .intersection(VIEWER_LOOKUPS)
            .is_empty();
        let mut admitted = HashMap::new();
        for &(_, kind, _) in posts {
            admitted
                .entry(kind)
                .or_insert_with(|| self.rule_engine.admit_post_kind(level, kind, &hydration));
        }
        posts
            .iter()
            .map(|&(post, kind, actor)| {
                let evaluation = match hydration.tweet(post) {
                    _ if viewer_failed => None,
                    Some(HydratedTweet::Unresolved { reason, .. }) => Some(unresolved(reason)),
                    Some(HydratedTweet::Resolved { candidate, .. })
                        if AuthorId(candidate.author.id) != actor =>
                    {
                        Some(Evaluation::NotFound(Lookup::Tweet))
                    }
                    Some(HydratedTweet::Resolved { .. }) => admitted
                        .get(&kind)
                        .and_then(Option::as_ref)
                        .and_then(|admitted| admitted.post(post, false, Placement::Standalone))
                        .map(|post| post.evaluate()),
                    None => None,
                };
                ((post, kind, actor), evaluation)
            })
            .collect()
    }

    pub(crate) async fn resolve_recipient_posts(
        &self,
        level: SafetyLevel,
        facts: RequestFacts,
        posts: Vec<TweetId>,
        metrics: &RequestMetrics,
    ) -> HashMap<TweetId, Result<(), Evaluation>> {
        let Some(plan) = self.rule_engine.recipient_posts_plan(level) else {
            warn!(
                ?level,
                "recipient posts resolved at a level whose kinds name none; failing them"
            );
            return posts
                .into_iter()
                .map(|post| (post, Err(Evaluation::Failed(Lookup::Tweet))))
                .collect();
        };
        let recipient = facts.viewer_id();
        let hydration = self.hydrate_posts(plan, facts, &posts, metrics).await;
        posts
            .into_iter()
            .map(|post| {
                let resolved = match hydration.tweet(post) {
                    Some(HydratedTweet::Resolved { candidate, .. })
                        if Some(candidate.author.id) == recipient =>
                    {
                        Ok(())
                    }
                    Some(HydratedTweet::Resolved { .. }) => {
                        Err(Evaluation::NotFound(Lookup::Tweet))
                    }
                    Some(HydratedTweet::Unresolved { reason, .. }) => Err(unresolved(reason)),
                    None => Err(Evaluation::Failed(Lookup::Tweet)),
                };
                (post, resolved)
            })
            .collect()
    }

    async fn hydrate_posts(
        &self,
        plan: &HydrationPlan,
        facts: RequestFacts,
        posts: &[TweetId],
        metrics: &RequestMetrics,
    ) -> Hydration {
        let raw = posts
            .iter()
            .map(|&tweet_id| RawCandidate {
                tweet_id,
                request_author_id: None,
            })
            .collect();
        plan.hydrate(
            &*self.sources,
            HydrationRequest::new(
                facts,
                raw,
                RetweetMerge::Unmerged,
                &self.fallback_caches,
                metrics,
            ),
        )
        .await
    }
}

fn unresolved(&Unresolved { lookup, cause }: &Unresolved) -> Evaluation {
    match cause {
        Cause::NotFound => Evaluation::NotFound(lookup),
        Cause::Failed => Evaluation::Failed(lookup),
    }
}
