use super::author::viewer_lookup_failed;
use super::{push_once, PostEvaluator};
use crate::hydration::{HydratedTweet, Hydration, HydrationPlan, HydrationRequest, Lookup};
use crate::ingress::{NotificationPost, NotificationUser};
use crate::metrics::RequestMetrics;
use crate::models::{
    AuthorId, Evaluation, NotificationKind, Placement, RawCandidate, RequestFacts, RetweetMerge,
    TweetId,
};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use tracing::warn;

pub(crate) struct UserNotifications {
    actors: HashMap<AuthorId, Option<Evaluation>>,
    recipient_posts: HashMap<TweetId, Result<(), Evaluation>>,
}

impl UserNotifications {
    pub(crate) fn evaluation(&self, user: NotificationUser) -> Option<&Evaluation> {
        match user
            .recipient_post
            .map(|post| self.recipient_posts.get(&post))
        {
            None | Some(Some(Ok(()))) => self.actors.get(&user.actor).and_then(Option::as_ref),
            Some(Some(Err(evaluation))) => Some(evaluation),
            Some(None) => None,
        }
    }
}

impl PostEvaluator {
    pub(crate) async fn evaluate_user_notifications(
        &self,
        level: SafetyLevel,
        facts: &RequestFacts,
        rows: &[(NotificationKind, NotificationUser)],
        metrics: &RequestMetrics,
    ) -> UserNotifications {
        let mut actors: Vec<AuthorId> = rows.iter().map(|&(_, user)| user.actor).collect();
        actors.sort_unstable_by_key(|actor| actor.0);
        actors.dedup();
        let mut posts: Vec<TweetId> = rows
            .iter()
            .filter_map(|&(_, user)| user.recipient_post)
            .collect();
        posts.sort_unstable();
        posts.dedup();
        let (actors, recipient_posts) = tokio::join!(
            self.evaluate_users(level, facts, actors, metrics),
            self.resolve_recipient_posts(level, facts, posts, metrics),
        );
        UserNotifications {
            actors,
            recipient_posts,
        }
    }

    pub(crate) async fn evaluate_post_notifications(
        &self,
        level: SafetyLevel,
        facts: &RequestFacts,
        rows: &[(NotificationKind, NotificationPost)],
        metrics: &RequestMetrics,
    ) -> Vec<Option<Evaluation>> {
        let Some(plan) = self.rule_engine.post_kinds_plan(level) else {
            warn!(
                ?level,
                "post notifications evaluated at a level without post kinds; failing them"
            );
            return rows.iter().map(|_| None).collect();
        };
        let mut ids: Vec<TweetId> = rows.iter().map(|(_, row)| row.post).collect();
        ids.sort_unstable();
        ids.dedup();
        let hydration = self.hydrate_posts(plan, facts, &ids, metrics).await;
        let viewer_failed = viewer_lookup_failed(&hydration);
        let mut admitted = HashMap::new();
        let mut firsts = (ids.len() < rows.len()).then(HashMap::new);
        let mut evaluations = Vec::with_capacity(rows.len());
        for &(kind, NotificationPost { post, actor }) in rows {
            push_once(
                &mut evaluations,
                firsts.as_mut(),
                (post, kind, actor),
                || match owned_post(&hydration, post, Some(actor)) {
                    Err(evaluation) => Some(evaluation),
                    Ok(()) if viewer_failed => None,
                    Ok(()) => admitted
                        .entry(kind)
                        .or_insert_with(|| {
                            self.rule_engine.admit_post_kind(level, kind, &hydration)
                        })
                        .as_ref()
                        .and_then(|admitted| admitted.post(post, Placement::Standalone))
                        .map(|post| post.evaluate()),
                },
            );
        }
        evaluations
    }

    async fn resolve_recipient_posts(
        &self,
        level: SafetyLevel,
        facts: &RequestFacts,
        posts: Vec<TweetId>,
        metrics: &RequestMetrics,
    ) -> HashMap<TweetId, Result<(), Evaluation>> {
        if posts.is_empty() {
            return HashMap::new();
        }
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
        let recipient = facts.viewer_id().map(AuthorId);
        let hydration = self.hydrate_posts(plan, facts, &posts, metrics).await;
        posts
            .into_iter()
            .map(|post| (post, owned_post(&hydration, post, recipient)))
            .collect()
    }

    async fn hydrate_posts(
        &self,
        plan: &HydrationPlan,
        facts: &RequestFacts,
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
                facts.clone(),
                raw,
                RetweetMerge::Unmerged,
                &self.fallback_caches,
                metrics,
            ),
        )
        .await
    }
}

fn owned_post(
    hydration: &Hydration,
    post: TweetId,
    owner: Option<AuthorId>,
) -> Result<(), Evaluation> {
    match hydration.tweet(post) {
        Some(HydratedTweet::Resolved { candidate, .. })
            if Some(AuthorId(candidate.author.id)) == owner =>
        {
            Ok(())
        }
        Some(HydratedTweet::Resolved { .. }) => Err(Evaluation::NotFound(Lookup::Tweet)),
        Some(HydratedTweet::Unresolved {
            reason, deletion, ..
        }) => Err(reason.evaluation(*deletion)),
        None => Err(Evaluation::Failed(Lookup::Tweet)),
    }
}
