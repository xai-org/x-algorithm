use super::{RequestRow, Store};
use crate::hydration::fetcher::AnyFetcher;
use crate::hydration::inventory::Answer;
use crate::hydration::sources::source;
use crate::hydration::{Cause, Hydrator, Hydrators};
use crate::models::{AuthorId, ReplyTo, TweetId};
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConversationRootAuthor {
    PureCore(AuthorId),
    ConversationControl(AuthorId),
    RootPost(TweetId),
    RootPureCoreFailed,
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RootAuthorBlock {
    Unread,
    ReadBy(Hydrator),
    FailedBy(Hydrator),
}

impl RootAuthorBlock {
    pub(super) fn unread(self) -> Hydrators {
        match self {
            Self::Unread => Hydrators::BLOCKED_BY_CONVERSATION_ROOT_AUTHOR,
            Self::ReadBy(node) | Self::FailedBy(node) => {
                Hydrators::BLOCKED_BY_CONVERSATION_ROOT_AUTHOR.difference(Hydrators::of(node))
            }
        }
    }

    pub(super) fn failed(self) -> Hydrators {
        match self {
            Self::FailedBy(node) => Hydrators::of(node),
            Self::Unread | Self::ReadBy(_) => Hydrators::empty(),
        }
    }
}

impl Store {
    pub(super) fn root_author_block(&self, row: &RequestRow) -> RootAuthorBlock {
        if self
            .callable
            .intersection(Hydrators::BLOCKED_BY_CONVERSATION_ROOT_AUTHOR)
            .is_empty()
        {
            return RootAuthorBlock::Unread;
        }
        match self.conversation_root_author(row) {
            None => RootAuthorBlock::Unread,
            Some(ConversationRootAuthor::PureCore(_)) => {
                RootAuthorBlock::ReadBy(Hydrator::BlockedByReplyRoot)
            }
            Some(ConversationRootAuthor::RootPureCoreFailed) => {
                RootAuthorBlock::FailedBy(Hydrator::BlockedByReplyRoot)
            }
            Some(
                ConversationRootAuthor::ConversationControl(_) | ConversationRootAuthor::Unresolved,
            ) => RootAuthorBlock::ReadBy(Hydrator::BlockedByConversationRoot),
            Some(ConversationRootAuthor::RootPost(_)) => {
                RootAuthorBlock::ReadBy(Hydrator::BlockedByConversationRootPost)
            }
        }
    }

    pub(super) fn conversation_root_author(
        &self,
        row: &RequestRow,
    ) -> Option<ConversationRootAuthor> {
        let id = row.tweet_id()?.0;
        let (conversation_id, parent_id) = match self
            .request_fetcher::<source::TesPureCoreData>()
            .get(id)?
            .reply_to?
        {
            ReplyTo::Root { root_author_id, .. } => {
                return Some(ConversationRootAuthor::PureCore(root_author_id));
            }
            ReplyTo::Nested {
                conversation_id,
                parent_id,
            } => (conversation_id, parent_id),
        };
        if let Some(author) = self.root_author_in_batch(conversation_id, parent_id) {
            return Some(ConversationRootAuthor::PureCore(author));
        }
        let controls = self.request_fetcher::<source::TesConversationControl>();
        let root = match (controls.get(id), controls.answer(id)) {
            (Some(control), _) => ConversationRootAuthor::ConversationControl(AuthorId(
                control.conversation_tweet_author_id,
            )),
            (None, Answer::NotFound) if !self.requested_tweet_ids.contains(&conversation_id) => {
                ConversationRootAuthor::RootPost(conversation_id)
            }
            (None, Answer::NotFound)
                if self.existence_miss(Hydrator::PureCore, conversation_id.0)
                    == Some(Cause::Failed) =>
            {
                ConversationRootAuthor::RootPureCoreFailed
            }
            (None, _) => ConversationRootAuthor::Unresolved,
        };
        Some(root)
    }

    fn root_author_in_batch(
        &self,
        conversation_id: TweetId,
        parent_id: TweetId,
    ) -> Option<AuthorId> {
        let row_core = |id: TweetId| {
            self.request_fetcher::<source::TesPureCoreData>()
                .get(id.0)
                .filter(|_| self.requested_tweet_ids.contains(&id))
        };
        row_core(conversation_id)
            .map(|root| root.author_id)
            .or_else(|| match row_core(parent_id)?.reply_to? {
                ReplyTo::Root {
                    conversation_id: parent_conversation_id,
                    root_author_id,
                } => (parent_conversation_id == conversation_id).then_some(root_author_id),
                ReplyTo::Nested { .. } => None,
            })
    }

    pub(super) fn root_not_following_viewer(&self, control: &ConversationControl) -> Option<u64> {
        if control.arm != ConversationControlArm::MyNetwork {
            return None;
        }
        let root = control.conversation_tweet_author_id;
        let holds = self.edge_holds(Hydrator::RootFollowsViewer, root)?;
        (!holds).then_some(root)
    }
}
