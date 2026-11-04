use super::{Role, Row, Subject};
use crate::hydration::Hydrator::{BlockedByAuthor, Blocks, MuteRetweets, Mutes};
use crate::models::LimitedEngagementReason;
use crate::rules::fixtures::{allow, candidate, controlled_root, dropped, limited, limited_for};
use xai_core_entities::entities::ConversationControlArm;
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "viewer_blocks_author",
            subject: Subject::Post(candidate().with_edge(Blocks).build()),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerBlocksAuthor,
                    "viewer_blocks_author/drop",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "viewer_mutes_author",
            subject: Subject::Post(candidate().with_edge(Mutes).build()),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerMutesAuthor,
                    "viewer_mutes_author/drop",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "viewer_blocks_and_mutes_author",
            subject: Subject::Post(candidate().with_edge(Blocks).with_edge(Mutes).build()),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerBlocksAuthor,
                    "viewer_blocks_author/drop",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "muted_retweets_retweet",
            subject: Subject::Post(candidate().with_edge(MuteRetweets).retweet_of(2).build()),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "viewer_mutes_retweets/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "muted_retweets_original",
            subject: Subject::Post(candidate().with_edge(MuteRetweets).build()),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "author_and_root_author_block",
            subject: Subject::Post(
                candidate()
                    .with_edge(BlockedByAuthor)
                    .root_author_blocks_viewer()
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                limited_for(
                    &[
                        LimitedEngagementReason::BlockedViewer,
                        LimitedEngagementReason::RootAuthorBlockedViewer,
                    ],
                    "blocked_viewer/limited_engagement",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "author_block",
            subject: Subject::Post(candidate().with_edge(BlockedByAuthor).build()),
            expect: vec![(Role::Author, allow())],
            at: vec![],
        },
        Row {
            name: "root_author_block",
            subject: Subject::Post(candidate().root_author_blocks_viewer().build()),
            expect: vec![(
                Role::Author,
                limited(
                    LimitedEngagementReason::RootAuthorBlockedViewer,
                    "blocked_viewer/limited_engagement/root_author_blocked_viewer",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "root_author_block_community_conversation",
            subject: Subject::Post(
                candidate()
                    .root_author_blocks_viewer()
                    .with_conversation_control(controlled_root(ConversationControlArm::Community))
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                limited_for(
                    &[
                        LimitedEngagementReason::RootAuthorBlockedViewer,
                        LimitedEngagementReason::ConversationControl,
                    ],
                    "blocked_viewer/limited_engagement/root_author_blocked_viewer",
                ),
            )],
            at: vec![],
        },
    ]
}
