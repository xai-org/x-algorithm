use super::{Role, Row, Subject, OUTER_TWEET_ID};
use crate::hydration::Hydrator::{
    AuthorBlocksOuterAuthor, Blocks, Mutes, OuterAuthorFollowsAuthor,
};
use crate::models::{EdgeSet, PostCandidate, Quote, QuoteLink, UserCandidate, Verdict};
use crate::rules::fixtures::{allow, author_candidate, candidate, dropped, AUTHOR_ID};
use crate::rules::SafetyLevel::{self, QuotedTweetRules};
use xai_visibility_filtering::models::FilteredReason;

const QUOTING_AUTHOR_ID: u64 = 200;

fn quoted_by(quoting_author: u64, pair_edges: EdgeSet) -> Quote {
    Quote {
        link: QuoteLink {
            outer: OUTER_TWEET_ID,
            quoting_author,
        },
        pair_edges,
    }
}

fn quoted() -> Quote {
    quoted_by(QUOTING_AUTHOR_ID, EdgeSet::default())
}

fn author(post: PostCandidate, quote: Quote) -> UserCandidate {
    UserCandidate {
        user: post.author,
        edges: post.edges,
        quote: Some(quote),
        failed: post.failed,
    }
}

fn inner_quoted(role: Role, expected: Verdict) -> (SafetyLevel, Role, Verdict) {
    (
        QuotedTweetRules,
        Role::InnerQuoted(Box::new(role)),
        expected,
    )
}

pub(super) fn rows() -> Vec<Row> {
    let author_state = [
        (
            "deactivated_author",
            author_candidate(|a| a.is_deactivated = true),
            FilteredReason::AuthorIsDeactivated,
            "deactivated_author/drop",
        ),
        (
            "erased_author",
            author_candidate(|a| a.is_erased = true),
            FilteredReason::AuthorAccountIsInactive,
            "erased_author/drop/inactive",
        ),
        (
            "offboarded_author",
            author_candidate(|a| a.is_offboarded = true),
            FilteredReason::AuthorAccountIsInactive,
            "offboarded_author/drop/inactive",
        ),
        (
            "suspended_author",
            author_candidate(|a| a.is_suspended = true),
            FilteredReason::AuthorIsSuspended,
            "suspended_author/drop",
        ),
    ];
    let mut rows: Vec<Row> = author_state
        .into_iter()
        .map(|(name, post, reason, by)| Row {
            name,
            subject: Subject::User(author(post, quoted())),
            expect: vec![],
            at: vec![
                inner_quoted(Role::NonFollower, dropped(reason, by)),
                inner_quoted(Role::Author, allow()),
            ],
        })
        .collect();
    rows.extend([
        Row {
            name: "viewer_blocks_author",
            subject: Subject::User(author(candidate().with_edge(Blocks).build(), quoted())),
            expect: vec![],
            at: vec![inner_quoted(
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerBlocksAuthor,
                    "viewer_blocks_author/drop",
                ),
            )],
        },
        Row {
            name: "viewer_mutes_author",
            subject: Subject::User(author(candidate().with_edge(Mutes).build(), quoted())),
            expect: vec![],
            at: vec![
                inner_quoted(
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ViewerMutesAuthor,
                        "quoted_viewer_mutes_author_not_following/drop",
                    ),
                ),
                inner_quoted(Role::Follower, allow()),
            ],
        },
    ]);
    let blocked = EdgeSet::default().with(AuthorBlocksOuterAuthor);
    let blocks_quoting_author = || {
        dropped(
            FilteredReason::AuthorBlockViewer,
            "quoted_author_blocks_quoting_author/drop/author_block_viewer",
        )
    };
    let quoting_author_follows = EdgeSet::default().with(OuterAuthorFollowsAuthor);
    let protected = |quote| author(author_candidate(|a| a.is_protected = true), quote);
    let protected_dropped = || {
        dropped(
            FilteredReason::AuthorIsProtected,
            "protected_quoted_author/drop",
        )
    };
    rows.extend([
        Row {
            name: "quoted_author_blocks_quoting_author",
            subject: Subject::User(author(
                candidate().build(),
                quoted_by(QUOTING_AUTHOR_ID, blocked),
            )),
            expect: vec![],
            at: vec![
                inner_quoted(Role::NonFollower, blocks_quoting_author()),
                inner_quoted(Role::LoggedOut, blocks_quoting_author()),
            ],
        },
        Row {
            name: "quoted_author_blocks_quoting_author_self_quote",
            subject: Subject::User(author(candidate().build(), quoted_by(AUTHOR_ID, blocked))),
            expect: vec![],
            at: vec![inner_quoted(Role::NonFollower, allow())],
        },
        Row {
            name: "protected_quoted_author",
            subject: Subject::User(protected(quoted())),
            expect: vec![],
            at: vec![
                inner_quoted(Role::NonFollower, protected_dropped()),
                inner_quoted(Role::LoggedOut, protected_dropped()),
                inner_quoted(Role::Author, protected_dropped()),
            ],
        },
        Row {
            name: "protected_quoted_author_quoting_author_follows",
            subject: Subject::User(protected(quoted_by(
                QUOTING_AUTHOR_ID,
                quoting_author_follows,
            ))),
            expect: vec![],
            at: vec![inner_quoted(Role::NonFollower, allow())],
        },
        Row {
            name: "protected_quoted_author_self_quote",
            subject: Subject::User(protected(quoted_by(AUTHOR_ID, EdgeSet::default()))),
            expect: vec![],
            at: vec![inner_quoted(Role::NonFollower, allow())],
        },
    ]);
    rows
}
