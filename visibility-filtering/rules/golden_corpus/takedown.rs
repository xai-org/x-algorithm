use super::{Role, Row, Subject};
use crate::models::{PostCandidate, RequestFacts, TweetFeatures, ViewerFeatures};
use crate::rules::fixtures::{
    allow, author_viewer, candidate, dropped, tweet_candidate, viewer_in_country,
};
use xai_core_entities::entities::TakedownReason;
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "legal_takedown_us",
            subject: Subject::Post(takedown_candidate(TakedownReason::LegalRequest {
                country_code: "us".to_string(),
            })),
            expect: vec![
                (
                    Role::As("in_us", viewer_in_country("us")),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "legal_takedown/drop/unspecified",
                    ),
                ),
                (
                    Role::each([
                        Role::NonFollower,
                        Role::As("in_fr", viewer_in_country("fr")),
                    ]),
                    allow(),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "legal_takedown_worldwide",
            subject: Subject::Post(takedown_candidate(TakedownReason::LegalRequest {
                country_code: "xx".to_string(),
            })),
            expect: vec![(
                Role::each([
                    Role::As("in_us", viewer_in_country("us")),
                    Role::NonFollower,
                ]),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "legal_takedown/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "legal_takedown_copyright",
            subject: Subject::Post(takedown_candidate(TakedownReason::LegalRequest {
                country_code: "xy".to_string(),
            })),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "unspecified_takedown_worldwide",
            subject: Subject::Post(takedown_candidate(TakedownReason::UnspecifiedReason {
                country_code: "xx".to_string(),
            })),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "legal_takedown/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "unspecified_takedown_copyright",
            subject: Subject::Post(takedown_candidate(TakedownReason::UnspecifiedReason {
                country_code: "xy".to_string(),
            })),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "legal_takedown/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "local_laws_takedown_worldwide",
            subject: Subject::Post(takedown_candidate(TakedownReason::BystanderReport {
                country_code: "xx".to_string(),
            })),
            expect: vec![(Role::As("in_us", viewer_in_country("us")), allow())],
            at: vec![],
        },
        Row {
            name: "dmca_takedown",
            subject: Subject::Post(takedown_candidate(TakedownReason::Dmca)),
            expect: vec![
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "legal_takedown/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "local_laws_takedown_de",
            subject: Subject::Post(takedown_candidate(TakedownReason::BystanderReport {
                country_code: "de".to_string(),
            })),
            expect: vec![
                (
                    Role::As("in_de", viewer_in_country("de")),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "local_laws_takedown/drop/unspecified",
                    ),
                ),
                (
                    Role::each([
                        Role::As("in_us", viewer_in_country("us")),
                        Role::NonFollower,
                        Role::As(
                            "author_in_de",
                            ViewerFeatures {
                                facts: RequestFacts::default().with_country("de"),
                                ..author_viewer()
                            },
                        ),
                    ]),
                    allow(),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "non_country_takedown",
            subject: Subject::Post(tweet_candidate(|t| {
                t.takedown_reasons = vec![TakedownReason::HatefulImagery, TakedownReason::Unknown]
            })),
            expect: vec![(Role::As("in_de", viewer_in_country("de")), allow())],
            at: vec![],
        },
        Row {
            name: "legal_takedown_uppercase_worldwide",
            subject: Subject::Post(takedown_candidate(TakedownReason::LegalRequest {
                country_code: "XX".to_string(),
            })),
            expect: vec![(
                Role::each([
                    Role::As("in_us", viewer_in_country("us")),
                    Role::NonFollower,
                ]),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "legal_takedown/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "legal_takedown_uppercase_copyright",
            subject: Subject::Post(takedown_candidate(TakedownReason::LegalRequest {
                country_code: "XY".to_string(),
            })),
            expect: vec![(
                Role::each([
                    Role::As("in_us", viewer_in_country("us")),
                    Role::NonFollower,
                ]),
                allow(),
            )],
            at: vec![],
        },
        Row {
            name: "local_laws_takedown_uppercase_copyright",
            subject: Subject::Post(takedown_candidate(TakedownReason::BystanderReport {
                country_code: "XY".to_string(),
            })),
            expect: vec![(
                Role::each([
                    Role::As("in_us", viewer_in_country("us")),
                    Role::NonFollower,
                ]),
                allow(),
            )],
            at: vec![],
        },
    ]
}

fn takedown_candidate(reason: TakedownReason) -> PostCandidate {
    candidate()
        .with_tweet_features(TweetFeatures {
            takedown_reasons: vec![reason],
            ..Default::default()
        })
        .build()
}
