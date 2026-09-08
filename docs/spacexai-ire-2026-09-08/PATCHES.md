# Touched files — SpaceXAI pack @ 902a06f

## Modified

| File | Pack |
|------|------|
| `home-mixer/params/param.rs` | A (comments), F1 params, F5 params |
| `home-mixer/filters/retweet_deduplication_filter.rs` | F1 |
| `home-mixer/filters/previously_served_posts_filter.rs` | F5 (defer when armed) |
| `home-mixer/filters/mod.rs` | F5 module export |
| `home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs` | F5 wire |
| `home-mixer/util/candidates_util.rs` | F5 `source_key` |
| `home-mixer/util/mod.rs` | D module export |

## Added

| File | Pack |
|------|------|
| `home-mixer/filters/controlled_source_reexposure_filter.rs` | F5 |
| `home-mixer/util/serve_card_classify.rs` | D |

## Standalone test crate (cache, not in x-algorithm)

| Path | Role |
|------|------|
| `x-algo-cache/merge-ready/spacexai-pack-tests/` | Pure F1/F5/D `cargo test` |
| `x-algo-cache/merge-ready/STATUS.md` | This status |
| `x-algo-cache/merge-ready/PATCHES.md` | This list |
| `x-algo-cache/merge-ready/0001-spacexai-f1-f5-a-d.patch` | `git diff` export |

## Params added (defaults = today's behavior)

- `EnablePreferOriginalRetweetDedup`: bool = **false**
- `EnableControlledSourceReexposure`: bool = **false**
- `ControlledSourceReexposureMaxServesK`: u32 = **1**
- `ControlledSourceReexposureMinRequestGap`: u32 = **0**
- `ControlledSourceReexposureInNetworkOnly`: bool = **true**

## Params NOT changed (PR-A)

- `FollowAuthorWeight` = 4.0  
- `ProfileClickWeight` = 0.0  
- `ShareViaCopyLinkWeight` = 20.0  
