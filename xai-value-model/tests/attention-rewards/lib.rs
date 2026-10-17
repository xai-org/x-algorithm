// Compile the actual public scoring modules without the unpublished protobuf crate.
// This does not compile or substitute for the production protobuf adapter.
#![allow(dead_code)]
#[path = "../../attention_rewards.rs"]
mod attention_rewards;
#[path = "../../inputs.rs"]
mod inputs;
#[path = "../../phoenix_scores.rs"]
mod phoenix_scores;
#[path = "../../scoring.rs"]
mod scoring;
#[path = "../../weights.rs"]
mod weights;
