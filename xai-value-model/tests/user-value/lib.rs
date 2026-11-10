// Compile actual public scoring modules; exclude the unpublished protobuf adapter.
#![allow(dead_code)]
#[path = "../../inputs.rs"]
mod inputs;
#[path = "../../phoenix_scores.rs"]
mod phoenix_scores;
#[path = "../../scoring.rs"]
mod scoring;
#[path = "../../user_value.rs"]
mod user_value;
#[path = "../../weights.rs"]
mod weights;
