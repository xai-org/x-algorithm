#![cfg_attr(not(test), deny(clippy::allow_attributes))]
#![deny(
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::clone_on_ref_ptr,
    clippy::dbg_macro,
    clippy::exit,
    clippy::expect_used,
    clippy::format_push_string,
    clippy::get_unwrap,
    clippy::indexing_slicing,
    clippy::large_futures,
    clippy::let_underscore_must_use,
    clippy::mem_forget,
    clippy::needless_pass_by_value,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::ref_option,
    clippy::string_slice,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::unused_self,
    clippy::unwrap_in_result,
    clippy::unwrap_used,
    clippy::wildcard_enum_match_arm
)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::dbg_macro,
        clippy::expect_used,
        clippy::get_unwrap,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::panic_in_result_fn,
        clippy::print_stderr,
        clippy::print_stdout,
        clippy::unreachable,
        clippy::unwrap_in_result,
        clippy::unwrap_used,
        reason = "fixtures may panic and cast freely"
    )
)]

pub(crate) mod caller_identity;
pub(crate) mod clients;
pub mod config;
pub mod dark_traffic_setup;
pub(crate) mod evaluate_tweets;
pub mod explain;
pub(crate) mod filter_notifications;
pub(crate) mod filter_tweets;
pub(crate) mod get_safety_labels;
pub(crate) mod hydration;
pub(crate) mod ingress;
pub(crate) mod metrics;
pub(crate) mod models;
pub mod params;
#[cfg(test)]
mod pinned;
pub(crate) mod post_card;
pub(crate) mod post_evaluator;
pub(crate) mod quote;
pub(crate) mod result_label;
pub(crate) mod rules;
#[cfg(test)]
mod scenario;
pub mod server;
pub(crate) mod server_deps;
pub mod staging;
pub(crate) mod thrift_read;
pub(crate) mod treatment;
pub mod verdict_gate;
