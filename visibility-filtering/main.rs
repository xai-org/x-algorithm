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

use clap::Parser;
use xai_visibility_filtering_service::server::{self, ServeArgs, VFServer};

#[derive(Parser, Debug)]
#[command(about = "Visibility Filtering gRPC Server")]
struct Args {
    #[command(flatten)]
    serve: ServeArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    server::serve::<VFServer>(Args::parse().serve).await
}
