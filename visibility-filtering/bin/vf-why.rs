#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

use clap::Parser;
use xai_visibility_filtering_service::why::{Ask, Why};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ask = Ask::parse();
    let datacenter = std::env::var("DATACENTER")?;
    let why = Why::connect(&datacenter).await;
    let levels = why.judge(&ask).await;
    println!(
        "post {} viewer {} in {datacenter}",
        ask.post,
        ask.viewer
            .map_or_else(|| "logged out".to_owned(), |viewer| viewer.to_string())
    );
    for level in levels {
        print!("{level}");
    }
    Ok(())
}
