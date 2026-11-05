#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

use std::io::{Read, Write};
use xai_visibility_filtering_service::verdict_gate::{parse_cases, tls_domain, GRPC_PORT};
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let cases = parse_cases(&input)?;
    let tls =
        TlsMode::mtls_from_env()?.with_domain_override(tls_domain(&std::env::var("DATACENTER")?));
    let channel = ChannelBuilder::new("vf-gate")
        .destination(format!("https://localhost:{GRPC_PORT}"))
        .tls(tls)
        .build()
        .await?;
    let mut out = std::io::stdout().lock();
    for case in &cases {
        let mut line = case.judge(&case.ask(&channel).await);
        if !line.passed() {
            line = case.judge(&case.ask(&channel).await);
        }
        serde_json::to_writer(&mut out, &line)?;
        writeln!(out)?;
    }
    Ok(())
}
