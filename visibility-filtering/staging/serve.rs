use crate::server::VFServer;
use crate::server_deps;
use crate::staging::reference;
use std::sync::Arc;
use tonic::metadata::MetadataMap;
use xai_x_service_builder::{ServiceContext, XService};

pub struct StagingServer(Arc<VFServer>);

#[tonic::async_trait]
impl XService for StagingServer {
    type Config = Option<MetadataMap>;

    async fn build(ctx: ServiceContext<Option<MetadataMap>>) -> Self {
        let deps = server_deps::build(&ctx.datacenter, ctx.config.as_ref()).await;
        let comparator =
            reference::build(deps.init_deadline, &deps.metrics, ctx.config.as_ref()).await;
        Self(Arc::new(deps.into_server(comparator)))
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        Arc::clone(&self.0).register(routes);
    }
}
