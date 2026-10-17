// Test-only stand-in for unpublished telemetry instrumentation.
// Preserve the entire annotated item; do not change pipeline business logic.
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn receive_stats(_attribute: TokenStream, item: TokenStream) -> TokenStream {
    item
}
