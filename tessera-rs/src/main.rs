//! `tessera-gateway` binary: serve the (mock) engine over HTTP.
//!
//! Configuration via env:
//!   TESSERA_ADDR             bind address (default 127.0.0.1:8080)
//!   TESSERA_MAX_CONCURRENCY  in-flight request budget (default 8)
//!   RUST_LOG                 tracing filter (e.g. info)

use std::sync::Arc;

use tessera_gateway::{app, MockEngine};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let addr = std::env::var("TESSERA_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let max_concurrency = std::env::var("TESSERA_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let engine = Arc::new(MockEngine::new());
    let router = app(engine, max_concurrency);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("tessera-gateway listening on http://{addr} (budget={max_concurrency})");
    axum::serve(listener, router).await?;
    Ok(())
}
