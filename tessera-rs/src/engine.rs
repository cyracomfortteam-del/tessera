//! The inference backend abstraction.
//!
//! The gateway is decoupled from *how* tokens are produced: it holds an `Arc<dyn Engine>`.
//! `MockEngine` is a deterministic in-process backend used by tests and local dev; the
//! real backend (`PyEngine`, behind the `python` feature) calls into the Tessera Python
//! engine via PyO3.

use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;

use crate::api::{GenerateRequest, GenerateResponse};

#[async_trait]
pub trait Engine: Send + Sync {
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GenerateResponse>;
}

/// Deterministic backend: useful for wiring/integration tests without a model.
#[derive(Default)]
pub struct MockEngine {
    counter: AtomicU64,
}

impl MockEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Engine for MockEngine {
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let prompt_tokens = req.prompt.split_whitespace().count().max(1);
        let output_tokens = req.params.max_new_tokens;
        // Echo the prompt and append a deterministic, greedy-looking continuation so tests
        // can assert on exact bytes.
        let text = format!("{} <gen {output_tokens} toks>", req.prompt.trim());
        Ok(GenerateResponse {
            request_id: format!("req-{id}"),
            text,
            prompt_tokens,
            output_tokens,
        })
    }
}
