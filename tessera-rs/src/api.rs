//! Wire types for the gateway. These mirror `tessera.serve.api` on the Python side so the
//! Rust front end and the Python engine speak the same JSON.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SamplingParams {
    #[serde(default = "default_max_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
}

fn default_max_tokens() -> usize {
    64
}
fn default_temperature() -> f32 {
    1.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_k: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub params: SamplingParams,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateResponse {
    pub request_id: String,
    pub text: String,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub queued: usize,
    pub running: usize,
}
