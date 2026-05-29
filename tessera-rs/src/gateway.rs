//! The axum HTTP surface: `POST /generate` and `GET /health`.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};

use crate::api::{GenerateRequest, HealthResponse};
use crate::engine::Engine;
use crate::scheduler::Scheduler;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub scheduler: Arc<Scheduler>,
}

pub fn app(engine: Arc<dyn Engine>, max_concurrency: usize) -> Router {
    let state = AppState {
        engine,
        scheduler: Arc::new(Scheduler::new(max_concurrency)),
    };
    Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        queued: state.scheduler.queued(),
        running: state.scheduler.running(),
    })
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let engine = state.engine.clone();
    // Run under the admission budget so a burst of requests applies back-pressure rather
    // than oversubscribing the backend.
    let result = state
        .scheduler
        .admit(async move { engine.generate(req).await })
        .await;

    match result {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(err) => {
            tracing::error!("generate failed: {err:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("error: {err}")).into_response()
        }
    }
}
