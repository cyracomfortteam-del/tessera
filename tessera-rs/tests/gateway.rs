//! Integration tests for the axum gateway using tower's `oneshot` (no real socket needed).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tessera_gateway::{app, MockEngine};
use tower::ServiceExt;

fn router() -> axum::Router {
    app(Arc::new(MockEngine::new()), 4)
}

#[tokio::test]
async fn health_returns_ok() {
    let resp = router()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["running"], 0);
}

#[tokio::test]
async fn generate_returns_echo_and_token_counts() {
    let body = serde_json::json!({
        "prompt": "hello world from tessera",
        "params": { "max_new_tokens": 5, "temperature": 0.0 }
    })
    .to_string();

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["output_tokens"], 5);
    assert_eq!(v["prompt_tokens"], 4);
    assert!(v["text"]
        .as_str()
        .unwrap()
        .contains("hello world from tessera"));
    assert!(v["request_id"].as_str().unwrap().starts_with("req-"));
}

#[tokio::test]
async fn generate_uses_defaults_when_params_omitted() {
    let body = serde_json::json!({ "prompt": "just a prompt" }).to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["output_tokens"], 64); // SamplingParams default
}
