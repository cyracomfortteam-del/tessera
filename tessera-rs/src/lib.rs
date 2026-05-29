//! Tessera serving gateway.
//!
//! An async tokio/axum front end that accepts generation requests, applies admission
//! control, and dispatches to a pluggable [`Engine`] backend. Build the pure-Rust gateway
//! with `cargo build`; build the Python extension (PyO3) with `--features python`.

pub mod api;
pub mod engine;
pub mod gateway;
pub mod scheduler;

#[cfg(feature = "python")]
mod py;

pub use engine::{Engine, MockEngine};
pub use gateway::{app, AppState};
pub use scheduler::Scheduler;
