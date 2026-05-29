# tessera-gateway (Rust)

An async **tokio + axum** serving gateway for the Tessera inference engine, with optional
**PyO3** bindings. It owns the network-facing concerns — HTTP, JSON, admission control /
back-pressure — and delegates token generation to a pluggable [`Engine`] backend.

```
client ──HTTP──▶ axum router ──▶ Scheduler (Semaphore budget) ──▶ dyn Engine
                  /generate                                         ├─ MockEngine (tests/dev)
                  /health                                           └─ PyEngine  (PyO3 → Python)
```

## Run

```bash
cargo run --release
# TESSERA_ADDR=127.0.0.1:8080  TESSERA_MAX_CONCURRENCY=8  RUST_LOG=info

curl -s localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"hello","params":{"max_new_tokens":16}}'
```

## Test / lint

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## PyO3 extension

The `python` feature builds an importable Python extension module (`import tessera_gateway`)
exposing latency-sensitive helpers (`whitespace_token_count`, `RequestBatcher`):

```bash
maturin develop -F python      # or: cargo build --features python
```

The pure-Rust gateway builds without it, so CI and `cargo test` need no libpython.

## Why Rust here

The gateway is the part that benefits most from a compiled async runtime: thousands of
concurrent connections, bounded memory, predictable tail latency. tokio's work-stealing
scheduler + a `Semaphore`-based admission budget give back-pressure for free, while the heavy
numerical work stays in the Python/CUDA engine reached over the `Engine` trait (in-process via
PyO3, or out-of-process over HTTP).
