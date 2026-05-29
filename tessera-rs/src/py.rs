//! PyO3 bindings (built with `--features python`, e.g. via `maturin develop -F python`).
//!
//! Exposes a couple of latency-sensitive helpers to Python so the hot request-shaping path
//! can run in Rust. The module name (`tessera_gateway`) matches the crate lib name so it
//! imports as `import tessera_gateway` from Python.

use pyo3::prelude::*;

/// Whitespace token count — a stand-in for a fast byte-level pre-tokenizer.
#[pyfunction]
fn whitespace_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Accumulate prompts until `max_batch`, then hand back a batch to forward to the engine.
/// Mirrors the admission/batching the gateway does, but callable from the Python engine.
#[pyclass]
struct RequestBatcher {
    max_batch: usize,
    buffer: Vec<String>,
}

#[pymethods]
impl RequestBatcher {
    #[new]
    fn new(max_batch: usize) -> Self {
        Self {
            max_batch,
            buffer: Vec::new(),
        }
    }

    /// Push a prompt; returns a full batch once `max_batch` is reached, else `None`.
    fn push(&mut self, prompt: String) -> Option<Vec<String>> {
        self.buffer.push(prompt);
        if self.buffer.len() >= self.max_batch {
            Some(std::mem::take(&mut self.buffer))
        } else {
            None
        }
    }

    /// Drain whatever is buffered (e.g. on a timeout tick).
    fn flush(&mut self) -> Vec<String> {
        std::mem::take(&mut self.buffer)
    }

    #[getter]
    fn pending(&self) -> usize {
        self.buffer.len()
    }
}

#[pymodule]
fn tessera_gateway(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(whitespace_token_count, m)?)?;
    m.add_class::<RequestBatcher>()?;
    Ok(())
}
