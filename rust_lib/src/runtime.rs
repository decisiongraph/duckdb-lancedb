//! Tokio runtime for blocking on LanceDB async operations.
//!
//! Uses a small thread pool (2 workers) since DuckDB manages its own parallelism.
//! We only need enough threads to handle async I/O from LanceDB/Lance.

use std::sync::LazyLock;
use tokio::runtime::Runtime;

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("lance-io")
        .enable_all()
        .build()
        .expect("failed to create tokio runtime")
});

/// Block on an async future using the shared tokio runtime.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    RUNTIME.block_on(future)
}
