//! Tokio runtime for blocking on LanceDB async operations.
//!
//! Thread count scales with available cores (capped at 4) since DuckDB manages
//! its own parallelism — we only need enough threads for async Lance I/O.

use std::sync::LazyLock;
use tokio::runtime::Runtime;

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .thread_name("lance-io")
        .enable_all()
        .build()
        .expect("failed to create tokio runtime")
});

/// Block on an async future using the shared tokio runtime.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    RUNTIME.block_on(future)
}
