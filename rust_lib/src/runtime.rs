//! Tokio runtime for blocking on LanceDB async operations.

use std::sync::LazyLock;
use tokio::runtime::Runtime;

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new().expect("failed to create tokio runtime")
});

/// Block on an async future using the shared tokio runtime.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    RUNTIME.block_on(future)
}
