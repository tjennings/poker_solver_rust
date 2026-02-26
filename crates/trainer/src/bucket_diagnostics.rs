//! Bucket diagnostic tool.
//!
//! Previously ran EHS bucket quality checks. Now that the postflop abstraction
//! uses 169 canonical hands directly (no clustering), bucket diagnostics are
//! not applicable.

use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use std::path::Path;

/// Bucket diagnostics are no longer applicable with 169-hand direct indexing.
/// Returns `true` (all checks pass) immediately.
pub fn run(_config: &PostflopModelConfig, _cache_dir: &Path, _json: bool) -> bool {
    eprintln!("Bucket diagnostics are not applicable: postflop uses 169 canonical hands directly.");
    true
}
