//! Re-export boundary trace module from the tauri-app crate.
//!
//! The canonical implementation lives in `poker_solver_tauri::boundary_trace`.
//! This module re-exports everything so that existing `compare_solve.rs` and
//! `main.rs` references continue to compile without changes.

pub use poker_solver_tauri::boundary_trace::*;
