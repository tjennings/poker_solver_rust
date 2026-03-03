# Clippy Cleanup Design

**Date:** 2026-03-02
**Status:** Approved

## Problem

179 clippy warnings + 1 hard error across the codebase. Distribution: 177 in `core`, 8 in `trainer`, 3 in `tauri-app`. The `core` crate has `clippy::pedantic` enabled.

## Approach: File-by-file sweep (bugs first)

### Phase 1: Bug Investigation (Sequential)
Investigate 10 "always returns zero" + 9 "no effect" + 1 hard error for real logic bugs. Key files:
- `preflop/tree.rs` — bulk of zero/no-effect warnings
- `tauri-app/exploration.rs` — `0 * n * n` error

### Phase 2: Mechanical Cleanup (4 parallel agents)

| Agent | Files | ~Warnings |
|-|-|-|
| 1 | `preflop/tree.rs`, `preflop/config.rs`, `info_key.rs` | ~35 |
| 2 | `postflop_abstraction.rs`, `postflop_exhaustive.rs`, `postflop_bundle.rs` | ~45 |
| 3 | `postflop_mccfr.rs`, `solver.rs`, `equity.rs`, `equity_table_cache.rs`, `postflop_tree.rs`, `postflop_model.rs` | ~40 |
| 4 | `agent.rs`, `simulation.rs`, `flops.rs`, `lib.rs`, `trainer/`, `tauri-app/`, tests | ~40 |

### Fix Patterns
- **format strings:** inline variables into `format!`/`println!`
- **raw string hashes:** `r#"..."#` → `r"..."`
- **doc backticks:** add backticks around code items in doc comments
- **cast truncation:** `#[allow(clippy::cast_possible_truncation)]` per-site
- **wildcard matches:** replace `_ =>` with explicit variant names
- **contains vs any:** use `.contains()` instead of `.iter().any(|x| x == val)`
- **float comparison:** use `(a - b).abs() < f64::EPSILON`
- **needless range loop:** convert to iterator patterns
- **other pedantic:** follow clippy's suggestions

### Decisions
- Cast truncation: `#[allow]` per-site (hot-path code, values always small)
- Float comparison: approx comparisons with epsilon
- Bug investigation before mechanical fixes
- Atomic commits per file group
