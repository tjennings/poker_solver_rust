# Postflop TUI Dashboard Design

**Date:** 2026-03-01
**Status:** Approved
**Scope:** `solve-postflop` subcommand console UI

## Overview

Replace the existing `indicatif` multi-bar progress display in `solve-postflop` with a full-screen alternate-terminal TUI dashboard (htop-style). The dashboard renders a fixed-frame view that refreshes at a configurable interval, showing real-time solver metrics including throughput, pruning rates, and per-flop convergence.

Future work: a preflop trainer variant using the same TUI infrastructure.

## Layout

```
┌─────────────────────────────────────────────────────────┐
│ SPR Progress            [███░░░░░░] 2/7 SPRs   01:23:45│
│ Flop Progress (SPR=3)   [██████░░░] 142/200 flops      │
│                                                         │
│ Traversals/sec    ▂▃▅▇█▇▅▆▇█▇▆▅▇  1.2M/s              │
│ Remaining pairs   █▇▇▆▅▅▄▃▃▂▂▁▁   842K                │
│ % Traversals pruned  ░▁▂▃▅▅▆▇▇▇█  62.3%               │
│ % Actions pruned     ░▁▂▃▄▅▆▆▇▇█  41.8%               │
│                                                         │
│ ── Active Flops (scrollable) ─────────────────────────  │
│ AhKs2d  expl ▇▆▅▄▃▂▂▁▁  12.4 mBB/h  iter 28/50       │
│ QcJd7h  expl █▇▆▅▄▃▃▂   31.7 mBB/h  iter 19/50       │
│ 9s8h3c  expl █▇▇▆▆▅▅▄   48.2 mBB/h  iter 11/50       │
│ Ts5d4c  expl █████▇▇▆▅   89.1 mBB/h  iter 5/50        │
│                                                         │
│ Elapsed: 01:23:45    ETA: 02:41:12                      │
└─────────────────────────────────────────────────────────┘
```

### Widget breakdown (top to bottom)

| Widget | Type | Data source |
|-|-|-|
| SPR Progress | Gauge | `current_spr` / `total_sprs` atomics |
| Flop Progress | Gauge | `flops_completed` / `total_flops` atomics |
| Traversals/sec | Sparkline | Delta of `traversal_count` between ticks |
| Remaining pairs | Sparkline | `total_expected - traversal_count`, descending to 0 |
| % Traversals pruned | Sparkline | `pruned_traversal_count` / `traversal_count` delta ratio |
| % Actions pruned | Sparkline | `pruned_action_slots` / `total_action_slots` delta ratio |
| Active Flops | Scrollable list | `DashMap<String, FlopTuiState>` — one sparkline row per active flop |
| Footer | Text | Wall-clock elapsed; ETA from overall completion rate |

### Active Flops section

- Dynamic rows: appear when rayon starts a flop, disappear on `FlopStage::Done`
- Each row: flop name, exploitability sparkline (history of mBB/h values), current exploitability, iteration N/M
- Scrollable via ratatui `List` with scroll state when active flops exceed visible area
- Sorted by iteration progress (most-progressed first)

## Data Flow

```
Solver threads (rayon)            TUI thread
──────────────────────            ──────────

exhaustive_solve_one_flop()       loop {
  ├─ AtomicU64::fetch_add            sample all atomics
  │  (traversal_count)               compute deltas / rates
  │  (pruned_traversal_count)        append to sparkline history
  │  (total_action_slots)            read DashMap snapshot
  │  (pruned_action_slots)           render frame
  │                                  sleep(refresh_interval)
  └─ on_progress callback        }
       └─ insert/update
          DashMap<flop_name,
                  FlopTuiState>
```

### Integration model: shared atomic counters

- Solver hot path increments lock-free `AtomicU64` counters — zero contention
- TUI thread samples atomics at its own refresh tick, computes rates from deltas
- Per-flop exploitability written from existing `on_progress` callback into a `DashMap`
- TUI thread reads `DashMap` snapshot each tick (read-only)

## Shared Metrics Struct

```rust
pub struct TuiMetrics {
    // Global atomics — incremented in traversal hot path
    pub traversal_count: AtomicU64,
    pub pruned_traversal_count: AtomicU64,
    pub total_action_slots: AtomicU64,
    pub pruned_action_slots: AtomicU64,

    // SPR-level — set by outer loop in build_postflop_with_progress
    pub current_spr: AtomicU32,
    pub total_sprs: AtomicU32,
    pub flops_completed: AtomicU32,
    pub total_flops: AtomicU32,

    // Per-flop convergence — written from on_progress callback
    pub flop_states: DashMap<String, FlopTuiState>,
}

pub struct FlopTuiState {
    pub exploitability_history: Vec<f64>,
    pub iteration: usize,
    pub max_iterations: usize,
}
```

Passed as `Arc<TuiMetrics>` — solver writes, TUI reads.

## CLI

New flag on `solve-postflop`:

```
--tui-refresh <seconds>    Refresh interval for TUI dashboard (default: 1.0)
```

TTY detection: when stderr is not a terminal (`!atty::is(Stderr)`), skip TUI and fall back to simple line-based progress logging. This allows piping output without alternate-screen artifacts.

## Dependencies

| Crate | Purpose |
|-|-|
| `ratatui` | TUI framework (Gauge, Sparkline, List, Layout widgets) |
| `crossterm` | Terminal backend (alternate screen, raw mode, event polling) |
| `dashmap` | Lock-free concurrent map for per-flop state |
| `atty` | TTY detection for graceful fallback |

Remove `indicatif` from `crates/trainer/Cargo.toml` if no longer used elsewhere.

## Files Changed

| File | Change |
|-|-|
| `crates/trainer/Cargo.toml` | Add ratatui, crossterm, dashmap, atty |
| `crates/trainer/src/tui.rs` (new) | TUI rendering loop, layout, widget composition, sparkline buffers |
| `crates/trainer/src/tui_metrics.rs` (new) | `TuiMetrics`, `FlopTuiState` definitions |
| `crates/trainer/src/main.rs` | Remove indicatif progress bars from `build_postflop_with_progress`; spawn TUI thread; wire `Arc<TuiMetrics>`; add `--tui-refresh` CLI arg |
| `crates/core/src/preflop/postflop_exhaustive.rs` | Increment atomic counters in `parallel_traverse_into` hot path |
| `crates/core/src/preflop/postflop_abstraction.rs` | Accept `Option<Arc<TuiMetrics>>` in `build_for_spr`; set SPR/flop counters; route callback data to flop_states |

## Non-goals (v1)

- Keyboard controls (q/p/+/- for interactive control)
- Memory usage display
- Active thread count
- Color-coded convergence thresholds
- Preflop trainer TUI variant (future work, same infrastructure)
