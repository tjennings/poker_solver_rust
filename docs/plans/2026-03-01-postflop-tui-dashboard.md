# Postflop TUI Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the indicatif progress bars in `solve-postflop` with a full-screen ratatui TUI dashboard showing real-time solver metrics.

**Architecture:** A shared `TuiMetrics` struct (atomic counters + DashMap) is passed into the solver. The solver hot path increments atomics; the `on_progress` callback updates per-flop exploitability state. A dedicated TUI thread samples these at a configurable refresh interval and renders the alternate-screen dashboard.

**Tech Stack:** ratatui, crossterm, dashmap, atty

**Design doc:** `docs/plans/2026-03-01-postflop-tui-dashboard-design.md`

---

## Agent Team & Execution Order

| Agent | Tasks | Parallel? |
|-|-|-|
| `rust-developer` A | Task 1 (deps), Task 2 (TuiMetrics struct) | Sequential foundation |
| `rust-developer` B | Task 3 (instrument hot path) | After Task 2 |
| `rust-developer` C | Task 4 (TUI renderer) | After Task 2 |
| `rust-developer` D | Task 5 (wire into main.rs) | After Tasks 3 + 4 |
| `rust-developer` E | Task 6 (TTY fallback) | After Task 5 |
| Review agents | `idiomatic-rust-enforcer` + `rust-perf-reviewer` | After Task 6 |

Tasks 3 and 4 can run in parallel (different crates). Task 5 integrates both.

---

### Task 1: Update dependencies

**Files:**
- Modify: `crates/trainer/Cargo.toml`

**Step 1: Update Cargo.toml**

Replace `indicatif = "0.17"` with:
```toml
ratatui = "0.29"
crossterm = "0.28"
dashmap = "6"
atty = "0.2"
```

Remove the `indicatif` line.

**Step 2: Verify it compiles**

Run: `cargo check -p poker-solver-trainer`
Expected: warnings about unused imports in main.rs (indicatif), but no errors from new deps

**Step 3: Commit**

```
feat(trainer): swap indicatif for ratatui/crossterm TUI deps
```

---

### Task 2: Create TuiMetrics shared state

**Files:**
- Create: `crates/trainer/src/tui_metrics.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod tui_metrics;`)

**Step 1: Write a test for TuiMetrics basic operations**

In `crates/trainer/src/tui_metrics.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn tui_metrics_atomic_increment() {
        let m = TuiMetrics::new(7, 200);
        m.traversal_count.fetch_add(100, Ordering::Relaxed);
        assert_eq!(m.traversal_count.load(Ordering::Relaxed), 100);
        assert_eq!(m.total_sprs.load(Ordering::Relaxed), 7);
        assert_eq!(m.total_flops.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn flop_state_lifecycle() {
        let m = TuiMetrics::new(1, 10);
        m.update_flop("AhKs2d", 0, 50, 120.5);
        m.update_flop("AhKs2d", 1, 50, 95.3);
        {
            let entry = m.flop_states.get("AhKs2d").unwrap();
            assert_eq!(entry.iteration, 1);
            assert_eq!(entry.exploitability_history.len(), 2);
        }
        m.remove_flop("AhKs2d");
        assert!(m.flop_states.get("AhKs2d").is_none());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer tui_metrics`
Expected: FAIL — module doesn't exist yet

**Step 3: Write TuiMetrics implementation**

`crates/trainer/src/tui_metrics.rs`:

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use dashmap::DashMap;

/// Per-flop convergence state, written from the on_progress callback.
pub struct FlopTuiState {
    pub exploitability_history: Vec<f64>,
    pub iteration: usize,
    pub max_iterations: usize,
}

/// Shared metrics between solver threads and the TUI renderer.
///
/// Solver threads increment atomic counters in the hot path.
/// The TUI thread samples them at its refresh interval.
pub struct TuiMetrics {
    // Global traversal counters (incremented per hand-pair traversal)
    pub traversal_count: AtomicU64,
    pub pruned_traversal_count: AtomicU64,
    pub total_action_slots: AtomicU64,
    pub pruned_action_slots: AtomicU64,

    // SPR-level progress (set by outer loop)
    pub current_spr: AtomicU32,
    pub total_sprs: AtomicU32,
    pub flops_completed: AtomicU32,
    pub total_flops: AtomicU32,

    // Per-flop exploitability (written from on_progress callback)
    pub flop_states: DashMap<String, FlopTuiState>,
}

impl TuiMetrics {
    pub fn new(total_sprs: u32, total_flops: u32) -> Self {
        Self {
            traversal_count: AtomicU64::new(0),
            pruned_traversal_count: AtomicU64::new(0),
            total_action_slots: AtomicU64::new(0),
            pruned_action_slots: AtomicU64::new(0),
            current_spr: AtomicU32::new(0),
            total_sprs: AtomicU32::new(total_sprs),
            flops_completed: AtomicU32::new(0),
            total_flops: AtomicU32::new(total_flops),
            flop_states: DashMap::new(),
        }
    }

    /// Reset per-SPR counters when starting a new SPR solve.
    pub fn start_spr(&self, spr_index: u32, total_flops: u32) {
        self.current_spr.store(spr_index, Ordering::Relaxed);
        self.flops_completed.store(0, Ordering::Relaxed);
        self.total_flops.store(total_flops, Ordering::Relaxed);
        self.traversal_count.store(0, Ordering::Relaxed);
        self.pruned_traversal_count.store(0, Ordering::Relaxed);
        self.total_action_slots.store(0, Ordering::Relaxed);
        self.pruned_action_slots.store(0, Ordering::Relaxed);
        self.flop_states.clear();
    }

    /// Update per-flop exploitability state (called from on_progress).
    pub fn update_flop(&self, flop_name: &str, iteration: usize, max_iterations: usize, exploitability: f64) {
        let mut entry = self.flop_states
            .entry(flop_name.to_string())
            .or_insert_with(|| FlopTuiState {
                exploitability_history: Vec::new(),
                iteration: 0,
                max_iterations,
            });
        entry.iteration = iteration;
        entry.max_iterations = max_iterations;
        entry.exploitability_history.push(exploitability);
    }

    /// Remove a flop entry when it finishes (called from on_progress).
    pub fn remove_flop(&self, flop_name: &str) {
        self.flop_states.remove(flop_name);
    }
}
```

Add `mod tui_metrics;` to `crates/trainer/src/main.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer tui_metrics`
Expected: 2 tests PASS

**Step 5: Commit**

```
feat(trainer): add TuiMetrics shared state for TUI dashboard
```

---

### Task 3: Instrument solver hot path with atomic counters

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`
- Modify: `crates/core/src/preflop/postflop_abstraction.rs`

This task adds an `Option<&TuiCounters>` parameter to the solver inner loop. To avoid coupling the core crate to the trainer's `TuiMetrics`, define a minimal counter trait or struct in core.

**Step 1: Add a lightweight counter struct to core**

Create a small struct in `crates/core/src/preflop/postflop_exhaustive.rs` (private to the module, exposed only through the `build_exhaustive` function signature):

```rust
use std::sync::atomic::{AtomicU64, Ordering};

/// Optional atomic counters for external progress monitoring (e.g. TUI).
/// When provided, the solver hot path increments these without blocking.
pub struct SolverCounters {
    pub traversal_count: AtomicU64,
    pub pruned_traversal_count: AtomicU64,
    pub total_action_slots: AtomicU64,
    pub pruned_action_slots: AtomicU64,
}
```

**Step 2: Add counter increments to `exhaustive_cfr_traverse`**

In `crates/core/src/preflop/postflop_exhaustive.rs`, at the Decision node hero branch (line ~216), add the counter parameter to the function signature and increment at the pruning decision point:

At the point where `prune_mask` is computed and actions are iterated (lines 238-258), add:

```rust
// After computing prune_mask, before the action loop:
if let Some(counters) = counters {
    let total = num_actions as u64;
    let pruned = prune_mask.count_ones() as u64;
    counters.total_action_slots.fetch_add(total, Ordering::Relaxed);
    counters.pruned_action_slots.fetch_add(pruned, Ordering::Relaxed);
}
```

Thread `Option<&SolverCounters>` through:
- `exhaustive_cfr_traverse` — add parameter, pass down in recursive calls
- `PostflopCfrCtx` — add field `counters: Option<&'a SolverCounters>`
- `traverse_pair` impl — increment `traversal_count` by 1 per call
- `build_exhaustive` — accept `Option<&SolverCounters>`, pass into `PostflopCfrCtx`
- `build_for_spr` — accept `Option<&SolverCounters>`, pass to `build_exhaustive`

**Step 3: Add a test verifying counters are incremented**

Add to `crates/core/src/preflop/postflop_exhaustive.rs` tests:

```rust
#[timed_test(5000)]
fn solver_counters_are_incremented() {
    use std::sync::atomic::Ordering;
    let config = PostflopModelConfig {
        max_flop_boards: 1,
        postflop_solve_iterations: 5,
        solve_type: PostflopSolveType::Exhaustive,
        ..PostflopModelConfig::default()
    };
    let counters = SolverCounters {
        traversal_count: AtomicU64::new(0),
        pruned_traversal_count: AtomicU64::new(0),
        total_action_slots: AtomicU64::new(0),
        pruned_action_slots: AtomicU64::new(0),
    };
    let _result = PostflopAbstraction::build_for_spr(
        &config, 3.0, None, None, Some(&counters), |_| {},
    );
    let traversals = counters.traversal_count.load(Ordering::Relaxed);
    assert!(traversals > 0, "traversal counter should be incremented");
    let total_actions = counters.total_action_slots.load(Ordering::Relaxed);
    assert!(total_actions > 0, "action counter should be incremented");
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core solver_counters_are_incremented`
Expected: PASS

Also run: `cargo test -p poker-solver-core`
Expected: All existing tests pass (no regressions from the new `Option` parameter defaulting to `None` at all existing call sites)

**Step 5: Commit**

```
feat(core): add atomic solver counters for TUI instrumentation

Thread optional SolverCounters through the exhaustive CFR hot path.
Increments traversal and action-slot counters without blocking.
Existing callers pass None — zero overhead when unused.
```

---

### Task 4: Build TUI renderer

**Files:**
- Create: `crates/trainer/src/tui.rs`

This is the largest task. The TUI thread owns all rendering state and reads from `Arc<TuiMetrics>`.

**Step 1: Write the TUI module skeleton with a smoke test**

The smoke test verifies the `TuiApp` can be constructed and that `render_to_buffer` produces non-empty output (using ratatui's `TestBackend`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tui_app_renders_without_panic() {
        let metrics = Arc::new(TuiMetrics::new(3, 200));
        let mut app = TuiApp::new(Arc::clone(&metrics));
        // Simulate one tick of data
        metrics.traversal_count.fetch_add(1000, Ordering::Relaxed);
        app.tick();
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }
}
```

**Step 2: Implement `TuiApp` struct**

```rust
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, Sparkline, List, ListItem, Paragraph};
use crate::tui_metrics::TuiMetrics;

const SPARKLINE_HISTORY: usize = 60;

pub struct TuiApp {
    metrics: Arc<TuiMetrics>,
    start_time: Instant,

    // Sparkline histories (ring buffers)
    traversals_per_sec: Vec<u64>,
    remaining_pairs: Vec<u64>,
    pct_traversals_pruned: Vec<u64>,
    pct_actions_pruned: Vec<u64>,

    // Previous tick's counter values for computing deltas
    prev_traversal_count: u64,
    prev_pruned_traversal_count: u64,
    prev_total_action_slots: u64,
    prev_pruned_action_slots: u64,

    // Total expected traversals for "remaining" sparkline
    total_expected_traversals: u64,
}
```

**Step 3: Implement `TuiApp::new()` and `TuiApp::tick()`**

`tick()` samples atomics, computes deltas, pushes to sparkline histories:

```rust
impl TuiApp {
    pub fn new(metrics: Arc<TuiMetrics>) -> Self {
        Self {
            metrics,
            start_time: Instant::now(),
            traversals_per_sec: Vec::with_capacity(SPARKLINE_HISTORY),
            remaining_pairs: Vec::with_capacity(SPARKLINE_HISTORY),
            pct_traversals_pruned: Vec::with_capacity(SPARKLINE_HISTORY),
            pct_actions_pruned: Vec::with_capacity(SPARKLINE_HISTORY),
            prev_traversal_count: 0,
            prev_pruned_traversal_count: 0,
            prev_total_action_slots: 0,
            prev_pruned_action_slots: 0,
            total_expected_traversals: 0,
        }
    }

    pub fn set_total_expected_traversals(&mut self, total: u64) {
        self.total_expected_traversals = total;
    }

    pub fn tick(&mut self) {
        let t = self.metrics.traversal_count.load(Ordering::Relaxed);
        let pt = self.metrics.pruned_traversal_count.load(Ordering::Relaxed);
        let ta = self.metrics.total_action_slots.load(Ordering::Relaxed);
        let pa = self.metrics.pruned_action_slots.load(Ordering::Relaxed);

        let dt = t.saturating_sub(self.prev_traversal_count);
        let dpt = pt.saturating_sub(self.prev_pruned_traversal_count);
        let dta = ta.saturating_sub(self.prev_total_action_slots);
        let dpa = pa.saturating_sub(self.prev_pruned_action_slots);

        self.push_sparkline(&mut self.traversals_per_sec.clone(), dt);
        // For remaining: total_expected - current count
        let remaining = self.total_expected_traversals.saturating_sub(t);
        self.push_sparkline(&mut self.remaining_pairs.clone(), remaining);
        // Percentage as 0-100 integer for sparkline
        let pct_trav = if dt > 0 { (dpt * 100) / dt } else { 0 };
        self.push_sparkline(&mut self.pct_traversals_pruned.clone(), pct_trav);
        let pct_act = if dta > 0 { (dpa * 100) / dta } else { 0 };
        self.push_sparkline(&mut self.pct_actions_pruned.clone(), pct_act);

        // NOTE: the above clone+push is a simplification for the plan.
        // Actual implementation should push directly:
        push_bounded(&mut self.traversals_per_sec, dt, SPARKLINE_HISTORY);
        push_bounded(&mut self.remaining_pairs, remaining, SPARKLINE_HISTORY);
        push_bounded(&mut self.pct_traversals_pruned, pct_trav, SPARKLINE_HISTORY);
        push_bounded(&mut self.pct_actions_pruned, pct_act, SPARKLINE_HISTORY);

        self.prev_traversal_count = t;
        self.prev_pruned_traversal_count = pt;
        self.prev_total_action_slots = ta;
        self.prev_pruned_action_slots = pa;
    }
}

fn push_bounded(buf: &mut Vec<u64>, value: u64, max_len: usize) {
    if buf.len() >= max_len {
        buf.remove(0);
    }
    buf.push(value);
}
```

**Step 4: Implement `TuiApp::render()`**

Layout using ratatui constraints:
1. SPR gauge (1 row)
2. Flop gauge (1 row)
3. 4 sparklines (1 row each)
4. Active flops scrollable list (remaining space)
5. Footer elapsed/ETA (1 row)

```rust
impl TuiApp {
    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // SPR gauge
                Constraint::Length(1),  // Flop gauge
                Constraint::Length(1),  // spacer
                Constraint::Length(2),  // traversals/sec sparkline
                Constraint::Length(2),  // remaining sparkline
                Constraint::Length(2),  // % traversals pruned sparkline
                Constraint::Length(2),  // % actions pruned sparkline
                Constraint::Length(1),  // spacer / section header
                Constraint::Min(3),    // active flops (scrollable)
                Constraint::Length(1),  // footer
            ])
            .split(area);

        // SPR gauge
        let spr_idx = self.metrics.current_spr.load(Ordering::Relaxed);
        let spr_total = self.metrics.total_sprs.load(Ordering::Relaxed);
        let spr_ratio = if spr_total > 0 { spr_idx as f64 / spr_total as f64 } else { 0.0 };
        let spr_gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(spr_ratio.min(1.0))
            .label(format!("SPR Progress: {}/{}", spr_idx, spr_total));
        frame.render_widget(spr_gauge, chunks[0]);

        // Flop gauge
        let flops_done = self.metrics.flops_completed.load(Ordering::Relaxed);
        let flops_total = self.metrics.total_flops.load(Ordering::Relaxed);
        let flop_ratio = if flops_total > 0 { flops_done as f64 / flops_total as f64 } else { 0.0 };
        let flop_gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(flop_ratio.min(1.0))
            .label(format!("Flop Progress: {}/{}", flops_done, flops_total));
        frame.render_widget(flop_gauge, chunks[1]);

        // Sparklines
        render_sparkline(frame, chunks[3], "Traversals/sec", &self.traversals_per_sec);
        render_sparkline(frame, chunks[4], "Remaining pairs", &self.remaining_pairs);
        render_sparkline(frame, chunks[5], "% Traversals pruned", &self.pct_traversals_pruned);
        render_sparkline(frame, chunks[6], "% Actions pruned", &self.pct_actions_pruned);

        // Active flops section header
        let header = Paragraph::new("── Active Flops ──")
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(header, chunks[7]);

        // Active flops list (scrollable)
        let mut items: Vec<ListItem> = Vec::new();
        for entry in self.metrics.flop_states.iter() {
            let name = entry.key();
            let state = entry.value();
            let latest_expl = state.exploitability_history.last().copied().unwrap_or(0.0);
            // Build mini sparkline text from history
            let spark_chars: String = state.exploitability_history.iter().rev().take(20).rev()
                .map(|&v| sparkline_char(v, &state.exploitability_history))
                .collect();
            let line = format!(
                "{:<8} expl {} {:>8.1} mBB/h  iter {}/{}",
                name, spark_chars, latest_expl, state.iteration, state.max_iterations,
            );
            items.push(ListItem::new(line));
        }
        // Sort by iteration progress descending
        items.sort_by(|a, b| b.to_string().cmp(&a.to_string()));
        let list = List::new(items)
            .block(Block::default().borders(Borders::NONE));
        frame.render_widget(list, chunks[8]);

        // Footer: elapsed + ETA
        let elapsed = self.start_time.elapsed();
        let eta = if flop_ratio > 0.01 {
            let total_est = elapsed.as_secs_f64() / flop_ratio;
            let remaining_secs = (total_est - elapsed.as_secs_f64()).max(0.0);
            format_duration(remaining_secs)
        } else {
            "calculating...".to_string()
        };
        let footer = Paragraph::new(format!(
            "Elapsed: {}    ETA: {}",
            format_duration(elapsed.as_secs_f64()),
            eta,
        ));
        frame.render_widget(footer, chunks[9]);
    }
}

fn render_sparkline(frame: &mut Frame, area: Rect, label: &str, data: &[u64]) {
    let sparkline = Sparkline::default()
        .block(Block::default().title(label))
        .data(data)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(sparkline, area);
}

fn format_duration(secs: f64) -> String {
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    let s = (secs % 60.0) as u64;
    format!("{h:02}:{m:02}:{s:02}")
}

fn sparkline_char(value: f64, history: &[f64]) -> char {
    let max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = history.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max - min).max(1e-9);
    let normalized = ((value - min) / range * 7.0) as usize;
    const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    CHARS[normalized.min(7)]
}
```

**Step 5: Implement the TUI event loop (the public entry point)**

```rust
use std::io;
use std::time::Duration;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;

/// Run the TUI in a dedicated thread. Returns a handle to join on completion.
/// The TUI runs until `done` is set to true by the solver thread.
pub fn run_tui(
    metrics: Arc<TuiMetrics>,
    refresh_interval: Duration,
    done: Arc<std::sync::atomic::AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_tui_inner(metrics, refresh_interval, done) {
            eprintln!("TUI error: {e}");
        }
    })
}

fn run_tui_inner(
    metrics: Arc<TuiMetrics>,
    refresh_interval: Duration,
    done: Arc<std::sync::atomic::AtomicBool>,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stderr();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = TuiApp::new(metrics);

    loop {
        app.tick();
        terminal.draw(|frame| app.render(frame))?;

        // Check for quit key or done signal
        if event::poll(refresh_interval)? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        if done.load(Ordering::Relaxed) {
            // Final render
            app.tick();
            terminal.draw(|frame| app.render(frame))?;
            // Brief pause so user can see final state
            std::thread::sleep(Duration::from_secs(2));
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
```

**Step 6: Run the smoke test**

Run: `cargo test -p poker-solver-trainer tui_app_renders`
Expected: PASS

**Step 7: Commit**

```
feat(trainer): add ratatui TUI renderer with sparklines and gauges

Full-screen alternate-terminal dashboard showing SPR/flop progress
gauges, traversal rate sparklines, pruning percentage sparklines,
per-flop exploitability rows, and elapsed/ETA footer.
```

---

### Task 5: Wire TUI into solve-postflop pipeline

**Files:**
- Modify: `crates/trainer/src/main.rs`

This is the integration task. Replace the indicatif progress bars with the TUI.

**Step 1: Remove indicatif imports and the entire `build_postflop_with_progress` function**

Remove from main.rs:
- `use indicatif::{MultiProgress, ProgressBar, ProgressStyle};` (line 13)
- The entire `build_postflop_with_progress` function (lines 316-519) including the `FlopSlotData` and `FlopBarState` structs

**Step 2: Add new imports**

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use crate::tui_metrics::TuiMetrics;
use crate::tui;
```

**Step 3: Add `--tui-refresh` CLI arg to SolvePostflop**

```rust
SolvePostflop {
    #[arg(short, long)]
    config: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    /// TUI dashboard refresh interval in seconds (default: 1.0)
    #[arg(long, default_value = "1.0")]
    tui_refresh: f64,
},
```

**Step 4: Rewrite `build_postflop_with_progress` to use TuiMetrics**

Replace with a new version that:
1. Creates `Arc<TuiMetrics>` and `Arc<SolverCounters>`
2. Spawns TUI thread via `tui::run_tui()`
3. In the SPR loop, calls `metrics.start_spr()` before each SPR
4. In the `on_progress` callback, routes `BuildPhase` events into `TuiMetrics`:
   - `FlopStage::Solving` → `metrics.update_flop()`
   - `FlopStage::Done` → `metrics.remove_flop()`
   - `MccfrFlopsCompleted` → `metrics.flops_completed.store()`
5. Passes `Some(&counters)` to `build_for_spr`
6. After all SPRs complete, sets `done` flag and joins TUI thread

**Step 5: Update `run_solve_postflop` to pass `tui_refresh`**

Thread the refresh interval from CLI args through to the new `build_postflop_with_progress`.

**Step 6: Verify compilation and basic run**

Run: `cargo build -p poker-solver-trainer --release`
Expected: Compiles clean

Run with a small config: `cargo run -p poker-solver-trainer --release -- solve-postflop -c sample_configurations/tiny.yaml -o /tmp/test_tui`
Expected: TUI dashboard appears, shows progress, exits cleanly

**Step 7: Commit**

```
feat(trainer): wire TUI dashboard into solve-postflop pipeline

Replace indicatif multi-bar progress with full-screen ratatui TUI.
Solver threads write to shared atomic counters; TUI thread samples
and renders at configurable interval (--tui-refresh flag).
```

---

### Task 6: TTY fallback for non-interactive use

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Add TTY detection**

At the start of the new `build_postflop_with_progress`, check if stderr is a TTY:

```rust
let use_tui = atty::is(atty::Stream::Stderr);
```

**Step 2: Implement simple fallback logger**

When `!use_tui`, instead of spawning the TUI thread, use the `on_progress` callback to print simple line-based progress to stderr:

```rust
if !use_tui {
    // Fallback: print progress lines
    eprintln!("SPR {}/{}: {} flops", spr_idx + 1, total_sprs, total_flops);
    // In on_progress callback:
    // MccfrFlopsCompleted → eprintln!("  {completed}/{total} flops completed")
}
```

**Step 3: Test with piped output**

Run: `cargo run -p poker-solver-trainer --release -- solve-postflop -c sample_configurations/tiny.yaml -o /tmp/test_tui 2>&1 | head -20`
Expected: Simple text progress lines, no ANSI escape sequences

**Step 4: Commit**

```
feat(trainer): add TTY fallback for non-interactive solve-postflop

When stderr is not a TTY (e.g. piped output), fall back to simple
line-based progress logging instead of the full-screen TUI.
```

---

### Task 7: Integration test

**Files:**
- Create: `crates/trainer/tests/tui_integration.rs`

**Step 1: Write an integration test that exercises the full pipeline**

```rust
use std::process::Command;

#[test]
#[ignore] // requires sample config; run with --ignored
fn solve_postflop_tui_completes() {
    let output = Command::new("cargo")
        .args(["run", "-p", "poker-solver-trainer", "--release", "--",
               "solve-postflop",
               "-c", "sample_configurations/tiny.yaml",
               "-o", "/tmp/tui_test_output",
               "--tui-refresh", "0.5"])
        .env("TERM", "dumb") // force non-TTY fallback
        .output()
        .expect("failed to run trainer");
    assert!(output.status.success(), "trainer failed: {}", String::from_utf8_lossy(&output.stderr));
    assert!(std::path::Path::new("/tmp/tui_test_output").exists());
}
```

**Step 2: Run the integration test**

Run: `cargo test -p poker-solver-trainer --test tui_integration -- --ignored`
Expected: PASS

**Step 3: Commit**

```
test(trainer): add integration test for solve-postflop TUI pipeline
```
