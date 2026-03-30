# TUI Live Config Reload — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Press `c` in TUI to re-read YAML config and hot-swap scenario/audit panels without stopping training. Update legend bar.

**Architecture:** The scenario boards and resolved audits move behind `Arc<RwLock<>>` so the trainer's existing refresh callbacks dynamically read current data. On reload, the new data is swapped in and pushed to the TUI via a `Mutex<Option<ReloadedTuiState>>` on metrics.

**Tech Stack:** Rust, ratatui, crossterm, Arc/RwLock/Mutex

---

### Task 1: Extract scenario/audit resolution into reusable functions

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Create: `crates/trainer/src/blueprint_tui_resolve.rs`

Extract the inline resolution logic from `main.rs` lines 280-370 into a new module with two public functions:

```rust
// crates/trainer/src/blueprint_tui_resolve.rs

use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::poker::Card;
use crate::blueprint_tui::ResolvedScenario;
use crate::blueprint_tui_audit;
use crate::blueprint_tui_audit_widget;
use crate::blueprint_tui_config::{ScenarioConfig, RegretAuditConfig};

pub struct ResolvedScenarios {
    pub scenarios: Vec<ResolvedScenario>,
    pub boards: Vec<Vec<Card>>,
}

pub struct ResolvedAudits {
    pub metas: Vec<blueprint_tui_audit_widget::AuditMeta>,
    pub audits: Vec<blueprint_tui_audit::ResolvedRegretAudit>,
    pub panel: Option<blueprint_tui_audit_widget::AuditPanelState>,
}

pub fn resolve_scenarios(
    tree: &GameTree,
    storage: &BlueprintStorage,
    configs: &[ScenarioConfig],
) -> ResolvedScenarios {
    // Move the scenario resolution logic from main.rs lines 281-330 here
}

pub fn resolve_audits(
    tree: &GameTree,
    storage: &BlueprintStorage,
    configs: &[RegretAuditConfig],
    sparkline_window: usize,
) -> ResolvedAudits {
    // Move the audit resolution logic from main.rs lines 333-370 here
}
```

Update `main.rs` to call these functions instead of inline code. Add `mod blueprint_tui_resolve;` to `main.rs`.

**Verification:** `cargo build -p poker-solver-trainer` — compiles. Training with TUI still works.

**Commit:** `refactor: extract scenario/audit resolution into blueprint_tui_resolve module`

---

### Task 2: Add reload trigger plumbing

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_metrics.rs`
- Modify: `crates/trainer/src/blueprint_tui.rs`

Add to `BlueprintTuiMetrics`:

```rust
pub config_reload_trigger: Arc<AtomicBool>,

// Shared state for pushing reloaded TUI config from trainer to TUI thread
pub reloaded_tui_state: Mutex<Option<ReloadedTuiState>>,
```

Add a struct (in metrics or a shared location):

```rust
pub struct ReloadedTuiState {
    pub scenarios: Vec<ResolvedScenario>,
    pub audit_panel: Option<AuditPanelState>,
}
```

Initialize in `BlueprintTuiMetrics::new()`.

Add method:

```rust
pub fn request_config_reload(&self) {
    self.config_reload_trigger.store(true, Ordering::Relaxed);
}
```

In `blueprint_tui.rs` event loop, add keybinding:

```rust
KeyCode::Char('c') => metrics.request_config_reload(),
```

**Verification:** `cargo build -p poker-solver-trainer`

**Commit:** `feat: add config reload trigger and shared state plumbing`

---

### Task 3: Move boards and audits behind Arc<RwLock<>>

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

The strategy refresh callback currently captures `boards_for_refresh` by value. The audit refresh callback captures `resolved_audits` by value. Both need to be behind shared references so reload can swap them.

In `main.rs`, after resolving scenarios and audits:

```rust
use std::sync::RwLock;

let shared_boards: Arc<RwLock<Vec<Vec<Card>>>> = Arc::new(RwLock::new(boards));
let shared_audits: Arc<RwLock<Vec<ResolvedRegretAudit>>> = Arc::new(RwLock::new(resolved_audits));
```

Update `on_strategy_refresh` callback to read from `shared_boards`:

```rust
let boards_for_refresh = Arc::clone(&shared_boards);
trainer.on_strategy_refresh = Some(Box::new(move |scenario_idx, node_idx, storage, tree, hand_evs| {
    let boards = boards_for_refresh.read().unwrap();
    if scenario_idx < boards.len() {
        let grid = extract_strategy_grid(tree, storage, node_idx, &boards[scenario_idx], Some(hand_evs));
        metrics_for_refresh.update_scenario_grid(scenario_idx, grid);
    }
}));
```

Update `on_audit_refresh` callback to read from `shared_audits`:

```rust
let audits_for_refresh = Arc::clone(&shared_audits);
trainer.on_audit_refresh = Some(Box::new(move |storage| {
    let mut audits = audits_for_refresh.write().unwrap();
    for audit in audits.iter_mut() {
        audit.tick(storage);
    }
    let snapshots: Vec<_> = audits.iter().map(|a| a.snapshot()).collect();
    metrics_for_audit.update_regret_audits(snapshots);
}));
```

Add `config_reload_trigger` to trainer fields and wire Arc in main.rs (same pattern as snapshot_trigger).

Add `config_path: PathBuf` to `BlueprintTrainer` so the reload callback can re-read the file.

**Verification:** `cargo test -p poker-solver-trainer` — existing TUI tests pass. Manual test: training with TUI works as before.

**Commit:** `refactor: move boards and audits behind Arc<RwLock<>> for hot-swap`

---

### Task 4: Implement reload logic on trainer

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`
- Modify: `crates/trainer/src/main.rs`

Add to `BlueprintTrainer`:

```rust
pub config_reload_trigger: Arc<AtomicBool>,
pub config_path: Option<PathBuf>,
/// Callback invoked when config reload is triggered. Re-reads YAML,
/// re-resolves scenarios/audits, returns new state for TUI.
pub on_config_reload: Option<Box<dyn FnMut(&GameTree, &BlueprintStorage) + Send>>,
```

In `check_timed_actions`, after the snapshot check:

```rust
if self.config_reload_trigger.swap(false, Ordering::Relaxed) {
    if let Some(ref mut reload_fn) = self.on_config_reload {
        reload_fn(&self.tree, &self.storage);
    }
    // Update scenario tracking after reload
    // (the callback updates shared_boards/shared_audits and pushes to metrics)
}
```

In `main.rs`, set up the reload callback:

```rust
let config_path_for_reload = config.clone(); // PathBuf from CLI
let shared_boards_for_reload = Arc::clone(&shared_boards);
let shared_audits_for_reload = Arc::clone(&shared_audits);
let metrics_for_reload = Arc::clone(&metrics);
let sparkline_window = tui_config.telemetry.sparkline_window;

trainer.on_config_reload = Some(Box::new(move |tree, storage| {
    // Re-read YAML
    let Ok(yaml) = std::fs::read_to_string(&config_path_for_reload) else { return };
    let new_tui_config = blueprint_tui_config::parse_tui_config(&yaml);

    // Re-resolve scenarios
    let resolved = blueprint_tui_resolve::resolve_scenarios(tree, storage, &new_tui_config.scenarios);

    // Re-resolve audits
    let audits = blueprint_tui_resolve::resolve_audits(
        tree, storage, &new_tui_config.regret_audits, sparkline_window,
    );

    // Swap shared data
    *shared_boards_for_reload.write().unwrap() = resolved.boards;
    *shared_audits_for_reload.write().unwrap() = audits.audits;

    // Push new UI state to TUI
    let state = ReloadedTuiState {
        scenarios: resolved.scenarios,
        audit_panel: audits.panel,
    };
    *metrics_for_reload.reloaded_tui_state.lock().unwrap() = Some(state);
}));
```

Note: After the callback runs, the trainer also needs to update `scenario_node_indices` and `scenario_ev_tracker`. Add a mechanism for the callback to return the new node indices, or have the callback update them via additional shared state.

Simplest: have the callback also update a `Arc<RwLock<Vec<u32>>>` for scenario_node_indices that the strategy refresh reads dynamically. Or better: the trainer reads the new indices from the reloaded state after the callback returns.

Add to the callback return: store new node indices in a `Arc<Mutex<Option<Vec<u32>>>>` that the trainer reads after the callback:

```rust
// In check_timed_actions, after calling reload_fn:
if let Some(new_indices) = self.reloaded_node_indices.lock().unwrap().take() {
    self.scenario_node_indices = new_indices.clone();
    self.scenario_ev_tracker.set_nodes(new_indices);
}
```

**Verification:** `cargo build -p poker-solver-trainer`

**Commit:** `feat: implement config reload logic in trainer and main`

---

### Task 5: TUI consumes reloaded state and update legend

**Files:**
- Modify: `crates/trainer/src/blueprint_tui.rs`

In `BlueprintTuiApp::tick()` (or at the top of the render loop), check for reloaded state:

```rust
// Check for config reload
if let Some(state) = self.metrics.reloaded_tui_state.lock().unwrap().take() {
    self.scenarios = state.scenarios;
    self.audit_panel = state.audit_panel;
    self.active_tab = 0; // reset to first tab
}
```

Update `render_hotkeys()`:

```rust
fn render_hotkeys(&self, frame: &mut Frame, area: Rect) {
    let text = if self.audit_panel.as_ref().is_some_and(|p| !p.metas.is_empty()) {
        "[p]ause [r]efresh [s]napshot [c]onfig \u{2190}/\u{2192} scenario \u{2191}/\u{2193} audit [q]uit"
    } else {
        "[p]ause [r]efresh [s]napshot [c]onfig \u{2190}/\u{2192} tab [q]uit"
    };
    frame.render_widget(
        Paragraph::new(text).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}
```

**Verification:** `cargo build -p poker-solver-trainer`. Manual test: start training, edit YAML, press `c`, panels update.

**Commit:** `feat: TUI consumes reloaded config state and updated legend bar`

---

### Task 6: Full workspace build and test

1. `cargo build` — full workspace
2. `cargo test` — all pass, < 1 minute
3. Verify the existing snapshot trigger (`s`) and strategy refresh (`r`) still work

**Commit:** None (validation only)
