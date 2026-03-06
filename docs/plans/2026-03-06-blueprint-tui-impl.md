# Blueprint Training TUI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a Ratatui-based live training dashboard to the Blueprint V2 `train-blueprint` command, showing telemetry (left panel) and configurable 13x13 colored strategy grids (right panel).

**Architecture:** Extends the existing postflop TUI pattern in `crates/trainer/src/`. A shared `BlueprintTuiMetrics` struct with atomics + mutexes bridges the training thread and a dedicated render thread. The TUI config is a new `tui` section in the `BlueprintV2Config` YAML. A `HandGridWidget` Ratatui widget renders strategy grids with action-based coloring.

**Tech Stack:** Rust, Ratatui 0.29, Crossterm 0.28, serde_yaml (already dependencies of `poker-solver-trainer`)

---

### Task 1: TUI Config Structs

**Files:**
- Create: `crates/trainer/src/blueprint_tui_config.rs`
- Modify: `crates/core/src/blueprint_v2/config.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui_config;`)
- Test: inline `#[cfg(test)]` in `blueprint_tui_config.rs`

**Step 1: Write the failing test**

In `crates/trainer/src/blueprint_tui_config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test(10)]
    fn parse_tui_config_from_yaml() {
        let yaml = r#"
enabled: true
refresh_rate_ms: 250
telemetry:
  exploitability_interval_minutes: 5
  strategy_delta_interval_seconds: 30
  sparkline_window: 60
scenarios:
  - name: "SB Open"
    player: SB
    actions: []
  - name: "BB vs 2.5x"
    player: BB
    actions: ["raise-0"]
  - name: "SB Cbet K72"
    player: SB
    actions: ["raise-0", "call"]
    board: ["Kh", "7s", "2d"]
    street: flop
random_scenario:
  enabled: true
  hold_minutes: 3
  pool: ["preflop", "flop", "turn"]
"#;
        let config: BlueprintTuiConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.refresh_rate_ms, 250);
        assert_eq!(config.scenarios.len(), 3);
        assert_eq!(config.scenarios[0].name, "SB Open");
        assert_eq!(config.scenarios[0].player, PlayerLabel::SB);
        assert!(config.scenarios[0].board.is_none());
        assert_eq!(config.scenarios[2].board.as_ref().unwrap().len(), 3);
        assert_eq!(config.scenarios[2].street, Some(StreetLabel::Flop));
        assert!(config.random_scenario.enabled);
        assert_eq!(config.random_scenario.hold_minutes, 3);
        assert_eq!(config.random_scenario.pool.len(), 3);
    }

    #[timed_test(10)]
    fn defaults_when_tui_absent() {
        let config = BlueprintTuiConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.refresh_rate_ms, 250);
        assert!(config.scenarios.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer blueprint_tui_config --no-run 2>&1 | head -20`
Expected: compilation error — module and types don't exist yet

**Step 3: Write implementation**

In `crates/trainer/src/blueprint_tui_config.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintTuiConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_refresh_rate")]
    pub refresh_rate_ms: u64,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub scenarios: Vec<ScenarioConfig>,
    #[serde(default)]
    pub random_scenario: RandomScenarioConfig,
}

impl Default for BlueprintTuiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            refresh_rate_ms: default_refresh_rate(),
            telemetry: TelemetryConfig::default(),
            scenarios: Vec::new(),
            random_scenario: RandomScenarioConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_exploitability_interval")]
    pub exploitability_interval_minutes: u64,
    #[serde(default = "default_strategy_delta_interval")]
    pub strategy_delta_interval_seconds: u64,
    #[serde(default = "default_sparkline_window")]
    pub sparkline_window: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            exploitability_interval_minutes: default_exploitability_interval(),
            strategy_delta_interval_seconds: default_strategy_delta_interval(),
            sparkline_window: default_sparkline_window(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioConfig {
    pub name: String,
    pub player: PlayerLabel,
    #[serde(default)]
    pub actions: Vec<String>,
    #[serde(default)]
    pub board: Option<Vec<String>>,
    #[serde(default)]
    pub street: Option<StreetLabel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PlayerLabel {
    SB,
    BB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreetLabel {
    Preflop,
    Flop,
    Turn,
    River,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomScenarioConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_hold_minutes")]
    pub hold_minutes: u64,
    #[serde(default)]
    pub pool: Vec<StreetLabel>,
}

impl Default for RandomScenarioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hold_minutes: default_hold_minutes(),
            pool: vec![StreetLabel::Preflop, StreetLabel::Flop, StreetLabel::Turn],
        }
    }
}

const fn default_refresh_rate() -> u64 { 250 }
const fn default_exploitability_interval() -> u64 { 5 }
const fn default_strategy_delta_interval() -> u64 { 30 }
const fn default_sparkline_window() -> usize { 60 }
const fn default_hold_minutes() -> u64 { 3 }
```

Then add the `tui` field to `BlueprintV2Config` in `crates/core/src/blueprint_v2/config.rs`:

Since `BlueprintTuiConfig` lives in the trainer crate (not core), we need the config to be optional and deserialized at the trainer level. **The cleanest approach**: add `#[serde(default, skip_serializing_if = "Option::is_none")] pub tui: Option<serde_yaml::Value>` to `BlueprintV2Config` so it doesn't fail on unknown fields, then parse it separately in the trainer. **Even simpler**: just use `#[serde(flatten)]` or `#[serde(deny_unknown_fields)]` is not set (it isn't), so serde_yaml will silently ignore the `tui` key. We parse TUI config separately from the same YAML in the trainer.

**Decision**: Keep `BlueprintV2Config` untouched. In the trainer, parse TUI config from the same YAML file as a separate top-level key. Since `BlueprintV2Config` doesn't use `deny_unknown_fields`, the `tui` key is silently ignored by core's parser.

In `crates/trainer/src/main.rs`, add `mod blueprint_tui_config;`.

Create a helper to extract TUI config from the raw YAML:

```rust
// In blueprint_tui_config.rs
/// Extract TUI config from a YAML string that may contain a top-level `tui` key.
pub fn parse_tui_config(yaml: &str) -> BlueprintTuiConfig {
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(default)]
        tui: BlueprintTuiConfig,
    }
    serde_yaml::from_str::<Wrapper>(yaml)
        .map(|w| w.tui)
        .unwrap_or_default()
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer blueprint_tui_config -- --nocapture`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_config.rs crates/trainer/src/main.rs
git commit -m "feat(tui): add BlueprintTuiConfig with YAML parsing

Serde structs for TUI section of Blueprint V2 config: scenarios,
telemetry intervals, random scenario rotation, refresh rate."
```

---

### Task 2: Blueprint TUI Metrics

**Files:**
- Create: `crates/trainer/src/blueprint_tui_metrics.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui_metrics;`)
- Test: inline `#[cfg(test)]`

**Step 1: Write the failing test**

In `crates/trainer/src/blueprint_tui_metrics.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use test_macros::timed_test;

    #[timed_test(10)]
    fn initial_state() {
        let m = BlueprintTuiMetrics::new(Some(1_000_000));
        assert_eq!(m.iterations.load(Ordering::Relaxed), 0);
        assert_eq!(m.target_iterations.unwrap(), 1_000_000);
        assert!(!m.paused.load(Ordering::Relaxed));
    }

    #[timed_test(10)]
    fn strategy_snapshot_lifecycle() {
        let m = BlueprintTuiMetrics::new(None);
        let scenario_id = 0;

        // Initially empty
        assert!(m.strategy_snapshots.lock().unwrap()[scenario_id].is_empty());

        // Update
        let probs = vec![0.3, 0.5, 0.2];
        m.update_scenario_strategy(scenario_id, probs.clone());
        let snap = m.strategy_snapshots.lock().unwrap();
        assert_eq!(snap[scenario_id].len(), 3);
    }

    #[timed_test(10)]
    fn exploitability_history_push() {
        let m = BlueprintTuiMetrics::new(None);
        m.push_exploitability(120.5);
        m.push_exploitability(95.3);
        let hist = m.exploitability_history.lock().unwrap();
        assert_eq!(hist.len(), 2);
        assert!((hist[1].1 - 95.3).abs() < 1e-6);
    }

    #[timed_test(10)]
    fn pause_resume() {
        let m = BlueprintTuiMetrics::new(None);
        assert!(!m.is_paused());
        m.toggle_pause();
        assert!(m.is_paused());
        m.toggle_pause();
        assert!(!m.is_paused());
    }

    #[timed_test(10)]
    fn trigger_flags() {
        let m = BlueprintTuiMetrics::new(None);
        assert!(!m.take_snapshot_trigger());
        m.request_snapshot();
        assert!(m.take_snapshot_trigger());
        assert!(!m.take_snapshot_trigger()); // consumed
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer blueprint_tui_metrics --no-run 2>&1 | head -20`
Expected: compilation error

**Step 3: Write implementation**

```rust
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Max number of strategy scenarios the TUI can track.
const MAX_SCENARIOS: usize = 16;

/// Shared metrics between the training thread and the TUI render thread.
pub struct BlueprintTuiMetrics {
    // -- Lock-free counters (hot path) --
    pub iterations: AtomicU64,
    pub target_iterations: Option<u64>,
    pub start_time: Instant,

    // -- Control flags --
    pub paused: AtomicBool,
    pub quit_requested: AtomicBool,
    snapshot_trigger: AtomicBool,
    exploitability_trigger: AtomicBool,

    // -- Infrequent bulk data (behind Mutex) --
    /// Per-scenario strategy vectors. Index = scenario order in config.
    pub strategy_snapshots: Mutex<Vec<Vec<f64>>>,
    /// Per-scenario previous strategy (for computing delta).
    pub prev_strategy_snapshots: Mutex<Vec<Vec<f64>>>,
    /// Strategy delta per scenario (average absolute change).
    pub strategy_deltas: Mutex<Vec<f64>>,
    /// Exploitability history: (elapsed_secs, mBB/hand).
    pub exploitability_history: Mutex<VecDeque<(f64, f64)>>,
    /// Random scenario descriptor (if active).
    pub random_scenario: Mutex<Option<RandomScenarioState>>,
}

/// State for the currently-displayed random scenario.
#[derive(Debug, Clone)]
pub struct RandomScenarioState {
    pub name: String,
    pub node_idx: u32,
    pub strategy: Vec<f64>,
    pub board_display: Option<String>,
    pub cluster_id: Option<u32>,
    pub action_path: Vec<String>,
    pub started_at: Instant,
}

impl BlueprintTuiMetrics {
    pub fn new(target_iterations: Option<u64>) -> Self {
        let mut snapshots = Vec::with_capacity(MAX_SCENARIOS);
        let mut prev_snapshots = Vec::with_capacity(MAX_SCENARIOS);
        let mut deltas = Vec::with_capacity(MAX_SCENARIOS);
        for _ in 0..MAX_SCENARIOS {
            snapshots.push(Vec::new());
            prev_snapshots.push(Vec::new());
            deltas.push(0.0);
        }
        Self {
            iterations: AtomicU64::new(0),
            target_iterations,
            start_time: Instant::now(),
            paused: AtomicBool::new(false),
            quit_requested: AtomicBool::new(false),
            snapshot_trigger: AtomicBool::new(false),
            exploitability_trigger: AtomicBool::new(false),
            strategy_snapshots: Mutex::new(snapshots),
            prev_strategy_snapshots: Mutex::new(prev_snapshots),
            strategy_deltas: Mutex::new(deltas),
            exploitability_history: Mutex::new(VecDeque::new()),
            random_scenario: Mutex::new(None),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn toggle_pause(&self) {
        let prev = self.paused.load(Ordering::Relaxed);
        self.paused.store(!prev, Ordering::Relaxed);
    }

    pub fn request_snapshot(&self) {
        self.snapshot_trigger.store(true, Ordering::Relaxed);
    }

    pub fn take_snapshot_trigger(&self) -> bool {
        self.snapshot_trigger.swap(false, Ordering::Relaxed)
    }

    pub fn request_exploitability(&self) {
        self.exploitability_trigger.store(true, Ordering::Relaxed);
    }

    pub fn take_exploitability_trigger(&self) -> bool {
        self.exploitability_trigger.swap(false, Ordering::Relaxed)
    }

    pub fn update_scenario_strategy(&self, scenario_idx: usize, probs: Vec<f64>) {
        let mut snaps = self.strategy_snapshots.lock().unwrap();
        if scenario_idx < snaps.len() {
            // Compute delta before overwriting
            let mut prev = self.prev_strategy_snapshots.lock().unwrap();
            let mut deltas = self.strategy_deltas.lock().unwrap();
            if !snaps[scenario_idx].is_empty() && snaps[scenario_idx].len() == probs.len() {
                let delta: f64 = snaps[scenario_idx]
                    .iter()
                    .zip(probs.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
                    / probs.len() as f64;
                deltas[scenario_idx] = delta;
                prev[scenario_idx] = snaps[scenario_idx].clone();
            }
            snaps[scenario_idx] = probs;
        }
    }

    pub fn push_exploitability(&self, mbb: f64) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let mut hist = self.exploitability_history.lock().unwrap();
        hist.push_back((elapsed, mbb));
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer blueprint_tui_metrics -- --nocapture`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_metrics.rs crates/trainer/src/main.rs
git commit -m "feat(tui): add BlueprintTuiMetrics shared state

Lock-free atomics for iteration counters, Mutex-guarded strategy
snapshots and exploitability history. Pause/resume and trigger flags
for snapshot and exploitability requests."
```

---

### Task 3: Hand Grid Widget

**Files:**
- Create: `crates/trainer/src/blueprint_tui_widgets.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui_widgets;`)
- Test: inline `#[cfg(test)]` using Ratatui `TestBackend`

**Step 1: Write the failing test**

In `crates/trainer/src/blueprint_tui_widgets.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use test_macros::timed_test;

    fn mock_grid_state() -> HandGridState {
        let mut cells = [[ACTION_EMPTY; 13]; 13];
        // AA: 80% raise, 15% call, 5% fold
        cells[0][0] = CellStrategy {
            actions: vec![
                ("fold".into(), 0.05),
                ("call".into(), 0.15),
                ("raise-0".into(), 0.80),
            ],
        };
        // 72o: 90% fold, 10% call
        cells[12][5] = CellStrategy {
            actions: vec![
                ("fold".into(), 0.90),
                ("call".into(), 0.10),
            ],
        };
        HandGridState {
            cells,
            scenario_name: "SB Open".into(),
            action_path: vec![],
            board_display: None,
            cluster_id: None,
            street_label: "Preflop".into(),
            iteration_at_snapshot: 100_000,
        }
    }

    #[timed_test(10)]
    fn action_color_mapping() {
        assert_eq!(action_color("fold"), Color::Rgb(180, 60, 60));
        assert_eq!(action_color("check"), Color::Rgb(100, 149, 237));
        assert_eq!(action_color("call"), Color::Rgb(60, 179, 113));
        assert_eq!(action_color("raise-0"), Color::Rgb(230, 190, 50));
    }

    #[timed_test(10)]
    fn dominant_action_picks_highest() {
        let cell = CellStrategy {
            actions: vec![
                ("fold".into(), 0.2),
                ("call".into(), 0.3),
                ("raise-0".into(), 0.5),
            ],
        };
        let (name, freq) = cell.dominant_action();
        assert_eq!(name, "raise-0");
        assert!((freq - 0.5).abs() < 1e-6);
    }

    #[timed_test(10)]
    fn widget_renders_without_panic() {
        let state = mock_grid_state();
        let backend = TestBackend::new(80, 30);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| {
            let area = frame.area();
            frame.render_widget(&HandGridWidget { state: &state }, area);
        }).unwrap();
    }
}
```

**Step 2: Run test to verify it fails**

Expected: compilation error — types/widget don't exist

**Step 3: Write implementation**

```rust
use ratatui::prelude::*;
use ratatui::widgets::Widget;

/// Per-cell strategy data.
#[derive(Debug, Clone)]
pub struct CellStrategy {
    pub actions: Vec<(String, f32)>,
}

const ACTION_EMPTY: CellStrategy = CellStrategy { actions: Vec::new() };

impl CellStrategy {
    /// Returns (action_name, frequency) of the most-frequent action.
    pub fn dominant_action(&self) -> (&str, f32) {
        self.actions
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(n, f)| (n.as_str(), *f))
            .unwrap_or(("", 0.0))
    }
}

/// Full state for rendering a 13x13 hand grid.
#[derive(Debug, Clone)]
pub struct HandGridState {
    pub cells: [[CellStrategy; 13]; 13],
    pub scenario_name: String,
    pub action_path: Vec<String>,
    pub board_display: Option<String>,
    pub cluster_id: Option<u32>,
    pub street_label: String,
    pub iteration_at_snapshot: u64,
}

/// Ratatui widget that renders a 13x13 hand grid.
pub struct HandGridWidget<'a> {
    pub state: &'a HandGridState,
}

const RANK_LABELS: [&str; 13] = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];

/// Map action name prefix to a color.
pub fn action_color(name: &str) -> Color {
    if name.starts_with("fold") {
        Color::Rgb(180, 60, 60) // Soft red
    } else if name.starts_with("check") {
        Color::Rgb(100, 149, 237) // Cornflower blue
    } else if name.starts_with("call") {
        Color::Rgb(60, 179, 113) // Medium sea green
    } else if name.contains("all") || name.contains("ai") {
        Color::Rgb(255, 255, 255) // White (on red bg)
    } else if name.starts_with("bet") || name.starts_with("raise") {
        // Aggression level by suffix digit
        let digit = name.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(0);
        match digit {
            0 => Color::Rgb(230, 190, 50),  // Yellow — small
            1 => Color::Rgb(230, 150, 40),  // Orange-yellow — medium-small
            2 => Color::Rgb(220, 120, 40),  // Orange — medium
            3 => Color::Rgb(200, 80, 120),  // Pink-magenta — large
            _ => Color::Rgb(180, 60, 180),  // Magenta — very large
        }
    } else {
        Color::DarkGray
    }
}

/// Dim a color to ~40% brightness for near-uniform strategies.
fn dim_color(color: Color, dominance: f32) -> Color {
    match color {
        Color::Rgb(r, g, b) => {
            // Scale brightness from 40% (at 0.33 dominance) to 100% (at 1.0)
            let factor = 0.4 + 0.6 * ((dominance - 0.33) / 0.67).clamp(0.0, 1.0);
            Color::Rgb(
                (r as f32 * factor) as u8,
                (g as f32 * factor) as u8,
                (b as f32 * factor) as u8,
            )
        }
        other => other,
    }
}

impl Widget for &HandGridWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Layout: 3 chars for row label, then 13 cells of (area.width - 3) / 13 each
        // Reserve 2 rows for header and 2 for footer
        let header_height = 2u16;
        let footer_height = 2u16;
        let grid_height = area.height.saturating_sub(header_height + footer_height).min(13);
        let cell_width = (area.width.saturating_sub(3)) / 13;

        if cell_width < 2 || grid_height < 1 {
            return; // Too small to render
        }

        // Title
        let title = format!(
            " {} — {}",
            self.state.scenario_name, self.state.street_label
        );
        buf.set_string(area.x, area.y, &title, Style::default().bold());

        // Column headers
        let header_y = area.y + 1;
        for col in 0..13 {
            let x = area.x + 3 + col as u16 * cell_width;
            let label = if cell_width >= 3 { RANK_LABELS[col] } else { &RANK_LABELS[col][..1] };
            buf.set_string(x, header_y, label, Style::default().fg(Color::DarkGray));
        }

        // Grid cells
        for row in 0..grid_height.min(13) as usize {
            let y = area.y + header_height + row as u16;
            // Row label
            buf.set_string(area.x, y, RANK_LABELS[row], Style::default().fg(Color::DarkGray));

            for col in 0..13 {
                let x = area.x + 3 + col as u16 * cell_width;
                let cell = &self.state.cells[row][col];

                if cell.actions.is_empty() {
                    let text = if cell_width >= 3 { " - " } else { "-" };
                    buf.set_string(x, y, text, Style::default().fg(Color::DarkGray));
                    continue;
                }

                let (dom_name, dom_freq) = cell.dominant_action();
                let color = action_color(dom_name);
                let bg = dim_color(color, dom_freq);
                let pct = (dom_freq * 100.0) as u32;
                let text = if cell_width >= 3 {
                    format!("{pct:>3}")
                } else {
                    format!("{pct:>2}")
                };
                let style = Style::default()
                    .fg(Color::White)
                    .bg(bg);
                buf.set_string(x, y, &text, style);
            }
        }

        // Footer
        let footer_y = area.y + header_height + grid_height;
        if footer_y < area.y + area.height {
            let board = self.state.board_display.as_deref().unwrap_or("--");
            let cluster = self.state.cluster_id.map_or("--".to_string(), |c| format!("#{c}"));
            let actions = if self.state.action_path.is_empty() {
                "[root]".to_string()
            } else {
                self.state.action_path.join(" > ")
            };
            let footer = format!(
                " Board: {}  Cluster: {}  Iter: {}  Actions: {}",
                board, cluster, self.state.iteration_at_snapshot, actions
            );
            buf.set_string(area.x, footer_y, &footer, Style::default().fg(Color::DarkGray));
        }

        // Legend
        if footer_y + 1 < area.y + area.height {
            let legend_parts = [
                ("FOLD", action_color("fold")),
                ("CHK", action_color("check")),
                ("CALL", action_color("call")),
                ("BET-S", action_color("bet-0")),
                ("BET-M", action_color("bet-2")),
                ("BET-L", action_color("bet-4")),
            ];
            let mut x = area.x + 1;
            for (label, color) in &legend_parts {
                let style = Style::default().fg(*color);
                buf.set_string(x, footer_y + 1, label, style);
                x += label.len() as u16 + 1;
            }
        }
    }
}
```

Note: `ACTION_EMPTY` as a `const` won't work because `Vec::new()` isn't const. Use a helper:

```rust
impl Default for CellStrategy {
    fn default() -> Self {
        Self { actions: Vec::new() }
    }
}
```

And change `cells` in `HandGridState` to use `Default` initialization in tests.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer blueprint_tui_widgets -- --nocapture`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_widgets.rs crates/trainer/src/main.rs
git commit -m "feat(tui): add HandGridWidget with action-based coloring

13x13 Ratatui widget rendering strategy frequencies as colored cells.
Color maps: red=fold, blue=check, green=call, yellow-magenta=bet/raise.
Intensity scales with dominance for early-training visibility."
```

---

### Task 4: Scenario Resolution (config → tree node)

**Files:**
- Create: `crates/trainer/src/blueprint_tui_scenarios.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui_scenarios;`)
- Test: inline `#[cfg(test)]`

This module resolves a `ScenarioConfig` action path to a game tree node index, and extracts 169-cell strategy grids from `BlueprintStorage`.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_v2::config::*;
    use poker_solver_core::blueprint_v2::game_tree::GameTree;
    use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
    use test_macros::timed_test;

    fn toy_tree() -> GameTree {
        GameTree::build(
            10.0, 0.5, 1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
            1,
        )
    }

    #[timed_test(10)]
    fn resolve_root_node() {
        let tree = toy_tree();
        let actions: Vec<String> = vec![];
        let node = resolve_action_path(&tree, &actions);
        assert!(node.is_some());
        assert_eq!(node.unwrap(), tree.root);
    }

    #[timed_test(10)]
    fn resolve_raise_call() {
        let tree = toy_tree();
        let actions = vec!["raise-0".to_string(), "call".to_string()];
        let node = resolve_action_path(&tree, &actions);
        assert!(node.is_some());
    }

    #[timed_test(10)]
    fn resolve_invalid_returns_none() {
        let tree = toy_tree();
        let actions = vec!["invalid-action".to_string()];
        let node = resolve_action_path(&tree, &actions);
        assert!(node.is_none());
    }

    #[timed_test(10)]
    fn extract_grid_returns_169_cells() {
        let tree = toy_tree();
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let node_idx = tree.root;
        let grid = extract_strategy_grid(&tree, &storage, node_idx, &bucket_counts);
        // 169 cells, each with actions
        let mut count = 0;
        for row in &grid {
            for cell in row {
                if !cell.actions.is_empty() {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 169);
    }
}
```

**Step 2: Run test to verify it fails**

Expected: compilation error

**Step 3: Write implementation**

```rust
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use crate::blueprint_tui_widgets::CellStrategy;

/// Parse an action string (e.g. "raise-0", "call", "check", "fold", "bet-2")
/// into a tree action matcher.
fn parse_action_label(s: &str) -> Option<ActionMatcher> {
    let s = s.trim().to_lowercase();
    if s == "fold" {
        Some(ActionMatcher::Fold)
    } else if s == "check" {
        Some(ActionMatcher::Check)
    } else if s == "call" {
        Some(ActionMatcher::Call)
    } else if s == "allin" || s == "all-in" {
        Some(ActionMatcher::AllIn)
    } else if let Some(idx_str) = s.strip_prefix("raise-").or_else(|| s.strip_prefix("bet-")) {
        let idx: usize = idx_str.parse().ok()?;
        if s.starts_with("raise") {
            Some(ActionMatcher::RaiseIdx(idx))
        } else {
            Some(ActionMatcher::BetIdx(idx))
        }
    } else {
        None
    }
}

#[derive(Debug)]
enum ActionMatcher {
    Fold,
    Check,
    Call,
    AllIn,
    BetIdx(usize),
    RaiseIdx(usize),
}

impl ActionMatcher {
    fn matches(&self, tree_actions: &[TreeAction], action_idx: usize) -> bool {
        let action = &tree_actions[action_idx];
        match self {
            ActionMatcher::Fold => matches!(action, TreeAction::Fold),
            ActionMatcher::Check => matches!(action, TreeAction::Check),
            ActionMatcher::Call => matches!(action, TreeAction::Call),
            ActionMatcher::AllIn => matches!(action, TreeAction::AllIn),
            ActionMatcher::BetIdx(idx) => {
                // Count Bet actions up to action_idx, match on the idx-th one
                let bet_count = tree_actions[..=action_idx]
                    .iter()
                    .filter(|a| matches!(a, TreeAction::Bet(_)))
                    .count();
                matches!(action, TreeAction::Bet(_)) && bet_count == idx + 1
            }
            ActionMatcher::RaiseIdx(idx) => {
                let raise_count = tree_actions[..=action_idx]
                    .iter()
                    .filter(|a| matches!(a, TreeAction::Raise(_)))
                    .count();
                matches!(action, TreeAction::Raise(_)) && raise_count == idx + 1
            }
        }
    }
}

/// Resolve a sequence of action labels to a game tree node index.
pub fn resolve_action_path(tree: &GameTree, actions: &[String]) -> Option<u32> {
    let mut node_idx = tree.root;

    for action_str in actions {
        let matcher = parse_action_label(action_str)?;

        match &tree.nodes[node_idx as usize] {
            GameNode::Decision { actions: tree_actions, children, .. } => {
                let mut found = false;
                for (i, _) in tree_actions.iter().enumerate() {
                    if matcher.matches(tree_actions, i) {
                        node_idx = children[i];
                        found = true;
                        break;
                    }
                }
                if !found {
                    return None;
                }
            }
            GameNode::Chance { child, .. } => {
                // Chance node: skip through to the child and retry this action
                node_idx = *child;
                // Re-try matching at the child
                match &tree.nodes[node_idx as usize] {
                    GameNode::Decision { actions: tree_actions, children, .. } => {
                        let mut found = false;
                        for (i, _) in tree_actions.iter().enumerate() {
                            if matcher.matches(tree_actions, i) {
                                node_idx = children[i];
                                found = true;
                                break;
                            }
                        }
                        if !found { return None; }
                    }
                    _ => return None,
                }
            }
            GameNode::Terminal { .. } => return None,
        }
    }

    Some(node_idx)
}

/// Format a TreeAction for display.
fn format_tree_action(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "fold".into(),
        TreeAction::Check => "check".into(),
        TreeAction::Call => "call".into(),
        TreeAction::AllIn => "all-in".into(),
        TreeAction::Bet(size) => format!("bet {size:.1}"),
        TreeAction::Raise(size) => format!("raise {size:.1}"),
    }
}

/// Extract a 13x13 strategy grid for a decision node.
///
/// For a preflop node, each cell maps to a canonical hand bucket.
/// For postflop nodes, each cell uses bucket 0 (simplified — in practice
/// the bucket depends on the specific board, which requires the clustering
/// data). The caller should provide the resolved bucket for postflop.
pub fn extract_strategy_grid(
    tree: &GameTree,
    storage: &BlueprintStorage,
    node_idx: u32,
    bucket_counts: &[u16; 4],
) -> [[CellStrategy; 13]; 13] {
    let mut grid: [[CellStrategy; 13]; 13] = std::array::from_fn(|_| {
        std::array::from_fn(|_| CellStrategy::default())
    });

    let (tree_actions, street) = match &tree.nodes[node_idx as usize] {
        GameNode::Decision { actions, street, .. } => (actions, street),
        _ => return grid,
    };

    let street_idx = *street as u8;
    let num_buckets = bucket_counts[street_idx as usize];

    for row in 0..13 {
        for col in 0..13 {
            // For preflop, bucket = canonical hand index (0..169)
            // For postflop, we'd need board-specific bucket lookup
            // For now, use canonical index mod num_buckets as approximation
            let canonical_idx = row * 13 + col;
            let bucket = (canonical_idx as u16) % num_buckets;

            let probs = storage.average_strategy(node_idx, bucket);
            let mut actions = Vec::with_capacity(probs.len());
            for (i, &p) in probs.iter().enumerate() {
                let name = format_tree_action(&tree_actions[i]);
                actions.push((name, p as f32));
            }
            grid[row][col] = CellStrategy { actions };
        }
    }

    grid
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer blueprint_tui_scenarios -- --nocapture`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_scenarios.rs crates/trainer/src/main.rs
git commit -m "feat(tui): scenario resolution and strategy grid extraction

Resolve YAML action paths to game tree nodes. Extract 13x13 strategy
grids from BlueprintStorage for display in the TUI."
```

---

### Task 5: Main TUI App — Layout and Render Loop

**Files:**
- Create: `crates/trainer/src/blueprint_tui.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui;`)
- Test: inline `#[cfg(test)]` with `TestBackend`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use ratatui::backend::TestBackend;
    use test_macros::timed_test;

    #[timed_test(10)]
    fn app_renders_without_panic() {
        let metrics = Arc::new(BlueprintTuiMetrics::new(Some(1_000_000)));
        let scenarios = vec![];
        let mut app = BlueprintTuiApp::new(
            metrics,
            scenarios,
            TelemetryConfig::default(),
        );
        app.tick();

        let backend = TestBackend::new(160, 50);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    #[timed_test(10)]
    fn tab_switching() {
        let metrics = Arc::new(BlueprintTuiMetrics::new(None));
        let scenarios = vec![
            ResolvedScenario { name: "A".into(), node_idx: 0, grid: Default::default() },
            ResolvedScenario { name: "B".into(), node_idx: 1, grid: Default::default() },
        ];
        let mut app = BlueprintTuiApp::new(metrics, scenarios, TelemetryConfig::default());
        assert_eq!(app.active_tab, 0);
        app.next_tab();
        assert_eq!(app.active_tab, 1);
        app.next_tab();
        assert_eq!(app.active_tab, 0); // wraps
        app.prev_tab();
        assert_eq!(app.active_tab, 1); // wraps backward
    }

    #[timed_test(10)]
    fn eta_calculation() {
        assert_eq!(format_eta(0, 100, 10.0), "10s");
        assert_eq!(format_eta(50, 100, 10.0), "5s");
    }
}
```

**Step 2: Run test to verify it fails**

Expected: compilation error

**Step 3: Write implementation**

```rust
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::io;

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Chart, Dataset, Gauge, Paragraph, Sparkline, Tabs};

use crate::blueprint_tui_config::TelemetryConfig;
use crate::blueprint_tui_metrics::BlueprintTuiMetrics;
use crate::blueprint_tui_widgets::{CellStrategy, HandGridState, HandGridWidget};

const SPARKLINE_HISTORY: usize = 60;

/// A scenario resolved to a tree node with its current grid state.
pub struct ResolvedScenario {
    pub name: String,
    pub node_idx: u32,
    pub grid: HandGridState,
}

impl Default for ResolvedScenario {
    fn default() -> Self {
        Self {
            name: String::new(),
            node_idx: 0,
            grid: HandGridState {
                cells: std::array::from_fn(|_| std::array::from_fn(|_| CellStrategy::default())),
                scenario_name: String::new(),
                action_path: vec![],
                board_display: None,
                cluster_id: None,
                street_label: "Preflop".into(),
                iteration_at_snapshot: 0,
            },
        }
    }
}

pub struct BlueprintTuiApp {
    pub metrics: Arc<BlueprintTuiMetrics>,
    pub scenarios: Vec<ResolvedScenario>,
    pub active_tab: usize,
    pub telemetry_config: TelemetryConfig,

    // Sparkline data
    iter_per_sec_history: Vec<u64>,
    prev_iterations: u64,
    peak_iter_per_sec: u64,
}

impl BlueprintTuiApp {
    pub fn new(
        metrics: Arc<BlueprintTuiMetrics>,
        scenarios: Vec<ResolvedScenario>,
        telemetry_config: TelemetryConfig,
    ) -> Self {
        Self {
            metrics,
            scenarios,
            active_tab: 0,
            telemetry_config,
            iter_per_sec_history: Vec::with_capacity(SPARKLINE_HISTORY),
            prev_iterations: 0,
            peak_iter_per_sec: 0,
        }
    }

    pub fn tick(&mut self) {
        let iters = self.metrics.iterations.load(Ordering::Relaxed);
        let delta = iters.saturating_sub(self.prev_iterations);
        // Assume tick called at refresh_rate_ms interval (250ms default = 4/sec)
        let ips = delta * 4; // approximate
        push_bounded(&mut self.iter_per_sec_history, ips, SPARKLINE_HISTORY);
        if ips > self.peak_iter_per_sec {
            self.peak_iter_per_sec = ips;
        }
        self.prev_iterations = iters;
    }

    pub fn next_tab(&mut self) {
        if !self.scenarios.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.scenarios.len();
        }
    }

    pub fn prev_tab(&mut self) {
        if !self.scenarios.is_empty() {
            self.active_tab = if self.active_tab == 0 {
                self.scenarios.len() - 1
            } else {
                self.active_tab - 1
            };
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        // Vertical split: left 45%, right 55%
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(area);

        self.render_left_panel(frame, chunks[0]);
        self.render_right_panel(frame, chunks[1]);
    }

    fn render_left_panel(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Training Progress ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // iterations counter
                Constraint::Length(1), // progress bar
                Constraint::Length(1), // runtime / ETA
                Constraint::Length(1), // spacer
                Constraint::Length(3), // throughput sparkline
                Constraint::Length(1), // spacer
                Constraint::Min(6),   // exploitability chart
                Constraint::Length(1), // delta-strat
                Constraint::Length(1), // hotkeys
            ])
            .split(inner);

        // Iterations counter
        let iters = self.metrics.iterations.load(Ordering::Relaxed);
        let target = self.metrics.target_iterations;
        let iter_text = match target {
            Some(t) => format!(" Iterations: {} / {}", format_count(iters), format_count(t)),
            None => format!(" Iterations: {}", format_count(iters)),
        };
        frame.render_widget(Paragraph::new(iter_text), chunks[0]);

        // Progress bar
        let ratio = target.map_or(0.0, |t| if t > 0 { iters as f64 / t as f64 } else { 0.0 });
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{:.1}%", ratio * 100.0));
        frame.render_widget(gauge, chunks[1]);

        // Runtime / ETA
        let elapsed = self.metrics.elapsed_secs();
        let eta = target.map_or("--".to_string(), |t| {
            format_eta(iters, t, elapsed)
        });
        let paused_marker = if self.metrics.is_paused() { " [PAUSED]" } else { "" };
        let runtime_text = format!(
            " Runtime: {}    ETA: {}{}",
            format_duration(elapsed), eta, paused_marker
        );
        frame.render_widget(Paragraph::new(runtime_text), chunks[2]);

        // Throughput sparkline
        let current_ips = self.iter_per_sec_history.last().copied().unwrap_or(0);
        let spark_title = format!(
            " Throughput: {} it/s  peak: {} it/s",
            format_count(current_ips),
            format_count(self.peak_iter_per_sec),
        );
        let sparkline = Sparkline::default()
            .block(Block::default().title(spark_title))
            .data(&self.iter_per_sec_history)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(sparkline, chunks[4]);

        // Exploitability chart
        let hist = self.metrics.exploitability_history.lock().unwrap();
        if !hist.is_empty() {
            let data: Vec<(f64, f64)> = hist.iter().copied().collect();
            let datasets = vec![
                Dataset::default()
                    .marker(ratatui::symbols::Marker::Braille)
                    .style(Style::default().fg(Color::Yellow))
                    .data(&data),
            ];
            let y_max = data.iter().map(|d| d.1).fold(0.0_f64, f64::max);
            let x_max = data.last().map(|d| d.0).unwrap_or(1.0);
            let chart = Chart::new(datasets)
                .block(Block::default().title(" Exploitability (mBB/hand) "))
                .x_axis(
                    ratatui::widgets::Axis::default()
                        .bounds([0.0, x_max])
                        .labels(vec![
                            Span::raw("0"),
                            Span::raw(format_duration(x_max)),
                        ]),
                )
                .y_axis(
                    ratatui::widgets::Axis::default()
                        .bounds([0.0, y_max * 1.1])
                        .labels(vec![
                            Span::raw("0"),
                            Span::raw(format!("{:.0}", y_max)),
                        ]),
                );
            frame.render_widget(chart, chunks[6]);
        } else {
            let placeholder = Paragraph::new(" Exploitability: waiting for first calculation...")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(placeholder, chunks[6]);
        }
        drop(hist);

        // Strategy delta
        let deltas = self.metrics.strategy_deltas.lock().unwrap();
        let avg_delta: f64 = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f64>() / deltas.len() as f64
        };
        drop(deltas);
        let delta_text = format!(" Avg strategy delta: {avg_delta:.6}");
        frame.render_widget(
            Paragraph::new(delta_text).style(Style::default().fg(Color::DarkGray)),
            chunks[7],
        );

        // Hotkeys
        let hotkeys = " [p]ause  [s]napshot  [e]xploitability  [?]help  [q]uit";
        frame.render_widget(
            Paragraph::new(hotkeys).style(Style::default().fg(Color::DarkGray)),
            chunks[8],
        );
    }

    fn render_right_panel(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Scenarios ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.scenarios.is_empty() {
            let msg = Paragraph::new(" No scenarios configured")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(msg, inner);
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(10)])
            .split(inner);

        // Tab bar
        let tab_titles: Vec<Line> = self
            .scenarios
            .iter()
            .map(|s| Line::from(s.name.clone()))
            .collect();
        let tabs = Tabs::new(tab_titles)
            .select(self.active_tab)
            .highlight_style(Style::default().fg(Color::Cyan).bold());
        frame.render_widget(tabs, chunks[0]);

        // Active grid
        if let Some(scenario) = self.scenarios.get(self.active_tab) {
            let widget = HandGridWidget { state: &scenario.grid };
            frame.render_widget(&widget, chunks[1]);
        }
    }
}

/// Launch the TUI on a dedicated thread.
pub fn run_blueprint_tui(
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedScenario>,
    telemetry_config: TelemetryConfig,
    refresh_interval: Duration,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_tui_inner(metrics, scenarios, telemetry_config, refresh_interval) {
            eprintln!("TUI error: {e}");
        }
    })
}

fn run_tui_inner(
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedScenario>,
    telemetry_config: TelemetryConfig,
    refresh_interval: Duration,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stderr = io::stderr();
    execute!(stderr, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stderr);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = BlueprintTuiApp::new(metrics.clone(), scenarios, telemetry_config);

    loop {
        app.tick();
        terminal.draw(|frame| app.render(frame))?;

        if event::poll(refresh_interval)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        metrics.quit_requested.store(true, Ordering::Relaxed);
                        break;
                    }
                    KeyCode::Right => app.next_tab(),
                    KeyCode::Left => app.prev_tab(),
                    KeyCode::Char('p') => metrics.toggle_pause(),
                    KeyCode::Char('s') => metrics.request_snapshot(),
                    KeyCode::Char('e') => metrics.request_exploitability(),
                    _ => {}
                }
            }
        }

        if metrics.quit_requested.load(Ordering::Relaxed) {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

fn push_bounded(buf: &mut Vec<u64>, value: u64, max_len: usize) {
    if buf.len() >= max_len {
        buf.remove(0);
    }
    buf.push(value);
}

fn format_duration(secs: f64) -> String {
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    let s = (secs % 60.0) as u64;
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

pub fn format_eta(current: u64, target: u64, elapsed_secs: f64) -> String {
    if current == 0 || elapsed_secs < 1.0 {
        return "calculating...".to_string();
    }
    let rate = current as f64 / elapsed_secs;
    let remaining = target.saturating_sub(current) as f64 / rate;
    format_duration(remaining)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer blueprint_tui::tests -- --nocapture`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui.rs crates/trainer/src/main.rs
git commit -m "feat(tui): main BlueprintTuiApp with split layout

Vertical split: left telemetry panel (progress bar, sparkline,
exploitability chart, hotkeys) + right tabbed strategy grids.
Input handling for tab switching, pause, snapshot, and quit."
```

---

### Task 6: Wire TUI into TrainBlueprint Command

**Files:**
- Modify: `crates/trainer/src/main.rs` — `Commands::TrainBlueprint` handler
- Modify: `crates/core/src/blueprint_v2/trainer.rs` — add metrics hooks
- Test: manual smoke test (run with a toy config)

**Step 1: Add `--no-tui` flag to CLI**

In the `Commands::TrainBlueprint` variant in `main.rs`:

```rust
TrainBlueprint {
    #[arg(short, long)]
    config: PathBuf,
    /// Disable the TUI dashboard (use text output instead)
    #[arg(long)]
    no_tui: bool,
},
```

**Step 2: Wire metrics into trainer**

In `crates/core/src/blueprint_v2/trainer.rs`, add a public `Arc<AtomicU64>` field for iteration count that the TUI can read:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// Add to BlueprintTrainer:
pub struct BlueprintTrainer {
    // ... existing fields ...

    /// Shared iteration counter for external consumers (TUI).
    pub shared_iterations: Arc<AtomicU64>,
    /// Pause flag — training loop spins when true.
    pub paused: Arc<AtomicBool>,
    /// Quit flag — training loop exits when true.
    pub quit_requested: Arc<AtomicBool>,
    /// Snapshot trigger — checked in timed_actions.
    pub snapshot_trigger: Arc<AtomicBool>,
}
```

Update `train()` loop:

```rust
pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
    while !self.should_stop() {
        // Check pause
        while self.paused.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if self.quit_requested.load(Ordering::Relaxed) {
                return Ok(());
            }
        }

        // ... existing traversal code ...

        self.iterations += 1;
        self.shared_iterations.store(self.iterations, Ordering::Relaxed);
        self.check_timed_actions()?;
    }
    Ok(())
}
```

And in `should_stop()`, add quit check:

```rust
fn should_stop(&self) -> bool {
    if self.quit_requested.load(Ordering::Relaxed) {
        return true;
    }
    // ... existing checks ...
}
```

In `check_timed_actions()`, add snapshot trigger check:

```rust
if self.snapshot_trigger.swap(false, Ordering::Relaxed) {
    self.save_snapshot()?;
}
```

**Step 3: Wire in the TrainBlueprint command handler**

```rust
Commands::TrainBlueprint { config, no_tui } => {
    let yaml = std::fs::read_to_string(&config)?;
    let bp_config: BlueprintV2Config = serde_yaml::from_str(&yaml)?;
    let tui_config = blueprint_tui_config::parse_tui_config(&yaml);

    eprintln!("Blueprint V2 Training");
    // ... existing eprintln! lines ...

    let mut trainer = BlueprintTrainer::new(bp_config);

    let use_tui = tui_config.enabled && !no_tui;

    if use_tui {
        let metrics = Arc::new(BlueprintTuiMetrics::new(
            trainer.config.training.iterations,
        ));

        // Share atomics between trainer and TUI metrics
        trainer.paused = Arc::clone(&metrics.paused);
        trainer.quit_requested = Arc::clone(&metrics.quit_requested);
        trainer.shared_iterations = Arc::clone(&metrics.iterations);
        trainer.snapshot_trigger = Arc::clone(&metrics.snapshot_trigger);

        // Resolve scenarios to tree nodes
        let scenarios: Vec<ResolvedScenario> = tui_config.scenarios.iter().map(|sc| {
            let node_idx = blueprint_tui_scenarios::resolve_action_path(
                &trainer.tree,
                &sc.actions,
            ).unwrap_or(trainer.tree.root);

            let grid = blueprint_tui_scenarios::extract_strategy_grid(
                &trainer.tree,
                &trainer.storage,
                node_idx,
                &trainer.storage.bucket_counts,
            );

            ResolvedScenario {
                name: sc.name.clone(),
                node_idx,
                grid: HandGridState {
                    cells: grid,
                    scenario_name: sc.name.clone(),
                    action_path: sc.actions.clone(),
                    board_display: sc.board.as_ref().map(|b| b.join(" ")),
                    cluster_id: None,
                    street_label: sc.street.map_or("Preflop".into(), |s| format!("{s:?}")),
                    iteration_at_snapshot: 0,
                },
            }
        }).collect();

        let refresh = Duration::from_millis(tui_config.refresh_rate_ms);
        let tui_handle = blueprint_tui::run_blueprint_tui(
            Arc::clone(&metrics),
            scenarios,
            tui_config.telemetry.clone(),
            refresh,
        );

        trainer.train()?;

        // Signal TUI to exit
        metrics.quit_requested.store(true, Ordering::Relaxed);
        let _ = tui_handle.join();
    } else {
        trainer.train()?;
    }

    eprintln!("\nTraining complete: {} iterations", trainer.iterations);
}
```

**Step 4: Run clippy and test suite**

Run: `cargo clippy -p poker-solver-trainer && cargo test -p poker-solver-trainer`
Expected: all pass, no warnings

**Step 5: Commit**

```bash
git add crates/trainer/src/main.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat(tui): wire TUI into train-blueprint command

Adds --no-tui flag. Trainer exposes shared atomics for iteration
count, pause, quit, and snapshot trigger. TUI thread launches
automatically when tui.enabled=true in config."
```

---

### Task 7: Strategy Refresh Loop

**Files:**
- Modify: `crates/trainer/src/blueprint_tui.rs` — add periodic strategy refresh
- Modify: `crates/trainer/src/main.rs` — pass storage reference

The TUI needs periodic access to `BlueprintStorage` to refresh strategy grids. Since storage is owned by the trainer (single-threaded), we use a "snapshot request" pattern: every `strategy_delta_interval_seconds`, the trainer copies strategy data for each scenario node into the shared metrics.

**Step 1: Add strategy refresh to trainer**

In `crates/core/src/blueprint_v2/trainer.rs`, add a new timed action:

```rust
// New field:
pub last_strategy_refresh_time: u64,
pub strategy_refresh_interval_secs: u64,
pub scenario_node_indices: Vec<u32>,
/// Callback to push strategy data to TUI metrics.
pub on_strategy_refresh: Option<Box<dyn Fn(usize, u32, &BlueprintStorage) + Send>>,
```

In `check_timed_actions()`, add:

```rust
let elapsed_secs = self.start_time.elapsed().as_secs();
if elapsed_secs >= self.last_strategy_refresh_time + self.strategy_refresh_interval_secs {
    if let Some(ref callback) = self.on_strategy_refresh {
        for (i, &node_idx) in self.scenario_node_indices.iter().enumerate() {
            callback(i, node_idx, &self.storage);
        }
    }
    self.last_strategy_refresh_time = elapsed_secs;
}
```

**Step 2: Wire callback in main.rs**

```rust
let metrics_for_refresh = Arc::clone(&metrics);
let tree_ref = &trainer.tree; // need to capture action labels
let bucket_counts = trainer.storage.bucket_counts;

// Store scenario node indices
trainer.scenario_node_indices = scenarios.iter().map(|s| s.node_idx).collect();
trainer.strategy_refresh_interval_secs = tui_config.telemetry.strategy_delta_interval_seconds;

trainer.on_strategy_refresh = Some(Box::new(move |scenario_idx, node_idx, storage| {
    let probs = storage.average_strategy(node_idx, 0); // bucket 0 for now
    metrics_for_refresh.update_scenario_strategy(scenario_idx, probs);
}));
```

**Step 3: Test**

Run: `cargo test -p poker-solver-trainer && cargo clippy -p poker-solver-trainer`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat(tui): periodic strategy refresh from trainer to TUI

Trainer pushes strategy snapshots for each scenario at configurable
intervals. TUI metrics compute strategy deltas automatically."
```

---

### Task 8: Integration Test

**Files:**
- Create: `crates/trainer/tests/blueprint_tui_integration.rs`

**Step 1: Write the test**

```rust
//! Integration test: verify TUI metrics are populated during training.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;

#[test]
fn tui_metrics_populated_during_training() {
    let config = BlueprintV2Config {
        game: GameConfig {
            players: 2,
            stack_depth: 10.0,
            small_blind: 0.5,
            big_blind: 1.0,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: StreetClusterConfig { buckets: 10 },
            flop: StreetClusterConfig { buckets: 10 },
            turn: StreetClusterConfig { buckets: 10 },
            river: StreetClusterConfig { buckets: 10 },
            seed: 42,
            kmeans_iterations: 50,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![vec!["2.5bb".into()]],
            flop: vec![vec![1.0]],
            turn: vec![vec![1.0]],
            river: vec![vec![1.0]],
            max_raises: 1,
        },
        training: TrainingConfig {
            cluster_path: "clusters/".into(),
            iterations: Some(100),
            time_limit_minutes: None,
            lcfr_warmup_minutes: 9999,
            lcfr_discount_interval: 1,
            prune_after_minutes: 9999,
            prune_threshold: -310_000_000,
            prune_explore_pct: 0.05,
            print_every_minutes: 9999,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 9999,
            snapshot_every_minutes: 9999,
            output_dir: "/tmp/test_tui_integration".into(),
        },
    };

    let mut trainer = BlueprintTrainer::new(config);

    // Verify shared_iterations is wired
    assert_eq!(trainer.shared_iterations.load(Ordering::Relaxed), 0);

    trainer.train().expect("training should complete");

    assert_eq!(trainer.shared_iterations.load(Ordering::Relaxed), 100);
    assert_eq!(trainer.iterations, 100);
}
```

**Step 2: Run test**

Run: `cargo test -p poker-solver-trainer tui_metrics_populated -- --nocapture`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/trainer/tests/blueprint_tui_integration.rs
git commit -m "test(tui): integration test for shared metrics during training"
```

---

### Task 9: Convergence Overlay + Color Intensity

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_widgets.rs`
- Test: extend existing tests

**Step 1: Write the failing test**

```rust
#[timed_test(10)]
fn convergence_border_shown_for_stable_cell() {
    // A cell where the previous and current strategy differ by < 0.01
    let cell = CellStrategy {
        actions: vec![("fold".into(), 0.50), ("call".into(), 0.50)],
    };
    let prev = CellStrategy {
        actions: vec![("fold".into(), 0.505), ("call".into(), 0.495)],
    };
    assert!(cell.is_converged(&prev, 0.01));
}

#[timed_test(10)]
fn convergence_border_hidden_for_moving_cell() {
    let cell = CellStrategy {
        actions: vec![("fold".into(), 0.50), ("call".into(), 0.50)],
    };
    let prev = CellStrategy {
        actions: vec![("fold".into(), 0.60), ("call".into(), 0.40)],
    };
    assert!(!cell.is_converged(&prev, 0.01));
}
```

**Step 2: Implement**

Add to `CellStrategy`:

```rust
pub fn is_converged(&self, previous: &CellStrategy, threshold: f32) -> bool {
    if self.actions.len() != previous.actions.len() {
        return false;
    }
    let delta: f32 = self.actions.iter()
        .zip(previous.actions.iter())
        .map(|((_, a), (_, b))| (a - b).abs())
        .sum::<f32>() / self.actions.len().max(1) as f32;
    delta < threshold
}
```

Update the `HandGridWidget::render` to add a bright border (e.g. `Color::White` underline style) on converged cells when a `prev_cells` field is available in `HandGridState`.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-trainer blueprint_tui_widgets -- --nocapture`
Expected: all PASS

**Step 4: Commit**

```bash
git add crates/trainer/src/blueprint_tui_widgets.rs
git commit -m "feat(tui): convergence overlay for stabilized cells

Cells where strategy delta < threshold get a bright border in the
hand grid, providing visual feedback on which hands converge first."
```

---

### Task 10: Sample Config + Documentation

**Files:**
- Modify: `sample_configurations/` — add TUI section to an existing config or create a new one
- Modify: `docs/training.md` — document TUI feature
- Modify: `docs/architecture.md` — mention TUI module

**Step 1: Add sample config**

Create `sample_configurations/blueprint_v2_with_tui.yaml` by copying a base Blueprint V2 config and adding the `tui:` section from the design doc.

**Step 2: Update docs/training.md**

Add a section under the Blueprint V2 training commands:

```markdown
### Training TUI Dashboard

When `tui.enabled: true` in the config, training launches a full-screen terminal dashboard.

**Left panel**: iteration progress, throughput sparkline, exploitability chart
**Right panel**: tabbed 13x13 strategy grids for configured scenarios

Hotkeys: `p` pause/resume, `s` snapshot, `e` exploitability, `←/→` switch tabs, `q` quit

Use `--no-tui` to disable and use text output instead.
```

**Step 3: Commit**

```bash
git add sample_configurations/blueprint_v2_with_tui.yaml docs/training.md docs/architecture.md
git commit -m "docs: Blueprint TUI dashboard usage and sample config"
```

---

## Task Dependency Graph

```
T1 (config) ─────┐
T2 (metrics) ─────┤
T3 (widgets) ─────┼── T5 (main app) ── T6 (wire to CLI) ── T7 (refresh loop) ── T8 (integration)
T4 (scenarios) ───┘                                                                     │
                                                                                   T9 (convergence)
                                                                                         │
                                                                                   T10 (docs)
```

Tasks 1-4 are independent and can be parallelized. Task 5 depends on all of 1-4. Tasks 6-10 are sequential.
