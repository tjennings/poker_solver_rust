# N-Player TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Adapt the blueprint trainer TUI for N-player support with position-aware spot encoding, dynamic position labels, and a 6-up tiled strategy grid layout with pagination.

**Architecture:** Add `position_label()`/`parse_position()` to `blueprint_mp/types.rs` (domain layer), create a new `mp_tui_widgets.rs` with a compact grid widget using half-block rendering (adapter layer), create `mp_tui.rs` as the N-player TUI app with tiled grid layout and pagination (coordination layer). Wire to the `train-blueprint-mp` CLI path. Existing `blueprint_v2` TUI untouched.

**Tech Stack:** Ratatui + Crossterm (already in Cargo.toml), Unicode half-block rendering, existing `CellStrategy`/`HandGridState` types reused.

---

## Task 1: Position Labels (Domain)

**Files:**
- Modify: `crates/core/src/blueprint_mp/types.rs`

**Step 1: Write failing tests**

```rust
// Add to existing tests module in types.rs

#[timed_test]
fn position_label_2_players() {
    assert_eq!(position_label(Seat::from_raw(0), 2), "btn");
    assert_eq!(position_label(Seat::from_raw(1), 2), "bb");
}

#[timed_test]
fn position_label_6_players() {
    assert_eq!(position_label(Seat::from_raw(0), 6), "utg");
    assert_eq!(position_label(Seat::from_raw(1), 6), "hj");
    assert_eq!(position_label(Seat::from_raw(2), 6), "co");
    assert_eq!(position_label(Seat::from_raw(3), 6), "btn");
    assert_eq!(position_label(Seat::from_raw(4), 6), "sb");
    assert_eq!(position_label(Seat::from_raw(5), 6), "bb");
}

#[timed_test]
fn position_label_3_players() {
    assert_eq!(position_label(Seat::from_raw(0), 3), "btn");
    assert_eq!(position_label(Seat::from_raw(1), 3), "sb");
    assert_eq!(position_label(Seat::from_raw(2), 3), "bb");
}

#[timed_test]
fn position_label_8_players() {
    assert_eq!(position_label(Seat::from_raw(0), 8), "utg");
    assert_eq!(position_label(Seat::from_raw(7), 8), "bb");
}

#[timed_test]
fn parse_position_round_trips() {
    for n in 2..=8u8 {
        for s in 0..n {
            let label = position_label(Seat::from_raw(s), n);
            let parsed = parse_position(label, n);
            assert_eq!(parsed, Some(Seat::from_raw(s)),
                "round-trip failed for seat {s} in {n}-player: label={label}");
        }
    }
}

#[timed_test]
fn parse_position_invalid() {
    assert_eq!(parse_position("xyz", 6), None);
    assert_eq!(parse_position("utg", 2), None); // utg doesn't exist in 2-player
}
```

**Step 2: Implement**

```rust
/// Position label tables indexed by seat for each player count.
const POS_2: &[&str] = &["btn", "bb"];
const POS_3: &[&str] = &["btn", "sb", "bb"];
const POS_4: &[&str] = &["co", "btn", "sb", "bb"];
const POS_5: &[&str] = &["hj", "co", "btn", "sb", "bb"];
const POS_6: &[&str] = &["utg", "hj", "co", "btn", "sb", "bb"];
const POS_7: &[&str] = &["utg", "mp", "hj", "co", "btn", "sb", "bb"];
const POS_8: &[&str] = &["utg", "utg1", "mp", "hj", "co", "btn", "sb", "bb"];

fn pos_table(num_players: u8) -> &'static [&'static str] {
    match num_players {
        2 => POS_2, 3 => POS_3, 4 => POS_4, 5 => POS_5,
        6 => POS_6, 7 => POS_7, 8 => POS_8,
        _ => panic!("unsupported player count: {num_players}"),
    }
}

/// Map a seat index to its position label (e.g., "utg", "btn", "bb").
#[must_use]
pub fn position_label(seat: Seat, num_players: u8) -> &'static str {
    pos_table(num_players)[seat.index() as usize]
}

/// Parse a position label back to a Seat. Returns None if the label
/// is not valid for the given player count.
#[must_use]
pub fn parse_position(label: &str, num_players: u8) -> Option<Seat> {
    let lower = label.to_ascii_lowercase();
    pos_table(num_players)
        .iter()
        .position(|&p| p == lower)
        .map(|i| Seat::from_raw(i as u8))
}
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core blueprint_mp::types::tests::position_label`
Expected: All PASS.

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_mp/types.rs
git commit -m "feat(blueprint_mp): add position_label and parse_position for N-player seat names"
```

---

## Task 2: Position-Aware Spot Resolver

**Files:**
- Create: `crates/trainer/src/mp_tui_scenarios.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod mp_tui_scenarios;`)

This is a clean-room spot resolver for `blueprint_mp` game trees. It mirrors `blueprint_tui_scenarios.rs` but uses position labels and `MpGameTree`/`MpGameNode` types.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn resolve_empty_spot_returns_root() {
        let (game_cfg, action_cfg) = test_config_6p();
        let tree = MpGameTree::build(&game_cfg, &action_cfg);
        let result = resolve_mp_spot(&tree, "", 6);
        assert!(result.is_some());
        let (node_idx, board) = result.unwrap();
        assert_eq!(node_idx, tree.root);
        assert!(board.is_empty());
    }

    #[timed_test]
    fn resolve_single_action() {
        let (game_cfg, action_cfg) = test_config_6p();
        let tree = MpGameTree::build(&game_cfg, &action_cfg);
        // UTG opens — should advance past UTG's decision
        let result = resolve_mp_spot(&tree, "utg:5bb", 6);
        assert!(result.is_some());
        let (node_idx, _) = result.unwrap();
        // Should be at HJ's decision node
        assert!(matches!(tree.nodes[node_idx as usize], MpGameNode::Decision { .. }));
    }

    #[timed_test]
    fn resolve_invalid_position_returns_none() {
        let (game_cfg, action_cfg) = test_config_6p();
        let tree = MpGameTree::build(&game_cfg, &action_cfg);
        assert!(resolve_mp_spot(&tree, "xyz:5bb", 6).is_none());
    }

    #[timed_test]
    fn resolve_fold_sequence() {
        let (game_cfg, action_cfg) = test_config_6p();
        let tree = MpGameTree::build(&game_cfg, &action_cfg);
        // Everyone folds to BB
        let result = resolve_mp_spot(&tree, "utg:fold,hj:fold,co:fold,btn:fold,sb:fold", 6);
        assert!(result.is_some());
    }
}
```

**Step 2: Implement `resolve_mp_spot`**

```rust
use poker_solver_core::blueprint_mp::game_tree::{MpGameNode, MpGameTree, TreeAction};
use poker_solver_core::blueprint_mp::types::parse_position;
use poker_solver_core::poker::{self, Card};

/// Walk the MP game tree following a spot notation string.
///
/// Format: segments separated by `|`.
/// - Action segments: `"position:label,position:label"` (e.g., `"utg:5bb,hj:fold"`)
/// - Board segments: pairs of rank+suit chars (e.g., `"Kh7s2d"`)
pub fn resolve_mp_spot(tree: &MpGameTree, spot: &str, num_players: u8) -> Option<(u32, Vec<Card>)> {
    let spot = spot.trim();
    if spot.is_empty() {
        return Some((tree.root, vec![]));
    }
    let mut node_idx = tree.root;
    let mut board = Vec::new();
    for segment in spot.split('|') {
        let segment = segment.trim();
        if segment.is_empty() { continue; }
        if segment.contains(':') {
            node_idx = resolve_action_segment(tree, node_idx, segment, num_players)?;
        } else {
            parse_board_segment(segment, &mut board)?;
        }
    }
    // Skip any trailing chance node to land on a decision
    node_idx = skip_chance_mp(tree, node_idx);
    Some((node_idx, board))
}
```

Break into helpers: `resolve_action_segment`, `match_action_label_mp`, `parse_board_segment`, `skip_chance_mp`. Each under 20 lines.

The action label matching logic: `"fold"` → `TreeAction::Fold`, `"check"` → `Check`, `"call"` → `Call`, `"all-in"` → `AllIn`, `"Nbb"` → match `Lead`/`Raise` by BB amount (chips/2), `"Nx"` → match multiplier raises.

**Step 3: Run tests and commit**

```bash
git add crates/trainer/src/mp_tui_scenarios.rs crates/trainer/src/main.rs
git commit -m "feat(trainer): add position-aware spot resolver for blueprint_mp TUI"
```

---

## Task 3: Compact Grid Widget (half-block rendering)

**Files:**
- Create: `crates/trainer/src/mp_tui_widgets.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod mp_tui_widgets;`)

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_tui_widgets::CellStrategy;
    use test_macros::timed_test;

    #[timed_test]
    fn compact_grid_dimensions() {
        assert_eq!(COMPACT_CELL_W, 4);
        assert_eq!(compact_grid_width(), 4 * 13 + 2); // 13 cells + rank label + padding
        assert_eq!(compact_grid_height(), 13 + 2);     // 13 rows + header + title
    }

    #[timed_test]
    fn action_to_half_blocks_fold_only() {
        let cell = CellStrategy {
            actions: vec![("fold".into(), 1.0)],
            ev: None,
        };
        let blocks = cell_to_half_blocks(&cell, 8);
        // All 8 slots should be fold color
        assert_eq!(blocks.len(), 8);
        assert!(blocks.iter().all(|&c| c == action_color_ranked("fold", 0, 1)));
    }

    #[timed_test]
    fn action_to_half_blocks_mixed() {
        let cell = CellStrategy {
            actions: vec![("fold".into(), 0.5), ("call".into(), 0.5)],
            ev: None,
        };
        let blocks = cell_to_half_blocks(&cell, 8);
        // 4 fold + 4 call
        assert_eq!(blocks.len(), 8);
    }

    #[timed_test]
    fn grids_per_row_calculation() {
        assert_eq!(compute_grids_per_row(180, compact_grid_width()), 3);
        assert_eq!(compute_grids_per_row(120, compact_grid_width()), 2);
        assert_eq!(compute_grids_per_row(60, compact_grid_width()), 1);
    }
}
```

**Step 2: Implement compact grid widget**

The key rendering technique from robopoker: each cell is 4 chars wide. With Unicode half-block (`▐`), each character position shows TWO colors (foreground = right half, background = left half). So 4 chars = 8 color slots per cell.

```rust
use ratatui::prelude::*;
use ratatui::widgets::Widget;
use crate::blueprint_tui_widgets::{CellStrategy, HandGridState, action_color_ranked};

pub const COMPACT_CELL_W: u16 = 4;

/// Total width of one compact grid (rank label + 13 cells + border).
pub fn compact_grid_width() -> u16 {
    2 + 13 * COMPACT_CELL_W  // 2 for rank label column + padding
}

/// Total height of one compact grid (title + header + 13 data rows).
pub fn compact_grid_height() -> u16 {
    15  // 1 title + 1 column header + 13 rows
}

/// How many grids fit side by side in the given terminal width.
pub fn compute_grids_per_row(terminal_width: u16, grid_width: u16) -> u16 {
    (terminal_width / grid_width).max(1)
}

/// Convert a cell's action distribution into `slots` color values.
pub fn cell_to_half_blocks(cell: &CellStrategy, slots: usize) -> Vec<Color> { ... }

/// Compact 13x13 grid widget using half-block rendering.
pub struct CompactGridWidget<'a> {
    pub state: &'a HandGridState,
}
```

The `Widget` impl renders:
1. Title row: scenario name (bold)
2. Column header: A K Q J T 9 8 7 6 5 4 3 2
3. 13 data rows: rank label + 13 cells using half-block pairs

Each cell renders as 4 characters. For each pair of adjacent color slots, emit one `▐` character with `fg = right_color, bg = left_color`. This gives 8 color resolution in 4 chars.

Reuse `action_color_ranked` from `blueprint_tui_widgets.rs` for color mapping.

**Step 3: Run tests and commit**

```bash
git add crates/trainer/src/mp_tui_widgets.rs crates/trainer/src/main.rs
git commit -m "feat(trainer): add compact half-block 13x13 grid widget for N-player TUI"
```

---

## Task 4: Tiled Grid Layout with Pagination

**Files:**
- Create: `crates/trainer/src/mp_tui.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod mp_tui;`)

This is the N-player TUI app. It reuses the metrics panel from `blueprint_tui.rs` but replaces the tabbed single-grid view with a tiled multi-grid layout.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn page_count_6_scenarios() {
        assert_eq!(page_count(6, 6), 1);
    }

    #[timed_test]
    fn page_count_12_scenarios() {
        assert_eq!(page_count(12, 6), 2);
    }

    #[timed_test]
    fn page_count_7_scenarios() {
        assert_eq!(page_count(7, 6), 2);
    }

    #[timed_test]
    fn page_scenarios_first_page() {
        let scenarios: Vec<String> = (0..12).map(|i| format!("s{i}")).collect();
        let page = page_scenarios(&scenarios, 0, 6);
        assert_eq!(page.len(), 6);
        assert_eq!(page[0], "s0");
        assert_eq!(page[5], "s5");
    }

    #[timed_test]
    fn page_scenarios_second_page() {
        let scenarios: Vec<String> = (0..12).map(|i| format!("s{i}")).collect();
        let page = page_scenarios(&scenarios, 1, 6);
        assert_eq!(page.len(), 6);
        assert_eq!(page[0], "s6");
    }

    #[timed_test]
    fn page_scenarios_partial_last_page() {
        let scenarios: Vec<String> = (0..8).map(|i| format!("s{i}")).collect();
        let page = page_scenarios(&scenarios, 1, 6);
        assert_eq!(page.len(), 2);
    }

    #[timed_test]
    fn next_page_wraps() {
        let mut app = MpTuiApp::test_with_scenarios(12);
        assert_eq!(app.current_page, 0);
        app.next_page();
        assert_eq!(app.current_page, 1);
        app.next_page();
        assert_eq!(app.current_page, 0); // wraps
    }

    #[timed_test]
    fn prev_page_wraps() {
        let mut app = MpTuiApp::test_with_scenarios(12);
        app.prev_page();
        assert_eq!(app.current_page, 1); // wraps backward
    }
}
```

**Step 2: Implement the N-player TUI app**

```rust
pub const GRIDS_PER_PAGE: usize = 6;

pub struct MpTuiApp {
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedMpScenario>,
    current_page: usize,
    num_players: u8,
    // ... sparkline histories (same as BlueprintTuiApp)
}

pub struct ResolvedMpScenario {
    pub name: String,
    pub node_idx: u32,
    pub grid: HandGridState,
}
```

**Rendering:**
- `render()` — same 3-part vertical layout (metrics, grids, hotkeys)
- `render_grid_page()` — compute which scenarios are on current page, tile them:
  ```rust
  fn render_grid_page(&self, frame: &mut Frame, area: Rect) {
      let page_scenarios = self.current_page_scenarios();
      let cols = compute_grids_per_row(area.width, compact_grid_width());
      let rows = (page_scenarios.len() as u16 + cols - 1) / cols;
      // Layout: rows of cols grids each
      for (i, scenario) in page_scenarios.iter().enumerate() {
          let grid_col = i as u16 % cols;
          let grid_row = i as u16 / cols;
          let x = area.x + grid_col * compact_grid_width();
          let y = area.y + grid_row * compact_grid_height();
          let rect = Rect::new(x, y, compact_grid_width(), compact_grid_height());
          let widget = CompactGridWidget { state: &scenario.grid };
          frame.render_widget(&widget, rect);
      }
  }
  ```

**Navigation:** Left/Right arrows call `prev_page()`/`next_page()`.

**Hotkey bar:** `[Page 1/N]  ← → pages  q quit  p pause  s snapshot`

**Step 3: Run tests and commit**

```bash
git add crates/trainer/src/mp_tui.rs crates/trainer/src/main.rs
git commit -m "feat(trainer): add N-player TUI with tiled 6-up grid layout and pagination"
```

---

## Task 5: Wire TUI to train-blueprint-mp

**Files:**
- Modify: `crates/trainer/src/main.rs` — update `run_train_blueprint_mp` to launch the TUI
- Modify: `crates/core/src/blueprint_mp/trainer.rs` — add TUI metrics integration

**Step 1: Add TUI config to BlueprintMpConfig**

Add an optional `tui` field to `BlueprintMpConfig`:
```rust
// In config.rs
pub struct BlueprintMpConfig {
    pub game: MpGameConfig,
    pub action_abstraction: MpActionAbstractionConfig,
    pub clustering: MpClusteringConfig,
    pub training: MpTrainingConfig,
    pub snapshots: MpSnapshotConfig,
    #[serde(default)]
    pub tui: Option<BlueprintTuiConfig>,  // reuse existing config type
}
```

**Step 2: Update trainer to push metrics**

The `train_blueprint_mp` function needs to accept an optional `Arc<BlueprintTuiMetrics>` and push iteration counts, strategy grids, etc. during training. This mirrors the pattern in `blueprint_v2/trainer.rs`.

**Step 3: Update CLI to parse TUI config, resolve scenarios, launch TUI thread**

```rust
fn run_train_blueprint_mp(path: &str) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(path)?;
    let config: BlueprintMpConfig = serde_yaml::from_str(&yaml)?;
    config.game.validate()?;

    let tui_config = parse_tui_config(&yaml);
    if tui_config.enabled {
        let tree = MpGameTree::build(&config.game, &config.action_abstraction);
        let scenarios = resolve_mp_scenarios(&tree, &tui_config.scenarios, config.game.num_players);
        let metrics = Arc::new(BlueprintTuiMetrics::new(...));
        let tui_handle = mp_tui::run_mp_tui(metrics.clone(), scenarios, ...);
        train_blueprint_mp_with_metrics(&config, Some(metrics));
        tui_handle.join().ok();
    } else {
        train_blueprint_mp(&config);
    }
    Ok(())
}
```

**Step 4: Test with sample config**

Update `sample_configurations/blueprint_mp_6player_ante.yaml` to include a TUI section:

```yaml
tui:
  enabled: true
  scenarios:
    - name: "UTG open"
      spot: ""
    - name: "HJ vs UTG"
      spot: "utg:5bb"
    - name: "CO vs UTG"
      spot: "utg:5bb,hj:fold"
    - name: "BTN vs UTG"
      spot: "utg:5bb,hj:fold,co:fold"
    - name: "SB vs UTG"
      spot: "utg:5bb,hj:fold,co:fold,btn:fold"
    - name: "BB vs UTG"
      spot: "utg:5bb,hj:fold,co:fold,btn:fold,sb:fold"
```

**Step 5: Verify**

```bash
cargo build -p poker-solver-trainer --release
```

**Step 6: Commit**

```bash
git add crates/trainer/src/main.rs crates/core/src/blueprint_mp/trainer.rs \
       crates/core/src/blueprint_mp/config.rs \
       sample_configurations/blueprint_mp_6player_ante.yaml
git commit -m "feat(trainer): wire N-player TUI to train-blueprint-mp with metrics integration"
```

---

## Task 6: Grid Refresh & Strategy Extraction

**Files:**
- Modify: `crates/trainer/src/mp_tui.rs` — add `tick()` method for live updates
- Modify: `crates/trainer/src/mp_tui_scenarios.rs` — add `extract_mp_grid()` function

**Step 1: Implement grid extraction**

```rust
/// Extract a 13x13 strategy grid from the MP storage at a given decision node.
pub fn extract_mp_grid(
    tree: &MpGameTree,
    storage: &MpStorage,
    node_idx: u32,
    iteration: u64,
) -> HandGridState {
    // For each of 169 canonical hands:
    //   1. Map hand to bucket (trivial: hand_index % bucket_count)
    //   2. Get average strategy from storage
    //   3. Map action indices to action labels
    //   4. Build CellStrategy { actions, ev: None }
}
```

**Step 2: Implement `tick()` on `MpTuiApp`**

Poll metrics, update sparklines, refresh grids from shared state. Same pattern as `BlueprintTuiApp::tick()`.

**Step 3: Test and commit**

```bash
git commit -m "feat(trainer): add grid refresh and strategy extraction for N-player TUI"
```

---

## Task 7: Documentation & Sample Configs

**Files:**
- Modify: `docs/training.md` — add TUI config section for N-player
- Update: `sample_configurations/blueprint_mp_6player_ante.yaml` — add TUI scenarios for 6-max opening ranges and 3-bets

**Step 1: Add comprehensive 6-max sample with 12 scenarios (2 pages)**

```yaml
tui:
  enabled: true
  refresh_rate_ms: 250
  scenarios:
    # Page 1: Opening ranges
    - name: "UTG open"
      spot: ""
    - name: "HJ open"
      spot: "utg:fold"
    - name: "CO open"
      spot: "utg:fold,hj:fold"
    - name: "BTN open"
      spot: "utg:fold,hj:fold,co:fold"
    - name: "SB open"
      spot: "utg:fold,hj:fold,co:fold,btn:fold"
    - name: "BB vs SB limp"
      spot: "utg:fold,hj:fold,co:fold,btn:fold,sb:call"
    # Page 2: 3-bet ranges
    - name: "HJ vs UTG"
      spot: "utg:5bb"
    - name: "CO vs UTG"
      spot: "utg:5bb,hj:fold"
    - name: "BTN vs UTG"
      spot: "utg:5bb,hj:fold,co:fold"
    - name: "SB vs UTG"
      spot: "utg:5bb,hj:fold,co:fold,btn:fold"
    - name: "BB vs UTG"
      spot: "utg:5bb,hj:fold,co:fold,btn:fold,sb:fold"
    - name: "BTN vs CO"
      spot: "utg:fold,hj:fold,co:5bb"
```

**Step 2: Update training.md**

Add a section under the existing `train-blueprint-mp` docs explaining TUI configuration for N-player, including the position label system and sample scenario configs.

**Step 3: Commit**

```bash
git commit -m "docs: add N-player TUI config documentation and 12-scenario 6-max sample"
```
