# TUI Spot Notation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the TUI scenario config format (`player`/`actions`/`board`/`street` fields) with Tauri's spot notation (`"sb:4bb,bb:call|Kh7s2d"`).

**Architecture:** Pure refactor within the trainer crate. The config layer (`ScenarioConfig`) is simplified to `{name, spot}`. The scenario resolution layer (`blueprint_tui_scenarios.rs`) gets a new `resolve_spot()` function that parses spot strings and walks the game tree using BB-based action labels. The old `resolve_action_path`/`match_action` functions are removed.

**Tech Stack:** Rust, serde/serde_yaml for config parsing, existing `GameTree`/`TreeAction` types from `poker_solver_core`.

---

### Task 1: Add `format_tree_action_bb` and `match_action_by_label`

These are the two new domain functions. `format_tree_action_bb` converts a `TreeAction` to its BB-based label (matching Tauri's `game_session::format_tree_action`). `match_action_by_label` finds an action at a node by case-insensitive label match.

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_scenarios.rs`

**Step 1: Write the failing tests**

Add these tests to the existing `mod tests` block in `blueprint_tui_scenarios.rs`, after the `extract_grid_returns_169_cells` test:

```rust
#[timed_test(10)]
fn format_tree_action_bb_labels() {
    assert_eq!(format_tree_action_bb(&TreeAction::Fold), "fold");
    assert_eq!(format_tree_action_bb(&TreeAction::Check), "check");
    assert_eq!(format_tree_action_bb(&TreeAction::Call), "call");
    assert_eq!(format_tree_action_bb(&TreeAction::AllIn), "all-in");
    // 10.0 chips / 2 = 5bb
    assert_eq!(format_tree_action_bb(&TreeAction::Bet(10.0)), "5bb");
    assert_eq!(format_tree_action_bb(&TreeAction::Raise(10.0)), "5bb");
    // Fractional: 5.0 chips / 2 = 2.5bb → rounds to 3bb
    assert_eq!(format_tree_action_bb(&TreeAction::Raise(5.0)), "3bb");
}

#[timed_test(10)]
fn match_action_by_label_finds_actions() {
    let actions = vec![
        TreeAction::Fold,
        TreeAction::Call,
        TreeAction::Raise(10.0), // 5bb
    ];
    assert_eq!(match_action_by_label("fold", &actions), Some(0));
    assert_eq!(match_action_by_label("call", &actions), Some(1));
    assert_eq!(match_action_by_label("5bb", &actions), Some(2));
    assert_eq!(match_action_by_label("Call", &actions), Some(1)); // case-insensitive
    assert_eq!(match_action_by_label("999bb", &actions), None);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-trainer match_action_by_label_finds_actions -- --nocapture 2>&1 | tail -5`
Expected: FAIL — `format_tree_action_bb` and `match_action_by_label` not found.

**Step 3: Implement `format_tree_action_bb` and `match_action_by_label`**

Add these two functions in `blueprint_tui_scenarios.rs`, right after the existing `format_tree_action` function (after line 54):

```rust
/// Format a `TreeAction` as a BB-based label matching Tauri's spot notation.
///
/// Chips are converted to big blinds by dividing by 2 and rounding.
/// Examples: `Raise(10.0)` → `"5bb"`, `Fold` → `"fold"`.
pub fn format_tree_action_bb(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "fold".to_string(),
        TreeAction::Check => "check".to_string(),
        TreeAction::Call => "call".to_string(),
        TreeAction::AllIn => "all-in".to_string(),
        TreeAction::Bet(chips) | TreeAction::Raise(chips) => {
            let bb = (chips / 2.0).round() as u64;
            format!("{bb}bb")
        }
    }
}

/// Find the position of an action in `node_actions` by BB-based label.
///
/// Case-insensitive. Returns `None` if no action matches.
fn match_action_by_label(label: &str, node_actions: &[TreeAction]) -> Option<usize> {
    let lower = label.to_ascii_lowercase();
    node_actions
        .iter()
        .position(|a| format_tree_action_bb(a) == lower)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer format_tree_action_bb -- --nocapture 2>&1 | tail -5`
Run: `cargo test -p poker-solver-trainer match_action_by_label -- --nocapture 2>&1 | tail -5`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_scenarios.rs
git commit -m "feat(tui): add format_tree_action_bb and match_action_by_label for spot notation"
```

---

### Task 2: Add `resolve_spot` function

This is the core parsing function that replaces `resolve_action_path`. It parses a spot string like `"sb:5bb,bb:call|Kh7s2d"` and walks the game tree.

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_scenarios.rs`

**Step 1: Write the failing tests**

Add to `mod tests`:

```rust
#[timed_test(10)]
fn resolve_spot_empty_string() {
    let tree = toy_tree();
    let (node_idx, board) = resolve_spot(&tree, "").unwrap();
    assert_eq!(node_idx, tree.root);
    assert!(board.is_empty());
}

#[timed_test(10)]
fn resolve_spot_single_action() {
    // toy_tree: SB opens with Raise(10.0) = 5bb
    let tree = toy_tree();
    let (node_idx, board) = resolve_spot(&tree, "sb:5bb").unwrap();
    // Should land on BB's decision node after SB raises
    assert_ne!(node_idx, tree.root);
    assert!(board.is_empty());
}

#[timed_test(10)]
fn resolve_spot_two_actions() {
    let tree = toy_tree();
    let result = resolve_spot(&tree, "sb:5bb,bb:call");
    assert!(result.is_some(), "sb:5bb,bb:call should resolve");
    let (_, board) = result.unwrap();
    assert!(board.is_empty());
}

#[timed_test(10)]
fn resolve_spot_with_board() {
    let tree = toy_tree();
    let result = resolve_spot(&tree, "sb:5bb,bb:call|Kh7s2d");
    assert!(result.is_some());
    let (_, board) = result.unwrap();
    assert_eq!(board.len(), 3);
}

#[timed_test(10)]
fn resolve_spot_invalid_label() {
    let tree = toy_tree();
    let result = resolve_spot(&tree, "sb:999bb");
    assert!(result.is_none());
}

#[timed_test(10)]
fn resolve_spot_position_ignored() {
    // Position labels (sb:/bb:) are stripped but not validated against
    // who is actually acting — the tree walk determines that.
    let tree = toy_tree();
    let result = resolve_spot(&tree, "sb:5bb");
    assert!(result.is_some());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-trainer resolve_spot -- --nocapture 2>&1 | tail -10`
Expected: FAIL — `resolve_spot` not found.

**Step 3: Implement `resolve_spot`**

Add this function in `blueprint_tui_scenarios.rs`, after `match_action_by_label`:

```rust
/// Parse a spot notation string and walk the game tree.
///
/// Format: `"position:label,position:label|BoardCards|position:label..."`
///
/// Segments separated by `|`. Segments containing `:` are action segments
/// (comma-separated `position:label` pairs). Segments without `:` are board
/// cards (pairs of rank+suit, e.g. `"Kh7s2d"`).
///
/// Returns `(node_idx, board_cards)` or `None` if any action label is invalid.
pub fn resolve_spot(tree: &GameTree, spot: &str) -> Option<(u32, Vec<Card>)> {
    let spot = spot.trim();
    if spot.is_empty() {
        return Some((tree.root, vec![]));
    }

    let mut node_idx = tree.root;
    let mut board = Vec::new();

    for segment in spot.split('|') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }

        if segment.contains(':') {
            // Action segment: "sb:5bb,bb:call"
            for action_str in segment.split(',') {
                let action_str = action_str.trim();
                let label = action_str
                    .split_once(':')
                    .map(|(_, l)| l)
                    .unwrap_or(action_str);

                // Skip through chance nodes before each action.
                node_idx = skip_chance(tree, node_idx);

                let GameNode::Decision {
                    actions: ref node_actions,
                    ref children,
                    ..
                } = tree.nodes[node_idx as usize]
                else {
                    return None;
                };

                let matched = match_action_by_label(label, node_actions)?;
                node_idx = children[matched];
            }
        } else {
            // Board segment: "Kh7s2d"
            let chars: Vec<char> = segment.chars().collect();
            for chunk in chars.chunks(2) {
                if chunk.len() == 2 {
                    let card_str: String = chunk.iter().collect();
                    if let Some(card) = poker_solver_core::poker::parse_card(&card_str) {
                        board.push(card);
                    }
                }
            }
        }
    }

    // Skip trailing chance node so we land on a decision or terminal.
    node_idx = skip_chance(tree, node_idx);
    Some((node_idx, board))
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer resolve_spot -- --nocapture 2>&1 | tail -10`
Expected: All 6 new tests PASS.

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_scenarios.rs
git commit -m "feat(tui): add resolve_spot to parse Tauri-style spot notation"
```

---

### Task 3: Simplify `ScenarioConfig` to `{name, spot}`

Replace the old `player`/`actions`/`board`/`street` fields with a single `spot` string.

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_config.rs`

**Step 1: Update `ScenarioConfig` struct**

Replace the current `ScenarioConfig` (lines 21-33) with:

```rust
/// A single scenario to display live strategy evolution in the TUI.
#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioConfig {
    pub name: String,
    #[serde(default)]
    pub spot: String,
}
```

**Step 2: Update tests in `blueprint_tui_config.rs`**

Replace the `parse_complete_tui_config` test YAML and assertions to use the new format:

```rust
#[timed_test(10)]
fn parse_complete_tui_config() {
    let yaml = r#"
tui:
  enabled: true
  refresh_rate_ms: 100
  telemetry:
    strategy_delta_interval_seconds: 15
    sparkline_window: 120
  scenarios:
    - name: "SB open"
      spot: ""
    - name: "BB vs raise"
      spot: "sb:5bb"
  random_scenario:
    enabled: true
    hold_minutes: 5
    pool: [preflop, flop, turn, river]
"#;
    let cfg = parse_tui_config(yaml);
    assert!(cfg.enabled);
    assert_eq!(cfg.refresh_rate_ms, 100);
    assert_eq!(cfg.telemetry.strategy_delta_interval_seconds, 15);
    assert_eq!(cfg.telemetry.sparkline_window, 120);
    assert_eq!(cfg.scenarios.len(), 2);

    let s0 = &cfg.scenarios[0];
    assert_eq!(s0.name, "SB open");
    assert_eq!(s0.spot, "");

    let s1 = &cfg.scenarios[1];
    assert_eq!(s1.name, "BB vs raise");
    assert_eq!(s1.spot, "sb:5bb");

    assert!(cfg.random_scenario.enabled);
    assert_eq!(cfg.random_scenario.hold_minutes, 5);
    assert_eq!(cfg.random_scenario.pool.len(), 4);
    assert_eq!(cfg.random_scenario.pool[3], StreetLabel::River);
}
```

Update the `extracts_tui_from_full_config` test:

```rust
#[timed_test(10)]
fn extracts_tui_from_full_config() {
    let yaml = r#"
game:
  players: 2
  stack_depth: 200.0
  small_blind: 1
  big_blind: 2
training:
  cluster_path: "/tmp/clusters"
tui:
  enabled: true
  refresh_rate_ms: 500
  scenarios:
    - name: "Check spot"
      spot: ""
"#;
    let cfg = parse_tui_config(yaml);
    assert!(cfg.enabled);
    assert_eq!(cfg.refresh_rate_ms, 500);
    assert_eq!(cfg.scenarios.len(), 1);
    assert_eq!(cfg.scenarios[0].name, "Check spot");
    assert_eq!(cfg.scenarios[0].spot, "");
    // Telemetry and random_scenario should be defaults.
    assert_eq!(cfg.telemetry.sparkline_window, 60);
    assert!(!cfg.random_scenario.enabled);
}
```

**Step 3: Remove `PlayerLabel` enum if unused elsewhere**

`PlayerLabel` was only used in `ScenarioConfig`. Check if anything else references it. If not, remove the `PlayerLabel` enum (lines 4-9) entirely. Keep `StreetLabel` — it is still used by `RandomScenarioConfig`.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-trainer blueprint_tui_config -- --nocapture 2>&1 | tail -10`
Expected: All config tests PASS. There will be compile errors in `main.rs` — that's expected, we fix that in Task 4.

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_config.rs
git commit -m "refactor(tui): simplify ScenarioConfig to {name, spot}"
```

---

### Task 4: Update `main.rs` scenario initialization

Replace the old board-parsing + `resolve_action_path` code with `resolve_spot`.

**Files:**
- Modify: `crates/trainer/src/main.rs` (lines ~277-353)

**Step 1: Replace the scenario initialization block**

Replace the block from `// Pre-compute board cards for each scenario.` (line 277) through to the end of the `let scenarios: Vec<...> = ...collect();` (line 334) with:

```rust
                // Resolve scenarios using spot notation.
                let scenarios: Vec<blueprint_tui::ResolvedScenario> = tui_config
                    .scenarios
                    .iter()
                    .map(|sc| {
                        let (node_idx, board) = blueprint_tui_scenarios::resolve_spot(
                            &trainer.tree,
                            &sc.spot,
                        )
                        .unwrap_or_else(|| {
                            eprintln!("WARNING: scenario '{}' spot '{}' failed to resolve, using root", sc.name, sc.spot);
                            (trainer.tree.root, vec![])
                        });
                        let grid = blueprint_tui_scenarios::extract_strategy_grid(
                            &trainer.tree,
                            &trainer.storage,
                            node_idx,
                            &board,
                            None,
                        );
                        let street_label = match &trainer.tree.nodes[node_idx as usize] {
                            poker_solver_core::blueprint_v2::game_tree::GameNode::Decision { street, .. } => {
                                format!("{street:?}")
                            }
                            _ => "Preflop".to_string(),
                        };
                        blueprint_tui::ResolvedScenario {
                            name: sc.name.clone(),
                            node_idx,
                            grid: blueprint_tui_widgets::HandGridState {
                                cells: grid,
                                prev_cells: None,
                                scenario_name: sc.name.clone(),
                                action_path: vec![sc.spot.clone()],
                                board_display: if board.is_empty() {
                                    None
                                } else {
                                    Some(board.iter().map(|c| format!("{c}")).collect::<Vec<_>>().join(" "))
                                },
                                cluster_id: None,
                                street_label,
                                iteration_at_snapshot: 0,
                            },
                        }
                    })
                    .collect();
```

**Step 2: Update the `scenario_boards` in the refresh callback**

The `boards_for_refresh` vector is used in the `on_strategy_refresh` callback (lines ~344-353). We need to extract boards from scenarios for the callback. Replace the `let boards_for_refresh = scenario_boards;` line and the closure to extract boards from the resolved scenarios:

```rust
                let boards_for_refresh: Vec<Vec<poker_solver_core::poker::Card>> = tui_config
                    .scenarios
                    .iter()
                    .map(|sc| {
                        blueprint_tui_scenarios::resolve_spot(&trainer.tree, &sc.spot)
                            .map(|(_, b)| b)
                            .unwrap_or_default()
                    })
                    .collect();
```

**Step 3: Run the full test suite**

Run: `cargo test 2>&1 | tail -5`
Expected: All tests PASS (no compile errors).

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "refactor(tui): use resolve_spot for scenario initialization"
```

---

### Task 5: Remove old `resolve_action_path` and `match_action`

Now that nothing references the old functions, remove them.

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_scenarios.rs`

**Step 1: Remove dead code**

Remove these functions and their tests:
- `resolve_action_path` (lines 19-42)
- `match_action` (lines 271-316)
- Test `resolve_root_node` (lines 337-342)
- Test `resolve_raise_call` (lines 344-349)
- Test `resolve_invalid_returns_none` (lines 352-358)

**Step 2: Run tests**

Run: `cargo test -p poker-solver-trainer -- --nocapture 2>&1 | tail -10`
Expected: All remaining tests PASS.

**Step 3: Commit**

```bash
git add crates/trainer/src/blueprint_tui_scenarios.rs
git commit -m "refactor(tui): remove old resolve_action_path and match_action"
```

---

### Task 6: Update YAML config and verify end-to-end

Update the sample config to use spot notation.

**Files:**
- Modify: `sample_configurations/blueprint_v2_200bkt_sapcfr.yaml`

**Step 1: Update the scenarios section**

Replace the current `scenarios:` block (lines 71-85) with:

```yaml
  scenarios:
    - name: "SB Open"
      spot: ""
    - name: "BB vs 2x"
      spot: "sb:4bb"
    - name: "SB vs 3-bet"
      spot: "sb:4bb,bb:14bb"
    - name: "SB Cbet K72"
      spot: "sb:4bb,bb:call|Kh7s2d"
```

Note: The exact BB labels depend on the tree's action abstraction. The config has preflop sizes `["4bb"]` at depth 0 and `["8bb", "14bb"]` at depth 1. After the SB opens with 4bb (= 8 chips), the action is `Raise(8.0)`, which formats as `(8/2).round() = 4` → `"4bb"`. BB's 3-bet options at depth 1 are 8bb and 14bb (= 16 and 28 chips), formatting as `"8bb"` and `"14bb"`.

**Step 2: Run full test suite**

Run: `cargo test 2>&1 | tail -5`
Expected: All tests PASS.

**Step 3: Commit**

```bash
git add sample_configurations/blueprint_v2_200bkt_sapcfr.yaml
git commit -m "config: update TUI scenarios to spot notation"
```

---

### Task 7: Final cleanup and verification

**Step 1: Run clippy**

Run: `cargo clippy 2>&1 | tail -20`
Expected: No new warnings.

**Step 2: Run full test suite**

Run: `cargo test 2>&1 | tail -5`
Expected: All tests PASS.

**Step 3: Verify no dead imports**

Check that removed functions don't leave behind unused imports in any file.

**Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: cleanup after TUI spot notation migration"
```
