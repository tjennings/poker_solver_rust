# Regret Audit Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a TUI panel that shows per-hand, per-action regret values at configured spots, with bucket trail, deltas, and trend indicators for debugging MCCFR and bucketing.

**Architecture:** Polled-snapshot approach — TUI reads existing `AtomicI32` regret values from `BlueprintStorage` on each tick. No MCCFR hot-path changes. New YAML config section `regret_audits` parsed alongside existing `scenarios`. Horizontal split in the upper TUI area: sparklines left, audit panel right.

**Tech Stack:** Rust, ratatui, serde/serde_yaml, crossterm

**Design doc:** `docs/plans/2026-03-27-regret-audit-panel-design.md`

---

### Task 1: Config — `RegretAuditConfig` struct and YAML parsing

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_config.rs`

**Step 1: Write the failing test**

Add to the existing `tests` module in `blueprint_tui_config.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PlayerLabel {
    Sb,
    Bb,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegretAuditConfig {
    pub name: String,
    #[serde(default)]
    pub spot: String,
    pub hand: String,
    pub player: PlayerLabel,
}
```

Test:

```rust
#[timed_test(10)]
fn parse_regret_audits() {
    let yaml = r#"
tui:
  enabled: true
  regret_audits:
    - name: "AKo SB open"
      spot: ""
      hand: "AKo"
      player: SB
    - name: "T9s flop cbet"
      spot: "sb:2bb,bb:call|AsTd9d"
      hand: "Ts9s"
      player: SB
"#;
    let cfg = parse_tui_config(yaml);
    assert_eq!(cfg.regret_audits.len(), 2);
    assert_eq!(cfg.regret_audits[0].name, "AKo SB open");
    assert_eq!(cfg.regret_audits[0].hand, "AKo");
    assert_eq!(cfg.regret_audits[0].player, PlayerLabel::Sb);
    assert_eq!(cfg.regret_audits[1].spot, "sb:2bb,bb:call|AsTd9d");
    assert_eq!(cfg.regret_audits[1].hand, "Ts9s");
}

#[timed_test(10)]
fn regret_audits_default_empty() {
    let cfg = parse_tui_config("");
    assert!(cfg.regret_audits.is_empty());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer parse_regret_audits -- --nocapture`
Expected: FAIL — `RegretAuditConfig` not defined, `regret_audits` field not on `BlueprintTuiConfig`.

**Step 3: Write minimal implementation**

Add to `blueprint_tui_config.rs`:

1. Add the `PlayerLabel` enum (before `ScenarioConfig`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PlayerLabel {
    Sb,
    Bb,
}
```

2. Add the `RegretAuditConfig` struct (after `ScenarioConfig`):

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct RegretAuditConfig {
    pub name: String,
    #[serde(default)]
    pub spot: String,
    pub hand: String,
    pub player: PlayerLabel,
}
```

3. Add `regret_audits` field to `BlueprintTuiConfig`:

```rust
#[serde(default)]
pub regret_audits: Vec<RegretAuditConfig>,
```

And add `regret_audits: Vec::new()` to `BlueprintTuiConfig::default()`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-trainer parse_regret_audits -- --nocapture`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test -p poker-solver-trainer`
Expected: All tests pass, existing tests unaffected.

**Step 6: Commit**

```bash
git add crates/trainer/src/blueprint_tui_config.rs
git commit -m "feat(tui): add RegretAuditConfig YAML parsing"
```

---

### Task 2: Audit resolution — resolve hand+spot to regret coordinates

**Files:**
- Create: `crates/trainer/src/blueprint_tui_audit.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui_audit;`)

This module resolves a `RegretAuditConfig` into the concrete `(node_idx, bucket, action_labels)` needed for reading regrets. It also computes the bucket trail across streets.

**Step 1: Write the failing test**

Create `crates/trainer/src/blueprint_tui_audit.rs` with:

```rust
//! Regret audit resolution: resolve a hand+spot config entry into the
//! concrete storage coordinates needed to read per-action regret values.

use std::collections::VecDeque;

use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::poker::{self, Card};

use crate::blueprint_tui_config::PlayerLabel;
use crate::blueprint_tui_scenarios::{format_tree_action, resolve_spot};

/// Trend direction derived from the smoothed delta ring buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Up,
    Down,
    Flat,
}

/// A resolved regret audit entry, ready for TUI polling.
pub struct ResolvedRegretAudit {
    pub name: String,
    pub node_idx: u32,
    pub player: u8,
    pub bucket: u16,
    pub bucket_trail: Vec<(Street, u16)>,
    pub action_labels: Vec<String>,
    pub num_actions: usize,
    /// Current cumulative regrets (scaled: raw_i32 / 1000.0).
    pub regrets: Vec<f64>,
    /// Previous tick's regrets for computing deltas.
    pub prev_regrets: Vec<f64>,
    /// Per-action ring buffer of deltas for smoothed trend.
    pub trend_buffers: Vec<VecDeque<f64>>,
    /// Error message if resolution failed.
    pub error: Option<String>,
}

/// Resolve a regret audit config entry against the game tree.
///
/// Returns a `ResolvedRegretAudit` with either valid coordinates or
/// an error message that the TUI can display.
pub fn resolve_regret_audit(
    tree: &GameTree,
    storage: &BlueprintStorage,
    name: &str,
    spot: &str,
    hand_str: &str,
    player: PlayerLabel,
    trend_window: usize,
) -> ResolvedRegretAudit {
    todo!()
}

impl ResolvedRegretAudit {
    /// Poll current regrets from storage, update deltas and trends.
    pub fn tick(&mut self, storage: &BlueprintStorage) {
        todo!()
    }

    /// Compute the derived strategy from current regrets (regret matching).
    pub fn strategy(&self) -> Vec<f64> {
        todo!()
    }

    /// Get the trend direction for a given action index.
    pub fn trend(&self, action_idx: usize) -> Trend {
        todo!()
    }

    /// Get the raw delta (current - previous) for a given action index.
    pub fn delta(&self, action_idx: usize) -> f64 {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    fn toy_tree() -> GameTree {
        GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    #[timed_test(10)]
    fn resolve_preflop_audit() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let audit = resolve_regret_audit(
            &tree, &storage, "AKo SB open", "", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_none(), "expected no error, got: {:?}", audit.error);
        assert_eq!(audit.node_idx, tree.root);
        assert_eq!(audit.player, 0);
        assert!(!audit.action_labels.is_empty());
        assert_eq!(audit.num_actions, audit.action_labels.len());
        assert_eq!(audit.regrets.len(), audit.num_actions);
        assert_eq!(audit.bucket_trail.len(), 1); // preflop only
        assert_eq!(audit.bucket_trail[0].0, Street::Preflop);
    }

    #[timed_test(10)]
    fn resolve_invalid_spot_returns_error() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let audit = resolve_regret_audit(
            &tree, &storage, "bad", "sb:999bb", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_some());
    }

    #[timed_test(10)]
    fn resolve_invalid_hand_returns_error() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let audit = resolve_regret_audit(
            &tree, &storage, "bad hand", "", "ZZo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_some());
    }

    #[timed_test(10)]
    fn tick_updates_regrets_and_deltas() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo SB open", "", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_none());

        // All regrets start at zero.
        for &r in &audit.regrets {
            assert_eq!(r, 0.0);
        }

        // Simulate regret accumulation by writing to storage.
        storage.add_regret(audit.node_idx, audit.bucket, 0, 5000); // fold: +5.0
        storage.add_regret(audit.node_idx, audit.bucket, 1, -3000); // call: -3.0

        audit.tick(&storage);
        assert!((audit.regrets[0] - 5.0).abs() < 0.01);
        assert!((audit.regrets[1] - (-3.0)).abs() < 0.01);
        assert!((audit.delta(0) - 5.0).abs() < 0.01);
        assert!((audit.delta(1) - (-3.0)).abs() < 0.01);
    }

    #[timed_test(10)]
    fn strategy_from_regrets() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo SB open", "", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_none());

        // Set some positive regrets.
        storage.add_regret(audit.node_idx, audit.bucket, 0, 0);     // fold: 0
        storage.add_regret(audit.node_idx, audit.bucket, 1, 1000);  // call: +1.0
        storage.add_regret(audit.node_idx, audit.bucket, 2, 3000);  // raise: +3.0

        audit.tick(&storage);
        let strat = audit.strategy();
        // fold regret=0 → not positive, so fold=0. call=1/(1+3)=0.25, raise=3/4=0.75
        assert!(strat[0] < 0.01, "fold should be ~0%");
        assert!((strat[1] - 0.25).abs() < 0.01, "call should be ~25%");
        assert!((strat[2] - 0.75).abs() < 0.01, "raise should be ~75%");
    }

    #[timed_test(10)]
    fn trend_detection() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo SB open", "", "AKo", PlayerLabel::Sb, 3,
        );
        assert!(audit.error.is_none());

        // Tick 1: add positive regret to action 0
        storage.add_regret(audit.node_idx, audit.bucket, 0, 1000);
        audit.tick(&storage);

        // Tick 2: add more
        storage.add_regret(audit.node_idx, audit.bucket, 0, 1000);
        audit.tick(&storage);

        // Tick 3: add more
        storage.add_regret(audit.node_idx, audit.bucket, 0, 1000);
        audit.tick(&storage);

        // Trend for action 0 should be Up (all deltas positive).
        assert_eq!(audit.trend(0), Trend::Up);
        // Action 1 never moved → Flat.
        assert_eq!(audit.trend(1), Trend::Flat);
    }
}
```

**Step 2: Register the module**

Add `mod blueprint_tui_audit;` to `crates/trainer/src/main.rs` alongside the other `blueprint_tui_*` mods.

**Step 3: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer resolve_preflop_audit -- --nocapture`
Expected: FAIL — `todo!()` panics.

**Step 4: Implement `resolve_regret_audit`**

Replace the `todo!()` in `resolve_regret_audit`:

```rust
pub fn resolve_regret_audit(
    tree: &GameTree,
    storage: &BlueprintStorage,
    name: &str,
    spot: &str,
    hand_str: &str,
    player: PlayerLabel,
    trend_window: usize,
) -> ResolvedRegretAudit {
    let player_idx: u8 = match player {
        PlayerLabel::Sb => 0,
        PlayerLabel::Bb => 1,
    };

    let make_error = |msg: String| ResolvedRegretAudit {
        name: name.to_string(),
        node_idx: 0,
        player: player_idx,
        bucket: 0,
        bucket_trail: vec![],
        action_labels: vec![],
        num_actions: 0,
        regrets: vec![],
        prev_regrets: vec![],
        trend_buffers: vec![],
        error: Some(msg),
    };

    // Resolve spot to node + board.
    let (node_idx, board) = match resolve_spot(tree, spot) {
        Some(result) => result,
        None => return make_error(format!("Spot failed to resolve: {spot}")),
    };

    // Get node info.
    let (street, actions) = match &tree.nodes[node_idx as usize] {
        GameNode::Decision { street, actions, .. } => (*street, actions.clone()),
        _ => return make_error(format!("Spot resolved to non-decision node")),
    };

    let num_actions = actions.len();
    let action_labels: Vec<String> = actions.iter().map(format_tree_action).collect();

    // Parse hand and compute bucket.
    let hand_str_trimmed = hand_str.trim();
    let bucket = if hand_str_trimmed.len() == 4 {
        // Specific combo like "Ts9s" — parse two cards.
        let c1 = match poker::parse_card(&hand_str_trimmed[0..2]) {
            Some(c) => c,
            None => return make_error(format!("Invalid card: {}", &hand_str_trimmed[0..2])),
        };
        let c2 = match poker::parse_card(&hand_str_trimmed[2..4]) {
            Some(c) => c,
            None => return make_error(format!("Invalid card: {}", &hand_str_trimmed[2..4])),
        };
        // Check cards don't conflict with board.
        if board.contains(&c1) || board.contains(&c2) {
            return make_error(format!("Hand {hand_str_trimmed} conflicts with board"));
        }
        compute_bucket_for_combo(storage, street, c1, c2, &board)
    } else {
        // Canonical hand like "AKo", "TT".
        match CanonicalHand::parse(hand_str_trimmed) {
            Ok(hand) => {
                if street == Street::Preflop {
                    (hand.index() as u16) % storage.bucket_counts[0]
                } else {
                    // For postflop canonical hand, pick first unblocked combo.
                    let combo = hand.combos().into_iter().find(|(c1, c2)| {
                        !board.contains(c1) && !board.contains(c2)
                    });
                    match combo {
                        Some((c1, c2)) => compute_bucket_for_combo(storage, street, c1, c2, &board),
                        None => return make_error(format!("Hand {hand_str_trimmed} blocked by board")),
                    }
                }
            }
            Err(e) => return make_error(format!("Invalid hand '{hand_str_trimmed}': {e}")),
        }
    };

    // Compute bucket trail (bucket at each street up to current).
    let bucket_trail = compute_bucket_trail(storage, hand_str_trimmed, street, &board);

    let window = trend_window.max(1);
    ResolvedRegretAudit {
        name: name.to_string(),
        node_idx,
        player: player_idx,
        bucket,
        bucket_trail,
        action_labels,
        num_actions,
        regrets: vec![0.0; num_actions],
        prev_regrets: vec![0.0; num_actions],
        trend_buffers: (0..num_actions).map(|_| VecDeque::with_capacity(window)).collect(),
        error: None,
    }
}

/// Compute the equity-based bucket for a specific combo at a postflop street.
fn compute_bucket_for_combo(
    storage: &BlueprintStorage,
    street: Street,
    c1: Card,
    c2: Card,
    board: &[Card],
) -> u16 {
    let street_idx = street as usize;
    let num_buckets = storage.bucket_counts[street_idx];
    if street == Street::Preflop {
        let hand = CanonicalHand::from_cards(c1, c2);
        return (hand.index() as u16) % num_buckets;
    }
    let equity = poker_solver_core::showdown_equity::compute_equity([c1, c2], board);
    ((equity * f64::from(num_buckets)) as u16).min(num_buckets - 1)
}

/// Compute the bucket trail from preflop through the target street.
fn compute_bucket_trail(
    storage: &BlueprintStorage,
    hand_str: &str,
    target_street: Street,
    board: &[Card],
) -> Vec<(Street, u16)> {
    let streets = [Street::Preflop, Street::Flop, Street::Turn, Street::River];
    let mut trail = Vec::new();

    // Parse cards for bucket computation.
    let (c1, c2) = if hand_str.len() == 4 {
        let c1 = poker::parse_card(&hand_str[0..2]).unwrap();
        let c2 = poker::parse_card(&hand_str[2..4]).unwrap();
        (c1, c2)
    } else if let Ok(hand) = CanonicalHand::parse(hand_str) {
        // Pick first unblocked combo.
        hand.combos().into_iter().find(|(c1, c2)| {
            !board.contains(c1) && !board.contains(c2)
        }).unwrap_or_else(|| hand.combos()[0])
    } else {
        return trail;
    };

    for &st in &streets {
        let board_slice = match st {
            Street::Preflop => &[][..],
            Street::Flop => {
                if board.len() >= 3 { &board[..3] } else { break; }
            }
            Street::Turn => {
                if board.len() >= 4 { &board[..4] } else { break; }
            }
            Street::River => {
                if board.len() >= 5 { &board[..5] } else { break; }
            }
        };
        let bucket = compute_bucket_for_combo(storage, st, c1, c2, board_slice);
        trail.push((st, bucket));
        if st == target_street {
            break;
        }
    }
    trail
}
```

**Step 5: Implement `tick`, `strategy`, `trend`, `delta`**

```rust
impl ResolvedRegretAudit {
    pub fn tick(&mut self, storage: &BlueprintStorage) {
        if self.error.is_some() {
            return;
        }
        self.prev_regrets.clone_from(&self.regrets);
        for i in 0..self.num_actions {
            let raw = storage.get_regret(self.node_idx, self.bucket, i);
            self.regrets[i] = f64::from(raw) / 1000.0;
        }
        // Update trend buffers with deltas.
        let window = self.trend_buffers[0].capacity();
        for i in 0..self.num_actions {
            let d = self.regrets[i] - self.prev_regrets[i];
            let buf = &mut self.trend_buffers[i];
            if buf.len() >= window {
                buf.pop_front();
            }
            buf.push_back(d);
        }
    }

    pub fn strategy(&self) -> Vec<f64> {
        let mut positive_sum = 0.0_f64;
        let positives: Vec<f64> = self.regrets.iter().map(|&r| r.max(0.0)).collect();
        positive_sum = positives.iter().sum();
        if positive_sum > 0.0 {
            positives.iter().map(|&r| r / positive_sum).collect()
        } else {
            vec![1.0 / self.num_actions as f64; self.num_actions]
        }
    }

    pub fn trend(&self, action_idx: usize) -> Trend {
        let buf = &self.trend_buffers[action_idx];
        if buf.is_empty() {
            return Trend::Flat;
        }
        let avg: f64 = buf.iter().sum::<f64>() / buf.len() as f64;
        if avg > 0.001 {
            Trend::Up
        } else if avg < -0.001 {
            Trend::Down
        } else {
            Trend::Flat
        }
    }

    pub fn delta(&self, action_idx: usize) -> f64 {
        self.regrets[action_idx] - self.prev_regrets[action_idx]
    }
}
```

**Step 6: Run tests to verify they pass**

Run: `cargo test -p poker-solver-trainer blueprint_tui_audit -- --nocapture`
Expected: All 6 tests pass.

**Step 7: Commit**

```bash
git add crates/trainer/src/blueprint_tui_audit.rs crates/trainer/src/main.rs
git commit -m "feat(tui): add regret audit resolution and tick logic"
```

---

### Task 3: Metrics bridge — pass audit data between trainer and TUI threads

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_metrics.rs`

**Step 1: Write the failing test**

Add to `blueprint_tui_metrics.rs` tests:

```rust
#[timed_test(10)]
fn regret_audit_snapshot_exchange() {
    let m = BlueprintTuiMetrics::new(None, None);
    // Push a snapshot.
    let snapshot = vec![
        crate::blueprint_tui_audit::AuditSnapshot {
            regrets: vec![1.0, -2.0, 3.0],
            deltas: vec![0.5, -0.1, 0.2],
            trends: vec![
                crate::blueprint_tui_audit::Trend::Up,
                crate::blueprint_tui_audit::Trend::Down,
                crate::blueprint_tui_audit::Trend::Flat,
            ],
            strategy: vec![0.0, 0.25, 0.75],
        },
    ];
    m.update_regret_audits(snapshot.clone());
    let taken = m.take_regret_audits();
    assert!(taken.is_some());
    assert_eq!(taken.unwrap()[0].regrets, vec![1.0, -2.0, 3.0]);
    // Second take returns None.
    assert!(m.take_regret_audits().is_none());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer regret_audit_snapshot_exchange -- --nocapture`
Expected: FAIL — `AuditSnapshot` and methods don't exist.

**Step 3: Implement**

Add `AuditSnapshot` to `blueprint_tui_audit.rs`:

```rust
/// Snapshot of audit state for transfer to the TUI thread.
#[derive(Debug, Clone)]
pub struct AuditSnapshot {
    pub regrets: Vec<f64>,
    pub deltas: Vec<f64>,
    pub trends: Vec<Trend>,
    pub strategy: Vec<f64>,
}

impl ResolvedRegretAudit {
    pub fn snapshot(&self) -> AuditSnapshot {
        AuditSnapshot {
            regrets: self.regrets.clone(),
            deltas: (0..self.num_actions).map(|i| self.delta(i)).collect(),
            trends: (0..self.num_actions).map(|i| self.trend(i)).collect(),
            strategy: self.strategy(),
        }
    }
}
```

Add to `BlueprintTuiMetrics`:

```rust
pub regret_audit_snapshots: Mutex<Option<Vec<AuditSnapshot>>>,
```

Initialize in `new()`:

```rust
regret_audit_snapshots: Mutex::new(None),
```

Add methods:

```rust
pub fn update_regret_audits(&self, snapshots: Vec<AuditSnapshot>) {
    let mut data = self.regret_audit_snapshots.lock().unwrap_or_else(|e| e.into_inner());
    *data = Some(snapshots);
}

pub fn take_regret_audits(&self) -> Option<Vec<AuditSnapshot>> {
    let mut data = self.regret_audit_snapshots.lock().unwrap_or_else(|e| e.into_inner());
    data.take()
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-trainer regret_audit_snapshot_exchange -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_audit.rs crates/trainer/src/blueprint_tui_metrics.rs
git commit -m "feat(tui): add audit snapshot exchange on metrics bridge"
```

---

### Task 4: TUI rendering — audit panel widget

**Files:**
- Create: `crates/trainer/src/blueprint_tui_audit_widget.rs`
- Modify: `crates/trainer/src/main.rs` (add `mod blueprint_tui_audit_widget;`)

**Step 1: Write the failing test**

Create `crates/trainer/src/blueprint_tui_audit_widget.rs`:

```rust
//! Ratatui widget for the regret audit panel.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};

use crate::blueprint_tui_audit::{AuditSnapshot, Trend};
use crate::blueprint_tui_config::PlayerLabel;
use poker_solver_core::blueprint_v2::Street;

/// Static metadata for one audit entry (set at startup, never changes).
#[derive(Debug, Clone)]
pub struct AuditMeta {
    pub name: String,
    pub hand: String,
    pub player: PlayerLabel,
    pub bucket_trail: Vec<(Street, u16)>,
    pub action_labels: Vec<String>,
    pub error: Option<String>,
}

/// Full state for rendering the audit panel.
pub struct AuditPanelState {
    pub metas: Vec<AuditMeta>,
    pub snapshots: Vec<AuditSnapshot>,
    pub active_tab: usize,
    pub iteration: u64,
}

impl AuditPanelState {
    pub fn next_tab(&mut self) {
        if !self.metas.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.metas.len();
        }
    }

    pub fn prev_tab(&mut self) {
        if !self.metas.is_empty() {
            self.active_tab = (self.active_tab + self.metas.len() - 1) % self.metas.len();
        }
    }
}

pub struct AuditPanelWidget<'a> {
    pub state: &'a AuditPanelState,
}

impl Widget for &AuditPanelWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.state.metas.is_empty() {
            return;
        }

        // Vertical: tab bar (1 line) + content
        let tab_area = Rect { height: 1, ..area };
        let content_area = Rect {
            y: area.y + 1,
            height: area.height.saturating_sub(1),
            ..area
        };

        // Tab bar
        let titles: Vec<Line<'_>> = self.state.metas.iter()
            .map(|m| Line::from(m.name.as_str()))
            .collect();
        let tabs = Tabs::new(titles)
            .select(self.state.active_tab)
            .highlight_style(Style::default().fg(Color::Cyan).bold())
            .divider("|");
        buf.set_style(tab_area, Style::default());
        tabs.render(tab_area, buf);

        // Content for active tab
        let idx = self.state.active_tab;
        let meta = &self.state.metas[idx];

        if let Some(ref err) = meta.error {
            let style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
            let y = content_area.y + 1;
            if y < content_area.y + content_area.height {
                buf.set_string(content_area.x + 1, y, err, style);
            }
            return;
        }

        let snapshot = self.state.snapshots.get(idx);
        let mut y = content_area.y;

        // Header: Hand + Player + Iter
        let player_str = match meta.player {
            PlayerLabel::Sb => "SB",
            PlayerLabel::Bb => "BB",
        };
        let header = format!("Hand: {}  Player: {}  Iter: {}", meta.hand, player_str, self.state.iteration);
        if y < content_area.y + content_area.height {
            buf.set_string(content_area.x + 1, y, &header, Style::default().fg(Color::White));
            y += 1;
        }

        // Bucket trail
        let trail: Vec<String> = meta.bucket_trail.iter().map(|(st, b)| {
            let label = match st {
                Street::Preflop => "pf",
                Street::Flop => "fl",
                Street::Turn => "tn",
                Street::River => "rv",
            };
            format!("{label}:{b}")
        }).collect();
        let trail_str = format!("Buckets: {}", trail.join(" → "));
        if y < content_area.y + content_area.height {
            buf.set_string(content_area.x + 1, y, &trail_str, Style::default().fg(Color::DarkGray));
            y += 2; // blank line
        }

        // Table header
        if y < content_area.y + content_area.height {
            let hdr = format!("{:<12} {:>8} {:>8}  {}", "Action", "Regret", "Δ/tick", "Trend");
            buf.set_string(content_area.x + 1, y, &hdr, Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD));
            y += 1;
        }

        // Table rows
        if let Some(snap) = snapshot {
            for (i, label) in meta.action_labels.iter().enumerate() {
                if y >= content_area.y + content_area.height {
                    break;
                }
                let regret = snap.regrets.get(i).copied().unwrap_or(0.0);
                let delta = snap.deltas.get(i).copied().unwrap_or(0.0);
                let trend = snap.trends.get(i).copied().unwrap_or(Trend::Flat);

                let (trend_arrow, trend_color) = match trend {
                    Trend::Up => ("↑", Color::Green),
                    Trend::Down => ("↓", Color::Red),
                    Trend::Flat => ("→", Color::DarkGray),
                };

                let regret_color = if regret > 0.0 { Color::Green } else if regret < 0.0 { Color::Red } else { Color::DarkGray };

                let label_trunc = if label.len() > 11 { &label[..11] } else { label.as_str() };
                buf.set_string(content_area.x + 1, y, &format!("{:<12}", label_trunc), Style::default().fg(Color::White));
                buf.set_string(content_area.x + 14, y, &format!("{:>8.1}", regret), Style::default().fg(regret_color));
                buf.set_string(content_area.x + 23, y, &format!("{:>+8.1}", delta), Style::default().fg(Color::DarkGray));
                buf.set_string(content_area.x + 33, y, trend_arrow, Style::default().fg(trend_color));
                y += 1;
            }
        }

        // Strategy line
        y += 1; // blank line
        if y < content_area.y + content_area.height {
            if let Some(snap) = snapshot {
                let parts: Vec<String> = meta.action_labels.iter().zip(snap.strategy.iter())
                    .map(|(label, &prob)| {
                        let short = if label.len() > 1 { &label[..1] } else { label.as_str() };
                        format!("{}:{:.0}%", short, prob * 100.0)
                    })
                    .collect();
                let strat_line = format!("Strategy: {}", parts.join(" "));
                buf.set_string(content_area.x + 1, y, &strat_line, Style::default().fg(Color::Cyan));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use test_macros::timed_test;

    fn mock_panel_state() -> AuditPanelState {
        AuditPanelState {
            metas: vec![
                AuditMeta {
                    name: "AKo SB open".to_string(),
                    hand: "AKo".to_string(),
                    player: PlayerLabel::Sb,
                    bucket_trail: vec![(Street::Preflop, 3)],
                    action_labels: vec!["fold".to_string(), "call".to_string(), "raise 5bb".to_string()],
                    error: None,
                },
            ],
            snapshots: vec![
                AuditSnapshot {
                    regrets: vec![-1.2, 0.3, 0.9],
                    deltas: vec![-0.03, 0.02, 0.01],
                    trends: vec![Trend::Down, Trend::Up, Trend::Up],
                    strategy: vec![0.0, 0.25, 0.75],
                },
            ],
            active_tab: 0,
            iteration: 1_200_000,
        }
    }

    #[timed_test(10)]
    fn audit_panel_renders_without_panic() {
        let state = mock_panel_state();
        let widget = AuditPanelWidget { state: &state };
        let backend = TestBackend::new(50, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| {
            frame.render_widget(&widget, frame.area());
        }).unwrap();
    }

    #[timed_test(10)]
    fn audit_panel_tab_switching() {
        let mut state = mock_panel_state();
        state.metas.push(AuditMeta {
            name: "TT 3bet".to_string(),
            hand: "TT".to_string(),
            player: PlayerLabel::Sb,
            bucket_trail: vec![(Street::Preflop, 5)],
            action_labels: vec!["fold".to_string(), "call".to_string()],
            error: None,
        });
        state.snapshots.push(AuditSnapshot {
            regrets: vec![0.0, 0.0],
            deltas: vec![0.0, 0.0],
            trends: vec![Trend::Flat, Trend::Flat],
            strategy: vec![0.5, 0.5],
        });

        assert_eq!(state.active_tab, 0);
        state.next_tab();
        assert_eq!(state.active_tab, 1);
        state.next_tab();
        assert_eq!(state.active_tab, 0);
        state.prev_tab();
        assert_eq!(state.active_tab, 1);
    }

    #[timed_test(10)]
    fn audit_panel_error_renders() {
        let state = AuditPanelState {
            metas: vec![AuditMeta {
                name: "bad".to_string(),
                hand: "AKo".to_string(),
                player: PlayerLabel::Sb,
                bucket_trail: vec![],
                action_labels: vec![],
                error: Some("Spot failed to resolve".to_string()),
            }],
            snapshots: vec![AuditSnapshot {
                regrets: vec![],
                deltas: vec![],
                trends: vec![],
                strategy: vec![],
            }],
            active_tab: 0,
            iteration: 0,
        };
        let widget = AuditPanelWidget { state: &state };
        let backend = TestBackend::new(50, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| {
            frame.render_widget(&widget, frame.area());
        }).unwrap();
    }

    #[timed_test(10)]
    fn empty_audits_renders_nothing() {
        let state = AuditPanelState {
            metas: vec![],
            snapshots: vec![],
            active_tab: 0,
            iteration: 0,
        };
        let widget = AuditPanelWidget { state: &state };
        let backend = TestBackend::new(50, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| {
            frame.render_widget(&widget, frame.area());
        }).unwrap();
    }
}
```

**Step 2: Register the module**

Add `mod blueprint_tui_audit_widget;` to `crates/trainer/src/main.rs`.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-trainer blueprint_tui_audit_widget -- --nocapture`
Expected: All 4 tests pass (widget code is complete, not `todo!()`).

**Step 4: Commit**

```bash
git add crates/trainer/src/blueprint_tui_audit_widget.rs crates/trainer/src/main.rs
git commit -m "feat(tui): add regret audit panel widget"
```

---

### Task 5: Layout integration — horizontal split and audit panel in TUI

**Files:**
- Modify: `crates/trainer/src/blueprint_tui.rs`

**Step 1: Write the failing test**

Add to `blueprint_tui.rs` tests:

```rust
#[timed_test(10)]
fn app_renders_with_audit_panel() {
    let metrics = make_metrics();
    let audit_state = crate::blueprint_tui_audit_widget::AuditPanelState {
        metas: vec![crate::blueprint_tui_audit_widget::AuditMeta {
            name: "AKo test".to_string(),
            hand: "AKo".to_string(),
            player: crate::blueprint_tui_config::PlayerLabel::Sb,
            bucket_trail: vec![(poker_solver_core::blueprint_v2::Street::Preflop, 3)],
            action_labels: vec!["fold".into(), "call".into(), "raise".into()],
            error: None,
        }],
        snapshots: vec![crate::blueprint_tui_audit::AuditSnapshot {
            regrets: vec![-1.0, 0.5, 0.5],
            deltas: vec![0.0, 0.1, -0.1],
            trends: vec![
                crate::blueprint_tui_audit::Trend::Down,
                crate::blueprint_tui_audit::Trend::Up,
                crate::blueprint_tui_audit::Trend::Flat,
            ],
            strategy: vec![0.0, 0.5, 0.5],
        }],
        active_tab: 0,
        iteration: 0,
    };
    let app = BlueprintTuiApp::new(
        metrics,
        vec![],
        TelemetryConfig::default(),
    );
    // TODO: set audit_state on app
    let backend = ratatui::backend::TestBackend::new(160, 50);
    let mut terminal = ratatui::Terminal::new(backend).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer app_renders_with_audit_panel -- --nocapture`
Expected: FAIL — `audit_state` not a field on `BlueprintTuiApp`.

**Step 3: Implement layout changes**

Modify `BlueprintTuiApp`:

1. Add field: `pub audit_panel: Option<AuditPanelState>`
2. Accept it in constructor (default `None`)
3. Modify `render()`: If `audit_panel` is `Some` and has entries, split the sparkline area horizontally 60/40. Sparklines go left, audit panel goes right.
4. Add keybindings: `KeyCode::Up` → `audit_panel.next_tab()`, `KeyCode::Down` → `audit_panel.prev_tab()`
5. In `tick()`: read audit snapshots from metrics via `take_regret_audits()` and update `audit_panel.snapshots`
6. Update hotkey bar to show `[↑/↓]audit` when audits are configured

Key layout change in `render()`:

```rust
// If audit panel configured, split sparkline area horizontally.
let (sparkline_area, audit_area) = if self.audit_panel.as_ref().is_some_and(|a| !a.metas.is_empty()) {
    let hsplit = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(sparkline_chunk);
    (hsplit[0], Some(hsplit[1]))
} else {
    (sparkline_chunk, None)
};
```

The sparkline rendering functions already accept an `area: Rect`, so they work in the narrower left column without changes.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-trainer -- --nocapture`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui.rs
git commit -m "feat(tui): integrate audit panel with horizontal split layout"
```

---

### Task 6: Wiring — connect config → resolution → trainer → TUI

**Files:**
- Modify: `crates/trainer/src/main.rs`

This is the glue that:
1. Reads `regret_audits` from YAML config
2. Resolves each to `ResolvedRegretAudit` at startup
3. Creates `AuditMeta` for the TUI panel
4. Adds a trainer callback that ticks audits and pushes snapshots to metrics
5. Passes `AuditPanelState` to `BlueprintTuiApp`

**Step 1: Add audit resolution after scenario resolution in `main.rs`**

After the existing scenario resolution block (around line 327), add:

```rust
// Resolve regret audits.
let (audit_metas, mut resolved_audits): (Vec<_>, Vec<_>) = tui_config
    .regret_audits
    .iter()
    .map(|ac| {
        let audit = blueprint_tui_audit::resolve_regret_audit(
            &trainer.tree,
            &trainer.storage,
            &ac.name,
            &ac.spot,
            &ac.hand,
            ac.player,
            tui_config.telemetry.sparkline_window,
        );
        let meta = blueprint_tui_audit_widget::AuditMeta {
            name: ac.name.clone(),
            hand: ac.hand.clone(),
            player: ac.player,
            bucket_trail: audit.bucket_trail.clone(),
            action_labels: audit.action_labels.clone(),
            error: audit.error.clone(),
        };
        (meta, audit)
    })
    .unzip();

let audit_panel = if !audit_metas.is_empty() {
    let initial_snapshots: Vec<_> = resolved_audits.iter().map(|a| a.snapshot()).collect();
    Some(blueprint_tui_audit_widget::AuditPanelState {
        metas: audit_metas,
        snapshots: initial_snapshots,
        active_tab: 0,
        iteration: 0,
    })
} else {
    None
};
```

**Step 2: Wire the audit refresh callback**

Add a callback on the trainer that ticks all resolved audits and pushes snapshots to metrics. This fires at the same interval as strategy refresh:

```rust
if !resolved_audits.is_empty() {
    let metrics_for_audit = Arc::clone(&metrics);
    trainer.on_audit_refresh = Some(Box::new(move |storage| {
        for audit in &mut resolved_audits {
            audit.tick(storage);
        }
        let snapshots: Vec<_> = resolved_audits.iter().map(|a| a.snapshot()).collect();
        metrics_for_audit.update_regret_audits(snapshots);
    }));
}
```

Note: `on_audit_refresh` is a new callback field on `BlueprintTrainer`. It needs to be added to the trainer struct and called at the same interval as strategy refresh. Check `crates/core/src/blueprint_v2/trainer.rs` for the callback pattern.

**Step 3: Pass audit_panel to BlueprintTuiApp**

Modify the `run_blueprint_tui` call to include the audit panel state.

**Step 4: Run full test suite**

Run: `cargo test`
Expected: All tests pass.

**Step 5: Manual verification**

Add regret audits to `sample_configurations/blueprint_v2_with_tui.yaml`:

```yaml
  regret_audits:
    - name: "AKo SB open"
      spot: ""
      hand: "AKo"
      player: SB
    - name: "72o SB open"
      spot: ""
      hand: "72o"
      player: SB
```

**Step 6: Commit**

```bash
git add crates/trainer/src/main.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat(tui): wire regret audit pipeline from config to TUI"
```

---

### Task 7: Trainer callback — add `on_audit_refresh` to BlueprintTrainer

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Add callback field**

Find the existing callback fields (`on_strategy_refresh`, `on_min_regret`, etc.) and add:

```rust
/// Called at strategy refresh interval to tick regret audits.
pub on_audit_refresh: Option<Box<dyn FnMut(&BlueprintStorage) + Send>>,
```

**Step 2: Call it at strategy refresh interval**

In the training loop, where `on_strategy_refresh` is called, add a call to `on_audit_refresh` with `&self.storage`.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat(trainer): add on_audit_refresh callback"
```

---

### Task 8: Sample config update

**Files:**
- Modify: `sample_configurations/blueprint_v2_with_tui.yaml`

Add `regret_audits` section to the sample config as a reference example.

**Step 1: Add the config**

```yaml
  regret_audits:
    - name: "AKo SB open"
      spot: ""
      hand: "AKo"
      player: SB
    - name: "72o SB open"
      spot: ""
      hand: "72o"
      player: SB
```

**Step 2: Commit**

```bash
git add sample_configurations/blueprint_v2_with_tui.yaml
git commit -m "docs: add regret_audits example to sample TUI config"
```

---

## Task Dependency Order

```
Task 1 (config parsing)
    ↓
Task 2 (audit resolution) ← depends on config types
    ↓
Task 3 (metrics bridge) ← depends on AuditSnapshot from Task 2
    ↓
Task 4 (audit widget) ← depends on AuditSnapshot and AuditMeta types
    ↓
Task 5 (layout integration) ← depends on widget from Task 4
    ↓
Task 7 (trainer callback) ← can be parallel with Task 4-5
    ↓
Task 6 (wiring) ← depends on all above
    ↓
Task 8 (sample config) ← final
```

**Parallelizable:** Tasks 4+5 and Task 7 can be done in parallel after Task 3.
