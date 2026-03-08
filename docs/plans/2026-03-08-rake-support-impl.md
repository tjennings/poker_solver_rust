# Rake Support + Blueprint Name — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add rake/rake_cap to blueprint MCCFR terminal payoffs and range solver Tauri UI, plus a required `name` field to blueprint config for selection/display.

**Architecture:** Three layers: (1) core config + terminal payoff changes in `poker-solver-core`, (2) Tauri backend threading rake through to range solver and exposing name/rake in API responses, (3) frontend displaying name and rake controls.

**Tech Stack:** Rust (serde, MCCFR), TypeScript/React (Tauri frontend)

---

### Task 1: Add `name`, `rake_rate`, `rake_cap` to `GameConfig`

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs:17-26`

**Step 1: Add fields to `GameConfig` struct**

In `crates/core/src/blueprint_v2/config.rs`, add three fields to `GameConfig`:

```rust
/// Core game parameters (stakes and stack depth).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    /// Human-readable name for this blueprint (shown in UI).
    pub name: String,
    pub players: u8,
    /// Stack depth in big blinds.
    pub stack_depth: f64,
    /// Small blind size in big blinds (typically 0.5).
    pub small_blind: f64,
    /// Big blind size in big blinds (typically 1.0).
    pub big_blind: f64,
    /// Rake as a fraction of the pot (0.0 = no rake, 0.05 = 5%).
    #[serde(default)]
    pub rake_rate: f64,
    /// Maximum rake in chips (min bet units). 0.0 = no cap.
    #[serde(default)]
    pub rake_cap: f64,
}
```

Note: `name` is required (no `#[serde(default)]`). `rake_rate` and `rake_cap` default to 0.0 for backward compatibility with existing configs.

**Step 2: Update the config deserialization test**

In the same file, update `test_deserialize_toy_config` — add `name: "Test Config"` to the YAML string and add assertions:

```rust
// In the YAML string, add under `game:`:
//   name: "Test Config"

// Add assertions:
assert_eq!(cfg.game.name, "Test Config");
assert!((cfg.game.rake_rate - 0.0).abs() < f64::EPSILON);
assert!((cfg.game.rake_cap - 0.0).abs() < f64::EPSILON);
```

Also add a separate test for rake fields:

```rust
#[test]
fn test_deserialize_with_rake() {
    let yaml = r#"
game:
  name: "NL50 5% rake"
  players: 2
  stack_depth: 50.0
  small_blind: 0.5
  big_blind: 1.0
  rake_rate: 0.05
  rake_cap: 3.0
clustering:
  algorithm: potential_aware_emd
  preflop: { buckets: 169 }
  flop: { buckets: 200 }
  turn: { buckets: 200 }
  river: { buckets: 200 }
action_abstraction:
  preflop: [["2.5bb"]]
  flop: [[0.5, 1.0]]
  turn: [[0.5, 1.0]]
  river: [[0.5, 1.0]]
training:
  cluster_path: "/tmp/clusters"
  iterations: 100
snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: "/tmp/out"
"#;
    let cfg: BlueprintV2Config = serde_yaml::from_str(yaml).expect("parse");
    assert_eq!(cfg.game.name, "NL50 5% rake");
    assert!((cfg.game.rake_rate - 0.05).abs() < f64::EPSILON);
    assert!((cfg.game.rake_cap - 3.0).abs() < f64::EPSILON);
}
```

**Step 3: Update `test_serialize_round_trip`**

Add `name`, `rake_rate`, `rake_cap` to the `GameConfig` construction:

```rust
game: GameConfig {
    name: "Round Trip Test".to_string(),
    players: 2,
    stack_depth: 50.0,
    small_blind: 0.5,
    big_blind: 1.0,
    rake_rate: 0.05,
    rake_cap: 3.0,
},
```

Add assertions:

```rust
assert_eq!(restored.game.name, "Round Trip Test");
assert!((restored.game.rake_rate - 0.05).abs() < f64::EPSILON);
assert!((restored.game.rake_cap - 3.0).abs() < f64::EPSILON);
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- blueprint_v2::config
```

Expected: All 3 tests pass (existing + new rake test).

**Step 5: Fix all compilation errors**

Search the codebase for every place `GameConfig` is constructed (tests, e2e, trainer) and add the `name`, `rake_rate`, `rake_cap` fields. Key locations:

- `crates/core/tests/blueprint_v2_e2e.rs` — the E2E pipeline test constructs a `GameConfig`
- `crates/core/src/blueprint_v2/mccfr.rs` — test helpers that build configs
- Any trainer code that constructs configs

For each, add:
```rust
name: "Test".to_string(),
rake_rate: 0.0,
rake_cap: 0.0,
```

**Step 6: Run full test suite**

```bash
cargo test
```

Expected: All tests pass. No clippy warnings on new fields.

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: add name, rake_rate, rake_cap to GameConfig"
```

---

### Task 2: Apply rake in MCCFR `terminal_value`

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:234-261` (terminal_value function)
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:167-180` (traverse_external call site)

**Step 1: Add rake parameters to `terminal_value`**

Change the signature and body:

```rust
/// Compute payoff at a terminal node from the traverser's perspective.
///
/// Rake is deducted from the winner's share at both fold and showdown
/// terminals: `rake = min(pot * rake_rate, rake_cap)`.
fn terminal_value(
    kind: TerminalKind,
    invested: &[f64; 2],
    traverser: u8,
    deal: &Deal,
    rake_rate: f64,
    rake_cap: f64,
) -> f64 {
    let t = traverser as usize;
    let o = 1 - t;
    let pot = invested[0] + invested[1];
    let rake = if rake_rate > 0.0 {
        (pot * rake_rate).min(if rake_cap > 0.0 { rake_cap } else { f64::MAX })
    } else {
        0.0
    };
    match kind {
        TerminalKind::Fold { winner } => {
            if winner == traverser {
                invested[o] - rake
            } else {
                -invested[t]
            }
        }
        TerminalKind::Showdown => {
            let rank_t = rank_hand(deal.hole_cards[t], &deal.board);
            let rank_o = rank_hand(deal.hole_cards[o], &deal.board);
            match rank_t.cmp(&rank_o) {
                Ordering::Greater => invested[o] - rake,
                Ordering::Less => -invested[t],
                Ordering::Equal => -rake / 2.0,
            }
        }
    }
}
```

Key behavior matching range-solver:
- Winner pays rake from their winnings
- Loser always loses their full investment (no rake deduction)
- Ties split the rake cost equally

**Step 2: Thread rake through `traverse_external`**

Add `rake_rate: f64` and `rake_cap: f64` parameters to `traverse_external` (after `rng`):

```rust
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
) -> (f64, PruneStats) {
```

Update the `Terminal` match arm:

```rust
GameNode::Terminal { kind, invested, .. } => {
    (terminal_value(*kind, invested, traverser, &deal.deal, rake_rate, rake_cap), PruneStats::default())
}
```

Update all recursive `traverse_external` calls within the function to pass `rake_rate, rake_cap` as trailing args. There are calls in:
- `Chance` arm (line ~184)
- `traverse_traverser` helper
- `traverse_opponent` helper

**Step 3: Thread rake through helper functions**

Add `rake_rate: f64, rake_cap: f64` to:
- `traverse_traverser(...)` signature and its recursive calls
- `traverse_opponent(...)` signature and its recursive calls

Each calls `traverse_external` recursively — pass the two rake params through.

**Step 4: Update the caller of `traverse_external`**

Find where `traverse_external` is called from the training loop. This will be in the MCCFR trainer or the parallel batch runner. Pass `config.game.rake_rate` and `config.game.rake_cap` from the `BlueprintV2Config`.

Search with:
```bash
grep -rn "traverse_external" crates/core/src/blueprint_v2/
```

**Step 5: Update existing tests**

Update existing `terminal_value` test calls to include `0.0, 0.0` for no rake:

```rust
#[test]
fn terminal_fold_payoff() {
    let invested = [0.5, 1.0];
    let deal = make_deal();
    let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 0, &deal, 0.0, 0.0);
    assert!((v - (-0.5)).abs() < 1e-10, "SB loses 0.5 on fold, got {v}");
    let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 1, &deal, 0.0, 0.0);
    assert!((v - 0.5).abs() < 1e-10, "BB wins 0.5 on SB fold, got {v}");
}
```

**Step 6: Add rake-specific terminal value tests**

```rust
#[test]
fn terminal_fold_with_rake() {
    let invested = [5.0, 5.0]; // pot = 10
    let deal = make_deal();
    // 5% rake, cap 1.0 → rake = min(10*0.05, 1.0) = 0.5
    let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 1, &deal, 0.05, 1.0);
    assert!((v - 4.5).abs() < 1e-10, "Winner gets 5 - 0.5 rake = 4.5, got {v}");
    // Loser still loses full investment
    let v = terminal_value(TerminalKind::Fold { winner: 1 }, &invested, 0, &deal, 0.05, 1.0);
    assert!((v - (-5.0)).abs() < 1e-10, "Loser loses 5.0, got {v}");
}

#[test]
fn terminal_showdown_with_rake_cap() {
    let invested = [50.0, 50.0]; // pot = 100
    let deal = make_deal_with_winner(); // needs a deal where player 0 wins
    // 5% rake, cap 3.0 → rake = min(100*0.05, 3.0) = 3.0 (capped)
    let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.05, 3.0);
    assert!((v - 47.0).abs() < 1e-10, "Winner gets 50 - 3.0 capped rake = 47, got {v}");
}

#[test]
fn terminal_tie_with_rake() {
    let invested = [5.0, 5.0]; // pot = 10
    let deal = make_deal(); // tie deal (existing helper makes equal hands)
    // 5% rake, no cap → rake = 0.5
    let v = terminal_value(TerminalKind::Showdown, &invested, 0, &deal, 0.05, 0.0);
    assert!((v - (-0.25)).abs() < 1e-10, "Tie splits 0.5 rake: -0.25 each, got {v}");
}
```

Note: You may need to create a `make_deal_with_winner()` helper that gives player 0 a better hand than player 1. Look at the existing `make_deal()` helper in the test module and create a variant where one player clearly wins.

**Step 7: Run tests**

```bash
cargo test -p poker-solver-core -- blueprint_v2::mccfr
```

Expected: All tests pass including new rake tests.

**Step 8: Run full suite**

```bash
cargo test
```

Expected: All tests pass (traverse_external callers updated in Step 4).

**Step 9: Commit**

```bash
git add -A && git commit -m "feat: apply rake deduction in MCCFR terminal_value"
```

---

### Task 3: Expose `name` and rake in `BundleInfo` and blueprint list

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:128-135` (BundleInfo struct)
- Modify: `crates/tauri-app/src/exploration.rs:48-55` (BlueprintListEntry struct)
- Modify: `crates/tauri-app/src/exploration.rs:634-672` (try_make_blueprint_entry)
- Modify: `crates/tauri-app/src/exploration.rs:1653-1668` (BlueprintV2 arm of get_bundle_info_core)
- Modify: `crates/tauri-app/src/exploration.rs:569-576` (load_blueprint_v2_core BundleInfo)
- Modify: `frontend/src/types.ts:1-8` (BundleInfo TS type)
- Modify: `frontend/src/types.ts:230-235` (BlueprintListEntry TS type)

**Step 1: Add rake fields to `BundleInfo`**

```rust
pub struct BundleInfo {
    pub name: Option<String>,
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
    pub info_sets: usize,
    pub iterations: u64,
    pub preflop_only: bool,
    pub rake_rate: f64,
    pub rake_cap: f64,
}
```

**Step 2: Update all `BundleInfo` construction sites**

Every place that constructs a `BundleInfo` (search for `BundleInfo {` in `exploration.rs`) must add `rake_rate: 0.0, rake_cap: 0.0` — except the `BlueprintV2` arms which should use the config values:

For the `BlueprintV2` arm in `get_bundle_info_core` (~line 1653):
```rust
StrategySource::BlueprintV2 { config, strategy, tree, .. } => {
    let (decision_nodes, _, _) = tree.node_counts();
    BundleInfo {
        name: Some(config.game.name.clone()),
        stack_depth: config.game.stack_depth as u32,
        bet_sizes: vec![],
        info_sets: decision_nodes,
        iterations: strategy.iterations,
        preflop_only: true,
        rake_rate: config.game.rake_rate,
        rake_cap: config.game.rake_cap,
    }
}
```

For `load_blueprint_v2_core` (~line 569):
```rust
let info = BundleInfo {
    name: Some(config.game.name.clone()),
    stack_depth: config.game.stack_depth as u32,
    bet_sizes: vec![],
    info_sets: decision_nodes,
    iterations: strategy.iterations,
    preflop_only: true,
    rake_rate: config.game.rake_rate,
    rake_cap: config.game.rake_cap,
};
```

For all other arms (Bundle, Agent, PreflopSolve, SubgameSolve), add:
```rust
rake_rate: 0.0,
rake_cap: 0.0,
```

**Step 3: Use config `name` in `BlueprintListEntry`**

In `try_make_blueprint_entry` (~line 634), use the config's name instead of the directory name:

```rust
fn try_make_blueprint_entry(dir: &Path) -> Option<BlueprintListEntry> {
    if !dir.join("config.yaml").exists() {
        return None;
    }

    let (name, stack_depth) = match v2_bundle::load_config(dir) {
        Ok(config) => (config.game.name.clone(), config.game.stack_depth),
        Err(e) => {
            eprintln!("Warning: failed to parse config in {}: {e}", dir.display());
            let fallback = dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            (fallback, 0.0)
        }
    };
    // ... rest unchanged
```

**Step 4: Update TypeScript types**

In `frontend/src/types.ts`, update `BundleInfo`:

```typescript
export interface BundleInfo {
  name: string | null;
  stack_depth: number;
  bet_sizes: number[];
  info_sets: number;
  iterations: number;
  preflop_only: boolean;
  rake_rate: number;
  rake_cap: number;
}
```

**Step 5: Run and verify**

```bash
cargo test -p poker-solver-tauri-app 2>/dev/null; cargo build -p poker-solver-tauri-app
```

Expected: Compiles clean.

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: expose blueprint name and rake in BundleInfo/BlueprintListEntry"
```

---

### Task 4: Thread rake from blueprint to range solver in Tauri

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:22-32` (PostflopConfig struct)
- Modify: `crates/tauri-app/src/postflop.rs:34-47` (Default impl)
- Modify: `crates/tauri-app/src/postflop.rs:672-677` (TreeConfig construction)
- Modify: `frontend/src/types.ts:159-168` (PostflopConfig TS type)

**Step 1: Add rake fields to Rust `PostflopConfig`**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopConfig {
    pub oop_range: String,
    pub ip_range: String,
    pub pot: i32,
    pub effective_stack: i32,
    pub oop_bet_sizes: String,
    pub oop_raise_sizes: String,
    pub ip_bet_sizes: String,
    pub ip_raise_sizes: String,
    pub rake_rate: f64,
    pub rake_cap: f64,
}
```

Update `Default`:

```rust
impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            oop_range: "QQ+,AKs,AKo".to_string(),
            ip_range: "TT+,AQs+,AKo".to_string(),
            pot: 30,
            effective_stack: 170,
            oop_bet_sizes: "25%,33%,75%,a".to_string(),
            oop_raise_sizes: "a".to_string(),
            ip_bet_sizes: "25%,33%,75%,a".to_string(),
            ip_raise_sizes: "a".to_string(),
            rake_rate: 0.0,
            rake_cap: 0.0,
        }
    }
}
```

**Step 2: Use config values in TreeConfig construction**

Replace the hardcoded values at ~line 676:

```rust
let tree_config = TreeConfig {
    initial_state,
    starting_pot: config.pot,
    effective_stack: config.effective_stack,
    rake_rate: config.rake_rate,
    rake_cap: config.rake_cap,
    // ... rest unchanged
};
```

**Step 3: Update TypeScript `PostflopConfig`**

```typescript
export interface PostflopConfig {
  oop_range: string;
  ip_range: string;
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
  rake_rate: number;
  rake_cap: number;
}
```

**Step 4: Build and verify**

```bash
cargo build -p poker-solver-tauri-app
```

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: thread rake from PostflopConfig to range solver TreeConfig"
```

---

### Task 5: Frontend — display name and rake, populate from blueprint

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx` (config card display, config modal inputs, blueprint init)
- Modify: `frontend/src/types.ts:142-155` (BlueprintConfig type)

**Step 1: Add rake to `BlueprintConfig` TypeScript type**

```typescript
export interface BlueprintConfig {
  oop_range: string;
  ip_range: string;
  oop_weights: number[];
  ip_weights: number[];
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
  blueprint_dir: string;
  board?: string[];
  rake_rate: number;
  rake_cap: number;
}
```

**Step 2: Populate rake from blueprint in PostflopExplorer init**

In `PostflopExplorer.tsx` (~line 45-54), where `blueprintConfig` is converted to `PostflopConfig`:

```typescript
if (blueprintConfig) {
    return {
        oop_range: blueprintConfig.oop_range,
        ip_range: blueprintConfig.ip_range,
        pot: blueprintConfig.pot,
        effective_stack: blueprintConfig.effective_stack,
        oop_bet_sizes: blueprintConfig.oop_bet_sizes,
        oop_raise_sizes: blueprintConfig.oop_raise_sizes,
        ip_bet_sizes: blueprintConfig.ip_bet_sizes,
        ip_raise_sizes: blueprintConfig.ip_raise_sizes,
        rake_rate: blueprintConfig.rake_rate,
        rake_cap: blueprintConfig.rake_cap,
    };
```

**Step 3: Display rake in config card**

In the config card display (~line 388-396), add rake info when non-zero:

```tsx
<div className="postflop-config-label">{blueprintConfig ? 'Blueprint' : 'Config'}</div>
<div className="postflop-config-summary">
  {config.pot} pot / {config.effective_stack} eff
  {config.rake_rate > 0 && ` / ${(config.rake_rate * 100).toFixed(1)}% rake`}
</div>
```

**Step 4: Add rake inputs to config modal**

Find the config modal in `PostflopExplorer.tsx` (look for input fields for pot, effective_stack, bet sizes). Add two new input fields after effective_stack:

```tsx
<label>
  Rake Rate (%)
  <input
    type="number"
    min="0"
    max="100"
    step="0.5"
    value={config.rake_rate * 100}
    onChange={(e) => setConfig({ ...config, rake_rate: parseFloat(e.target.value) / 100 || 0 })}
  />
</label>
<label>
  Rake Cap (chips)
  <input
    type="number"
    min="0"
    step="0.5"
    value={config.rake_cap}
    onChange={(e) => setConfig({ ...config, rake_cap: parseFloat(e.target.value) || 0 })}
  />
</label>
```

**Step 5: Verify the Rust backend sends rake in BlueprintConfig**

Check `exploration.rs` where `BlueprintConfig` is constructed for the frontend. Search for where `BlueprintConfig` or the equivalent response is built when a blueprint is loaded for postflop exploration. Ensure `rake_rate` and `rake_cap` from the `BlueprintV2Config.game` are included.

**Step 6: Build frontend**

```bash
cd frontend && npm run build
```

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: display blueprint name and rake in explorer UI"
```

---

### Task 6: Update sample configs and explorer display

**Files:**
- Modify: `sample_configurations/blueprint_v2_toy.yaml`
- Modify: `sample_configurations/blueprint_v2_realistic.yaml`
- Modify: `sample_configurations/blueprint_v2_with_tui.yaml`
- Modify: `frontend/src/Explorer.tsx` (~line 1269-1278, blueprint picker display)

**Step 1: Add `name` to all sample YAML configs**

For `blueprint_v2_toy.yaml`:
```yaml
game:
  name: "HU 10BB Toy"
  players: 2
  ...
```

For `blueprint_v2_realistic.yaml`:
```yaml
game:
  name: "HU 100BB Realistic"
  players: 2
  ...
```

For `blueprint_v2_with_tui.yaml`:
```yaml
game:
  name: "HU 100BB TUI"
  players: 2
  ...
```

No need to add `rake_rate`/`rake_cap` — they default to 0.0.

**Step 2: Use blueprint name in Explorer picker**

In `Explorer.tsx` (~line 1269), the blueprint picker currently shows `bp.name` which comes from the directory name via `BlueprintListEntry`. Now that `try_make_blueprint_entry` uses the config's `name` field, this will automatically show the config name. Verify the picker item renders correctly — it should already display `bp.name`.

**Step 3: Use `bundleInfo.name` in the Explorer header**

Find where `bundleInfo` is displayed in `Explorer.tsx` and ensure the name is shown prominently. Search for how `bundleInfo` is rendered and make sure `bundleInfo.name` appears as a header/label.

**Step 4: Run full test suite**

```bash
cargo test
```

Expected: All tests pass. Config changes are backward-compatible via serde defaults.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add names to sample configs, use in explorer picker"
```

---

### Task 7: Final validation

**Step 1: Run full test suite**

```bash
cargo test
```

**Step 2: Run clippy**

```bash
cargo clippy
```

**Step 3: Verify frontend builds**

```bash
cd frontend && npm run build
```

**Step 4: Manual smoke test (if devserver available)**

```bash
cargo run -p poker-solver-devserver &
cd frontend && npm run dev
```

Open browser, load a blueprint, verify:
- Blueprint name shows in picker and header
- Rake rate/cap show in config card (if non-zero)
- Range solver config modal has rake inputs
- Solving works with rake values

**Step 5: Commit any final fixes**

```bash
git add -A && git commit -m "chore: final cleanup for rake support"
```
