# Per-Flop Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Enable per-flop blueprint training with locked preflop strategy, using per-flop turn/river buckets. The per-flop clustering pipeline already exists — this plan adds the training and integration pieces.

**Architecture:** Per-flop clustering already works via `run_per_flop_pipeline` and `cluster_single_flop` in `cluster_pipeline.rs`. This plan adds: (1) config fields for mode/flops selection, (2) locked preflop + fixed flop in MCCFR training, (3) Tauri integration for loading per-flop data during subgame solving.

**Tech Stack:** Rust, existing MCCFR trainer, existing per-flop clustering pipeline, `BlueprintV2Strategy`, `AllBuckets`

**Design doc:** `docs/plans/2026-03-31-per-flop-pipeline-design.md`

**Existing code to build on:**
- `cluster_pipeline.rs:1012` — `cluster_single_flop()` clusters turn/river for one flop
- `cluster_pipeline.rs:1197` — `run_per_flop_pipeline()` batches all 1,755 canonical flops
- `config.rs:51` — `PerFlopConfig` struct with turn/river bucket counts
- `config.rs:90` — `ClusteringConfig.per_flop: Option<PerFlopConfig>`
- `trainer/src/main.rs:541` — CLI already dispatches per-flop clustering when `per_flop` is set

---

### Task 1: Add `mode` and `flops` config fields to ClusteringConfig

Refactor the existing `per_flop: Option<PerFlopConfig>` into a cleaner `mode`/`flops` design.

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs:71-91`
- Modify: `crates/trainer/src/main.rs:541-592` (adjust CLI dispatch)

**Step 1: Write the failing test**

Add in `config.rs` tests:

```rust
#[test]
fn test_per_flop_mode_config() {
    let yaml = r#"
game:
  name: test
  players: 2
  stack_depth: 200
  small_blind: 1
  big_blind: 2
clustering:
  mode: per_flop
  flops: ["QhTh7d", "AsKs2d"]
  output_dir: ./per_flop_buckets
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 500
  river:
    buckets: 500
"#;
    let cfg: BlueprintV2Config = serde_yaml::from_str(yaml).expect("parse per_flop config");
    assert_eq!(cfg.clustering.mode, ClusteringMode::PerFlop);
    assert_eq!(cfg.clustering.flops.as_ref().unwrap().len(), 2);
    assert_eq!(cfg.clustering.output_dir.as_ref().unwrap(), "./per_flop_buckets");
}

#[test]
fn test_global_mode_default() {
    let yaml = r#"
game:
  name: test
  players: 2
  stack_depth: 200
  small_blind: 1
  big_blind: 2
clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200
"#;
    let cfg: BlueprintV2Config = serde_yaml::from_str(yaml).expect("parse global config");
    assert_eq!(cfg.clustering.mode, ClusteringMode::Global);
}
```

**Step 2: Implement**

Add to `ClusteringConfig`:

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringMode {
    #[default]
    Global,
    PerFlop,
}

// In ClusteringConfig:
#[serde(default)]
pub mode: ClusteringMode,
/// Flops to cluster in per_flop mode. "all" or a list of flop strings.
/// Ignored in global mode.
#[serde(default)]
pub flops: Option<Vec<String>>,
/// Output directory for per-flop bucket files. Used in per_flop mode.
#[serde(default)]
pub output_dir: Option<String>,
```

Keep `per_flop: Option<PerFlopConfig>` for backward compatibility (the turn/river bucket counts for per-flop mode). Or migrate: in per-flop mode, use `turn.buckets` and `river.buckets` from the existing street configs.

Update `main.rs` CLI dispatch to use the new `mode` field instead of `per_flop.is_some()`. Support `flops: ["QhTh7d"]` by parsing flop strings into `[Card; 3]` and passing as `flop_limit` or a filter to `run_per_flop_pipeline`.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core -- test_per_flop_mode`
Expected: PASS

**Step 4: Commit**

```bash
git commit -m "feat: add mode/flops/output_dir config fields for per-flop clustering"
```

---

### Task 2: Add `lock_preflop` config and skip preflop regret updates

When `lock_preflop: true`, the MCCFR traversal uses the blueprint's average strategy at preflop nodes but does NOT update regrets or strategy sums there.

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs` (add `lock_preflop` to `TrainingConfig`)
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` (skip updates at preflop nodes)
- Modify: `crates/core/src/blueprint_v2/trainer.rs` (load preflop strategy from global blueprint)

**Step 1: Write the failing test**

```rust
#[test]
fn lock_preflop_skips_regret_updates_at_preflop() {
    let mut config = toy_config();
    config.training.lock_preflop = true;
    config.training.iterations = Some(100);
    let mut trainer = BlueprintTrainer::new(config).expect("trainer");
    trainer.train().expect("train");
    // Preflop regrets should be zero (never updated).
    // Postflop regrets should be non-zero.
    // ... (check specific infoset regrets at preflop vs postflop nodes)
}
```

**Step 2: Implement**

Add to `TrainingConfig`:

```rust
/// When true, preflop strategy is locked from the global blueprint.
/// Regrets and strategy sums at preflop decision nodes are NOT updated.
/// Used for per-flop blueprint training.
#[serde(default)]
pub lock_preflop: bool,
```

In `traverse_traverser` and `traverse_opponent` in `mccfr.rs`, check the node's street. If `street == Street::Preflop && lock_preflop`:
- **Traverser nodes:** Still traverse all children (to compute cfv for the traverser), but skip `storage.add_regret()` and `storage.set_prediction()` calls.
- **Opponent nodes:** Use the average strategy (from storage) but skip `storage.add_strategy_sum()`.
- The current strategy at preflop nodes comes from the loaded global blueprint (seeded into storage at startup).

The `lock_preflop` flag needs to be passed through the traversal. The simplest way: add it to the existing parameter set or as a field on the trainer that's read during traversal.

**Step 3: Commit**

```bash
git commit -m "feat: add lock_preflop to skip regret updates at preflop nodes"
```

---

### Task 3: Add `fixed_flop` config and constrained deal sampling

When a specific flop is set, `sample_deal_with_rng` always deals that flop instead of random.

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs` (add `fixed_flop` to `TrainingConfig`)
- Modify: `crates/core/src/blueprint_v2/trainer.rs:53-66` (modify `sample_deal_with_rng`)

**Step 1: Write the failing test**

```rust
#[test]
fn fixed_flop_always_deals_specified_board() {
    let flop = parse_flop("QhTh7d"); // helper to parse "QhTh7d" -> [Card; 3]
    let mut rng = SmallRng::seed_from_u64(42);
    for _ in 0..100 {
        let deal = sample_deal_with_fixed_flop(&mut rng, &flop);
        assert_eq!(deal.board[0], flop[0]);
        assert_eq!(deal.board[1], flop[1]);
        assert_eq!(deal.board[2], flop[2]);
        // Turn and river should be random but not overlap with flop or hole cards
        assert!(!flop.contains(&deal.board[3]));
        assert!(!flop.contains(&deal.board[4]));
    }
}
```

**Step 2: Implement**

Add to `TrainingConfig`:

```rust
/// Fix the flop to these specific cards for per-flop training.
/// Format: "QhTh7d". When set, every deal uses this flop.
#[serde(default)]
pub fixed_flop: Option<String>,
```

Add a new deal sampling function:

```rust
fn sample_deal_with_fixed_flop(rng: &mut impl Rng, flop: &[Card; 3]) -> Deal {
    let mut deck: Vec<Card> = CANONICAL_DECK.iter()
        .filter(|c| !flop.contains(c))
        .copied()
        .collect(); // 49 cards
    // Shuffle for hole cards + turn + river
    for i in 0..6 { // 4 hole cards + 2 remaining board cards
        let j = rng.random_range(i..deck.len());
        deck.swap(i, j);
    }
    Deal {
        hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
        board: [flop[0], flop[1], flop[2], deck[4], deck[5]],
    }
}
```

In the trainer's `sample_deal` method, check `config.training.fixed_flop` and dispatch accordingly.

**Step 3: Commit**

```bash
git commit -m "feat: add fixed_flop config for per-flop deal sampling"
```

---

### Task 4: Add `preflop_blueprint` config and load preflop strategy on startup

When `lock_preflop: true`, the trainer needs to load the global blueprint's preflop strategy and seed it into storage before training begins.

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs` (add `preflop_blueprint` path)
- Modify: `crates/core/src/blueprint_v2/trainer.rs` (load and seed preflop strategy)

**Step 1: Implement**

Add to `TrainingConfig`:

```rust
/// Path to global blueprint snapshot directory for preflop strategy.
/// Required when `lock_preflop` is true.
#[serde(default)]
pub preflop_blueprint: Option<String>,
```

In `BlueprintTrainer::new()`, when `lock_preflop` is true:
1. Load `BlueprintV2Strategy` from `preflop_blueprint` path
2. For each preflop decision node, write the blueprint's average strategy into the storage's strategy sums (same as seeding, but only preflop nodes)
3. The locked preflop strategy is then used by the MCCFR traversal at preflop nodes

**Step 2: Commit**

```bash
git commit -m "feat: load and seed preflop strategy from global blueprint"
```

---

### Task 5: Per-flop game tree without flop buckets

In per-flop mode, flop decision nodes should use individual combos (no bucketing) since the flop is fixed. The game tree needs ~1000 "buckets" (one per valid combo) at flop nodes instead of the configured flop bucket count.

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs` (bucket count override for per-flop)

**Step 1: Implement**

When `fixed_flop` is set, override `bucket_counts[Street::Flop]` to be the number of valid combos for that flop (typically ~1000). Each combo maps to its own "bucket" (its combo index). The `AllBuckets::get_bucket()` for flop returns the combo's index rather than a clustered bucket.

This may require a special `AllBuckets` mode where flop lookups return combo indices. The simplest approach: set `flop_buckets` to 1326 and use the combo index directly. The storage allocates slots for 1326 flop buckets per node, most unused (blocked combos). Memory overhead is modest for a single-flop tree.

**Step 2: Commit**

```bash
git commit -m "feat: use per-combo flop bucketing in fixed_flop mode"
```

---

### Task 6: Tauri integration — load per-flop blueprint and buckets

When the explorer opens a flop spot, check for per-flop data and use it for seeding and boundary rollouts.

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` (check for per-flop data on bundle load)
- Modify: `crates/tauri-app/src/game_session.rs` (use per-flop data in solve)

**Step 1: Implement**

On bundle load (in `exploration.rs`), check if a per-flop directory exists alongside the bundle:
```
{bundle_dir}/per_flop/{canonical_flop_key}/strategy.bin
{bundle_dir}/per_flop/{canonical_flop_key}/turn.buckets
{bundle_dir}/per_flop/{canonical_flop_key}/river.buckets
```

If found, store in the exploration state. When `game_solve_core` runs:
1. Canonicalize the current flop
2. Check for per-flop data matching that canonical flop
3. If found:
   - Seed with the per-flop blueprint strategy (instead of global)
   - Create `AllBuckets` with per-flop turn/river bucket files for boundary rollouts
4. If not found: fall back to global (current behavior)

**Step 2: Commit**

```bash
git commit -m "feat: load per-flop blueprint and buckets in Tauri explorer"
```

---

### Task 7: End-to-end integration test

**Files:**
- Create: test in `crates/core/tests/` or `crates/trainer/`

**Step 1: Write test**

```rust
#[test]
fn per_flop_training_produces_differentiated_strategies() {
    // 1. Build a small global blueprint (few iterations)
    // 2. Run per-flop clustering for one test flop
    // 3. Train per-flop blueprint with lock_preflop + fixed_flop
    // 4. Verify:
    //    - Preflop regrets are zero (locked)
    //    - Postflop regrets are non-zero (trained)
    //    - Strategy differentiates flush draws from non-flush draws on the fixed flop
}
```

**Step 2: Commit**

```bash
git commit -m "test: add per-flop training end-to-end integration test"
```
