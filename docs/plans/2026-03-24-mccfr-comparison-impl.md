# Phase 2: MCCFR Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add an MCCFR solver adapter to the convergence harness that runs the blueprint MCCFR solver on Flop Poker, lifts strategies to combo-level, and compares against the exact baseline.

**Architecture:** The MCCFR adapter creates a `BlueprintTrainer` (from core) to get its game tree and storage, then runs its own training loop calling `traverse_external()` directly. Deals are sampled with a fixed flop (QhJdTh). Buckets use 169 canonical preflop hand indices for all streets. At configurable checkpoints, the adapter lifts the bucket strategy into a range-solver game tree and computes exploitability via best-response.

**Tech Stack:** poker-solver-core (BlueprintTrainer, GameTree, BlueprintStorage, traverse_external, CanonicalHand), range-solver (PostFlopGame, lock_current_strategy, compute_exploitability)

---

### Task 1: Fixed-flop deal sampling

**Files:**
- Create: `crates/convergence-harness/src/solvers/mccfr.rs`
- Modify: `crates/convergence-harness/src/solvers/mod.rs`

We need a deal sampler that always produces QhJdTh as the flop, with random hole cards and turn/river.

**Key references:**
- `crates/core/src/blueprint_v2/trainer.rs:52-65` — `sample_deal_with_rng()` for the deal format
- `crates/core/src/blueprint_v2/mccfr.rs:45-55` — `Deal` struct: `hole_cards: [[Card; 2]; 2], board: [Card; 5]`
- `crates/core/src/game/card.rs` — `Card` struct with `value: Value` and `suit: Suit` enums
- `crates/core/src/hands.rs:69-72` — `CanonicalHand::from_cards(card1, card2).index()` returns 0-168

**Step 1: Add module declaration**

In `crates/convergence-harness/src/solvers/mod.rs`, add:
```rust
pub mod mccfr;
```

**Step 2: Write fixed-flop deal sampler and test**

Create `crates/convergence-harness/src/solvers/mccfr.rs`:

```rust
use poker_solver_core::blueprint_v2::mccfr::{Deal, DealWithBuckets};
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::game::card::{Card, Value, Suit};
use rand::Rng;

/// The fixed flop cards: Qh Jd Th
fn flop_cards() -> [Card; 3] {
    [
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Ten, Suit::Heart),
    ]
}

/// Sample a deal with a fixed flop (QhJdTh).
/// Hole cards, turn, and river are random from the remaining 49 cards.
fn sample_fixed_flop_deal(rng: &mut impl Rng) -> Deal {
    let blocked = flop_cards();
    let mut deck: Vec<Card> = Vec::with_capacity(49);
    for card_id in 0..52u8 {
        let card = Card::from_id(card_id);
        if !blocked.contains(&card) {
            deck.push(card);
        }
    }

    // Partial Fisher-Yates: shuffle first 6 positions (2+2 hole + turn + river)
    for i in 0..6 {
        let j = rng.random_range(i..deck.len());
        deck.swap(i, j);
    }

    Deal {
        hole_cards: [
            [deck[0], deck[1]],
            [deck[2], deck[3]],
        ],
        board: [blocked[0], blocked[1], blocked[2], deck[4], deck[5]],
    }
}

/// Assign canonical preflop hand index as bucket for ALL streets.
/// This ignores board interaction — pipeline validation only.
fn canonical_buckets(deal: &Deal) -> [[u16; 4]; 2] {
    let mut result = [[0u16; 4]; 2];
    for (player, row) in result.iter_mut().enumerate() {
        let hole = deal.hole_cards[player];
        let hand = CanonicalHand::from_cards(hole[0], hole[1]);
        let idx = hand.index() as u16;
        // Same bucket for all streets
        row[0] = idx; // preflop
        row[1] = idx; // flop
        row[2] = idx; // turn
        row[3] = idx; // river
    }
    result
}
```

**IMPORTANT:** Check the exact Card API in `crates/core/src/game/card.rs`. The `Card` type may use `Card::new(Value, Suit)` or `Card { value, suit }` or `Card::from_id(u8)`. Read the file to confirm. Also check how `Deal` is imported — it may be at `poker_solver_core::blueprint_v2::mccfr::Deal` or re-exported elsewhere.

**Step 3: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_fixed_flop_deal_has_correct_board() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let expected = flop_cards();
        assert_eq!(deal.board[0], expected[0]);
        assert_eq!(deal.board[1], expected[1]);
        assert_eq!(deal.board[2], expected[2]);
    }

    #[test]
    fn test_fixed_flop_deal_no_duplicate_cards() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let mut all_cards = vec![
            deal.hole_cards[0][0], deal.hole_cards[0][1],
            deal.hole_cards[1][0], deal.hole_cards[1][1],
            deal.board[0], deal.board[1], deal.board[2],
            deal.board[3], deal.board[4],
        ];
        all_cards.sort();
        all_cards.dedup();
        assert_eq!(all_cards.len(), 9, "All 9 cards should be unique");
    }

    #[test]
    fn test_canonical_buckets_same_for_all_streets() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let buckets = canonical_buckets(&deal);
        for player in 0..2 {
            // All 4 streets should have the same bucket
            assert_eq!(buckets[player][0], buckets[player][1]);
            assert_eq!(buckets[player][1], buckets[player][2]);
            assert_eq!(buckets[player][2], buckets[player][3]);
            // Bucket should be a valid canonical hand index
            assert!(buckets[player][0] < 169);
        }
    }
}
```

**Step 4: Verify tests pass**

Run: `cargo test -p convergence-harness mccfr -- --nocapture`

**Step 5: Commit**

```bash
git add crates/convergence-harness/src/solvers/mccfr.rs crates/convergence-harness/src/solvers/mod.rs
git commit -m "feat(convergence-harness): fixed-flop deal sampling and canonical bucketing"
```

---

### Task 2: MCCFR solver adapter — construction and solve_step

**Files:**
- Modify: `crates/convergence-harness/src/solvers/mccfr.rs`
- Modify: `crates/convergence-harness/Cargo.toml` (may need `rand`, `rayon`)

Build the `MccfrSolver` struct that wraps a `BlueprintTrainer` and implements `ConvergenceSolver`.

**Key references:**
- `crates/core/src/blueprint_v2/trainer.rs:210-234` — `BlueprintTrainer::new(config)` builds tree + storage
- `crates/core/src/blueprint_v2/config.rs` — `BlueprintV2Config`, `GameConfig`, `ClusteringConfig`, `TrainingConfig`, `ActionAbstractionConfig`, `SnapshotConfig`
- `crates/core/src/blueprint_v2/mccfr.rs:625` — `traverse_external()` free function
- `crates/core/src/blueprint_v2/game_tree.rs:165` — `GameTree::build()`
- `crates/convergence-harness/src/game.rs` — `FlopPokerConfig`
- `crates/convergence-harness/src/solver_trait.rs` — `ConvergenceSolver` trait

**Step 1: Build BlueprintV2Config from FlopPokerConfig**

Write a function that translates our game config into a `BlueprintV2Config`:

```rust
fn build_mccfr_config(config: &FlopPokerConfig) -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "Flop Poker convergence test".into(),
            players: 2,
            stack_depth: config.effective_stack as f64,
            small_blind: 0.5,
            big_blind: 1.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: StreetClusterConfig { buckets: 169, ..Default::default() },
            flop: StreetClusterConfig { buckets: 169, ..Default::default() },
            turn: StreetClusterConfig { buckets: 169, ..Default::default() },
            river: StreetClusterConfig { buckets: 169, ..Default::default() },
            seed: 42,
            kmeans_iterations: 0,
            ..Default::default()
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![vec!["2bb".into()]],  // Trivial: single preflop action
            flop: vec![vec![0.67]],    // 67% pot bet, matching range-solver
            turn: vec![vec![0.67]],
            river: vec![vec![0.67]],
        },
        training: TrainingConfig {
            cluster_path: None,  // No bucket files — we assign buckets manually
            iterations: Some(1_000_000),
            batch_size: 1000,
            lcfr_warmup_iterations: 10_000,
            lcfr_discount_interval: 10_000,
            prune_after_iterations: 50_000,
            prune_threshold: -300,
            prune_explore_pct: 0.05,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            ..Default::default()
        },
        snapshots: SnapshotConfig {
            output_dir: "/tmp/convergence-harness-mccfr".into(),
            resume: false,
            ..Default::default()
        },
    }
}
```

**IMPORTANT:** Read the actual `Default` impls and field names in `config.rs` — the above is approximate. The config structs may not implement `Default`. If not, you must fill in every field explicitly.

**Step 2: Implement MccfrSolver struct**

```rust
pub struct MccfrSolver {
    trainer: BlueprintTrainer,
    config: FlopPokerConfig,
    iteration: u64,
    batch_size: u64,
    rng: StdRng,
}

impl MccfrSolver {
    pub fn new(config: FlopPokerConfig) -> Self {
        let mccfr_config = build_mccfr_config(&config);
        let mut trainer = BlueprintTrainer::new(mccfr_config);
        trainer.skip_bucket_validation = true;

        Self {
            trainer,
            config,
            iteration: 0,
            batch_size: 1000,
            rng: StdRng::seed_from_u64(42),
        }
    }
}
```

**Step 3: Implement solve_step — our own training loop**

```rust
impl ConvergenceSolver for MccfrSolver {
    fn name(&self) -> &str { "MCCFR (169 canonical buckets)" }

    fn solve_step(&mut self) {
        // Run one batch of iterations
        let tree = &self.trainer.tree;
        let storage = &self.trainer.storage;

        for _ in 0..self.batch_size {
            let mut rng = SmallRng::seed_from_u64(self.rng.random());
            let deal = sample_fixed_flop_deal(&mut rng);
            let buckets = canonical_buckets(&deal);
            let deal = DealWithBuckets { deal, buckets };

            // Traverse for both players (external sampling)
            traverse_external(
                tree, storage, &deal, 0, tree.root,
                false, 0, &mut rng, 0.0, 0.0, None, None,
            );
            traverse_external(
                tree, storage, &deal, 1, tree.root,
                false, 0, &mut rng, 0.0, 0.0, None, None,
            );
        }

        self.iteration += self.batch_size;
    }

    fn iterations(&self) -> u64 { self.iteration }

    fn average_strategy(&self) -> StrategyMap {
        StrategyMap::new()  // Stub — filled in Task 3
    }

    fn combo_evs(&self) -> ComboEvMap {
        ComboEvMap::new()  // Stub
    }

    fn self_reported_metrics(&self) -> SolverMetrics {
        SolverMetrics::default()  // Stub — filled in Task 4
    }
}
```

**NOTE:** The `traverse_external` function signature has many parameters. Read `crates/core/src/blueprint_v2/mccfr.rs:625-640` for the exact signature. Key params:
- `prune: bool` — set `false` initially (no pruning)
- `prune_threshold: i32` — `0` when not pruning
- `rake_rate: f64` — `0.0`
- `rake_cap: f64` — `0.0`
- `ev_tracker: Option<&ScenarioEvTracker>` — `None`
- `full_ev_tracker: Option<&FullTreeEvTracker>` — `None`

**Step 4: Write test**

```rust
#[test]
fn test_mccfr_solver_runs_iterations() {
    let config = FlopPokerConfig::default();
    let mut solver = MccfrSolver::new(config);
    assert_eq!(solver.iterations(), 0);

    solver.solve_step();
    assert_eq!(solver.iterations(), 1000); // batch_size = 1000
}
```

**Step 5: Verify and commit**

Run: `cargo test -p convergence-harness mccfr`

```bash
git add crates/convergence-harness/
git commit -m "feat(convergence-harness): MccfrSolver with custom training loop"
```

---

### Task 3: Strategy lifting — bucket to combo

**Files:**
- Modify: `crates/convergence-harness/src/solvers/mccfr.rs`

Implement `average_strategy()` on MccfrSolver: extract bucket-level strategies and lift to combo-level.

**Key references:**
- `crates/core/src/blueprint_v2/storage.rs:197` — `BlueprintStorage::average_strategy(node_idx, bucket) -> Vec<f64>` returns per-action probabilities for a bucket at a node
- `crates/core/src/blueprint_v2/game_tree.rs:64-73` — `GameTree` and `GameNode` enum (Decision, Chance, Terminal)
- `crates/convergence-harness/src/solver_trait.rs` — `StrategyMap = BTreeMap<u64, Vec<f32>>`
- `crates/range-solver/src/game/interpreter.rs:180` — `PostFlopGame::private_cards(player) -> &[(Card, Card)]`

**Strategy lifting procedure:**

For each decision node in the blueprint game tree:
1. Get `node.blueprint_decision_idx` (this is the storage index, not the arena index)
2. For each canonical hand bucket 0-168:
   - Get `storage.average_strategy(decision_idx, bucket)` → `Vec<f64>`
3. Build a combo-level strategy: for each combo in the range-solver's `private_cards(0)`:
   - Map the combo to its canonical hand index
   - Copy the bucket's strategy into this combo's slot

**Node ID mapping:** The blueprint tree and range-solver tree have different node structures. We need to map between them. The simplest approach: walk both trees in parallel, matching by action sequence. At each decision node, verify the action count matches, then extract/lift the strategy.

**This is complex.** For the initial implementation, extract the strategy at the BLUEPRINT tree level (node_idx → bucket → strategy), and provide a separate `lift_strategy()` function that maps it to the range-solver's combo-level format.

**Step 1: Implement strategy extraction from blueprint storage**

```rust
impl MccfrSolver {
    /// Extract the current average strategy from blueprint storage.
    /// Returns: BTreeMap<blueprint_decision_idx, BTreeMap<bucket, Vec<f64>>>
    fn extract_blueprint_strategy(&self) -> BTreeMap<u32, BTreeMap<u16, Vec<f64>>> {
        let mut result = BTreeMap::new();
        for (idx, node) in self.trainer.tree.nodes.iter().enumerate() {
            if let GameNode::Decision { blueprint_decision_idx: Some(did), .. } = node {
                let mut bucket_strategies = BTreeMap::new();
                for bucket in 0..169u16 {
                    let strat = self.trainer.storage.average_strategy(*did as u32, bucket);
                    bucket_strategies.insert(bucket, strat);
                }
                result.insert(*did, bucket_strategies);
            }
        }
        result
    }
}
```

**Step 2: Implement combo-level lifting**

Build a function that maps blueprint strategies to the range-solver's combo format:

```rust
/// Lift a bucket-level strategy to combo-level for the range-solver.
/// Returns a strategy vector compatible with PostFlopGame::lock_current_strategy().
fn lift_to_combo_strategy(
    bucket_strategy: &BTreeMap<u16, Vec<f64>>,
    private_cards: &[(Card, Card)],  // From range-solver
    num_actions: usize,
) -> Vec<f32> {
    let num_hands = private_cards.len();
    let mut result = vec![0.0f32; num_actions * num_hands];

    for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
        let hand = CanonicalHand::from_cards(
            /* convert range-solver Card to core Card */
        );
        let bucket = hand.index() as u16;

        if let Some(strat) = bucket_strategy.get(&bucket) {
            for (action_idx, &prob) in strat.iter().enumerate() {
                if action_idx < num_actions {
                    result[action_idx * num_hands + hand_idx] = prob as f32;
                }
            }
        } else {
            // Uniform fallback
            for action_idx in 0..num_actions {
                result[action_idx * num_hands + hand_idx] = 1.0 / num_actions as f32;
            }
        }
    }
    result
}
```

**IMPORTANT — Card type mismatch:** The range-solver uses its own `Card` type (u8, encoding: 4*rank+suit) while the core crate uses `poker_solver_core::game::card::Card` (struct with Value and Suit enums). You MUST convert between them. Read both card types and write a conversion function. Check `crates/range-solver/src/card.rs` for the range-solver's encoding and `crates/core/src/game/card.rs` for the core's.

**Step 3: Wire into ConvergenceSolver trait**

Fill in `average_strategy()` on MccfrSolver. For now, return the root node strategy only (full tree lifting is Task 5).

**Step 4: Test**

```rust
#[test]
fn test_mccfr_strategy_after_iterations() {
    let config = FlopPokerConfig::default();
    let mut solver = MccfrSolver::new(config);
    for _ in 0..10 {
        solver.solve_step();
    }
    let strategy = solver.average_strategy();
    assert!(!strategy.is_empty(), "Should have at least one node's strategy");
}
```

**Step 5: Commit**

```bash
git commit -m "feat(convergence-harness): strategy lifting from buckets to combos"
```

---

### Task 4: Exploitability computation via strategy injection

**Files:**
- Modify: `crates/convergence-harness/src/solvers/mccfr.rs`
- May create: `crates/convergence-harness/src/exploitability.rs`

Inject the MCCFR lifted strategy into a range-solver `PostFlopGame` and compute exploitability.

**Key references:**
- `crates/range-solver/src/game/query.rs:676-691` — `PostFlopGame::lock_current_strategy(&mut self, strategy: &[f32])` — locks strategy at current node. Panics if terminal, chance, or already solved.
- `crates/range-solver/src/utility.rs:359` — `compute_exploitability(&game) -> f32`
- `crates/convergence-harness/src/game.rs` — `build_flop_poker_game_with_config()`

**Procedure:**

```rust
pub fn compute_mccfr_exploitability(
    mccfr_solver: &MccfrSolver,
    config: &FlopPokerConfig,
) -> f64 {
    // 1. Build a range-solver game with same params
    let mut game = build_flop_poker_game_with_config(config).unwrap();
    game.allocate_memory(false);

    // 2. Walk the range-solver tree, locking MCCFR strategy at each decision node
    //    (parallel walk of both trees, matching by action sequence)
    lock_mccfr_strategy(&mut game, mccfr_solver);

    // 3. Compute exploitability
    compute_exploitability(&game) as f64
}
```

**Tree correspondence challenge:**

The range-solver tree (navigated via `play(action_idx)`) and the blueprint tree (arena of `GameNode`) have different structures. We need to walk both in sync:

1. Start at both roots
2. At a decision node: verify action counts match, lock the lifted strategy, then recurse into each child
3. At a chance node: iterate over possible cards, recurse
4. At terminal: return

The mapping function:

```rust
fn lock_mccfr_strategy(
    game: &mut PostFlopGame,
    solver: &MccfrSolver,
    // ... tree traversal state
) {
    if game.is_terminal_node() { return; }
    if game.is_chance_node() {
        let possible = game.possible_cards();
        for card in 0..52u8 {
            if possible & (1u64 << card) != 0 {
                game.play(card as usize);
                lock_mccfr_strategy(game, solver, ...);
                game.back_to_root();
                game.apply_history(&history);
            }
        }
        return;
    }

    // Decision node: lift and lock strategy
    let strategy = lift_to_combo_strategy(...);
    game.lock_current_strategy(&strategy);

    // Recurse into children
    let num_actions = game.available_actions().len();
    for action_idx in 0..num_actions {
        game.play(action_idx);
        lock_mccfr_strategy(game, solver, ...);
        game.back_to_root();
        game.apply_history(&history);
    }
}
```

**IMPORTANT:** This is the hardest part of the whole feature. The tree correspondence must be exact. If the blueprint tree has a different number of actions at any node (due to different all-in thresholds, etc.), the comparison is invalid. Add assertions to catch mismatches early.

**Step 1: Implement `compute_mccfr_exploitability`**

**Step 2: Test with a simple case**

```rust
#[test]
fn test_exploitability_after_mccfr_training() {
    let config = FlopPokerConfig { effective_stack: 10, bet_sizes: "a".into(), raise_sizes: "a".into(), ..Default::default() };
    let mut solver = MccfrSolver::new(config.clone());
    for _ in 0..50 {
        solver.solve_step();
    }
    let expl = compute_mccfr_exploitability(&solver, &config);
    assert!(expl > 0.0, "Exploitability should be positive");
    assert!(expl < 100.0, "Exploitability should be finite and reasonable");
}
```

**Step 3: Commit**

```bash
git commit -m "feat(convergence-harness): compute MCCFR exploitability via strategy injection"
```

---

### Task 5: Self-reported metrics

**Files:**
- Modify: `crates/convergence-harness/src/solvers/mccfr.rs`

Add strategy delta and avg positive regret to `self_reported_metrics()`.

**Key references:**
- `crates/core/src/blueprint_v2/storage.rs:275` — `strategy_delta(&self, prev_sums: &[i64]) -> (f64, f64)` returns (delta, pct_moving)
- `crates/core/src/blueprint_v2/storage.rs:315` — `avg_pos_regret(&self, iterations: u64) -> f64`

**Implementation:**

Add a `prev_strategy_sums: Vec<i64>` field to `MccfrSolver`. After each `solve_step()`, snapshot strategy sums. On `self_reported_metrics()`, compute delta.

```rust
fn self_reported_metrics(&self) -> SolverMetrics {
    let mut values = BTreeMap::new();
    if !self.prev_strategy_sums.is_empty() {
        let (delta, pct_moving) = self.trainer.storage.strategy_delta(&self.prev_strategy_sums);
        values.insert("strategy_delta".into(), delta);
        values.insert("pct_moving".into(), pct_moving);
    }
    let avg_regret = self.trainer.storage.avg_pos_regret(self.iteration);
    values.insert("avg_pos_regret".into(), avg_regret);
    SolverMetrics { values }
}
```

**Test and commit:**
```bash
git commit -m "feat(convergence-harness): MCCFR self-reported metrics (strategy delta, avg regret)"
```

---

### Task 6: run-solver CLI subcommand

**Files:**
- Modify: `crates/convergence-harness/src/main.rs`
- Modify: `crates/convergence-harness/src/lib.rs`

Add the `RunSolver` subcommand to the CLI.

**Step 1: Add subcommand to CLI**

```rust
/// Run a solver against the Flop Poker game and compare to baseline
RunSolver {
    /// Solver to run
    #[arg(long)]
    solver: String,  // "mccfr" for now

    /// Maximum iterations
    #[arg(long, default_value_t = 1_000_000)]
    iterations: u64,

    /// Exploitability checkpoint iterations (comma-separated)
    #[arg(long, default_value = "1000,10000,100000,500000,1000000")]
    checkpoints: String,

    /// Path to baseline directory
    #[arg(long, default_value = "baselines/flop_poker_v1")]
    baseline_dir: String,

    /// Output directory for results
    #[arg(long, default_value = "results/mccfr_run")]
    output_dir: String,
},
```

**Step 2: Implement the handler**

The handler:
1. Parses checkpoints from comma-separated string to `Vec<u64>`
2. Creates `MccfrSolver` with default `FlopPokerConfig`
3. Runs a loop: call `solve_step()`, check if at a checkpoint
4. At each checkpoint: compute exploitability, print progress, record metrics
5. On completion: print strategy matrix, save results as `Baseline` format, auto-run comparison

**Step 3: Test the CLI parsing**

```rust
#[test]
fn parse_run_solver_defaults() {
    let cli = Cli::parse_from(["convergence-harness", "run-solver", "--solver", "mccfr"]);
    // verify defaults
}
```

**Step 4: Commit**

```bash
git commit -m "feat(convergence-harness): run-solver CLI subcommand with configurable checkpoints"
```

---

### Task 7: Integration test — full MCCFR pipeline

**Files:**
- Modify: `crates/convergence-harness/tests/integration.rs`

Test the complete pipeline: train MCCFR → extract strategy → compute exploitability → compare to baseline.

```rust
#[test]
fn test_mccfr_pipeline_end_to_end() {
    // Use a tiny config for speed
    let config = FlopPokerConfig {
        effective_stack: 10,
        bet_sizes: "a".into(),
        raise_sizes: "a".into(),
        ..Default::default()
    };

    // 1. Generate a baseline with exhaustive DCFR
    let baseline = generate_baseline_with_config(&config, 20, 10.0).unwrap();
    let dir = tempfile::TempDir::new().unwrap();
    baseline.save(dir.path()).unwrap();

    // 2. Run MCCFR solver
    let mut solver = MccfrSolver::new(config.clone());
    for _ in 0..10 {
        solver.solve_step(); // 10 * batch_size iterations
    }

    // 3. Compute exploitability
    let expl = compute_mccfr_exploitability(&solver, &config);
    assert!(expl > 0.0);

    // 4. Extract strategy and compare
    let strategy = solver.average_strategy();
    assert!(!strategy.is_empty());
}
```

**Step: Run and commit**

```bash
cargo test -p convergence-harness --test integration test_mccfr_pipeline -- --nocapture
git commit -m "feat(convergence-harness): MCCFR end-to-end integration test"
```

---

### Task 8: Clippy + smoke test

**Step 1: Run clippy**

```bash
cargo clippy -p convergence-harness
```

Fix any warnings.

**Step 2: Run all tests**

```bash
cargo test -p convergence-harness
```

**Step 3: Smoke test the CLI**

```bash
# Generate baseline first (if not already done)
cargo run -p convergence-harness --release -- generate-baseline --iterations 100

# Run MCCFR with small iteration count
cargo run -p convergence-harness --release -- run-solver --solver mccfr --iterations 10000 --checkpoints 1000,5000,10000
```

**Step 4: Commit**

```bash
git commit -m "chore(convergence-harness): clippy fixes and smoke test verification"
```

---

## Summary of Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 1 | Fixed-flop deal sampling | `sample_fixed_flop_deal()`, `canonical_buckets()` |
| 2 | MCCFR solver adapter | `MccfrSolver` struct with custom training loop |
| 3 | Strategy lifting | Bucket-to-combo mapping via `CanonicalHand` |
| 4 | Exploitability computation | `compute_mccfr_exploitability()` via `lock_current_strategy` |
| 5 | Self-reported metrics | Strategy delta, avg positive regret |
| 6 | run-solver CLI | `RunSolver` subcommand with configurable checkpoints |
| 7 | Integration test | Full MCCFR pipeline end-to-end |
| 8 | Clippy + smoke test | Final polish |
