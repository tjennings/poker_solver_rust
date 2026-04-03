# Turn Datagen Domain-Driven Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a clean domain model for turn data generation — Situation, SituationGenerator, GameBuilder, Game, BoundaryEvaluator, Solver, SolvedGame — that encapsulates all complexity behind simple interfaces.

**Architecture:** New `domain/` module inside `crates/cfvnet/src/datagen/`. Pure domain types wrapping existing range-solver primitives. No I/O, no threading, no GPU concerns in the domain layer. Each type is testable in isolation.

**Tech Stack:** Rust, range-solver crate (PostFlopGame, solve_step, compute_exploitability, finalize)

**Design doc:** `docs/plans/2026-04-03-turn-datagen-ddd-design.md`

---

## Context for Implementer

**Existing types you'll wrap:**
- `Situation` in `crates/cfvnet/src/datagen/sampler.rs:9` — already a pub struct
- `sample_situation()` in `sampler.rs:30` — produces Situations
- `PostFlopGame` in `range-solver` crate — the game tree
- `solve_step(game, iter)` in `range-solver::solver` — one DCFR iteration
- `finalize(game)` in `range-solver::solver` — averages strategy after solving
- `compute_exploitability(game)` in `range-solver::utility` — computes exploitability
- `TrainingRecord` in `crates/cfvnet/src/datagen/storage.rs:25` — output format
- `NUM_COMBOS` = 1326 — number of hole card combinations

**Existing functions to extract logic from:**
- `build_turn_game()` in `turn_generate.rs:565` — builds PostFlopGame from situation
- `evaluate_game_boundaries()` in `turn_generate.rs:426` — evaluates all boundaries via LeafEvaluator
- `extract_results()` in `turn_generate.rs:618` — extracts CFVs from solved game into TrainingRecord format

**Key crate boundary:** The domain module lives in the `cfvnet` crate but depends on `range-solver` for PostFlopGame, solve_step, etc. This is the same dependency the existing code has.

---

### Task 1: Create domain module with Game type

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/mod.rs`
- Create: `crates/cfvnet/src/datagen/domain/game.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs` (add `pub mod domain;`)

**Step 1: Create the module structure**

Create `crates/cfvnet/src/datagen/domain/mod.rs`:
```rust
pub mod game;

pub use game::{Game, GameBuilder};
```

**Step 2: Create Game and GameBuilder**

Create `crates/cfvnet/src/datagen/domain/game.rs`:
```rust
use range_solver::game::PostFlopGame;
use crate::datagen::sampler::Situation;
use crate::datagen::range_gen::NUM_COMBOS;

/// A turn game tree built from a situation.
/// Thin wrapper over PostFlopGame with domain-level boundary access.
pub struct Game {
    pub situation: Situation,
    pub tree: PostFlopGame,
}

impl Game {
    pub fn num_boundaries(&self) -> usize {
        self.tree.num_boundary_nodes()
    }

    pub fn boundary_pot(&self, ordinal: usize) -> i32 {
        self.tree.boundary_pot(ordinal)
    }

    pub fn boundary_cfvs_empty(&self, ordinal: usize, player: usize) -> bool {
        self.tree.boundary_cfvs_empty(ordinal, player)
    }

    pub fn set_boundary_cfvs(&self, ordinal: usize, player: usize, cfvs: Vec<f32>) {
        self.tree.set_boundary_cfvs(ordinal, player, cfvs);
    }

    pub fn boundary_reach(&self, ordinal: usize, player: usize) -> Vec<f32> {
        self.tree.boundary_reach(ordinal, player)
    }
}
```

The `GameBuilder` wraps the existing `build_turn_game` logic. For now, keep it simple — extract later if the build logic is complex:

```rust
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::{BetSize, BetSizeOptions};
use range_solver::card::{CardConfig, NOT_DEALT};
use range_solver::range::Range as RsRange;

pub struct GameBuilder {
    pub bet_sizes: Vec<Vec<f64>>,
    pub depth_limit: Option<usize>,
}

impl GameBuilder {
    pub fn new(bet_sizes: Vec<Vec<f64>>) -> Self {
        Self { bet_sizes, depth_limit: Some(0) }
    }

    pub fn build(&self, sit: &Situation) -> Option<Game> {
        // Delegate to existing build_turn_game logic.
        // For now, call the existing function and wrap the result.
        let tree = super::super::turn_generate::build_turn_game(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &self.bet_sizes,
        )?;
        Some(Game { situation: sit.clone(), tree })
    }
}
```

Note: `build_turn_game` is currently a private function. You'll need to make it `pub(crate)` or `pub` so the domain module can call it.

**Step 3: Add module to datagen/mod.rs**

Add `pub mod domain;` to `crates/cfvnet/src/datagen/mod.rs`.

**Step 4: Write tests**

In `crates/cfvnet/src/datagen/domain/game.rs`, add:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::sampler::sample_situation;
    use crate::config::DatagenConfig;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn game_builder_produces_game_with_boundaries() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        if sit.effective_stack <= 0 { return; }
        let game = builder.build(&sit).expect("should build");
        assert!(game.num_boundaries() > 0, "turn game should have boundaries");
    }

    #[test]
    fn game_boundary_methods_delegate_correctly() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        if sit.effective_stack <= 0 { return; }
        let game = builder.build(&sit).expect("should build");
        let pot = game.boundary_pot(0);
        assert!(pot > 0, "boundary pot should be positive");
        assert!(game.boundary_cfvs_empty(0, 0), "cfvs should start empty");
    }
}
```

**Step 5: Verify**

Run: `cargo test -p cfvnet domain 2>&1 | tail -5`

**Step 6: Commit**

```
git commit -m "feat: add Game and GameBuilder domain types"
```

---

### Task 2: Add SituationGenerator

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/situation.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`

**Step 1: Create SituationGenerator**

```rust
use crate::config::DatagenConfig;
use crate::datagen::sampler::{sample_situation, Situation};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Produces random Situations from config. Implements Iterator.
/// Skips degenerate situations (effective_stack <= 0) internally.
pub struct SituationGenerator {
    config: DatagenConfig,
    initial_stack: i32,
    board_size: usize,
    rng: ChaCha8Rng,
    remaining: u64,
}

impl SituationGenerator {
    pub fn new(config: &DatagenConfig, initial_stack: i32, board_size: usize, seed: u64, count: u64) -> Self {
        Self {
            config: config.clone(),
            initial_stack,
            board_size,
            rng: ChaCha8Rng::seed_from_u64(seed),
            remaining: count,
        }
    }
}

impl Iterator for SituationGenerator {
    type Item = Situation;

    fn next(&mut self) -> Option<Situation> {
        while self.remaining > 0 {
            self.remaining -= 1;
            let sit = sample_situation(&self.config, self.initial_stack, self.board_size, &mut self.rng);
            if sit.effective_stack > 0 {
                return Some(sit);
            }
        }
        None
    }
}
```

**Step 2: Update mod.rs**

```rust
pub mod game;
pub mod situation;

pub use game::{Game, GameBuilder};
pub use situation::SituationGenerator;
```

**Step 3: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;

    #[test]
    fn generator_produces_valid_situations() {
        let config = DatagenConfig::default();
        let gen = SituationGenerator::new(&config, 200, 4, 42, 10);
        let situations: Vec<_> = gen.collect();
        assert!(!situations.is_empty());
        for sit in &situations {
            assert!(sit.effective_stack > 0);
            assert_eq!(sit.board_size, 4);
        }
    }

    #[test]
    fn generator_respects_count() {
        let config = DatagenConfig::default();
        let gen = SituationGenerator::new(&config, 200, 4, 42, 3);
        let count = gen.count();
        assert!(count <= 3);
    }
}
```

**Step 4: Verify and commit**

```
git commit -m "feat: add SituationGenerator domain type"
```

---

### Task 3: Add BoundaryEvaluator trait and BoundaryCfvs

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/evaluator.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`

**Step 1: Create the trait and types**

```rust
use super::game::Game;

/// Counterfactual values for a single boundary node.
pub struct BoundaryCfvs {
    pub ordinal: usize,
    pub player: usize,
    pub cfvs: Vec<f32>,
}

/// Evaluates boundary CFVs for a game's boundary nodes.
/// Implementations: neural net (GPU), exact (solve to showdown).
pub trait BoundaryEvaluator: Send + Sync {
    /// Evaluate all boundary nodes and return CFVs for each (ordinal, player).
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs>;
}
```

**Step 2: Add a trivial ExactEvaluator** (wraps existing `evaluate_game_boundaries` logic)

```rust
use poker_solver_core::blueprint_v2::LeafEvaluator;
use crate::eval::river_net_evaluator::RiverNetEvaluator;

/// Evaluates boundaries using a neural net model.
/// Wraps the existing evaluate_game_boundaries logic.
pub struct NeuralNetEvaluator<B: burn::tensor::backend::Backend> {
    evaluator: RiverNetEvaluator<B>,
}
```

Actually, the neural net evaluator needs GPU types (burn backend) which makes it non-domain. Keep the trait in domain, put implementations outside:

```rust
// In evaluator.rs — just the trait + BoundaryCfvs (pure domain)
```

**Step 3: Update mod.rs**

```rust
pub mod evaluator;
pub mod game;
pub mod situation;

pub use evaluator::{BoundaryEvaluator, BoundaryCfvs};
pub use game::{Game, GameBuilder};
pub use situation::SituationGenerator;
```

**Step 4: Test that the trait compiles and a mock works**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockEvaluator;
    impl BoundaryEvaluator for MockEvaluator {
        fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs> {
            (0..game.num_boundaries())
                .flat_map(|ord| {
                    (0..2).map(move |player| BoundaryCfvs {
                        ordinal: ord,
                        player,
                        cfvs: vec![0.0; game.tree.num_private_hands(player)],
                    })
                })
                .collect()
        }
    }

    #[test]
    fn mock_evaluator_produces_cfvs_for_all_boundaries() {
        use crate::datagen::sampler::sample_situation;
        use crate::config::DatagenConfig;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 { return; }
        let builder = super::super::game::GameBuilder::new(vec![vec![0.5, 1.0]]);
        let game = builder.build(&sit).expect("build");
        let eval = MockEvaluator;
        let results = eval.evaluate(&game);
        assert_eq!(results.len(), game.num_boundaries() * 2);
    }
}
```

**Step 5: Commit**

```
git commit -m "feat: add BoundaryEvaluator trait and BoundaryCfvs"
```

---

### Task 4: Add Solver and SolvedGame

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/solver.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`

**Step 1: Create SolverConfig**

```rust
/// Configuration for the DCFR solver.
pub struct SolverConfig {
    pub max_iterations: u32,
    pub target_exploitability: Option<f32>,
    pub leaf_eval_interval: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            target_exploitability: None,
            leaf_eval_interval: 0, // 0 = evaluate once at start
        }
    }
}
```

**Step 2: Create SolvedGame**

```rust
use super::game::Game;
use crate::datagen::storage::TrainingRecord;
use crate::datagen::range_gen::NUM_COMBOS;
use range_solver::card::card_pair_to_index;

pub struct SolvedGame {
    pub game: Game,
    pub exploitability: f32,
}

impl SolvedGame {
    pub fn extract_records(&self) -> Vec<TrainingRecord> {
        // Extract CFVs from the solved game — delegates to existing logic.
        // Uses game.tree.expected_values(), maps to 1326-indexed arrays.
        // This is the logic from extract_results() in turn_generate.rs.
        let tree = &self.game.tree;
        let sit = &self.game.situation;
        let pot = f64::from(sit.pot);
        let half_pot = pot / 2.0;
        let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

        let raw_oop = tree.expected_values(0);
        let raw_ip = tree.expected_values(1);
        let oop_hands = tree.private_cards(0);
        let ip_hands = tree.private_cards(1);

        let mut oop_cfvs = [0.0_f32; NUM_COMBOS];
        let mut ip_cfvs = [0.0_f32; NUM_COMBOS];
        let mut valid_mask = [0_u8; NUM_COMBOS];

        for (i, &(c0, c1)) in oop_hands.iter().enumerate() {
            let idx = card_pair_to_index(c0, c1);
            oop_cfvs[idx] = ((f64::from(raw_oop[i]) - half_pot) / norm) as f32;
            valid_mask[idx] = 1;
        }
        for (i, &(c0, c1)) in ip_hands.iter().enumerate() {
            let idx = card_pair_to_index(c0, c1);
            ip_cfvs[idx] = ((f64::from(raw_ip[i]) - half_pot) / norm) as f32;
            valid_mask[idx] = 1;
        }

        let oop_gv: f32 = sit.ranges[0].iter().zip(oop_cfvs.iter()).map(|(&r, &c)| r * c).sum();
        let ip_gv: f32 = sit.ranges[1].iter().zip(ip_cfvs.iter()).map(|(&r, &c)| r * c).sum();

        let board = sit.board_cards().to_vec();
        vec![
            TrainingRecord {
                board: board.clone(),
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                player: 0,
                game_value: oop_gv,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                cfvs: oop_cfvs,
                valid_mask,
            },
            TrainingRecord {
                board,
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                player: 1,
                game_value: ip_gv,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                cfvs: ip_cfvs,
                valid_mask,
            },
        ]
    }
}
```

**Step 3: Create Solver**

```rust
use std::sync::Arc;
use super::evaluator::{BoundaryEvaluator, BoundaryCfvs};
use super::game::Game;
use range_solver::{solve_step, finalize, compute_exploitability};

pub struct Solver {
    game: Game,
    config: SolverConfig,
    evaluator: Arc<dyn BoundaryEvaluator>,
    iteration: u32,
    boundaries_set: bool,
}

impl Solver {
    pub fn new(game: Game, config: SolverConfig, evaluator: Arc<dyn BoundaryEvaluator>) -> Self {
        Self {
            game,
            config,
            evaluator,
            iteration: 0,
            boundaries_set: false,
        }
    }

    /// Run one DCFR iteration. Returns Some(SolvedGame) when done.
    pub fn step(&mut self) -> Option<SolvedGame> {
        // Evaluate boundaries if needed.
        let needs_eval = !self.boundaries_set
            || (self.config.leaf_eval_interval > 0
                && self.iteration > 0
                && self.iteration % self.config.leaf_eval_interval == 0);

        if needs_eval {
            let cfvs = self.evaluator.evaluate(&self.game);
            for bc in cfvs {
                self.game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs);
            }
            self.boundaries_set = true;
        }

        // Run one DCFR iteration.
        solve_step(&self.game.tree, self.iteration);
        self.iteration += 1;

        // Check if done.
        if self.iteration >= self.config.max_iterations {
            return Some(self.finish());
        }

        // Check exploitability target (every 10 iterations to avoid overhead).
        if let Some(target) = self.config.target_exploitability {
            if self.iteration % 10 == 0 {
                // Need to evaluate boundaries for exploitability computation.
                let cfvs = self.evaluator.evaluate(&self.game);
                for bc in &cfvs {
                    self.game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs.clone());
                }
                let exploit = compute_exploitability(&self.game.tree);
                let abs_target = target * self.game.situation.pot as f32;
                if exploit <= abs_target {
                    return Some(self.finish());
                }
            }
        }

        None
    }

    fn finish(&mut self) -> SolvedGame {
        // Ensure boundaries are set for final exploitability computation.
        let cfvs = self.evaluator.evaluate(&self.game);
        for bc in cfvs {
            self.game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs);
        }

        finalize(&mut self.game.tree);
        self.game.tree.back_to_root();
        self.game.tree.cache_normalized_weights();

        let exploit = compute_exploitability(&self.game.tree);

        SolvedGame {
            game: std::mem::replace(&mut self.game, unsafe { std::mem::zeroed() }),
            exploitability: exploit,
        }
    }
}
```

Note: The `std::mem::replace` with zeroed is ugly — we should use `Option<Game>` instead. Let me fix:

```rust
pub struct Solver {
    game: Option<Game>,
    // ...
}

fn finish(&mut self) -> SolvedGame {
    let mut game = self.game.take().expect("finish called twice");
    // ... finalize, compute exploit ...
    SolvedGame { game, exploitability: exploit }
}
```

**Step 4: Update mod.rs**

```rust
pub mod evaluator;
pub mod game;
pub mod situation;
pub mod solver;

pub use evaluator::{BoundaryEvaluator, BoundaryCfvs};
pub use game::{Game, GameBuilder};
pub use situation::SituationGenerator;
pub use solver::{Solver, SolverConfig, SolvedGame};
```

**Step 5: Write end-to-end test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::domain::evaluator::BoundaryCfvs;
    use crate::datagen::domain::game::GameBuilder;
    use crate::datagen::sampler::sample_situation;
    use crate::config::DatagenConfig;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::Arc;

    struct ZeroEvaluator;
    impl BoundaryEvaluator for ZeroEvaluator {
        fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs> {
            (0..game.num_boundaries())
                .flat_map(|ord| {
                    (0..2).map(move |player| BoundaryCfvs {
                        ordinal: ord,
                        player,
                        cfvs: vec![0.0; game.tree.num_private_hands(player)],
                    })
                })
                .collect()
        }
    }

    #[test]
    fn solver_produces_solved_game_with_finite_records() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 { return; }

        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        let game = builder.build(&sit).expect("build");

        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(ZeroEvaluator);
        let solver_config = SolverConfig {
            max_iterations: 50,
            ..Default::default()
        };
        let mut solver = Solver::new(game, solver_config, eval);

        let solved = loop {
            match solver.step() {
                None => continue,
                Some(sg) => break sg,
            }
        };

        assert!(solved.exploitability.is_finite());
        let records = solved.extract_records();
        assert_eq!(records.len(), 2); // OOP + IP
        for rec in &records {
            assert_eq!(rec.board.len(), 4);
            assert!(rec.game_value.is_finite());
            for &cfv in &rec.cfvs {
                assert!(cfv.is_finite());
            }
        }
    }
}
```

**Step 6: Verify and commit**

```
git commit -m "feat: add Solver, SolverConfig, and SolvedGame domain types"
```

---

### Task 5: End-to-end integration test using domain model

**Files:**
- Test in: `crates/cfvnet/src/datagen/domain/mod.rs` or a separate integration test

**Step 1: Write the full pipeline test using domain objects**

```rust
#[test]
fn full_domain_pipeline_produces_training_records() {
    // SituationGenerator → GameBuilder → Solver → SolvedGame → TrainingRecords
    let datagen_config = DatagenConfig::default();
    let gen = SituationGenerator::new(&datagen_config, 200, 4, 42, 5);
    let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
    let eval: Arc<dyn BoundaryEvaluator> = Arc::new(ZeroEvaluator);

    let mut total_records = 0;
    for sit in gen {
        let game = match builder.build(&sit) {
            Some(g) => g,
            None => continue,
        };
        let mut solver = Solver::new(game, SolverConfig { max_iterations: 20, ..Default::default() }, Arc::clone(&eval));
        let solved = loop {
            match solver.step() {
                None => continue,
                Some(sg) => break sg,
            }
        };
        let records = solved.extract_records();
        assert_eq!(records.len(), 2);
        total_records += records.len();
    }
    assert!(total_records > 0, "should produce at least some records");
}
```

**Step 2: Verify and commit**

```
git commit -m "test: end-to-end domain pipeline integration test"
```

---

## Important Notes

1. **Make `build_turn_game` pub(crate)** — GameBuilder needs to call it. Change visibility in `turn_generate.rs` from `fn` to `pub(crate) fn`.

2. **The Solver's `finish()` method** must ensure boundary_cfvs are populated before `compute_exploitability`. The design doc's invariant #2 is enforced here.

3. **`solve_step` takes `&T` where T: Game** — PostFlopGame uses interior mutability. The Solver can call solve_step on `&self.game.tree` without `&mut`.

4. **`finalize` and `back_to_root` need `&mut`** — the Solver takes the Game out of the Option to get mutable access for finalization.

5. **Don't optimize yet** — no threading, no GPU batching, no channels. Pure sequential domain logic. The pipeline wrapper adds parallelism later.

6. **The `ZeroEvaluator` in tests** produces zero boundary CFVs. This is correct for testing — the solver will converge to a strategy, just not an optimal one. The test validates the pipeline works end-to-end, not solution quality.
