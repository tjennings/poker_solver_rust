# Domain Pipeline Parity — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Close all feature and performance gaps between the domain pipeline and the old pipeline — 7 phases from domain encapsulation to parallel GPU batching.

**Architecture:** Phases 1-4 extend domain types (Game, GameBuilder, Solver, RecordWriter, BoundaryEvaluator). Phases 5-7 add pipeline-level parallelism outside the domain. Each phase is independently testable.

**Tech Stack:** Rust, range-solver, burn (wgpu), std::thread::scope, std::sync::mpsc

**Design doc:** `docs/plans/2026-04-04-domain-parity-design.md`

---

## Context for Implementer

The domain module is at `crates/cfvnet/src/datagen/domain/`. Current files:
- `game.rs` — `Game` (pub fields: situation, tree), `GameBuilder`
- `solver.rs` — `Solver`, `SolverConfig`, `SolvedGame`
- `evaluator.rs` — `BoundaryEvaluator` trait, `BoundaryCfvs`
- `situation.rs` — `SituationGenerator`, `RangeSource`
- `writer.rs` — `RecordWriter`
- `neural_net_evaluator.rs` — `NeuralNetEvaluator` (implements BoundaryEvaluator)
- `pipeline.rs` — `DomainPipeline::run` (sequential coordinator)
- `mod.rs` — re-exports

**Key range-solver functions currently used directly:**
- `range_solver::solve_step(&game.tree, iter)` — in solver.rs
- `range_solver::finalize(&mut game.tree)` — in solver.rs
- `range_solver::compute_exploitability(&game.tree)` — in solver.rs
- `game.tree.back_to_root()` / `game.tree.cache_normalized_weights()` — in solver.rs
- `game.tree.expected_values(player)` / `game.tree.private_cards(player)` — in solver.rs and neural_net_evaluator.rs
- `game.tree.num_private_hands(player)` — in evaluator tests

All of these need to be routed through `Game` methods.

---

### Task 1: Game encapsulation — make tree private

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/game.rs`
- Modify: `crates/cfvnet/src/datagen/domain/solver.rs`
- Modify: `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs` (tests)

**Step 1: Add solver methods to Game**

In `game.rs`, change `pub tree` to `tree` (private) and `pub situation` to `situation` (private). Add accessor and solver methods:

```rust
pub struct Game {
    situation: Situation,  // was pub
    tree: PostFlopGame,    // was pub
}

impl Game {
    // Existing boundary methods stay unchanged.

    // New accessors:
    pub fn situation(&self) -> &Situation { &self.situation }
    pub fn num_private_hands(&self, player: usize) -> usize {
        self.tree.num_private_hands(player)
    }
    pub fn private_cards(&self, player: usize) -> &[(u8, u8)] {
        self.tree.private_cards(player)
    }

    // New solver operations:
    pub fn solve_step(&self, iteration: u32) {
        range_solver::solve_step(&self.tree, iteration);
    }
    pub fn finalize(&mut self) {
        range_solver::finalize(&mut self.tree);
    }
    pub fn compute_exploitability(&self) -> f32 {
        range_solver::compute_exploitability(&self.tree)
    }
    pub fn expected_values(&self, player: usize) -> Vec<f32> {
        self.tree.expected_values(player)
    }
    pub fn back_to_root(&mut self) {
        self.tree.back_to_root();
    }
    pub fn cache_normalized_weights(&mut self) {
        self.tree.cache_normalized_weights();
    }
}
```

**Step 2: Update solver.rs to use Game methods**

Replace all `range_solver::solve_step(&game.tree, ...)` with `game.solve_step(...)`.
Replace `range_solver::finalize(&mut game.tree)` with `game.finalize()`.
Replace `range_solver::compute_exploitability(&game.tree)` with `game.compute_exploitability()`.
Replace `game.tree.back_to_root()` with `game.back_to_root()`.
Replace `game.tree.cache_normalized_weights()` with `game.cache_normalized_weights()`.
Replace `game.tree.expected_values(p)` with `game.expected_values(p)`.

Remove `use range_solver::{solve_step, finalize, compute_exploitability}` from solver.rs.

**Step 3: Update neural_net_evaluator.rs**

Replace `game.tree.private_cards(0)` with `game.private_cards(0)`.
Replace `game.tree.num_private_hands(player)` with `game.num_private_hands(player)`.
Replace `game.situation.board_cards()` with `game.situation().board_cards()`.
Replace `game.situation.ranges[...]` with `game.situation().ranges[...]`.
Replace `game.situation.pot` with `game.situation().pot`.
Replace `game.situation.effective_stack` with `game.situation().effective_stack`.

**Step 4: Update tests in mod.rs and other test files**

Replace `game.tree.num_private_hands(0)` with `game.num_private_hands(0)` in evaluator tests.
Replace `game.situation.board_size` with `game.situation().board_size` etc.

**Step 5: Verify**

Run: `cargo test -p cfvnet domain 2>&1 | tail -5`
All domain tests must pass with no `range_solver::` imports outside `game.rs`.

**Step 6: Commit**

```
git commit -m "refactor: encapsulate PostFlopGame inside Game — tree becomes private"
```

---

### Task 2: SolveStrategy + GameBuilder fuzz

**Files:**
- Create or modify: `crates/cfvnet/src/datagen/domain/evaluator.rs` — add SolveStrategy
- Modify: `crates/cfvnet/src/datagen/domain/game.rs` — GameBuilder takes strategy + fuzz
- Modify: `crates/cfvnet/src/datagen/domain/solver.rs` — Solver takes SolveStrategy
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` — construct SolveStrategy

**Step 1: Add SolveStrategy to evaluator.rs**

```rust
use std::sync::Arc;

pub enum SolveStrategy {
    /// Solve to showdown — no boundaries, no evaluator.
    Exact,
    /// Depth-limited with boundary evaluation.
    DepthLimited { evaluator: Arc<dyn BoundaryEvaluator> },
}
```

**Step 2: Update GameBuilder**

```rust
pub struct GameBuilder {
    bet_sizes: Vec<Vec<f64>>,
    exact: bool,
    fuzz: f64,
}

impl GameBuilder {
    pub fn new(bet_sizes: Vec<Vec<f64>>, strategy: &SolveStrategy) -> Self {
        Self {
            bet_sizes,
            exact: matches!(strategy, SolveStrategy::Exact),
            fuzz: 0.0,
        }
    }

    pub fn with_fuzz(mut self, fuzz: f64) -> Self {
        self.fuzz = fuzz;
        self
    }

    pub fn build(&self, sit: &Situation, rng: &mut impl Rng) -> Option<Game> {
        let sizes = if self.fuzz > 0.0 {
            fuzz_bet_sizes(&self.bet_sizes, self.fuzz, rng)
        } else {
            self.bet_sizes.clone()
        };
        let build_fn = if self.exact {
            super::super::turn_generate::build_turn_game_exact
        } else {
            super::super::turn_generate::build_turn_game
        };
        let tree = build_fn(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &sizes,
        )?;
        Some(Game::new(sit.clone(), tree))
    }
}
```

Note: `build_turn_game_exact` is currently private — make it `pub(crate)`.
Note: `fuzz_bet_sizes` is in turn_generate.rs — make it `pub(crate)`.
Note: `build` now takes `&mut impl Rng` for fuzzing. Update callers.

**Step 3: Update Solver to take SolveStrategy**

```rust
pub struct Solver {
    game: Option<Game>,
    config: SolverConfig,
    strategy: SolveStrategy,
    iteration: u32,
    boundaries_set: bool,
}

impl Solver {
    pub fn new(game: Game, config: &SolverConfig, strategy: SolveStrategy) -> Self { ... }

    pub fn step(&mut self) -> Option<SolvedGame> {
        let game = self.game.as_ref().expect("step called after finish");

        match &self.strategy {
            SolveStrategy::Exact => {
                // No boundary eval needed.
            }
            SolveStrategy::DepthLimited { evaluator } => {
                let needs_eval = !self.boundaries_set
                    || (self.config.leaf_eval_interval > 0
                        && self.iteration > 0
                        && self.iteration % self.config.leaf_eval_interval == 0);
                if needs_eval {
                    let cfvs = evaluator.evaluate(game);
                    for bc in cfvs {
                        game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs);
                    }
                    self.boundaries_set = true;
                }
            }
        }

        game.solve_step(self.iteration);
        self.iteration += 1;
        // ... rest unchanged
    }
}
```

**Step 4: Update pipeline.rs**

Construct `SolveStrategy` from config, pass to `GameBuilder` and `Solver`.
The pipeline loop's `builder.build(&sit)` becomes `builder.build(&sit, &mut rng)`.

**Step 5: Update tests**

All test constructors need updating for new signatures.

**Step 6: Verify and commit**

```
git commit -m "feat: add SolveStrategy, GameBuilder fuzz, exact mode support"
```

---

### Task 3: RecordWriter rotation

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/writer.rs`
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs`

**Step 1: Update RecordWriter for per-file rotation**

```rust
pub struct RecordWriter {
    base_path: PathBuf,
    per_file: Option<u64>,
    writer: BufWriter<File>,
    records_in_file: u64,
    file_index: u32,
    total_count: u64,
}

impl RecordWriter {
    pub fn create(path: &Path, per_file: Option<u64>) -> Result<Self, String> {
        let (writer, actual_path) = Self::open_file(path, 0)?;
        Ok(Self {
            base_path: path.to_path_buf(),
            per_file,
            writer,
            records_in_file: 0,
            file_index: 0,
            total_count: 0,
        })
    }

    pub fn write(&mut self, records: &[TrainingRecord]) -> Result<(), String> {
        for rec in records {
            // Rotate if needed (check before write, not after).
            if let Some(limit) = self.per_file {
                if self.records_in_file >= limit {
                    self.rotate()?;
                }
            }
            write_record(&mut self.writer, rec)
                .map_err(|e| format!("write: {e}"))?;
            self.records_in_file += 1;
            self.total_count += 1;
        }
        Ok(())
    }

    fn rotate(&mut self) -> Result<(), String> {
        self.flush()?;
        self.file_index += 1;
        let (new_writer, path) = Self::open_file(&self.base_path, self.file_index)?;
        self.writer = new_writer;
        self.records_in_file = 0;
        eprintln!("[writer] rotated to {}", path.display());
        Ok(())
    }

    fn open_file(base: &Path, index: u32) -> Result<(BufWriter<File>, PathBuf), String> {
        let stem = base.file_stem().unwrap_or_default().to_string_lossy();
        let parent = base.parent().unwrap_or(Path::new("."));
        let path = if index == 0 {
            parent.join(format!("{stem}.bin"))
        } else {
            parent.join(format!("{stem}_{index:05}.bin"))
        };
        let file = File::create(&path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        Ok((BufWriter::with_capacity(1 << 20, file), path))
    }
}
```

**Step 2: Update pipeline to pass per_file**

```rust
let mut writer = RecordWriter::create(output_path, config.datagen.per_file)?;
```

**Step 3: Write tests for rotation**

Test: write 10 records with per_file=3, verify 4 files created with correct record counts.

**Step 4: Verify and commit**

```
git commit -m "feat: add per-file rotation to RecordWriter"
```

---

### Task 4: BoundaryEvaluator batch

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/evaluator.rs`
- Modify: `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs`

**Step 1: Add evaluate_batch to trait**

```rust
pub trait BoundaryEvaluator: Send + Sync {
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs>;

    /// Evaluate multiple games in one batch. Default calls evaluate per-game.
    fn evaluate_batch(&self, games: &[&Game]) -> Vec<Vec<BoundaryCfvs>> {
        games.iter().map(|g| self.evaluate(g)).collect()
    }
}
```

**Step 2: Override in NeuralNetEvaluator**

Implement `evaluate_batch` that builds one combined tensor for all games' boundaries, one GPU forward pass, scatters results back. This is the logic from the existing `evaluate_pool_boundaries` / `build_game_inputs` / `decode_boundary_cfvs` functions.

The key: all games' boundary inputs are concatenated into one tensor, one `model.forward()`, results are split back per-game.

**Step 3: Test**

Build 3 games, call `evaluate_batch`, verify results match calling `evaluate` per-game.

**Step 4: Verify and commit**

```
git commit -m "feat: add evaluate_batch to BoundaryEvaluator with GPU batching"
```

---

### Task 5: Pipeline threading — parallel solve

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs`

**Step 1: Replace sequential loop with threaded solve**

Use `std::thread::scope` with N solver threads. Each thread runs the domain loop independently: pull situation from shared iterator (Mutex-wrapped), build game, solve, write records.

```rust
let sit_gen = Arc::new(Mutex::new(sit_gen));
let writer = Arc::new(Mutex::new(writer));

std::thread::scope(|s| {
    for _ in 0..threads {
        let sit_gen = Arc::clone(&sit_gen);
        let writer = Arc::clone(&writer);
        let builder = &builder;  // shared ref
        let solver_config = &solver_config;
        let strategy = &strategy;
        let pb = &pb;

        s.spawn(move || {
            range_solver::set_force_sequential(true);
            let mut rng = ChaCha8Rng::seed_from_u64(thread_seed);
            loop {
                let sit = {
                    let mut gen = sit_gen.lock().unwrap();
                    gen.next()
                };
                let sit = match sit {
                    Some(s) => s,
                    None => return,
                };
                let game = match builder.build(&sit, &mut rng) {
                    Some(g) => g,
                    None => { pb.inc(1); continue; }
                };
                let mut solver = Solver::new(game, solver_config, strategy.clone());
                let solved = loop {
                    match solver.step() {
                        None => continue,
                        Some(sg) => break sg,
                    }
                };
                let records = solved.extract_records();
                writer.lock().unwrap().write(&records).unwrap();
                pb.inc(1);
            }
        });
    }
});
```

Note: `SolveStrategy` needs `Clone`. Implement Clone for SolveStrategy (Arc clones are cheap).

**Step 2: Test**

Run with `threads: 4`, verify output has correct number of records, no panics.

**Step 3: Verify and commit**

```
git commit -m "perf: add parallel solving to DomainPipeline"
```

---

### Task 6: Pipeline deal gen thread

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs`

**Step 1: Move situation generation to a dedicated thread**

Instead of Mutex-wrapped iterator, use a channel. A deal gen thread runs `SituationGenerator` + `GameBuilder`, sends `Game` objects to solver threads via a bounded channel.

```rust
let (deal_tx, deal_rx) = std::sync::mpsc::sync_channel::<Game>(256);
let deal_rx = Arc::new(Mutex::new(deal_rx));

// Deal gen thread.
let deal_thread = std::thread::spawn(move || {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for sit in sit_gen {
        let game = match builder.build(&sit, &mut rng) {
            Some(g) => g,
            None => continue,
        };
        if deal_tx.send(game).is_err() {
            break;
        }
    }
});

// Solver threads pull from deal_rx.
```

Note: This moves `Game` through a channel — which we proved is safe with the move test. But the Game now has `tree` private, so moves are through the struct, not the inner PostFlopGame. Should be fine.

**Step 2: Verify and commit**

```
git commit -m "perf: add deal gen thread to DomainPipeline"
```

---

### Task 7: Pipeline GPU batching

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs`

**Step 1: Batch boundary evaluation across games**

Instead of each solver thread calling `evaluator.evaluate()` per-game, collect games that need boundary eval and call `evaluator.evaluate_batch()` once.

This requires restructuring the pipeline: solver threads solve until they need boundary eval, then submit the game to a GPU queue. The GPU thread processes the queue, calls `evaluate_batch`, and returns games to solvers.

This is the same architectural pattern as the old pipeline's Stage 2. Use a bounded channel for the GPU queue.

**However**: Given the move-safety concerns we investigated, this should be validated carefully. Use `active_pool_size` to control the batch size.

**Step 2: Verify with exploitability check**

The batch pipeline must produce the same exploitability as the sequential pipeline.

**Step 3: Commit**

```
git commit -m "perf: add GPU batching to DomainPipeline via evaluate_batch"
```

---

## Important Notes

1. **Remove diagnostic probes** before committing Phase 1. The `[SOLVER]`, `[EVAL]`, `[RANGE]` eprintln calls in solver.rs and neural_net_evaluator.rs should be removed or put behind a `cfg(debug_assertions)` guard.

2. **`build_turn_game_exact` needs pub(crate)** — currently private in turn_generate.rs.

3. **`fuzz_bet_sizes` needs pub(crate)** — currently private in turn_generate.rs.

4. **SolveStrategy needs Clone** — for sharing across solver threads. `Arc<dyn BoundaryEvaluator>` already clones cheaply.

5. **Game moves through channels (Task 6)** — we proved this is safe with a unit test. But validate exploitability after implementing to be sure.

6. **Task 7 is optional** — Tasks 1-6 get us to full feature parity and ~30-50/s with parallel solving. GPU batching adds another 2x but is complex.
