# Domain Pipeline Parity — Design

**Date**: 2026-04-04
**Status**: Approved

## Problem

The domain pipeline (`mode: "domain"`) is functionally correct but missing features and ~350x slower than the old pipeline. Need to close gaps while keeping all new functionality as proper domain concepts.

## Domain Model Extensions

### SolveStrategy (new)

Captures the choice between depth-limited (GPU boundary eval) and exact (solve to showdown).

```rust
pub enum SolveStrategy {
    Exact,
    DepthLimited { evaluator: Arc<dyn BoundaryEvaluator> },
}
```

Used by both `GameBuilder` (to set depth_limit) and `Solver` (to decide whether to call evaluator). One config object, one decision point.

### Game (extended)

Encapsulate range-solver completely. `tree` becomes private. All solver operations exposed as domain methods:

```rust
pub struct Game {
    situation: Situation,
    tree: PostFlopGame,  // private
}

impl Game {
    // Boundary accessors
    pub fn num_boundaries(&self) -> usize;
    pub fn boundary_pot(&self, ordinal: usize) -> i32;
    pub fn boundary_cfvs_empty(&self, ordinal: usize, player: usize) -> bool;
    pub fn set_boundary_cfvs(&self, ordinal: usize, player: usize, cfvs: Vec<f32>);
    pub fn boundary_reach(&self, ordinal: usize, player: usize) -> Vec<f32>;

    // Solver operations (new — wraps range_solver)
    pub fn solve_step(&self, iteration: u32);
    pub fn finalize(&mut self);
    pub fn compute_exploitability(&self) -> f32;
    pub fn expected_values(&self, player: usize) -> Vec<f32>;
    pub fn back_to_root(&mut self);
    pub fn cache_normalized_weights(&mut self);

    // Accessors
    pub fn situation(&self) -> &Situation;
    pub fn num_private_hands(&self, player: usize) -> usize;
    pub fn private_cards(&self, player: usize) -> &[(u8, u8)];
}
```

Nobody outside `game.rs` imports from `range_solver`. The solver crate becomes a hidden implementation detail.

### GameBuilder (extended)

Takes `SolveStrategy` for depth_limit config. Adds bet size fuzzing.

```rust
pub struct GameBuilder {
    bet_sizes: Vec<Vec<f64>>,
    strategy: SolveStrategy,  // Exact → depth_limit: None, DepthLimited → Some(0)
    fuzz: f64,                // 0.0 = no fuzzing
}

impl GameBuilder {
    pub fn new(bet_sizes: Vec<Vec<f64>>, strategy: &SolveStrategy) -> Self;
    pub fn with_fuzz(mut self, fuzz: f64) -> Self;
    pub fn build(&self, sit: &Situation, rng: &mut impl Rng) -> Option<Game>;
}
```

`build` takes `&mut Rng` for fuzzing. When `fuzz > 0`, each bet size is perturbed by `1.0 + uniform(-fuzz, +fuzz)`.

### BoundaryEvaluator (extended)

Add batch evaluation with default per-game fallback:

```rust
pub trait BoundaryEvaluator: Send + Sync {
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs>;

    fn evaluate_batch(&self, games: &[&Game]) -> Vec<Vec<BoundaryCfvs>> {
        games.iter().map(|g| self.evaluate(g)).collect()
    }
}
```

`NeuralNetEvaluator` overrides `evaluate_batch` with a single GPU forward pass for all games.

### Solver (adjusted)

Takes `SolveStrategy` instead of `Arc<dyn BoundaryEvaluator>`:

```rust
pub struct Solver {
    game: Option<Game>,
    config: SolverConfig,
    strategy: SolveStrategy,
    iteration: u32,
    boundaries_set: bool,
}

impl Solver {
    pub fn new(game: Game, config: &SolverConfig, strategy: SolveStrategy) -> Self;
    pub fn step(&mut self) -> Option<SolvedGame>;
}
```

- `Exact` mode: never calls evaluator, just `game.solve_step()`.
- `DepthLimited` mode: calls `evaluator.evaluate(&game)` at configured intervals.

### RecordWriter (extended)

Add per-file rotation:

```rust
pub struct RecordWriter {
    base_path: PathBuf,
    per_file: Option<u64>,
    current_writer: BufWriter<File>,
    current_count: u64,
    file_index: u32,
    total_count: u64,
}

impl RecordWriter {
    pub fn create(path: &Path, per_file: Option<u64>) -> Result<Self, String>;
    pub fn write(&mut self, records: &[TrainingRecord]) -> Result<(), String>;
    pub fn flush(&mut self) -> Result<(), String>;
    pub fn count(&self) -> u64;
}
```

When `per_file` is set, rotates to `{stem}_{index:05}.bin` at threshold.

### SituationGenerator (unchanged)

Already has `RangeSource` for blueprint ranges. No changes needed.

## Pipeline (outside domain)

The pipeline coordinator (`DomainPipeline`) is NOT a domain concept. It handles:

- **Threading**: `std::thread::scope` with N threads, each running the sequential domain loop
- **Deal gen thread**: separate thread running `SituationGenerator` + `GameBuilder`, feeding a channel
- **GPU batching**: pipeline collects games needing boundary eval, calls `evaluator.evaluate_batch`, distributes results
- **Progress bar**: tracking throughput and exploitability

The domain types are thread-safe: `Game` is `Send`, `BoundaryEvaluator` is `Send + Sync`, `Solver` is `Send`.

## Implementation Order

Each phase is independently testable:

1. **Game encapsulation** — make tree private, add solver methods. No behavior change.
2. **SolveStrategy + GameBuilder fuzz** — new type, wire into builder and solver.
3. **RecordWriter rotation** — add per_file support.
4. **BoundaryEvaluator batch** — add evaluate_batch to trait + NeuralNetEvaluator override.
5. **Pipeline threading** — parallel solve in DomainPipeline.
6. **Pipeline deal gen** — separate deal gen thread.
7. **Pipeline GPU batching** — use evaluate_batch from pipeline.

Phases 1-4 are domain changes. Phases 5-7 are pipeline optimization.

## Files

Domain changes:
- `crates/cfvnet/src/datagen/domain/game.rs` — encapsulate tree, add solver methods
- `crates/cfvnet/src/datagen/domain/solver.rs` — take SolveStrategy
- `crates/cfvnet/src/datagen/domain/evaluator.rs` — add evaluate_batch, SolveStrategy
- `crates/cfvnet/src/datagen/domain/writer.rs` — add rotation
- `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs` — override evaluate_batch

Pipeline changes:
- `crates/cfvnet/src/datagen/domain/pipeline.rs` — threading, deal gen, batching
