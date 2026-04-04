# Turn Datagen Domain-Driven Redesign

**Date**: 2026-04-03
**Status**: Approved

## Problem

`turn_generate.rs` is 4000+ lines with no domain abstractions. Pipeline plumbing, solver internals, GPU batching, and serialization are tangled together. Boundary state management is ad-hoc (flush timing bugs, uninitialized memory reads). The code is unmaintainable and impossible to debug.

## Solution

Rebuild turn data generation around a clean domain model. Get the domain objects working first, optimize later.

## Domain Model

### Situation
Pure data — the random deal.
```rust
struct Situation {
    board: [u8; 5],
    board_size: usize,
    pot: i32,
    effective_stack: i32,
    ranges: [[f32; 1326]; 2],  // OOP, IP
}
```
Already exists in `sampler.rs`.

### SituationGenerator
Produces situations from config + RNG. Implements `Iterator<Item = Situation>`.
Wraps existing `sample_situation`. Skips degenerate situations (effective_stack <= 0) internally.

### GameBuilder
Builds a game tree from a situation + config.
```rust
struct GameBuilder { game_config: GameConfig }
fn build(&self, sit: &Situation) -> Option<Game>
```
Wraps existing `build_turn_game`. Returns `None` for degenerate cases.

### Game
Owns the `PostFlopGame`. Provides domain-level access to boundary nodes.
```rust
struct Game {
    situation: Situation,
    tree: PostFlopGame,
}
impl Game {
    fn num_boundaries(&self) -> usize;
    fn boundary_pot(&self, ordinal: usize) -> i32;
    fn boundary_cfvs(&self, ordinal: usize, player: usize) -> Vec<f32>;
    fn boundary_reach(&self, ordinal: usize, player: usize) -> Vec<f32>;
    fn set_boundary_cfvs(&self, ordinal: usize, player: usize, cfvs: Vec<f32>);
}
```
Thin wrapper — delegates to `PostFlopGame` but provides a clean interface.

### BoundaryEvaluator
Trait — evaluates boundary CFVs. Two implementations:
- **NeuralNetEvaluator** — wraps the river model + wgpu device
- **ExactEvaluator** — solves river subgames to showdown

```rust
trait BoundaryEvaluator {
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs>;
}

struct BoundaryCfvs {
    ordinal: usize,
    player: usize,
    cfvs: Vec<f32>,
}
```

### Solver
Owns the game, drives DCFR iteration. Iterator-like step interface.
```rust
struct Solver {
    game: Game,
    config: SolverConfig,
    evaluator: Arc<dyn BoundaryEvaluator>,
    iteration: u32,
}

impl Solver {
    fn new(game: Game, config: SolverConfig, evaluator: Arc<dyn BoundaryEvaluator>) -> Self;
    fn step(&mut self) -> Option<SolvedGame>;
}
```

`step()` runs one DCFR iteration. Calls `evaluator.evaluate()` at `leaf_eval_interval` boundaries. Returns `Some(SolvedGame)` when `iteration >= max_iterations` or exploitability target met.

### SolverConfig
Full solver parameterization including algorithm-specific knobs.
```rust
struct SolverConfig {
    max_iterations: u32,
    target_exploitability: Option<f32>,
    leaf_eval_interval: u32,
    // DCFR-specific
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
}
```

### SolvedGame
The result — ready for extraction.
```rust
struct SolvedGame {
    game: Game,
    exploitability: f32,
}

impl SolvedGame {
    fn extract_records(&self) -> Vec<TrainingRecord>;
}
```

## Usage

```rust
let gen = SituationGenerator::new(datagen_config, rng);
let builder = GameBuilder::new(game_config);
let evaluator = NeuralNetEvaluator::load(model_path)?;

for situation in gen.take(num_samples) {
    let game = builder.build(&situation)?;
    let mut solver = Solver::new(game, solver_config, &evaluator);
    let solved = loop {
        match solver.step() {
            None => continue,
            Some(game) => break game,
        }
    };
    let records = solved.extract_records();
    writer.write(records);
}
```

No pipeline stages, no channels, no GPU batching. The domain objects encapsulate all complexity. Parallelism and batching are optimizations applied on top of this clean interface.

## Key Invariants

1. **Boundary CFVs must be set before solve_step** — the Solver is responsible for calling the evaluator before each iteration (or at eval_interval boundaries).
2. **Boundary CFVs must be populated for compute_exploitability** — SolvedGame ensures this by evaluating boundaries one final time before computing exploitability.
3. **Boundary reaches update automatically** — `solve_step` writes reaches during traversal. No manual flush needed.

## Files

New module: `crates/cfvnet/src/datagen/domain/`
- `mod.rs` — re-exports
- `situation.rs` — Situation, SituationGenerator
- `game.rs` — Game, GameBuilder
- `solver.rs` — Solver, SolverConfig, SolvedGame
- `evaluator.rs` — BoundaryEvaluator trait, BoundaryCfvs, NeuralNetEvaluator, ExactEvaluator

The old `turn_generate.rs` pipeline code wraps these domain objects for parallel execution.
