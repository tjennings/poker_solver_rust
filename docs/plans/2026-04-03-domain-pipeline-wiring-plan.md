# Domain Pipeline Wiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Wire the domain model (SituationGenerator, GameBuilder, Solver, etc.) into a working datagen pipeline that reads `CfvnetConfig` YAML and produces training records, replacing the old pipeline for the `mode: "model"` code path.

**Architecture:** Three layers: (1) Domain types already built — Game, Solver, BoundaryEvaluator, etc. (2) New `NeuralNetEvaluator` adapter — implements BoundaryEvaluator using the wgpu river model. (3) New `DomainPipeline` coordinator — reads config, constructs domain objects, orchestrates the generate loop. The CLI dispatches to DomainPipeline for `mode: "domain"`.

**Tech Stack:** Rust, burn (wgpu backend), range-solver, existing CfvnetConfig

**Design doc:** `docs/plans/2026-04-03-turn-datagen-ddd-design.md`

---

## Context for Implementer

**Domain types (already built, in `crates/cfvnet/src/datagen/domain/`):**
- `SituationGenerator::new(config, initial_stack, board_size, seed, count)` → Iterator<Item=Situation>
- `GameBuilder::new(bet_sizes)` → `.build(&sit) → Option<Game>`
- `Game` — wraps PostFlopGame + Situation, has boundary accessors
- `BoundaryEvaluator` trait — `fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs>`
- `Solver::new(game, config, evaluator)` → `.step() → Option<SolvedGame>`
- `SolvedGame` — `.extract_records() → Vec<TrainingRecord>`

**Existing config types (in `crates/cfvnet/src/config.rs`):**
- `CfvnetConfig { game: GameConfig, datagen: DatagenConfig, training, evaluation }`
- `GameConfig { initial_stack, bet_sizes, board_size, river_model_path, ... }`
- `DatagenConfig { num_samples, street, mode, solver_iterations, threads, seed, leaf_eval_interval, ... }`

**Existing infrastructure to reuse:**
- `parse_bet_sizes_all(config)` in `turn_generate.rs` — converts BetSizeConfig → Vec<Vec<f64>>
- `write_record(writer, record)` in `storage.rs` — serializes TrainingRecord to binary
- `load_river_model(config, label)` in `turn_generate.rs` — loads CfvNet model, returns (model, device)
- `evaluate_game_boundaries()` in `turn_generate.rs` — bridges LeafEvaluator → set_boundary_cfvs

**New domain type to add:** `RecordWriter` — a domain concept for writing training records. Currently `write_record` is a raw function. Wrapping it makes the pipeline cleaner and testable.

---

### Task 1: Add RecordWriter domain type

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/writer.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`

A domain concept for writing training records. Wraps BufWriter + write_record.

```rust
// writer.rs
use std::io::BufWriter;
use std::path::Path;
use crate::datagen::storage::{write_record, TrainingRecord};

pub struct RecordWriter {
    writer: BufWriter<std::fs::File>,
    count: u64,
}

impl RecordWriter {
    pub fn create(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, file),
            count: 0,
        })
    }

    pub fn write(&mut self, records: &[TrainingRecord]) -> Result<(), String> {
        for rec in records {
            write_record(&mut self.writer, rec)
                .map_err(|e| format!("write record: {e}"))?;
            self.count += 1;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        use std::io::Write;
        self.writer.flush().map_err(|e| format!("flush: {e}"))
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}
```

Add to `mod.rs`:
```rust
pub mod writer;
pub use writer::RecordWriter;
```

**Tests:** Write records to a temp file, read back with `storage::read_record`, verify roundtrip.

**Commit:** `feat: add RecordWriter domain type`

---

### Task 2: Add NeuralNetEvaluator (implements BoundaryEvaluator)

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`

This adapter implements `BoundaryEvaluator` by wrapping the existing `RiverNetEvaluator` + `evaluate_game_boundaries` logic. It bridges the domain trait to the GPU model.

```rust
use burn::backend::wgpu::Wgpu;
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::Card;
use range_solver::card::card_pair_to_index;

use crate::eval::river_net_evaluator::RiverNetEvaluator;
use crate::datagen::range_gen::NUM_COMBOS;
use super::evaluator::{BoundaryEvaluator, BoundaryCfvs};
use super::game::Game;

type B = Wgpu;

pub struct NeuralNetEvaluator {
    evaluator: RiverNetEvaluator<B>,
}

impl NeuralNetEvaluator {
    pub fn load(model_path: &str, config: &CfvnetConfig) -> Result<Self, String> {
        // Load model using existing load_river_model logic.
        // ...
    }
}

impl BoundaryEvaluator for NeuralNetEvaluator {
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs> {
        // Bridge: convert Game's boundary info → LeafEvaluator::evaluate_boundaries call
        // This is the logic from evaluate_game_boundaries() in turn_generate.rs
        // Extract: combos, board, ranges, boundary (pot, eff_stack, player) requests
        // Call self.evaluator.evaluate_boundaries(...)
        // Convert Vec<f64> results → Vec<BoundaryCfvs>
    }
}
```

The key logic to port from `evaluate_game_boundaries`:
1. Convert board u8 → Card
2. Get combos from game.tree.private_cards(0)
3. Build per-combo range arrays from sit.ranges
4. Collect (pot, eff_stack, player) requests per boundary
5. Call evaluate_boundaries (batched GPU call)
6. Map results to BoundaryCfvs

**Make `parse_bet_sizes_all` pub(crate)** — needed by the pipeline but currently private.

**Tests:** Load a tiny untrained model, evaluate a game, verify BoundaryCfvs are non-empty and finite.

**Commit:** `feat: add NeuralNetEvaluator implementing BoundaryEvaluator`

---

### Task 3: Add DomainPipeline coordinator

**Files:**
- Create: `crates/cfvnet/src/datagen/domain/pipeline.rs`
- Modify: `crates/cfvnet/src/datagen/domain/mod.rs`

This is the coordination layer — reads config, constructs domain objects, runs the generate loop. Pure sequential for now (no threading).

```rust
use std::path::Path;
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::CfvnetConfig;
use super::{
    SituationGenerator, GameBuilder, Solver, SolverConfig,
    BoundaryEvaluator, RecordWriter,
};
use super::neural_net_evaluator::NeuralNetEvaluator;

pub struct DomainPipeline;

impl DomainPipeline {
    pub fn run(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        let num_samples = config.datagen.num_samples;
        let seed = crate::config::resolve_seed(config.datagen.seed);
        let initial_stack = config.game.initial_stack;
        let board_size = config.game.board_size;

        // Parse bet sizes from config.
        let bet_sizes = crate::datagen::turn_generate::parse_bet_sizes_all(&config.game.bet_sizes);
        if bet_sizes.is_empty() {
            return Err("no valid bet sizes".into());
        }

        // Load boundary evaluator (neural net or exact based on board_size).
        let evaluator: Arc<dyn BoundaryEvaluator> = if board_size < 5 {
            Arc::new(NeuralNetEvaluator::load(
                config.game.river_model_path.as_deref()
                    .ok_or("river_model_path required for turn datagen")?,
                config,
            )?)
        } else {
            // River mode: no GPU needed, use exact evaluator or zero evaluator.
            // For now, a ZeroBoundaryEvaluator (river games have no boundaries).
            Arc::new(ZeroBoundaryEvaluator)
        };

        // Construct domain objects.
        let sit_gen = SituationGenerator::new(
            &config.datagen, initial_stack, board_size, seed, num_samples,
        );
        let builder = GameBuilder::new(bet_sizes);
        let solver_config = SolverConfig {
            max_iterations: config.datagen.solver_iterations,
            target_exploitability: config.datagen.target_exploitability,
            leaf_eval_interval: config.datagen.leaf_eval_interval,
        };

        let mut writer = RecordWriter::create(output_path)?;

        // Progress bar.
        let pb = ProgressBar::new(num_samples);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
                .expect("valid template"),
        );

        // Generate loop.
        let mut exploit_sum = 0.0f64;
        let mut exploit_count = 0u64;

        for sit in sit_gen {
            let game = match builder.build(&sit) {
                Some(g) => g,
                None => { pb.inc(1); continue; }
            };

            let mut solver = Solver::new(game, solver_config.clone(), Arc::clone(&evaluator));
            let solved = loop {
                match solver.step() {
                    None => continue,
                    Some(sg) => break sg,
                }
            };

            // Track exploitability.
            if solved.exploitability.is_finite() {
                let bb = initial_stack as f32 / 100.0;
                let mbb = if bb > 0.0 { solved.exploitability / bb * 1000.0 } else { 0.0 };
                exploit_sum += mbb as f64;
                exploit_count += 1;
            }

            let records = solved.extract_records();
            writer.write(&records)?;
            pb.inc(1);

            // Update progress bar.
            let avg_exploit = if exploit_count > 0 { exploit_sum / exploit_count as f64 } else { 0.0 };
            pb.set_message(format!(
                "expl:{avg_exploit:.1} mbb/h  written:{}",
                writer.count()
            ));
        }

        writer.flush()?;
        pb.finish_with_message("done");

        eprintln!("Wrote {} records to {}", writer.count(), output_path.display());
        Ok(())
    }
}

/// Zero evaluator for river games (no boundary nodes).
struct ZeroBoundaryEvaluator;
impl BoundaryEvaluator for ZeroBoundaryEvaluator {
    fn evaluate(&self, _game: &Game) -> Vec<BoundaryCfvs> {
        Vec::new()
    }
}
```

**Make `parse_bet_sizes_all` pub(crate)** in `turn_generate.rs`.

**Tests:** Run the pipeline with a tiny config (5 samples, small model), verify output file has records.

**Commit:** `feat: add DomainPipeline coordinator`

---

### Task 4: Wire into CLI as mode "domain"

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs` — add dispatch for mode "domain"
- OR: modify `crates/cfvnet/src/main.rs` — add dispatch before existing pipeline

Add a new mode check in `generate_turn_training_data`:

```rust
pub fn generate_turn_training_data(config, output_path, backend) -> Result<(), String> {
    if config.datagen.mode == "domain" {
        return crate::datagen::domain::pipeline::DomainPipeline::run(config, output_path);
    }
    // ... existing dispatch ...
}
```

**Config to test with:**
```yaml
datagen:
  mode: "domain"
  street: "turn"
  solver_iterations: 100
  num_samples: 100
  threads: 1
```

**Test:** Run `cargo run -p cfvnet --release -- generate -c <config>` and verify:
- Output file exists with valid records
- Exploitability is reported in progress bar
- No panics (boundary_cfvs populated before compute_exploitability)

**Commit:** `feat: wire DomainPipeline into CLI as mode "domain"`

---

### Task 5: Validate against existing pipeline

**No code changes.** Run both pipelines on the same config and compare:

1. Run with `mode: "model"` (existing pipeline) — note exploitability and record count
2. Run with `mode: "domain"` (new pipeline) — note exploitability and record count
3. Both should produce similar exploitability (~20 mbb/h) and identical record format

This validates the domain pipeline produces correct results before we start optimizing.

---

## Domain Model Investments

During wiring, these new domain concepts should be added:

1. **RecordWriter** (Task 1) — replaces raw BufWriter+write_record
2. **NeuralNetEvaluator** (Task 2) — proper BoundaryEvaluator implementation, not ad-hoc GPU code
3. **DomainPipeline** (Task 3) — clean coordinator that composes domain objects
4. **SolverConfig should derive Clone** — needed by pipeline loop

If additional domain gaps are discovered during implementation (e.g., missing GameBuilder config options, bet size fuzzing), add them as domain methods rather than pipeline hacks.

## Important Notes

1. **Make these functions pub(crate):** `parse_bet_sizes_all` in `turn_generate.rs` (for GameBuilder/pipeline config parsing).

2. **SolverConfig needs Clone** — the pipeline creates one config and clones it per game.

3. **The NeuralNetEvaluator** bridges two interfaces: the domain's `BoundaryEvaluator::evaluate(&Game)` and the existing `LeafEvaluator::evaluate_boundaries(combos, board, ranges, requests)`. The bridge code is extracted from `evaluate_game_boundaries()`.

4. **Sequential only for now** — no threading, no rayon, no GPU batching. The domain pipeline runs one game at a time. Optimization is a separate task after correctness is validated.

5. **The "domain" mode is additive** — existing modes ("model", "exact", "iterative") continue to work unchanged. The new mode exists alongside them until validated.
