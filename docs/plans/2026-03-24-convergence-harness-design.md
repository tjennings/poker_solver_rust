# Convergence Validation Harness — Design

## Goal

Build a validation harness to test convergence of CFR algorithms against an exact baseline. A small, tractable game ("Flop Poker") is solved exhaustively once to produce a golden baseline. Future algorithms run on the same game and are evaluated against that baseline.

## Game Definition: "Flop Poker"

- **Board:** QhJdTh (flop), all turn/river runouts dealt by the solver
- **Ranges:** All non-conflicting combos (~1,176 per player, full deck)
- **Starting pot:** 2bb
- **Effective stack:** 50bb
- **Bet sizes:** 33% pot, 67% pot (both players, all streets)
- **All-in:** Always available
- **Raise cap:** 1 raise per street
- **Streets:** Flop → Turn → River (full street-to-street transitions)

If all-combo solving proves too slow, the fallback is trimming the deck rather than filtering ranges.

## Architecture

### New crate: `convergence-harness`

```
crates/convergence-harness/
  src/
    main.rs                # CLI entry point
    game.rs                # FlopPokerGame definition, game tree construction
    solver_trait.rs         # ConvergenceSolver trait
    solvers/
      mod.rs
      exhaustive.rs        # Wraps range-solver DCFR
      mccfr.rs             # Wraps blueprint MCCFR (Phase 2)
    baseline.rs            # Serialize/deserialize baseline artifacts
    evaluator.rs           # Exploitability, L1 distance, combo EV diff
    reporter.rs            # Terminal output, CSV/JSON writers, human summary
```

**Dependencies:**
- `range-solver` — game tree construction, DCFR solving, best-response/exploitability
- `poker-solver-core` — blueprint MCCFR, bucketing, DCFR params (Phase 2)

### Pluggable solver interface

```rust
trait ConvergenceSolver {
    fn name(&self) -> &str;
    fn solve_step(&mut self);           // advance one iteration (or one batch)
    fn iterations(&self) -> u64;
    fn average_strategy(&self) -> StrategyMap;
    fn combo_evs(&self) -> ComboEvMap;
    fn self_reported_metrics(&self) -> SolverMetrics;
}
```

The harness drives the loop for every solver:
1. Call `solve_step()`
2. At configured intervals, snapshot strategy and combo EVs
3. Compute exploitability using range solver's best-response code
4. Record convergence curve and timing
5. At the end, compare against the persisted baseline

### Baseline generation

The range solver (exhaustive DCFR) solves Flop Poker once to near-zero exploitability. The harness captures:

- **Convergence curve:** `Vec<(iteration, exploitability, elapsed_ms)>` sampled densely
- **Final average strategy:** action probabilities at every info set for every combo
- **Combo EVs:** per-player, per-combo EV at every info set
- **Final exploitability**
- **Timing:** wall-clock time per milestone

Persisted as a versioned baseline artifact:

```
baselines/flop_poker_v1/
  game_config.yaml        # exact game definition (reproducibility)
  convergence.csv         # iteration, exploitability, elapsed_ms
  strategy.bin            # bincode: full strategy map
  combo_ev.bin            # per-player, per-combo EV at every info set
  summary.json            # final exploitability, total iterations, total time
```

The baseline is solved once and kept as a fixed dataset for all future comparisons.

### Comparison & reporting

When a solver completes, the harness compares against the persisted baseline:

**Metrics:**
- **Exploitability** of the solver's strategy in the full game (via range solver best-response)
- **L1 strategy distance** vs baseline, per info set, reach-weighted average
- **Combo EV difference** vs baseline — per-player, per-combo, at every info set
- **Convergence rate** — exploitability vs iterations and vs wall-clock time

**Output:**
- **Terminal:** progress during the run, final summary
- **CSV/JSON artifacts:** convergence curve, per-node strategy distances, combo EV diffs
- **Human summary:** plain-English report with key findings

```
results/<solver_name>_<run_id>/
  convergence.csv
  strategy_distance.csv    # per-node L1 vs baseline
  combo_ev_diff.csv        # per-node, per-combo EV delta vs baseline
  summary.json             # machine-readable metrics
  report.txt               # human-readable summary
```

### CLI subcommands

- `generate-baseline` — solve Flop Poker with exhaustive DCFR, persist baseline
- `run-solver --solver <name>` — run a solver, compute metrics vs baseline
- `compare` — re-compare a saved solver result against the baseline (no re-solving)

## Phase Breakdown

### Phase 1 (this effort)
- New `convergence-harness` crate
- `FlopPokerGame` definition
- `ConvergenceSolver` trait
- Exhaustive solver adapter (wraps range-solver)
- Baseline generation, serialization, loading
- Evaluator (exploitability, L1 distance, combo EV diff)
- Reporter (terminal, CSV/JSON, human summary)
- `generate-baseline` and `compare` CLI subcommands

### Phase 2 (future)
- Run clustering pipeline on Flop Poker to produce buckets
- MCCFR solver adapter (wraps blueprint solver with bucket-to-combo lifting)
- `run-solver --solver mccfr` subcommand
- Ablation support (vary bucket counts, compare exploitability vs abstraction granularity)

### Phase 3 (future)
- Additional CFR variant adapters (vanilla CFR, CFR+, Linear, PCFR+, new algorithms)
- Convergence rate comparison charts across algorithms
- Parameterized game definitions (different boards, stack depths, bet structures)

## Key Design Decisions

1. **Baseline is a fixed artifact** — solved once with the range solver, persisted, never re-solved. All future algorithms compare against this fixed dataset.

2. **Action trees must match** — the exact solver and any algorithm under test must use identical game trees (same bet sizes, raise caps, etc.) to isolate card abstraction error from action abstraction error.

3. **Exploitability computed uniformly** — regardless of solver, exploitability is always computed by the range solver's best-response code on the full (unabstracted) game tree. This ensures apples-to-apples comparison.

4. **Algorithm-agnostic trait** — the `ConvergenceSolver` trait is minimal and imposes no assumptions about how a solver works internally. This supports future algorithms without redesigning the harness.

5. **L1 over L2 for strategy distance** — L1 equals total variation distance, which has direct game-theoretic interpretation: it bounds maximum EV difference achievable by switching strategies.
