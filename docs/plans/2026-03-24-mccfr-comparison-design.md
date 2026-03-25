# Phase 2: Blueprint MCCFR Comparison — Design

## Goal

Run the blueprint MCCFR solver on the same Flop Poker game used for the exact baseline, and compare its strategy against the golden baseline. This validates the MCCFR implementation and measures abstraction error.

## Game Setup

Same Flop Poker game as Phase 1:
- Board: QhJdTh, all combos (~1176/player), 20bb effective, 2bb pot
- Bet sizes: 67% pot, all-in via thresholds
- Full street transitions: flop → turn → river

## Approach: Wrap BlueprintTrainer

### MCCFR Adapter (`solvers/mccfr.rs`)

Wraps `BlueprintTrainer` from `poker-solver-core`, implementing the `ConvergenceSolver` trait.

**Configuration:**
- Builds a `BlueprintV2Config` from `FlopPokerConfig`
- Trivial preflop: single action (limp) so every deal goes straight to the flop
- Fixed flop: QhJdTh
- **Bucketing: 169 canonical preflop hand indices for ALL streets** (flop/turn/river). Ignores board interaction — this is a pipeline validation, not a realistic abstraction. Proper clustering is a separate future phase.
- Default 1M iterations, configurable

**`solve_step()`:** Runs one batch of MCCFR iterations via `traverse_external()`.

**`average_strategy()`:** Extracts bucket-level strategy from `BlueprintStorage::average_strategy(node_idx, bucket)`, lifts to combo-level by mapping each combo to its canonical hand bucket (0-168). Returns `StrategyMap` compatible with baseline.

**`self_reported_metrics()`:** Reports strategy delta and avg positive regret from the trainer's built-in tracking.

### Strategy Lifting

All combos in the same bucket get the same action distribution (simple assignment, no reach-weighting). This is definitionally what the abstracted strategy produces — the forced uniformity IS the abstraction error we want to measure.

### Exploitability Computation

Evaluate the MCCFR strategy in the full (unabstracted) game:

1. Build a range-solver `PostFlopGame` with identical parameters
2. Allocate memory, do NOT solve
3. Walk the tree — at each decision node:
   - Map each combo to its canonical hand bucket
   - Get MCCFR strategy for that bucket
   - Assemble combo-level strategy vector
   - Lock via `game.lock_current_strategy()`
4. Call `compute_exploitability(&game)` — runs best-response against locked strategy

**Node correspondence:** Both solvers built from the same `FlopPokerConfig`, so action trees align. Verified by assertion on action counts per node.

### Configurable Exploitability Checkpoints

A list of iteration milestones where the harness pauses MCCFR, lifts the full strategy, computes exploitability in the real game, then resumes training.

- Default: `[1_000, 10_000, 100_000, 500_000, 1_000_000]`
- Overridable via `--checkpoints` CLI flag
- Self-reported metrics (strategy delta, avg regret) logged continuously between checkpoints

## CLI

```
cargo run -p convergence-harness --release -- run-solver \
  --solver mccfr \
  --iterations 1000000 \
  --checkpoints 1000,10000,100000,500000,1000000 \
  --baseline-dir baselines/flop_poker_v1 \
  --output-dir results/mccfr_169bkt_run1
```

## Output

**During run:** Self-reported metrics periodically. Exploitability + L1 distance at each checkpoint.

**On exit:** SB strategy matrix (colored 13x13), then saves:

```
results/mccfr_169bkt_run1/
  summary.json
  convergence.csv       # iteration, exploitability, strategy_delta, elapsed_ms
  strategy.bin           # combo-level lifted strategy
  combo_ev.bin           # empty or bucket-averaged
  comparison/
    report.txt
    strategy_distance.csv
    combo_ev_diff.csv
    summary.json
```

Output directory is a valid `Baseline` format — `compare` can re-run against it later.

## Key Design Decisions

1. **Trivial preflop** — Use existing full-game tree builder with a single preflop action. Avoids modifying core crate.
2. **169 canonical buckets for all streets** — Pipeline validation only. Ignores board interaction. Proper clustering is a separate phase.
3. **Wall-clock time as primary comparison** — MCCFR iterations are orders of magnitude cheaper than DCFR. Comparing iteration counts is misleading.
4. **Configurable checkpoints** — Exploitability is expensive to compute (full tree walk + best-response). Compute at milestones, not every iteration.
5. **Strategy lifting = simple assignment** — All combos in a bucket get the same strategy. No reach-weighting.

## Expected Results

- With 169 buckets (preflop-only abstraction), expect high exploitability (~200+ mbb/hand) since board interaction is ignored
- MCCFR convergence will be slower than DCFR per wall-clock time on this small game (MCCFR's advantage is on games too large for full traversal)
- The 169-bucket result validates the full pipeline: training → extraction → lifting → exploitability → comparison
