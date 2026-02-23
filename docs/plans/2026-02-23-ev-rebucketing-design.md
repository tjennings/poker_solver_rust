# EV-Based Postflop Rebucketing

## Problem

Current postflop bucketing clusters hands on **equity histograms** (P(win) distribution over future boards). This is pot-size-agnostic: a nut flush and second-nut flush both show ~95%+ equity, but their realized EV diverges dramatically. Nut hands extract value *from* strong-but-second-best hands in large pots, while second-best hands pay off in those same spots.

We need buckets that capture **how much value a hand extracts given realistic postflop play**, not just how often it wins.

## Solution: Post-Solve Rebucketing Loop

After the initial EHS-based solve, extract per-hand EV histograms from the converged strategy, re-cluster on those EV features, and re-solve. Repeat for N configurable rounds.

```
For each rebucket_round (1..=N):
  1. Cluster hands into buckets
     - Round 1: EHS equity histograms (current behavior)
     - Round 2+: EV histograms from previous round's converged strategy
  2. Compute bucket-pair equity (pairwise hand-vs-hand evaluation)
  3. Solve postflop CFR per flop
     - Stop at iteration limit OR max-regret-delta threshold, whichever first
  4. Extract per-hand EV histograms from converged strategy
  5. If not last round: re-cluster using EV histograms, go to step 2
  6. Final round: compute PostflopValues from converged strategy
```

With `rebucket_rounds: 1`, behavior is identical to today (pure EHS, no rebucketing).

## EV Histogram Feature

For each (hand, flop), after CFR converges, build an **EV histogram CDF**: the distribution of realized EVs across opponent buckets.

For hero hand `h` in bucket `b_h`, iterate over all opponent buckets `b_opp` and record the EV from `eval_with_avg_strategy(b_h, b_opp)`. Bin these EVs into a 10-bin histogram CDF (same format as equity histograms).

This captures the full EV profile:
- **Nut hands**: high EV even vs strong opponents (extraction)
- **Second-nut hands**: bimodal distribution (high EV vs weak, negative vs nuts)
- **Medium hands**: low variance, moderate EV (can fold cheaply)

## Early Stopping: Max Regret Delta

Each per-flop CFR solve tracks the **max absolute change in regret-matched strategy** between consecutive iterations. When this drops below a configurable threshold (default 0.001), the solve stops early. Otherwise it runs to the iteration limit.

This replaces the current fixed-iteration-count approach for all postflop solves, not just rebucketing rounds.

## Config Changes

New fields in `PostflopModelConfig`:

```yaml
postflop_model:
  rebucket_rounds: 1              # 1 = EHS only (backward compat), 2+ = EV rebucketing
  rebucket_delta_threshold: 0.001 # max-regret-delta for early CFR stopping
  postflop_sprs: [3.5]            # list of SPRs, starting with one (100bb 3bet/call)
```

The existing `postflop_spr: f64` field is replaced by `postflop_sprs: Vec<f64>` with backward-compatible deserialization (scalar auto-wraps to single-element vec).

## Progress Reporting

Uses `indicatif::MultiProgress` with parallel bars per concurrent flop solve. Each phase gets labeled output with timing:

```
Phase 1/2: EHS Bucketing
⠋ Hand buckets [########>------] 120/169 (12s)

Phase 1/2: Solving (EHS)
⠋ Flop 'AhKd7s' [####>---] iter 45/200 δ=0.0032 (8.2s)
⠋ Flop 'Tc9h2d' [######>-] iter 87/200 δ=0.0018 (14.1s)
⠋ Flop '5s5h3c' converged at iter 38 δ=0.0009 (6.7s) ✓

Phase 2/2: Rebucketing (EV)
⠋ EV histograms [########>------] 120/169
⠋ Flop 'AhKd7s' [####>---] iter 52/200 δ=0.0021 (9.1s)
```

### BuildPhase Extensions

```rust
enum BuildPhase {
    // Existing
    HandBuckets(usize, usize),
    EquityTable,
    Trees,
    Layout,
    ComputingValues,

    // New/updated
    SolvingPostflop {
        round: u16,
        total_rounds: u16,
        flop_name: String,
        iteration: usize,
        max_iterations: usize,
        delta: f64,
    },
    ExtractingEv(usize, usize),   // (hands_done, total)
    Rebucketing(u16, u16),         // (round, total_rounds)
}
```

## Architecture Changes

| Component | Change |
|-|-|
| `PostflopModelConfig` | Add `rebucket_rounds`, `rebucket_delta_threshold`; replace `postflop_spr` with `postflop_sprs: Vec<f64>` |
| `PostflopAbstraction::build()` | Outer loop over rebucket rounds; extract EV histograms after each solve |
| `solve_one_flop()` | Add early-stopping via max-regret-delta check per iteration; return delta alongside strategy_sum |
| `hand_buckets.rs` | New `build_ev_histograms()` to extract EV distribution per (hand, flop) from converged strategy; new `cluster_ev_histograms()` reusing existing k-means infra |
| `BuildPhase` | Richer variants with round/flop/delta info |
| Trainer `main.rs` | Switch to `MultiProgress` with per-flop bars; handle new `BuildPhase` variants |

## What Doesn't Change

- k-means clustering infrastructure (reused for EV histograms)
- Postflop tree building
- Bucket-pair equity computation (pairwise hand-vs-hand method)
- `PostflopValues` format and how preflop solver queries it
- Round 1 behavior when `rebucket_rounds: 1` (identical to current EHS-only path)

## Testing Strategy

- Unit: EV histogram extraction produces valid CDFs
- Unit: rebucketing with rounds=1 produces identical results to current code
- Unit: early stopping triggers when delta is below threshold
- Unit: early stopping does not trigger when delta is above threshold
- Integration: rebucket_rounds=2 produces different bucket assignments than rounds=1
- Integration: nut hands and second-nut hands land in different EV buckets on coordinated boards
