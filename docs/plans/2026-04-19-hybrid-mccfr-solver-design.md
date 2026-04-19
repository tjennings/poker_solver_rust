# Hybrid MCCFR Solver — Design

**Date:** 2026-04-19
**Bean:** `poker_solver_rust-mx1j`
**Status:** design-approved, ready for implementation planning
**Supersedes:** current Subgame mode (Modicum K=4 rollout precompute — broken per `docs/plans/2026-04-19-subgame-next-steps.md`)

## Background

See `docs/research/2026-04-19-subgame-solving-literature.md` for the research context. Short version: our current Subgame mode implements Modicum-style K=4 biased-continuation precompute via MCCFR rollouts. Measured problems:

- **depth=0 catastrophically exploitable** — 11,354 mbb/hand vs Exact's 38.6 (structural shallowness, not a bug)
- **depth=1 blocked on precompute** — 11,960 serial rollouts × ~100 ms = 20 minutes before DCFR iter 1
- **Modicum paper uses table lookups, not rollouts** — our implementation is paying 5 orders of magnitude more per boundary than any published system

Rather than pursue incremental fixes to the rollout precompute, we are replacing the Subgame solver in place.

## Goals

- Replace the broken K=4 rollout precompute with live MCCFR-at-boundary sampling.
- Keep near-tree algorithm identical to Exact (DCFR with full regret/strategy tables) — so Exact-vs-Hybrid comparison isolates boundary evaluation as the only variable.
- Three config knobs: `depth_limit`, `boundary_refresh_interval`, `samples_per_refresh`.
- Delete all K=4 biased-continuation precompute code paths. Keep `RolloutLeafEvaluator` and `BiasType` for `validate-rollout` diagnostics.
- `compare-solve` works unchanged — it selects backend via mode enum.
- Tauri UI and metrics surface the new knobs and expose MCCFR performance telemetry.

## Non-goals (v1)

- No K=4 biased continuations. Revisit as a 4th knob only if empirical data shows a safety gap.
- No variance reduction (baselines, antithetic, control variates).
- No neural boundary evaluator (`cfvnet` integration is separate future work).
- No cross-iteration sample caching beyond the refresh interval.

## Architecture

```
┌─────────────────── HybridMCCFR solver ──────────────────────────┐
│                                                                 │
│  Near tree (depth ≤ d):                                         │
│    ┌──────────────────────────────────────────────┐             │
│    │  Standard DCFR traversal                     │             │
│    │  (exact same code path as Exact mode)        │             │
│    │  - regret / strategy tables                  │             │
│    │  - α/β/γ discounting                         │             │
│    └──────────────────┬───────────────────────────┘             │
│                       │  at each boundary visit:                │
│                       ▼                                         │
│           ┌──────────────────────────────┐                      │
│           │ HybridBoundaryEvaluator      │                      │
│           │   .compute_cfvs(bdry,        │                      │
│           │                 oop_range,   │                      │
│           │                 ip_range,    │                      │
│           │                 iter)        │                      │
│           └──────────────┬───────────────┘                      │
│                          │                                      │
│         ┌────────────────┴───────────────────┐                  │
│         │ refresh stale?  (iter % R == 0)    │                  │
│         │   yes → re-sample N trajectories   │                  │
│         │   no  → return cached CFV vectors  │                  │
│         └────────────────────────────────────┘                  │
│                          │                                      │
│         ┌────────────────┴───────────────────┐                  │
│         │ Sampler: N trajectories from       │                  │
│         │   boundary → showdown using        │                  │
│         │   blueprint strategy (unbiased).   │                  │
│         │   Amortize across both sides.      │                  │
│         │   Output: (oop_cfvs, ip_cfvs).     │                  │
│         └────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Design invariants

- **Exact equivalence**: when `depth_limit` ≥ tree depth, Hybrid produces byte-identical strategies to Exact (no boundaries reached, sampling never invoked).
- **Boundary freeze between refreshes**: within a refresh window, boundary CFVs are constant, so DCFR converges against stable values (Modicum-style stability). At each refresh, boundary values re-adapt to the current strategy.
- **Knob degeneracies**:
  - `samples_per_refresh = ∞` + `refresh_interval = ∞` → exhaustive (Exact)
  - `samples_per_refresh = 1` + `refresh_interval = 1` → pure external-sampling MCCFR

## Components

### New

**`HybridBoundaryEvaluator`** (new module, or in `postflop.rs`)

```rust
pub struct HybridBoundaryEvaluator {
    sampler: RolloutLeafEvaluator,
    refresh_interval: u32,
    samples_per_refresh: u32,
    cached_cfvs: RwLock<HashMap<BoundaryId, BoundaryCfvEntry>>,
    current_iter: AtomicU32,
    metrics: Mutex<HybridSolveMetrics>,
}

struct BoundaryCfvEntry {
    cfvs: BoundaryCfvs,
    refreshed_at_iter: u32,
}

pub struct BoundaryCfvs {
    pub oop_cfvs: Vec<f32>,
    pub ip_cfvs:  Vec<f32>,
}

impl HybridBoundaryEvaluator {
    pub fn compute_cfvs(
        &self,
        boundary: &Boundary,
        oop_range: &[f32],
        ip_range:  &[f32],
        iteration: u32,
    ) -> BoundaryCfvs;
}
```

**`HybridSolveConfig`** (new struct, threaded through `build_solve_game`)

```rust
pub struct HybridSolveConfig {
    pub depth_limit: u8,
    pub boundary_refresh_interval: u32,
    pub samples_per_refresh: u32,
}
```

Defaults: `depth_limit=1, boundary_refresh_interval=10, samples_per_refresh=100`.

**`HybridRefreshMetrics`** + **`HybridSolveMetrics`** (new — see "Metrics" section).

### Reused (unchanged)

- `RolloutLeafEvaluator` — single-trajectory blueprint rollout; new entry point `sample_boundary(oop_range, ip_range, n_samples) → BoundaryCfvs`.
- Near-tree DCFR solver — zero changes.
- `compute_boundary_reach` — extend to return both players' reaches in one tree walk.

### Deletions

- K-continuation precompute loop at `game_session.rs:1921-1974` (~50 lines)
- `set_boundary_cfvs_multi` + multi-K indexing — replaced by single-K `set_boundary_cfvs`
- K=4 dispatch logic in `SolveBoundaryEvaluator` at `game_session.rs:1416+`
- `compare_solve.rs`'s `--dump-boundary-cfvs` K=4 output → single-K output

### Kept for diagnostics

- `RolloutLeafEvaluator` (still used by `validate-rollout`)
- `BiasType::{Fold, Call, Raise}` + `bias_strategy()` (future re-introduction, validate-rollout)

## Data flow (per DCFR iteration)

```
iter_n start
  │
  ├─ if n == 0 or n % refresh_interval == 0:
  │    record refresh start
  │    for each boundary in parallel (rayon):
  │      oop_range_n = compute_boundary_range(boundary, σ_n, OOP)
  │      ip_range_n  = compute_boundary_range(boundary, σ_n, IP)
  │      cfvs_n      = sampler.sample_boundary(
  │                      boundary, oop_range_n, ip_range_n,
  │                      samples_per_refresh)
  │      cached[boundary] = (cfvs_n, n)
  │    emit HybridRefreshMetrics event
  │
  ├─ DCFR near-tree traversal:
  │    at each boundary visit:
  │      return cached[boundary].cfvs
  │
  └─ DCFR updates regrets + strategy σ_{n+1}
```

## Error handling

- **Cache miss on iter 0**: bulk refresh happens before any DCFR traversal — no cache-miss path needed.
- **`samples_per_refresh = 0`**: reject at config-load with clear error.
- **`boundary_refresh_interval = 0`**: normalize to 1 with warning.
- **Range size mismatch**: panic (per `feedback_no_fallbacks`).
- **Blueprint rollout failure**: panic — indicates tree inconsistency, not recoverable.

## Testing

1. **Unit: `HybridBoundaryEvaluator::compute_cfvs`**
   - Seeded RNG + known ranges + degenerate blueprint → CFVs match analytic equity within tolerance.
   - Convergence: CFV at N=1 within variance bound; N=10k within 1% of analytic.

2. **Integration: Exact-equivalence**
   - `depth_limit` > tree depth → byte-identical strategies to Exact.

3. **Regression: `compare-solve` on izod repro**
   - `./compare-solve ... --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d"`
   - Success criterion v1: **depth=1 Hybrid within 500 mbb/hand of Exact, completing in ≤2× Exact wall-time.** Anything better is gravy.
   - Depth=0 Hybrid must be strictly better than broken-Subgame depth=0 (< 11,354 mbb/hand).

4. **Perf smoke**: `bench-rollout` reports per-refresh latency.

## Migration

- **Config keys**: `subgame_depth_limit` → `hybrid_depth_limit` (with legacy read compat for one release).
  - New: `hybrid_boundary_refresh_interval`, `hybrid_samples_per_refresh`
- **Tauri UI**: existing "Depth limit" input stays. Two new inputs below. Group under "Hybrid Solver" section. Mode radio label: "Subgame" → "Hybrid". Internal enum: `SolveMode::Hybrid`.
- **CLI**: `compare-solve --subgame-depth-limit` aliased to `--hybrid-depth-limit`; new `--hybrid-refresh-interval`, `--hybrid-samples-per-refresh`.
- **No snapshot migration**: snapshots store blueprint strategies, not solver configs.

## Tauri configuration panel

New "Hybrid Solver" group in Settings, adjacent to current `subgame_depth_limit`:

```
┌─ Hybrid Solver ─────────────────────────────────────┐
│                                                     │
│   Depth limit            [  1 ]  ⓘ                  │
│   Refresh interval       [ 10 ]  iters  ⓘ           │
│   Samples per refresh    [100 ]  ⓘ                  │
│                                                     │
│   ┌─ Preset ──┐                                     │
│   │ Fast      │ depth=0, interval=20, samples=50    │
│   │ Balanced  │ depth=1, interval=10, samples=100   │
│   │ Accurate  │ depth=2, interval=5,  samples=500   │
│   └───────────┘                                     │
└─────────────────────────────────────────────────────┘
```

Presets populate all three at once.

## MCCFR performance metrics

### Per-refresh telemetry

```rust
pub struct HybridRefreshMetrics {
    pub iteration: u32,
    pub refresh_wall_ms: f32,
    pub boundaries_refreshed: u32,
    pub total_samples_drawn: u64,
    pub mean_cfv_variance: f32,   // estimator noise across boundaries
    pub mean_cfv_drift:    f32,   // L2 delta vs previous refresh
    pub samples_per_sec:   f32,
}
```

### Per-solve aggregate

```rust
pub struct HybridSolveMetrics {
    pub refresh_count: u32,
    pub total_refresh_ms: f32,
    pub total_dcfr_ms: f32,
    pub refresh_overhead_pct: f32,
    pub final_mean_cfv_variance: f32,
    pub cfv_drift_trajectory: Vec<f32>,
}
```

### Rationale

- `mean_cfv_drift`: key knob-tuning signal. High → interval too long or samples too low. Rapidly → 0 means over-refreshing.
- `refresh_overhead_pct`: Amdahl's-law indicator. If dominant, slow down refreshes.
- `mean_cfv_variance`: separates sampling noise from strategy drift in exploitability trace.

### Surfaces

1. **Live stream** — extend existing solve-progress event channel with per-iter + per-refresh payloads.
2. **"Solver Telemetry" panel** (collapsible, default collapsed) in explorer. Live iteration, exploitability, refresh overhead %. Sparklines for `cfv_drift` and `mean_cfv_variance`. End-of-solve summary.
3. **CLI** — `compare-solve --verbose` prints aggregate metrics. `--metrics-json <path>` dumps per-refresh trajectory.

## Downstream value

- **Auto-tuning groundwork**: logged `HybridSolveMetrics` enables future heuristic knob suggestions.
- **Comparison harness**: `compare-solve` dumps Exact + Hybrid metrics side-by-side.
- **Tuning doc**: after ship, sweep knob grid and write `docs/hybrid-solver-tuning.md` with spot-type recommendations.

## Success criteria

1. Depth=1 Hybrid exploitability on izod repro within 500 mbb/hand of Exact (stretch: within 100 mbb).
2. Depth=1 Hybrid wall-time ≤ 2× Exact (stretch: ≤ 1×).
3. First DCFR iteration begins within 2 seconds of solve start (no long precompute).
4. Byte-equivalence with Exact when `depth_limit` ≥ tree depth.
5. No regressions in `validate-rollout` or existing `compare-solve` diagnostics.

## Open follow-ups (post-ship)

- `BoundaryNet` training on generated (board, ranges, pot, stack) samples — replaces rollout sampler with neural evaluator (µs per boundary).
- Variance reduction via MCCFR baselines (Davis et al. 2020) if MC noise dominates.
- Safe subgame solving (Libratus gift construction) as optional 4th knob for exploitability-sensitive users.

## References

- Research report: `docs/research/2026-04-19-subgame-solving-literature.md`
- Prior handoff: `docs/plans/2026-04-19-subgame-next-steps.md`
- Bean: `poker_solver_rust-mx1j`
