---
# poker_solver_rust-jpwu
title: Parallelize Modicum K-continuation precompute in subgame solver
status: todo
type: task
priority: high
created_at: 2026-04-19T02:01:01Z
updated_at: 2026-04-19T02:01:01Z
---

Subgame solver startup is dominated by the Modicum K-continuation precompute loop in `crates/tauri-app/src/game_session.rs:1915-1974`. The triple-nested loop runs 8N rollout calls (N boundaries × 2 players × 4 biases) serially before DCFR iteration 1 begins.

**Observed impact:** On a 4bet-called flop spot (SB:2bb, BB:10bb, SB:22bb, BB:call | Jd9d7d), each rollout call takes ~1.3s. User reports 75+ seconds of precompute with no DCFR iterations yet — Exact mode finishes the whole solve in ~2min on the same spot. Subgame shouldn't lose to Exact.

**Fix:** parallelize the inner bias loop (and optionally the outer boundary loop) with rayon. Each rollout call inside the loop is independent — they write to distinct `game.set_boundary_cfvs_multi(ordinal, player, ki, cfvs)` slots.

**Structure:**
```rust
// Current
for ordinal in 0..n_boundaries {
    for player in 0..2 {
        for (ki, &bias) in biases.iter().enumerate().take(k) {
            // build fresh evaluator with this bias, call evaluate_boundaries
            // write to game.set_boundary_cfvs_multi(ordinal, player, ki, cfvs)
        }
    }
}

// Proposed (parallel over (ordinal, player, ki) tuples)
let work: Vec<_> = (0..n_boundaries)
    .flat_map(|o| (0..2).flat_map(move |p| (0..k).map(move |ki| (o, p, ki))))
    .collect();
let results: Vec<((usize, usize, usize), Vec<f32>)> = work.par_iter()
    .map(|&(ordinal, player, ki)| {
        // construct evaluator with biases[ki], call evaluate_boundaries
        ((ordinal, player, ki), cfvs)
    })
    .collect();
// Serial apply results (set_boundary_cfvs_multi probably requires &mut game)
for ((o, p, ki), cfvs) in results {
    game.set_boundary_cfvs_multi(o, p, ki, cfvs);
}
```

**Expected speedup:** on an 8-core machine, 4-8× reduction in precompute wall-time.

**Risks:**
- Each rollout must use its own RNG state (already does via `Arc<AtomicU64> call_counter`).
- The `RolloutLeafEvaluator` construction copies Arc references; no contention expected.
- Memory: parallel evaluator instances each hold their own scratch. For 8 threads × 4 biases in flight, memory footprint rises modestly.

**Validation:**
- Compare per-boundary CFVs between serial and parallel implementations on a small scenario — must match within RNG-seed tolerance.
- Re-time the 4bet flop spot — expect 75s+ precompute to drop to 10-20s.

**Out of scope (for future beans):**
- Lazy/on-demand boundary CFV computation (only compute CFVs DCFR actually needs).
- Caching CFVs across snapshot navigations within a session.

Context: identified during 2026-04-19 debugging session with user. Exact mode outperforming Subgame mode on a real spot is the concrete pain point.
