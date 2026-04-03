# Iterative Pipeline Performance Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the iterative boundary re-evaluation pipeline from 0.9/s to 30-70/s while preserving ~20 mbb/h exploitability, using incremental phases with validation gates.

**Architecture:** Phase 1 adds eval_interval to the existing lockstep (no arch changes). Phase 2 replaces lockstep with two queues and persistent solver threads, with flush logic moved to the GPU thread. Phase 3 is config tuning. Each phase is validated before proceeding.

**Tech Stack:** Rust, burn (wgpu), range-solver, std::sync::mpsc channels, std::thread

**Design doc:** `docs/plans/2026-04-02-iterative-pipeline-perf-design.md`
**Branch:** `feat/iterative-pipeline-perf`
**Baseline:** commit `a1fc4d51` — lockstep, 19.6 mbb/h, 0.9/s

---

## Context for Implementer

The working lockstep loop is at `crates/cfvnet/src/datagen/turn_generate.rs:1574`. It does:

```
while !active.is_empty() {
    evaluate_pool_boundaries(...)   // GPU eval ALL games every iteration
    thread::scope { solve_step }    // 1 iteration per game, parallel
    flush_boundary_caches()         // clear reaches + cfvs
    graduate/inject
}
```

Key functions:
- `evaluate_pool_boundaries()` at line ~221 — batch GPU eval using `build_iterative_game_inputs`
- `flush_boundary_caches()` — clears both `boundary_reach` and `boundary_cfvs` on a game
- `solve_step(game, iter)` — one DCFR iteration (from range_solver crate)
- `extract_results(game, pot, ranges, exploit)` — extracts CFVs after finalize

**CRITICAL RULE: After each phase, you MUST run the iterative pipeline and verify exploitability is ~20 mbb/h (not 29,000). If exploitability regresses, STOP and investigate. Do not proceed to the next phase.**

Test command:
```bash
cargo run -p cfvnet --release -- generate -c sample_configurations/turn_cfvnet_test.yaml
```

Config must have `mode: "iterative"`, `solver_iterations: 100` (or 300), `leaf_eval_interval: 1` (for validation). Look for `expl:XX.X mbb/h` in the progress bar. It should be < 100 mbb/h, ideally ~20.

---

### Task 1: Phase 1 — Add eval_interval to lockstep

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs` (the lockstep loop at line ~1574)

**Step 1: Read eval_interval from config**

At the top of `generate_turn_training_data_iterative`, after the existing config reads (~line 1474), add:

```rust
let eval_interval = if config.datagen.leaf_eval_interval == 0 {
    1 // 0 means every iteration
} else {
    config.datagen.leaf_eval_interval
};
```

**Step 2: Guard the GPU eval and flush**

Replace the lockstep loop body (lines ~1574-1650) with:

```rust
while !active.is_empty() {
    // 1. GPU: evaluate boundaries only at eval_interval boundaries.
    //    Check if ANY game in the pool needs eval this iteration.
    let any_needs_eval = active.iter().any(|(_, _, iter)| *iter % eval_interval == 0);
    if any_needs_eval {
        evaluate_pool_boundaries(&model, &device, &mut active);
    }

    // 2. Solve: one iteration per game, parallel via std::thread::scope.
    let chunk_size = (active.len() + threads - 1) / threads;
    std::thread::scope(|s| {
        for chunk in active.chunks_mut(chunk_size.max(1)) {
            s.spawn(|| {
                range_solver::set_force_sequential(true);
                for (_sit, game, iter) in chunk.iter_mut() {
                    solve_step(game, *iter);
                    *iter += 1;
                }
            });
        }
    });

    // 3. Flush boundary caches ONLY for games whose next iteration needs GPU eval.
    //    This clears stale CFVs so the GPU sets fresh ones, while preserving
    //    CFVs for games that will reuse them next iteration.
    for (_sit, game, iter) in &active {
        if *iter % eval_interval == 0 {
            game.flush_boundary_caches();
        }
    }

    // (progress bar + graduate/inject unchanged)
    // ...
}
```

**Step 3: Add eval_interval to progress bar and startup log**

After the pool filled message, add:
```rust
eprintln!("[iterative] leaf_eval_interval={eval_interval}");
```

**Step 4: Verify compilation**

Run: `cargo build -p cfvnet --release 2>&1 | tail -5`

**Step 5: Validate exploitability with eval_interval=1**

Set config: `leaf_eval_interval: 1` (equivalent to current lockstep — GPU eval every iteration).

Run: `cargo run -p cfvnet --release -- generate -c sample_configurations/turn_cfvnet_test.yaml`

**GATE: Exploitability must be ~20 mbb/h (same as baseline). If not, STOP.**

**Step 6: Test eval_interval=10**

Set config: `leaf_eval_interval: 10`.

Run the same command. Note throughput and exploitability.

**GATE: Exploitability must still be < 100 mbb/h. Throughput should be ~5-10x better than baseline.**

**Step 7: Commit**

```
git commit -m "perf(phase1): add eval_interval to lockstep iterative pipeline"
```

---

### Task 2: Phase 1 — Run eval_interval iterations per scope

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

Currently the loop does 1 iteration per scope call, checking eval_interval each time. The scope spawn/join overhead happens every iteration. Instead, run eval_interval iterations inside each scope, then do GPU eval.

**Step 1: Restructure the loop**

Replace the lockstep loop with:

```rust
while !active.is_empty() {
    // 1. GPU: evaluate boundaries for games that need it.
    evaluate_pool_boundaries(&model, &device, &mut active);

    // 2. Solve: run eval_interval iterations per game.
    let iters_this_round = eval_interval;
    let chunk_size = (active.len() + threads - 1) / threads;
    std::thread::scope(|s| {
        for chunk in active.chunks_mut(chunk_size.max(1)) {
            s.spawn(|| {
                range_solver::set_force_sequential(true);
                for (_sit, game, iter) in chunk.iter_mut() {
                    let target = (*iter + iters_this_round).min(solver_iterations);
                    while *iter < target {
                        solve_step(game, *iter);
                        *iter += 1;
                    }
                }
            });
        }
    });

    // 3. Flush boundary caches for next GPU eval.
    for (_sit, game, _iter) in &active {
        game.flush_boundary_caches();
    }

    // (progress bar + graduate/inject unchanged)
}
```

This reduces scope spawn/join from `solver_iterations` times to `solver_iterations / eval_interval` times.

**Step 2: Validate exploitability**

Run with `leaf_eval_interval: 1` — must match baseline ~20 mbb/h.
Run with `leaf_eval_interval: 10` — must stay < 100 mbb/h.

**GATE: Exploitability must hold. Throughput should improve due to reduced scope overhead.**

**Step 3: Commit**

```
git commit -m "perf(phase1): run eval_interval iterations per scope call"
```

---

### Task 3: Phase 2 — Replace lockstep with queues

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

This is the core architectural change. Replace the lockstep loop with persistent solver threads and two queues.

**Step 1: Create the queue infrastructure**

Replace the lockstep loop and surrounding code (from `// --- Fill initial active pool ---` through the end of the loop) with:

```rust
type ActiveGame = (Situation, PostFlopGame, u32);

// Two queues: GPU pushes to ready, solvers pop from ready.
//             Solvers push to eval, GPU pops from eval.
let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<ActiveGame>(pool_size * 2);
let (eval_tx, eval_rx) = std::sync::mpsc::sync_channel::<ActiveGame>(pool_size * 2);
let ready_rx = Arc::new(Mutex::new(ready_rx));

// Seed: all games need initial GPU eval.
for _ in 0..pool_size {
    match deal_rx.recv() {
        Ok((sit, game)) => eval_tx.send((sit, game, 0)).expect("seed eval queue"),
        Err(_) => break,
    }
}

let active_count = Arc::new(AtomicU64::new(pool_size as u64));
```

**Step 2: Create persistent solver threads**

```rust
let mut solver_handles = Vec::with_capacity(threads);
for _ in 0..threads {
    let ready_rx = Arc::clone(&ready_rx);
    let eval_tx = eval_tx.clone();
    let storage_tx = storage_tx.clone();
    let pb = pb.clone();
    let exploit_sum = Arc::clone(&exploit_sum);
    let exploit_count = Arc::clone(&exploit_count);
    let active_count = Arc::clone(&active_count);

    solver_handles.push(std::thread::spawn(move || {
        range_solver::set_force_sequential(true);
        loop {
            let (sit, mut game, mut iter) = match {
                let rx = ready_rx.lock().expect("ready_rx lock");
                rx.recv()
            } {
                Ok(item) => item,
                Err(_) => return,
            };

            // Run eval_interval iterations.
            let target = (iter + eval_interval).min(solver_iterations);
            while iter < target {
                solve_step(&game, iter);
                iter += 1;
            }

            if iter >= solver_iterations {
                // Graduate.
                range_solver::finalize(&mut game);
                let exploit = range_solver::compute_exploitability(&game);
                let pot = f64::from(sit.pot);
                let (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _) =
                    extract_results(&mut game, pot, &sit.ranges, exploit);

                if exploit >= 0.0 {
                    let bb = initial_stack as f32 / 100.0;
                    let exploit_mbb = if bb > 0.0 { exploit / bb * 1000.0 } else { 0.0 };
                    exploit_sum.fetch_add((exploit_mbb * 100.0) as u64, Ordering::Relaxed);
                    exploit_count.fetch_add(1, Ordering::Relaxed);
                }
                pb.inc(1);
                let _ = storage_tx.send((sit, oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv));
                active_count.fetch_sub(1, Ordering::Relaxed);
            } else {
                // Send to eval queue — DO NOT flush here.
                // Reaches stay intact for GPU to read.
                if eval_tx.send((sit, game, iter)).is_err() {
                    return;
                }
            }
        }
    }));
}
drop(eval_tx);  // solver clones are the remaining senders
drop(storage_tx);
```

**Step 3: Create GPU loop (main thread)**

```rust
loop {
    // Drain eval queue into batch.
    let mut batch: Vec<ActiveGame> = Vec::with_capacity(pool_size);
    match eval_rx.recv() {
        Ok(item) => batch.push(item),
        Err(_) => break,
    }
    while batch.len() < pool_size {
        match eval_rx.try_recv() {
            Ok(item) => batch.push(item),
            Err(_) => break,
        }
    }

    // GPU-side flush: read reaches THEN flush THEN eval THEN set CFVs.
    // Step A: build inputs (reads boundary_reach from each game).
    let mut all_inputs: Vec<f32> = Vec::new();
    let mut requests: Vec<BoundaryRequest> = Vec::new();
    let mut rows_per: Vec<usize> = Vec::new();
    for (gi, (sit, game, _iter)) in batch.iter().enumerate() {
        build_iterative_game_inputs(gi, game, sit, &mut all_inputs, &mut requests, &mut rows_per);
    }

    // Step B: flush caches (AFTER reading reaches, BEFORE setting new CFVs).
    for (_sit, game, _iter) in &batch {
        game.flush_boundary_caches();
    }

    // Step C: GPU forward pass + scatter.
    if !all_inputs.is_empty() {
        use burn::tensor::{Tensor, TensorData};
        let total_rows = all_inputs.len() / INPUT_SIZE;
        let data = TensorData::new(all_inputs, [total_rows, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &device);
        let output = model.forward(input_tensor);
        let out_vec: Vec<f32> = output.into_data().to_vec().expect("output tensor conversion");
        for (gi, ordinal, player, cfvs) in decode_boundary_cfvs(&out_vec, &requests, &rows_per) {
            batch[gi].1.set_boundary_cfvs(ordinal, player, cfvs);
        }
    }

    // Push evaluated games to ready queue.
    for game in batch {
        if ready_tx.send(game).is_err() {
            break;
        }
    }

    // Inject new deals.
    while active_count.load(Ordering::Relaxed) < pool_size as u64 {
        match deal_rx.try_recv() {
            Ok((sit, game)) => {
                // New game needs GPU eval — process inline.
                let mut single = vec![(sit, game, 0u32)];
                evaluate_pool_boundaries(&model, &device, &mut single);
                let item = single.into_iter().next().unwrap();
                active_count.fetch_add(1, Ordering::Relaxed);
                if ready_tx.send(item).is_err() { break; }
            }
            Err(_) => break,
        }
    }

    // Update progress bar.
    let ac = active_count.load(Ordering::Relaxed);
    if ac == 0 { break; }
}

// Shutdown.
drop(ready_tx);
for (i, h) in solver_handles.into_iter().enumerate() {
    h.join().map_err(|e| format!("solver {i} panicked: {e:?}"))?;
}
```

**IMPORTANT**: The critical difference from the failed async attempt:
- Flush happens on the GPU thread AFTER `build_iterative_game_inputs` reads reaches
- Previously, flush/clear happened on the solver thread BEFORE sending to eval queue

**Step 4: Validate exploitability with eval_interval=1**

Run: `cargo run -p cfvnet --release -- generate -c sample_configurations/turn_cfvnet_test.yaml`

Config: `mode: "iterative"`, `leaf_eval_interval: 1`, `solver_iterations: 100`

**GATE: Exploitability must be ~20 mbb/h. If it's 29,000 — STOP. The channel transfer is the problem. Do NOT proceed.**

**Step 5: If validation passes, test with eval_interval=10**

Note throughput and exploitability. Both GPU and CPU should show steady utilization.

**Step 6: Commit**

```
git commit -m "perf(phase2): replace lockstep with queues and GPU-side flush"
```

---

### Task 4: Phase 3 — Config tuning and final validation

**No code changes.** This is pure configuration experimentation.

**Step 1: Sweep parameters**

Test the following combinations and record throughput + exploitability:

| active_pool_size | leaf_eval_interval | solver_iterations | Expected behavior |
|-----|------|------|-----|
| 64  | 1    | 300  | Baseline: ~20 mbb/h, low throughput |
| 64  | 10   | 300  | Better throughput, verify exploitability |
| 64  | 30   | 300  | Best throughput, verify exploitability |
| 256 | 10   | 300  | Larger batches, verify GPU utilization |
| 256 | 30   | 300  | Max throughput config |

**Step 2: Find the sweet spot**

The optimal config balances:
- GPU batch size (larger active_pool → better GPU utilization)
- Eval frequency (lower interval → better exploitability, lower throughput)
- Solver threads (match to CPU cores minus GPU/deal/write threads)

**Step 3: Update default config**

Update `sample_configurations/turn_cfvnet_test.yaml` with the best-performing settings.

**Step 4: Commit**

```
git commit -m "chore(phase3): document optimal iterative pipeline settings"
```

---

## Validation Protocol

After EVERY phase:

1. Build: `cargo build -p cfvnet --release`
2. Test: `cargo test -p cfvnet 2>&1 | grep "test result:"`
3. Run iterative pipeline with `leaf_eval_interval: 1`, `solver_iterations: 100`
4. **Check exploitability in progress bar** — must be < 100 mbb/h
5. If exploitability is ~29,000 mbb/h → boundary re-eval is broken, STOP

## Important Notes

1. **`flush_boundary_caches` in Phase 2 MUST happen on the GPU thread** — after reading `boundary_reach` via `build_iterative_game_inputs`, before setting new CFVs. This is the key correctness fix.

2. **New deal injection** in Phase 2: new games (iter=0) need GPU eval before they can solve. The GPU thread handles this inline — eval the new game immediately and push to ready.

3. **`solve_step` takes `&T`** (not `&mut T`) — uses interior mutability. Safe to call from solver threads after receiving from channel.

4. **`range_solver::set_force_sequential(true)`** is thread-local — each solver thread must call it independently.

5. **The `evaluate_pool_boundaries` function** calls `build_iterative_game_inputs` which reads `game.boundary_reach()`. In Phase 2, the GPU loop inlines this logic instead of calling `evaluate_pool_boundaries`, because it needs to flush BETWEEN reading reaches and setting CFVs.
