# Phase 5: Training UI Design

**Date:** 2026-02-04
**Goal:** Add convergence chart showing exploitability over iterations for Kuhn Poker training

---

## Scope

Enhance the Phase 4 MVP with:
- Checkpoints input to control chart granularity
- Convergence chart (exploitability vs iterations)
- Updated results display with final exploitability

**Not included (future work):**
- Async/background training
- HUNL Preflop training
- Config file loading
- Device selection

---

## Rust Command

New command in `crates/tauri-app/src/commands.rs`:

```rust
#[derive(Serialize)]
pub struct Checkpoint {
    pub iteration: u64,
    pub exploitability: f64,
    pub elapsed_ms: u64,
}

#[derive(Serialize)]
pub struct TrainingResultWithCheckpoints {
    pub checkpoints: Vec<Checkpoint>,
    pub strategies: HashMap<String, Vec<f64>>,
    pub total_iterations: u64,
    pub total_elapsed_ms: u64,
}

#[tauri::command]
pub fn train_with_checkpoints(
    total_iterations: u64,
    checkpoint_interval: u64,
) -> Result<TrainingResultWithCheckpoints, String>
```

**Behavior:**
- Trains Kuhn Poker using MCCFR
- Pauses every `checkpoint_interval` iterations to calculate exploitability
- Returns all checkpoints plus final strategies

---

## Frontend UI

```
┌─────────────────────────────────────────┐
│  Poker Solver                           │
├─────────────────────────────────────────┤
│  Iterations: [10000]  Checkpoints: [20] │
│  [Train Kuhn Poker]                     │
├─────────────────────────────────────────┤
│  Convergence Chart                      │
│  ┌─────────────────────────────────┐    │
│  │ Exploitability (log scale)      │    │
│  │ 0.5 ┤ •                         │    │
│  │     │  •                        │    │
│  │ 0.1 ┤   • •                     │    │
│  │     │      • • • • • • • • •    │    │
│  │ 0.0 └────────────────────────── │    │
│  │     0        5000        10000  │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│  Training Results                       │
│  Time: 245ms | Final Exploitability: 0.001│
│  [Strategy Table]                       │
└─────────────────────────────────────────┘
```

**Controls:**
- Iterations input (default 10000)
- Checkpoints input (default 20)
- Interval calculated as `iterations / checkpoints`

**Chart:**
- recharts LineChart
- X-axis: iteration count
- Y-axis: exploitability (consider log scale for better visualization)

---

## Dependencies

**Frontend:**
```json
"recharts": "^2.12.0"
```

---

## Files to Modify

- `crates/tauri-app/src/commands.rs` - Add `train_with_checkpoints`
- `crates/tauri-app/src/lib.rs` - Export new command
- `crates/tauri-app/src/main.rs` - Register handler
- `frontend/src/types.ts` - Add TypeScript types
- `frontend/src/App.tsx` - Add chart and checkpoints input
- `frontend/src/App.css` - Chart styling

---

## Success Criteria

1. Training 10,000 iterations with 20 checkpoints completes in < 1 second
2. Chart shows exploitability decreasing from ~0.5 to < 0.01
3. Final strategies match known Kuhn Nash equilibrium values
4. UI remains responsive during training
