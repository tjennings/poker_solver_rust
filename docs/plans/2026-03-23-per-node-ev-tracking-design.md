# Per-Node EV Tracking

**Date:** 2026-03-23

## Problem

The `ScenarioEvTracker` only tracks EVs for a small set of configured scenario nodes (typically 4-6). When training resumes from a snapshot, EVs reset to zero. Users want to see EVs at every decision node in the tree, persisted across sessions — like GTO Wizard's per-spot EV display.

## Design

### Core: Full-tree EV accumulator

Store per-node, per-player, per-hand EV accumulators for ALL decision nodes in the tree.

**Data structure:**

```rust
pub struct FullTreeEvTracker {
    /// Number of decision nodes in the tree.
    num_nodes: usize,
    /// ev_sum[node_decision_idx][player][hand_index] — accumulated EV × 1000 (i64 for atomics).
    /// Indexed by decision node index (0..num_decision_nodes), not arena index.
    ev_sum: Vec<[[AtomicI64; 169]; 2]>,
    /// ev_count[node_decision_idx][player][hand_index] — sample count.
    ev_count: Vec<[[AtomicU64; 169]; 2]>,
}
```

**Memory footprint:** `num_decisions × 2 × 169 × (8 + 8)` bytes = `num_decisions × 5408` bytes. For 5000 decision nodes: ~27MB. Acceptable.

### Accumulation during MCCFR

In `traverse_traverser` (mccfr.rs), after computing `node_value`, accumulate:

```rust
// After line 692 (existing scenario tracker)
if let Some(ev_tracker) = full_ev_tracker {
    let decision_idx = decision_map[node_idx as usize];
    if decision_idx != u32::MAX {
        ev_tracker.accumulate(decision_idx as usize, traverser as usize, hand_index, node_value);
    }
}
```

This runs for EVERY decision node during traversal, not just configured scenarios. The overhead is one atomic add per node per traversal — same cost as regret updates.

### Persistence in snapshots

Save as `hand_ev.bin` alongside `strategy.bin` and `regrets.bin` in each snapshot directory.

**Format:**
```
[u32: num_decision_nodes]
[u32: num_players (2)]
[u32: num_hands (169)]
For each decision node (0..num_decision_nodes):
  For each player (0..2):
    For each hand (0..169):
      [f64: average_ev]  // ev_sum / ev_count, or 0 if count == 0
```

File size: `12 + num_decisions × 2 × 169 × 8` bytes. For 5000 nodes: ~13.5MB.

### Loading on resume

When `try_resume()` loads a snapshot:
1. Load `hand_ev.bin` if present
2. Convert average EVs back to sum/count: `ev_sum = avg_ev × count`, where `count` is loaded from a companion field or set to a fixed value (e.g., 1000000) to weight the historical data appropriately
3. New MCCFR iterations accumulate on top of the loaded values

**Simpler approach:** Store sum and count directly (not average). File is larger (2× values) but resume is exact — no count estimation needed.

```
For each decision node:
  For each player:
    For each hand:
      [i64: ev_sum_times_1000]
      [u64: ev_count]
```

File size: `12 + num_decisions × 2 × 169 × 16` = ~27MB for 5000 nodes.

### Display

#### TUI

The existing `on_strategy_refresh` callback passes `hand_evs` for configured scenarios. Extend it to also pass the full EV tracker (or query EVs for the current scenario node from the full tracker).

No change to TUI rendering — it already displays EVs per hand. The data source changes from `ScenarioEvTracker` to `FullTreeEvTracker`.

The `ScenarioEvTracker` can be removed once `FullTreeEvTracker` is in place, since it's a strict subset.

#### Explorer

The `StrategyMatrix` already has `reaching_p1`/`reaching_p2`. Add `ev_per_hand: Vec<f32>` (169 values for the acting player) to the response. The frontend renders EVs in each cell (already supported via `cell.ev`).

For blueprint loading: read `hand_ev.bin` from the bundle and serve EVs for any requested node.

### Integration with existing code

| Component | Change |
|-----------|--------|
| `BlueprintTrainer` | Owns `FullTreeEvTracker`, passes to MCCFR traversal |
| `traverse_traverser` (mccfr.rs) | Accumulates EV at every node |
| `save_snapshot` (trainer.rs) | Writes `hand_ev.bin` |
| `try_resume` (trainer.rs) | Loads `hand_ev.bin` |
| `on_strategy_refresh` callback | Reads EVs from full tracker |
| `ScenarioEvTracker` | Replaced by `FullTreeEvTracker` |
| Explorer `get_strategy_matrix` | Includes per-hand EVs from loaded blueprint |
| `BlueprintV2Strategy` or new struct | Stores loaded EVs for exploration |

### What doesn't change

- MCCFR traversal logic (regrets, strategy sums)
- CFR convergence properties (EV tracking is observation-only)
- Tree structure
- Bucket/clustering pipeline

### Validation

- EVs at the preflop root should match the current `ScenarioEvTracker` output
- Fold EV should display as 0 (with the existing offset)
- AA EV should be consistent across resume boundaries (save at 1B, resume, check at 1.1B — EVs should be similar, not reset)
- Sum of all hand EVs weighted by dealing probability should be ~0 (zero-sum game)
