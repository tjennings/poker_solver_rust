# Per-Street Pruning Control

**Date**: 2026-03-29
**Status**: Approved

## Problem

Regret-based pruning is applied uniformly across all streets. Pluribus explicitly
skipped pruning on preflop because the cost of traversing all preflop actions is
negligible (only 169 canonical hands), but pruning a preflop action kills the
entire subtree below it. The current implementation has no way to exclude streets.

Additionally, the current `prune_threshold: -2` (= -1 BB) is far too aggressive.
For pruning only actions worse than -100 BB, the correct chip-unit value is -200
(since 100 BB = 200 chips at SB=1/BB=2, and the threshold is scaled internally
by REGRET_SCALE=100,000).

## Design

### Config

Add `prune_streets` to `TrainingConfig`:

```yaml
training:
  prune_streets: [flop, turn, river]   # omit preflop to skip pruning there
  prune_threshold: -200                # only prune actions with regret < -100 BB
```

- **Type**: `Option<Vec<String>>` in serde, parsed to `[bool; 4]` bitmask indexed by `Street as usize`
- **Default**: `None` → `[true; 4]` (all streets, preserves current behavior)
- **Valid values**: `preflop`, `flop`, `turn`, `river` (case-insensitive)

### Code Changes

1. **`config.rs`** — Add `prune_streets` field to `TrainingConfig`. Add helper
   method `prune_street_mask(&self) -> [bool; 4]` that parses the string list
   into a bitmask.

2. **`trainer.rs`** — Compute `prune_streets: [bool; 4]` once at trainer init.
   Pass it into `traverse_external` alongside `prune` and `prune_threshold`.

3. **`mccfr.rs: traverse_external`** — Add `prune_streets: [bool; 4]` parameter.
   In the `Decision` arm, compute:
   ```rust
   let node_prune = prune && prune_streets[street as usize];
   ```
   Pass `node_prune` (not `prune`) to `traverse_traverser` / `traverse_opponent`.
   Pass `prune` (original) through recursion so deeper streets still check their
   own mask.

4. **`traverse_traverser` / `traverse_opponent`** — No signature changes. They
   receive a pre-resolved `prune` bool as before.

### Validation

1. Unit test: config parsing (default, subset, empty)
2. Unit test: bitmask generation
3. Unit test: prune gating logic at preflop vs postflop nodes
4. Full `cargo test` passes in < 1 minute
5. Full `cargo build` workspace compilation check
