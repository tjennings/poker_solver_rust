# MCCFR Real Clustering — Design

## Goal

Replace the 169-canonical-hand bucketing in the MCCFR convergence harness with real potential-aware clustering via `cluster_single_flop()`.

## Approach

- At `MccfrSolver::new()`, call `cluster_single_flop(flop, turn_buckets, river_buckets, 50, seed)` (~2-10s one-time cost)
- Store the `PerFlopBucketFile` on the solver
- Replace `canonical_buckets()` with per-flop file lookups:
  - Preflop: `CanonicalHand::from_cards().index()` (169, lossless)
  - Flop: 169 canonical (single flop, doesn't need real flop clustering)
  - Turn: `pf.get_turn_bucket(turn_idx, combo_idx)`
  - River: `pf.get_river_bucket(turn_idx, river_idx, combo_idx)`
- Update `BlueprintTrainer` bucket_counts to `[169, 169, turn_buckets, river_buckets]`
- Add `--turn-buckets` and `--river-buckets` CLI flags (default 200)
