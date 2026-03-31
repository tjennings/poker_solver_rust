---
# poker_solver_rust-65n9
title: 'KRW-EMD: Historical-context-aware clustering for turn/river'
status: todo
type: feature
created_at: 2026-03-31T14:38:18Z
updated_at: 2026-03-31T14:38:18Z
---

Implement k-Recall Winrate EMD clustering from arXiv 2511.12089 (KrwEmd, Nov 2025). Adds ancestor PAWF features to the clustering pipeline so strategically different hands that look identical on the current street (but arrived via different histories) are no longer merged. Paper shows ~30% exploitability reduction vs standard PAAEMD on reduced games. Start with k=1 on turn only, measure before expanding.

## References

- Paper: `docs/papers/KrwEmd.pdf` (arXiv 2511.12089, Nov 2025)
- Reference impl: [robopoker](https://github.com/krukah/robopoker) (Rust, Pluribus-style)
- Our pipeline: `crates/core/src/blueprint_v2/cluster_pipeline.rs`, `clustering.rs`

## Background

Standard PAAEMD (our current approach) builds a histogram over next-street bucket IDs and clusters with EMD. This entirely discards historical information — two hands with identical future distributions but different flop histories get the same bucket. KRW-EMD fixes this by concatenating (lose, tie, win) triples from the current street AND k ancestor streets into the feature vector.

Key insight: for heads-up, PAWF = (lose_rate, tie_rate, win_rate) — just 3 floats per street. EMD on 3 bins is O(1). The feature vector for k=1 on turn is only 6 floats vs ~200 histogram bins.

Best weights from paper: **(16, 4, 1)** — sharply decreasing. Current street dominates but history adds meaningful discrimination.

## Implementation Plan

### Phase 1: PAWF Computation Module

- [ ] Create `crates/core/src/blueprint_v2/pawf.rs`
- [ ] Define `PawfFile` struct: stores (lose, tie, win) per (board, combo), indexed by `PackedBoard` + `combo_index`
- [ ] Wire format: header + flat `[f32; 3]` array, board-indexed like `BucketFile`
- [ ] Implement `compute_river_pawfs()`: decompose existing equity into (lose, tie, win) by counting showdown outcomes against all opponent hands. Reuse `compute_board_equities` logic but track 3 components instead of 1.
- [ ] Implement `compute_turn_pawfs()`: for each (turn_board, combo), enumerate all 48 river cards, compute showdown vs all opponents, aggregate into (lose, tie, win)
- [ ] Implement `compute_flop_pawfs()`: for each (flop_board, combo), enumerate all turn+river runouts, aggregate into (lose, tie, win)
- [ ] Add tests: verify win+tie+lose = 1.0 for all entries, verify equity ≈ win + 0.5*tie

### Phase 2: KRW Feature Construction

- [ ] Implement `build_krw_features_turn()`: for each (turn_board, combo), retrieve turn PAWF + flop PAWF by board prefix lookup. Return `Vec<[f32; 6]>`
- [ ] Ancestor lookup: flop board = `turn_board[..3]`, canonicalize, look up in flop `PawfFile`
- [ ] Handle edge case: combo blocked by board cards → skip (same as current histogram pipeline)

### Phase 3: KRW-EMD Distance & K-Means

- [ ] Implement `krw_emd_distance(a: &[f32], b: &[f32], weights: &[f32]) -> f32`: weighted sum of EMD over each street's 3-element distribution
- [ ] Implement `emd_3(p: &[f32], q: &[f32]) -> f32`: L1 of CDF difference on 3 bins
- [ ] Implement `elkan_krw_emd()`: Elkan-accelerated k-means using KRW-EMD distance on `[f32; 6]` feature vectors. Triangle inequality holds (weighted sum of metrics is a metric).
- [ ] k-means++ initialization adapted for KRW-EMD distance

### Phase 4: Pipeline Integration

- [ ] Add `ClusteringAlgorithm::KrwEmd { recall_depth: u8, weights: Vec<f32> }` variant to config
- [ ] Modify `run_clustering_pipeline()`:
  1. River: equity-based 1-D k-means (unchanged)
  2. **NEW**: Compute PAWFs at all streets (river, turn, flop)
  3. Turn: if KRW-EMD enabled, use `build_krw_features_turn()` + `elkan_krw_emd()` instead of histogram pipeline
  4. Flop: keep histogram-over-turn-buckets (unchanged for now)
  5. Preflop: 169 canonical (unchanged)
- [ ] Add YAML config support: `algorithm: krw_emd`, `krw_recall_depth: 1`, `krw_weights: [16, 4]`

### Phase 5: Validation & Comparison

- [ ] Add cluster diagnostic: intra-bucket equity variance comparison (KRW-EMD vs PAAEMD at same bucket count)
- [ ] Run both pipelines on same config, compare bucket quality metrics
- [ ] Verify MCCFR training works with KRW-EMD-produced bucket files (no changes needed — BucketFile format is the same)
- [ ] Benchmark: clustering time, memory usage, file sizes

### Future Extensions (separate beans)

- k=2 on river (add turn + flop PAWFs to river clustering)
- Hybrid approach: concatenate histogram features WITH ancestor PAWFs for both future distribution AND historical context
- Per-flop KRW clustering
- Tuning: grid search over weight configurations

## Estimated Scope

- Phase 1-3: Core implementation (~2-3 sessions)
- Phase 4: Integration (~1 session)
- Phase 5: Validation (~1 session)
- No changes to MCCFR solver, range solver, or explorer — BucketFile format stays the same
