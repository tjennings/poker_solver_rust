# Use Bucket Files for MCCFR Bucket Lookups — Design

## Problem

The MCCFR trainer computes `compute_equity()` 6 times per deal (3 streets × 2 players) to derive bucket assignments via uniform equity binning. This has two problems:

1. **Correctness:** The clustering pipeline produces potential-aware EMD bucket assignments for flop and turn, but MCCFR ignores them and uses equity bins. Training against the wrong abstraction.
2. **Performance:** `compute_equity()` enumerates ~1000 opponent combos per call. This dominates iteration cost and nullifies regret-based pruning gains.

Additionally, `build_bucket_histogram_u8` silently skips next-street boards not present in the bucket file, so non-exhaustive bucket files produce corrupted histograms for upstream streets.

## Bucketing Strategy Per Street (Pluribus-style)

All streets use exhaustive precomputed bucket files with O(1) lookup during MCCFR. Zero `compute_equity` calls at runtime.

| Street | Clustering Method | Lookup at MCCFR Runtime |
|--------|------------------|------------------------|
| Preflop | 169 canonical hands | Deterministic index (unchanged) |
| Flop | Potential-aware EMD over turn buckets | BucketFile O(1) lookup |
| Turn | Potential-aware EMD over river buckets | BucketFile O(1) lookup |
| River | 1-D k-means on raw equity | BucketFile O(1) lookup |

## Solution

### Part A: Exhaustive Clustering Pipeline

The pipeline is bottom-up. Each upstream street builds histograms over the downstream street's bucket assignments.

- **River:** Exhaustive (~700K canonical boards). 1-D k-means on raw equity with board weights. Must cover all canonical rivers so turn histogram building can look up every possible runout.
- **Turn:** Use `cluster_turn_canonical()` (already exists, enumerates all ~15K canonical turns). Potential-aware EMD histograms over river buckets.
- **Flop:** Already exhaustive (all 1,755 canonical flops). Potential-aware EMD histograms over turn buckets. With exhaustive turn file, histograms are now complete.

Pipeline change: `run_clustering_pipeline` uses canonical/exhaustive variants by default. `sample_boards` config respected when set but defaults to exhaustive.

### Part B: Wire BucketFile Lookups into MCCFR

Replace ALL `compute_equity` calls in `AllBuckets::precompute_buckets()` with O(1) bucket file lookups.

**Lookup flow (all postflop streets):**
```
deal.board[..N]
  → CanonicalBoard::from_cards(board)  // suit isomorphism
  → canonical_key(canonical.cards)     // sorted pack → PackedBoard
  → board_maps[street].get(packed)     // HashMap → board_idx
  → canonical.canonicalize_holding(hole_cards)  // apply same suit mapping
  → combo_index(c0, c1)               // triangular index → combo_idx
  → bucket_file.get_bucket(board_idx, combo_idx)  // flat array lookup
```

**Changes to `AllBuckets`:**
- Add `board_maps: [Option<HashMap<PackedBoard, u32>>; 4]` built at construction from `bucket_file.board_index_map()`
- Rewrite `precompute_buckets()` to use bucket file lookups for all postflop streets
- Fall back to equity binning if bucket file missing or board not found (graceful degradation for incomplete files)
- Remove `equity_cache`, `delta_bins`, `expected_delta` fields — no longer needed

**Changes to `Trainer`:**
- Remove equity cache loading logic (no longer needed)
- Load all 4 bucket files at startup, build board maps
- `AllBuckets::new()` builds board maps from loaded bucket files

## Storage (loaded during MCCFR)

| Street | Boards | Bucket data | Board table | Total |
|--------|--------|-------------|-------------|-------|
| Flop | 1,755 | ~4.5 MB | ~14 KB | ~4.5 MB |
| Turn | ~15K | ~39 MB | ~120 KB | ~39 MB |
| River | ~700K | ~1.8 GB | ~5.6 MB | ~1.8 GB |

~1.85 GB total in memory during training.

## Performance Impact

- Eliminates ALL 6 `compute_equity` calls per deal (~6000 `rank_hand` evaluations)
- Replaces with 6 HashMap lookups + 6 array accesses + 3 `CanonicalBoard::from_cards` calls
- Expected: 5-10x improvement in iterations/sec
- Regret-based pruning should now show real throughput gains since traversal becomes the dominant cost

## Risks

- **River memory:** ~1.8 GB in RAM. Acceptable for training machines.
- **Exhaustive river clustering time:** ~700K boards × 990 combos × equity computation. Hours, not days. One-time cost.
- **Fallback path:** If bucket file is missing or board not found, equity binning still works. No hard failure.
- **Suit canonicalization correctness:** Must apply the same `SuitMapping` to both board and hole cards. Verified by existing `CanonicalBoard::canonicalize_holding`.

## Out of Scope

- Memory-mapping bucket files (follow-up if memory is tight)
- Preflop changes (already uses 169 canonical hands)
