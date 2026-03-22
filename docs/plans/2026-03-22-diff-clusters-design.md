# diff-clusters CLI Command Design

**Date:** 2026-03-22
**Status:** Approved

## Problem

After changing the clustering algorithm (e.g., L2 → EMD fix), we need a way to
compare two bucket sets to measure quality improvement and clustering similarity.

## Solution

New `diff-clusters` subcommand that compares two bucket directories and reports:
1. Quality comparison — intra-bucket equity std, bucket size distribution (per street, side-by-side)
2. Similarity — Adjusted Rand Index measuring clustering agreement

## CLI Interface

```
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200 \
  --verbose
```

- `--dir-a` / `--dir-b` — required, paths to directories containing `.buckets` files
- `--sample-boards` — boards to sample for equity audit (default 200)
- `--verbose` — show per-bucket breakdown and equity histogram comparison

## Default Output

```
=== Cluster Diff: River ===
                        Dir A           Dir B
Bucket count:           200             200
Mean intra-bkt std:     0.0823          0.0614  (-25.4%)
Max intra-bkt std:      0.1847          0.1203  (-34.9%)
Bucket size std:        142.3           98.7    (-30.6%)
Empty buckets:          3               0
Adjusted Rand Index:    0.742

=== Cluster Diff: Turn ===
...
```

## Verbose Output

Appends 10-bin equity histogram showing how many buckets have mean equity in
each range [0.0-0.1), [0.1-0.2), ..., for each clustering side-by-side.

## Architecture

### Core Library (`cluster_diagnostics.rs`)

**`ClusterDiffReport`** struct:
- `street: Street`
- `quality_a: EquityAuditReport` (existing type)
- `quality_b: EquityAuditReport`
- `ari: Option<f64>` (None if degenerate — all combos in one bucket)
- `bucket_count: u16`

**`diff_bucket_files(a, b, sample_boards, seed)`** → `ClusterDiffReport`:
- Validates: same street, same bucket count, same board count
- Calls existing `audit_bucket_equity()` on each bucket file
- Computes ARI by sampling

**ARI computation** (sampling-based):
- Sample N random (board, combo) pairs (N = ~100k)
- For each pair (i, j), record: same-bucket-in-A? same-bucket-in-B?
- Build contingency table (a, b, c, d) counts
- Compute ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
- Sampling avoids O(n^2) full pairwise comparison

### CLI (`main.rs`)

New `DiffClusters` variant in `Commands` enum. Loads bucket files from both
dirs, calls `diff_bucket_files` per street, prints report.

## Edge Cases

- Mismatched bucket counts → error with clear message
- Mismatched board sets → error
- Missing street in one dir → skip with warning, diff whatever's in both
- All combos in one bucket → ARI = N/A (degenerate)

## Testing

- Unit: ARI on known clusterings (identical → 1.0, random → ~0.0, permuted IDs → 1.0)
- Unit: `diff_bucket_files` with synthetic bucket files
- Integration: `cargo test -p poker-solver-core`
