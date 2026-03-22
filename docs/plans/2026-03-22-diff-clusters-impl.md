# diff-clusters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a `diff-clusters` CLI command that compares two bucket sets and reports quality metrics (intra-bucket equity std, bucket sizes) and similarity (Adjusted Rand Index).

**Architecture:** Domain layer in `cluster_diagnostics.rs` (ARI computation + diff report struct), CLI adapter in `main.rs`. Reuses existing `audit_bucket_equity()` for quality metrics. ARI is sampling-based to avoid O(n^2).

**Tech Stack:** Rust, Rayon (parallelism), clap (CLI), existing `BucketFile` + `EquityAuditReport` types

**Design doc:** `docs/plans/2026-03-22-diff-clusters-design.md`

---

### Task 1: Add ARI computation function

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs` (add before `#[cfg(test)]` at line 1350)
- Test: same file, `mod tests`

**Step 1: Write the failing tests**

Add to `mod tests`:

```rust
#[test]
fn ari_identical_clusterings() {
    let a = vec![0u16, 0, 1, 1, 2, 2];
    let b = vec![0u16, 0, 1, 1, 2, 2];
    let ari = adjusted_rand_index(&a, &b, 42);
    assert!(
        (ari.unwrap() - 1.0).abs() < 0.05,
        "identical clusterings should have ARI ~1.0, got {ari:?}"
    );
}

#[test]
fn ari_permuted_ids_is_one() {
    // Same grouping, different IDs: {0,0,1,1} vs {5,5,3,3}
    let a = vec![0u16, 0, 1, 1];
    let b = vec![5u16, 5, 3, 3];
    let ari = adjusted_rand_index(&a, &b, 42);
    assert!(
        (ari.unwrap() - 1.0).abs() < 0.05,
        "permuted IDs should have ARI ~1.0, got {ari:?}"
    );
}

#[test]
fn ari_completely_different() {
    // A groups by pairs, B groups odds/evens — maximally different for 4 items
    let a = vec![0u16, 0, 1, 1];
    let b = vec![0u16, 1, 0, 1];
    let ari = adjusted_rand_index(&a, &b, 42);
    // For 4 items with these groupings, ARI should be negative or near zero
    assert!(
        ari.unwrap() < 0.3,
        "completely different clusterings should have low ARI, got {ari:?}"
    );
}

#[test]
fn ari_degenerate_single_cluster() {
    let a = vec![0u16, 0, 0, 0];
    let b = vec![0u16, 1, 2, 3];
    let ari = adjusted_rand_index(&a, &b, 42);
    assert!(ari.is_none(), "degenerate clustering should return None");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core ari_ -- --nocapture 2>&1 | head -20`
Expected: FAIL — `adjusted_rand_index` not found

**Step 3: Write the implementation**

Add before `#[cfg(test)]` in `cluster_diagnostics.rs`:

```rust
// ---------------------------------------------------------------------------
// Adjusted Rand Index (sampling-based)
// ---------------------------------------------------------------------------

/// Compute the Adjusted Rand Index between two clusterings.
///
/// ARI measures agreement between two clusterings, adjusted for chance:
/// - 1.0 = identical groupings (regardless of label permutation)
/// - 0.0 = agreement expected by random chance
/// - negative = less agreement than random
///
/// Uses sampling to avoid O(n^2) pairwise comparison: draws random pairs
/// of indices, checks same-cluster agreement, builds a contingency table.
///
/// Returns `None` if degenerate (all items in one cluster in either input).
#[must_use]
pub fn adjusted_rand_index(a: &[u16], b: &[u16], seed: u64) -> Option<f64> {
    assert_eq!(a.len(), b.len(), "clusterings must have same length");
    let n = a.len();
    if n < 2 {
        return None;
    }

    // Check for degenerate cases: all items in one cluster.
    let a_unique: std::collections::HashSet<u16> = a.iter().copied().collect();
    let b_unique: std::collections::HashSet<u16> = b.iter().copied().collect();
    if a_unique.len() <= 1 || b_unique.len() <= 1 {
        return None;
    }

    // Sample random pairs and build contingency counts.
    // a = same in A and same in B
    // b = same in A, different in B
    // c = different in A, same in B
    // d = different in A, different in B
    let num_samples = (100_000).min(n * (n - 1) / 2);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tp = 0_u64; // same-same
    let mut fp = 0_u64; // same in A, diff in B
    let mut fn_ = 0_u64; // diff in A, same in B
    let mut tn = 0_u64; // diff-diff

    for _ in 0..num_samples {
        let i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        let same_a = a[i] == a[j];
        let same_b = b[i] == b[j];
        match (same_a, same_b) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }

    let total = (tp + fp + fn_ + tn) as f64;
    if total == 0.0 {
        return None;
    }

    // ARI = (RI - Expected_RI) / (Max_RI - Expected_RI)
    // where RI = (tp + tn) / total
    // Expected = ((tp+fp)*(tp+fn_) + (fn_+tn)*(fp+tn)) / total^2
    // Max = ((tp+fp) + (tp+fn_)) / (2 * total)
    let ri = (tp + tn) as f64 / total;
    let sum_a_same = (tp + fp) as f64;
    let sum_b_same = (tp + fn_) as f64;
    let sum_a_diff = (fn_ + tn) as f64;
    let sum_b_diff = (fp + tn) as f64;
    let expected_ri = (sum_a_same * sum_b_same + sum_a_diff * sum_b_diff) / (total * total);
    let max_ri = (sum_a_same + sum_b_same) / (2.0 * total);

    let denom = max_ri - expected_ri;
    if denom.abs() < 1e-15 {
        return None;
    }

    Some((ri - expected_ri) / denom)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core ari_ -- --nocapture 2>&1 | head -20`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_diagnostics.rs
git commit -m "feat: add sampling-based adjusted_rand_index for cluster comparison"
```

---

### Task 2: Add `ClusterDiffReport` and `diff_bucket_files`

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs` (add after ARI function)
- Test: same file, `mod tests`

**Step 1: Write the failing test**

Add to `mod tests`:

```rust
#[test]
fn diff_report_identical_files() {
    let bf = make_test_bucket_file(3, vec![0, 0, 1, 1, 2, 2]);
    let report = diff_bucket_files(&bf, &bf, 0, 42);
    assert_eq!(report.bucket_count, 3);
    assert!(
        report.ari.unwrap() > 0.95,
        "identical files should have ARI ~1.0, got {:?}",
        report.ari
    );
    assert_eq!(report.street, "River");
}

#[test]
fn diff_report_different_groupings() {
    let a = make_test_bucket_file(2, vec![0, 0, 1, 1]);
    let b = make_test_bucket_file(2, vec![0, 1, 0, 1]);
    let report = diff_bucket_files(&a, &b, 0, 42);
    assert_eq!(report.bucket_count, 2);
    assert!(
        report.ari.unwrap() < 0.3,
        "different groupings should have low ARI, got {:?}",
        report.ari
    );
}

#[test]
fn diff_report_size_stats() {
    let a = make_test_bucket_file(3, vec![0, 0, 0, 1, 2, 2]);
    let b = make_test_bucket_file(3, vec![0, 1, 1, 1, 2, 2]);
    let report = diff_bucket_files(&a, &b, 0, 42);
    // a: sizes [3, 1, 2], b: sizes [1, 3, 2]
    assert_eq!(report.size_a.bucket_sizes, vec![3, 1, 2]);
    assert_eq!(report.size_b.bucket_sizes, vec![1, 3, 2]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core diff_report -- --nocapture 2>&1 | head -20`
Expected: FAIL — `diff_bucket_files` not found

**Step 3: Write the implementation**

Add after the ARI function:

```rust
// ---------------------------------------------------------------------------
// Cluster diff
// ---------------------------------------------------------------------------

/// Side-by-side comparison of two bucket files.
#[derive(Debug)]
pub struct ClusterDiffReport {
    pub street: String,
    pub bucket_count: u16,
    /// Bucket size report for dir A.
    pub size_a: ClusterReport,
    /// Bucket size report for dir B.
    pub size_b: ClusterReport,
    /// Equity audit for dir A (None if sample_boards == 0).
    pub equity_a: Option<EquityAuditReport>,
    /// Equity audit for dir B (None if sample_boards == 0).
    pub equity_b: Option<EquityAuditReport>,
    /// Adjusted Rand Index (None if degenerate).
    pub ari: Option<f64>,
}

impl ClusterDiffReport {
    /// Format as a human-readable summary.
    #[must_use]
    pub fn summary(&self, verbose: bool) -> String {
        use std::fmt::Write;
        let mut s = format!("=== Cluster Diff: {} ===\n", self.street);
        let _ = writeln!(s, "{:<28} {:<16} {:<16}", "", "Dir A", "Dir B");
        let _ = writeln!(
            s, "{:<28} {:<16} {:<16}",
            "Bucket count:", self.size_a.bucket_count, self.size_b.bucket_count
        );

        // Bucket size stats
        let _ = writeln!(
            s, "{:<28} {:<16.1} {:<16.1}",
            "Bucket size std:", self.size_a.size_stats.std_dev, self.size_b.size_stats.std_dev
        );
        let empty_a = self.size_a.bucket_sizes.iter().filter(|&&c| c == 0).count();
        let empty_b = self.size_b.bucket_sizes.iter().filter(|&&c| c == 0).count();
        let _ = writeln!(
            s, "{:<28} {:<16} {:<16}",
            "Empty buckets:", empty_a, empty_b
        );

        // Equity quality (if audited)
        if let (Some(ea), Some(eb)) = (&self.equity_a, &self.equity_b) {
            let pct = if ea.mean_intra_bucket_std > 0.0 {
                (eb.mean_intra_bucket_std - ea.mean_intra_bucket_std)
                    / ea.mean_intra_bucket_std
                    * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                s, "{:<28} {:<16.4} {:<12.4} ({:+.1}%)",
                "Mean intra-bkt std:", ea.mean_intra_bucket_std, eb.mean_intra_bucket_std, pct
            );
            let pct_max = if ea.max_intra_bucket_std > 0.0 {
                (eb.max_intra_bucket_std - ea.max_intra_bucket_std)
                    / ea.max_intra_bucket_std
                    * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                s, "{:<28} {:<16.4} {:<12.4} ({:+.1}%)",
                "Max intra-bkt std:", ea.max_intra_bucket_std, eb.max_intra_bucket_std, pct_max
            );
        }

        // ARI
        match self.ari {
            Some(ari) => { let _ = writeln!(s, "{:<28} {:.3}", "Adjusted Rand Index:", ari); }
            None => { let _ = writeln!(s, "{:<28} N/A", "Adjusted Rand Index:"); }
        }

        // Verbose: equity histogram comparison
        if verbose {
            if let (Some(ea), Some(eb)) = (&self.equity_a, &self.equity_b) {
                let _ = writeln!(s, "\n  Equity histogram (buckets per equity bin):");
                let _ = writeln!(s, "  {:<16} {:<10} {:<10}", "Equity Range", "Dir A", "Dir B");
                for bin in 0..10 {
                    let lo = bin as f64 * 0.1;
                    let hi = lo + 0.1;
                    let count_a = ea.buckets.iter().filter(|b| b.count > 0 && b.mean_equity >= lo && b.mean_equity < hi).count();
                    let count_b = eb.buckets.iter().filter(|b| b.count > 0 && b.mean_equity >= lo && b.mean_equity < hi).count();
                    let _ = writeln!(s, "  [{:.1}, {:.1})       {:<10} {:<10}", lo, hi, count_a, count_b);
                }
            }
        }

        s
    }
}

/// Compare two bucket files: size distribution, equity quality, and ARI.
///
/// Both files must have the same bucket count, board count, and street.
/// If `sample_boards > 0`, runs equity audit on both files.
///
/// # Panics
///
/// Panics if the files have different bucket counts, board counts, or streets.
#[must_use]
pub fn diff_bucket_files(
    a: &BucketFile,
    b: &BucketFile,
    sample_boards: usize,
    seed: u64,
) -> ClusterDiffReport {
    assert_eq!(
        a.header.bucket_count, b.header.bucket_count,
        "bucket count mismatch: {} vs {}",
        a.header.bucket_count, b.header.bucket_count
    );
    assert_eq!(
        a.header.board_count, b.header.board_count,
        "board count mismatch: {} vs {}",
        a.header.board_count, b.header.board_count
    );
    assert_eq!(
        a.header.street, b.header.street,
        "street mismatch"
    );

    let size_a = ClusterReport::from_bucket_file(a);
    let size_b = ClusterReport::from_bucket_file(b);

    let (equity_a, equity_b) = if sample_boards > 0 {
        (
            Some(audit_bucket_equity(a, sample_boards, seed)),
            Some(audit_bucket_equity(b, sample_boards, seed)),
        )
    } else {
        (None, None)
    };

    let ari = adjusted_rand_index(&a.buckets, &b.buckets, seed);

    let street = format!("{:?}", a.header.street);

    ClusterDiffReport {
        street,
        bucket_count: a.header.bucket_count,
        size_a,
        size_b,
        equity_a,
        equity_b,
        ari,
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core diff_report -- --nocapture 2>&1 | head -20`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_diagnostics.rs
git commit -m "feat: add ClusterDiffReport and diff_bucket_files"
```

---

### Task 3: Add `DiffClusters` CLI command

**Files:**
- Modify: `crates/trainer/src/main.rs` (add `DiffClusters` variant + handler)
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs` (ensure `diff_bucket_files`, `ClusterDiffReport` are public — they already are)

**Step 1: Add the command variant**

In `main.rs`, add after the `ValidateBlueprint` variant (line ~143) inside the `Commands` enum:

```rust
    /// Compare two sets of cluster bucket files
    DiffClusters {
        /// Directory A containing .buckets files
        #[arg(long)]
        dir_a: PathBuf,
        /// Directory B containing .buckets files
        #[arg(long)]
        dir_b: PathBuf,
        /// Number of boards to sample for equity audit (0 = skip equity)
        #[arg(long, default_value = "200")]
        sample_boards: usize,
        /// Show per-bucket equity histogram breakdown
        #[arg(long)]
        verbose: bool,
    },
```

**Step 2: Add the handler**

In the `match cli.command` block, add a new arm:

```rust
        Commands::DiffClusters {
            dir_a,
            dir_b,
            sample_boards,
            verbose,
        } => {
            use poker_solver_core::blueprint_v2::cluster_diagnostics::{
                diff_bucket_files, ClusterDiffReport,
            };
            use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

            let streets = ["river", "turn", "flop", "preflop"];
            let mut any_found = false;

            for street_name in &streets {
                let path_a = dir_a.join(format!("{street_name}.buckets"));
                let path_b = dir_b.join(format!("{street_name}.buckets"));

                if !path_a.exists() && !path_b.exists() {
                    continue;
                }
                if !path_a.exists() {
                    eprintln!("warning: {street_name}.buckets missing from dir-a, skipping");
                    continue;
                }
                if !path_b.exists() {
                    eprintln!("warning: {street_name}.buckets missing from dir-b, skipping");
                    continue;
                }

                let bf_a = BucketFile::load(&path_a)?;
                let bf_b = BucketFile::load(&path_b)?;

                if bf_a.header.bucket_count != bf_b.header.bucket_count {
                    eprintln!(
                        "error: {street_name} bucket count mismatch: {} vs {}",
                        bf_a.header.bucket_count, bf_b.header.bucket_count
                    );
                    continue;
                }

                eprintln!("diffing {street_name}...");
                let report = diff_bucket_files(&bf_a, &bf_b, sample_boards, 42);
                println!("{}", report.summary(verbose));

                any_found = true;
            }

            if !any_found {
                eprintln!("no matching .buckets files found in both directories");
            }
        }
```

**Step 3: Ensure imports compile**

Check that `diff_bucket_files` and `ClusterDiffReport` are exported from the core crate's public API. They should be — `cluster_diagnostics` functions are already `pub`.

Run: `cargo build -p poker-solver-trainer 2>&1 | tail -10`
Expected: Compiles successfully

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core 2>&1 | grep "test result"`
Expected: All tests pass

**Step 5: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat: add diff-clusters CLI command"
```

---

### Task 4: Update docs and bean

**Files:**
- Modify: `docs/training.md` (add `diff-clusters` section after `diag-clusters`)

**Step 1: Add documentation**

Add after the `diag-clusters` section in `docs/training.md`:

```markdown
### diff-clusters

Compare two sets of bucket files to measure quality improvement and clustering similarity.

```bash
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200

# Verbose mode with equity histogram
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200 \
  --verbose
```

- `--dir-a`, `--dir-b` -- directories containing `.buckets` files to compare
- `--sample-boards` -- boards to sample for equity audit (default 200, 0 = skip)
- `--verbose` -- show per-equity-bin bucket histogram

Reports per-street: bucket size stats, intra-bucket equity std (lower = better), and Adjusted Rand Index (1.0 = identical groupings, 0.0 = random agreement).
```

**Step 2: Update bean**

```bash
beans update poker_solver_rust-9zzq -s completed --body-append "## Summary of Changes

- Added sampling-based adjusted_rand_index() function
- Added ClusterDiffReport struct and diff_bucket_files() function
- Added diff-clusters CLI command to poker-solver-trainer
- Updated docs/training.md with usage"
```

**Step 3: Commit**

```bash
git add docs/training.md .beans/
git commit -m "docs: add diff-clusters to training.md, complete bean"
```

---

### Task 5: Final verification

**Step 1: Run full test suite**

Run: `cargo test -p poker-solver-core 2>&1 | grep "test result"`
Expected: All tests pass

**Step 2: Verify CLI compiles and shows help**

Run: `cargo run -p poker-solver-trainer -- diff-clusters --help 2>&1`
Expected: Shows usage with `--dir-a`, `--dir-b`, `--sample-boards`, `--verbose`

**Step 3: Run clippy**

Run: `cargo clippy -p poker-solver-core -p poker-solver-trainer 2>&1 | tail -5`
Expected: No new warnings
