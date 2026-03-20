# Transition Coherence Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Improve transition consistency diagnostics with normalized EMD, persisted centroids, and a centroid separation ratio that answers "how good is this clustering?"

**Architecture:** Modify `audit_transition_consistency` to accept optional centroids and normalize all EMD values. Add centroid separation computation. Update CLI to load centroid files and pass them through. Update output formatting.

**Tech Stack:** Rust, existing `cluster_diagnostics.rs` and `centroid_file.rs` modules.

---

### Task 1: Normalize EMD and add new report fields

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs`

**Step 1: Write the failing test**

Add a test module at the bottom of `cluster_diagnostics.rs` (or add to an existing one):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_emd_divides_by_k_minus_1() {
        // With 500 child buckets, raw EMD of 24.95 should normalize to 24.95 / 499
        let raw = 24.95;
        let k = 500;
        let normalized = raw / (k as f64 - 1.0);
        assert!((normalized - 0.05).abs() < 0.001);
    }
}
```

**Step 2: Run test to verify it passes (trivial arithmetic — this bootstraps the test module)**

Run: `cargo test -p poker-solver-core normalize_emd`
Expected: PASS.

**Step 3: Add fields to `TransitionConsistencyReport`**

Add three new fields to the struct at line 378:

```rust
pub struct TransitionConsistencyReport {
    pub from_street: String,
    pub to_street: String,
    pub bucket_count: u16,
    pub child_bucket_count: u16,     // NEW: K for normalization context
    pub sample_boards: usize,
    pub buckets: Vec<BucketTransitionStats>,
    pub mean_emd: f64,               // now normalized
    pub max_emd: f64,                // now normalized
    pub mean_between_emd: Option<f64>,   // NEW: mean centroid-to-centroid EMD (normalized)
    pub separation_ratio: Option<f64>,   // NEW: between / within
}
```

**Step 4: Normalize all EMD values in `audit_transition_consistency`**

After computing `emds` at line 544-547, normalize by `(next_k - 1)`:

```rust
            let norm = if next_k > 1 { 1.0 / (next_k as f64 - 1.0) } else { 1.0 };
            let emds: Vec<f64> = hists
                .iter()
                .map(|h| emd_1d(h, &centroid) * norm)
                .collect();
```

Also normalize the global `mean_emd` and `max_emd` at lines 572-581 (they are already computed from the per-bucket `mean_emd_to_centroid` which will now be normalized, so no extra normalization needed there).

Set the new fields to `None` for now (centroids task fills them in):

```rust
    TransitionConsistencyReport {
        from_street,
        to_street,
        bucket_count: current_bf.header.bucket_count,
        child_bucket_count: next_bf.header.bucket_count,
        sample_boards: num_sample_boards,
        buckets: bucket_stats,
        mean_emd,
        max_emd,
        mean_between_emd: None,
        separation_ratio: None,
    }
```

**Step 5: Update `summary()` output format**

Replace the existing summary format string (line 394-397) with:

```rust
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let non_empty: Vec<_> = self.buckets.iter().filter(|b| b.count > 0).collect();
        let mut s = format!(
            "{} → {}: {} buckets, {} sample boards\n  Within-bucket EMD: mean={:.4}, max={:.4} (normalized)",
            self.from_street, self.to_street, non_empty.len(), self.sample_boards,
            self.mean_emd, self.max_emd,
        );
        if let Some(between) = self.mean_between_emd {
            let _ = write!(s, "\n  Between-centroid EMD: mean={:.4}", between);
        }
        if let Some(ratio) = self.separation_ratio {
            let _ = write!(s, "\n  Separation ratio: {:.2} (higher = better)", ratio);
        }
        s.push_str("\n  Per-bucket (sorted by mean EMD):");
        // ... rest of per-bucket output unchanged ...
```

**Step 6: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: PASS (may need to fix compilation of any code that constructs `TransitionConsistencyReport` directly).

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_diagnostics.rs
git commit -m "feat: normalize transition EMD to [0,1] and add separation ratio fields"
```

---

### Task 2: Accept persisted centroids and compute separation ratio

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn centroid_separation_ratio() {
    // Two centroids far apart should give high separation.
    // Centroid 0: mass at bucket 0. Centroid 1: mass at bucket 2.
    let centroids = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let between = mean_pairwise_centroid_emd(&centroids);
    // EMD between [1,0,0] and [0,0,1] on 3 buckets: CDF diffs = [1.0, 1.0], sum = 2.0
    // Normalized by (3-1) = 2: 2.0 / 2.0 = 1.0
    assert!((between - 1.0).abs() < 1e-10);
}

#[test]
fn centroid_separation_close() {
    // Two adjacent centroids should give low separation.
    let centroids = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let between = mean_pairwise_centroid_emd(&centroids);
    // EMD = 1.0, normalized by 2 = 0.5
    assert!((between - 0.5).abs() < 1e-10);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core centroid_separation`
Expected: FAIL — `mean_pairwise_centroid_emd` doesn't exist.

**Step 3: Implement `mean_pairwise_centroid_emd`**

Add a helper function:

```rust
/// Mean pairwise EMD between all centroid pairs, normalized by (K-1).
///
/// Computes K*(K-1)/2 pairwise EMD values between centroids and returns
/// the mean. Normalized by (dim-1) where dim is the centroid dimension
/// (= number of child-street buckets).
#[must_use]
pub fn mean_pairwise_centroid_emd(centroids: &[Vec<f64>]) -> f64 {
    let k = centroids.len();
    if k < 2 {
        return 0.0;
    }
    let dim = centroids[0].len();
    let norm = if dim > 1 { 1.0 / (dim as f64 - 1.0) } else { 1.0 };
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..k {
        for j in (i + 1)..k {
            total += emd_1d(&centroids[i], &centroids[j]) * norm;
            count += 1;
        }
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core centroid_separation`
Expected: PASS.

**Step 5: Add `centroid_file` parameter to `audit_transition_consistency`**

Change the signature:

```rust
pub fn audit_transition_consistency(
    current_bf: &BucketFile,
    next_bf: &BucketFile,
    num_sample_boards: usize,
    seed: u64,
    centroid_file: Option<&CentroidFile>,  // NEW
) -> TransitionConsistencyReport {
```

Add `use super::centroid_file::CentroidFile;` at the top of the file.

When `centroid_file` is `Some`, use its centroids instead of reconstructing:

In the per-bucket stats computation (line 529-541), replace the centroid reconstruction with:

```rust
            // Use persisted centroid if available, otherwise reconstruct from samples.
            let centroid = if let Some(cf) = centroid_file {
                if (id) < cf.centroids().len() {
                    cf.centroids()[id].clone()
                } else {
                    // Fallback: reconstruct
                    reconstruct_centroid(hists, dim)
                }
            } else {
                reconstruct_centroid(hists, dim)
            };
```

Extract the existing centroid reconstruction into a helper:

```rust
fn reconstruct_centroid(hists: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut centroid = vec![0.0_f64; dim];
    for h in hists {
        for (j, &v) in h.iter().enumerate() {
            centroid[j] += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / hists.len() as f64;
    for v in &mut centroid {
        *v *= inv;
    }
    centroid
}
```

At the end of the function, compute separation ratio when centroids are available:

```rust
    let (mean_between_emd, separation_ratio) = if let Some(cf) = centroid_file {
        if cf.centroids().len() >= 2 {
            let between = mean_pairwise_centroid_emd(cf.centroids());
            let ratio = if mean_emd > 0.0 { between / mean_emd } else { f64::INFINITY };
            (Some(between), Some(ratio))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
```

**Step 6: Run full test suite**

Run: `cargo test -p poker-solver-core`
Expected: Compilation errors from callers passing 4 args instead of 5. Fix any callers inside the crate by passing `None` for the new parameter.

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_diagnostics.rs
git commit -m "feat: accept persisted centroids and compute separation ratio in transition audit"
```

---

### Task 3: Wire centroids into CLI

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Update the CLI transition audit block to load centroids**

In `main.rs` around line 644-657, change the transition audit section:

```rust
            if transition_audit {
                use poker_solver_core::blueprint_v2::cluster_diagnostics::audit_transition_consistency;
                use poker_solver_core::blueprint_v2::centroid_file::CentroidFile;
                let pairs = [("flop", "turn"), ("turn", "river")];
                for (from_name, to_name) in &pairs {
                    let from_path = cluster_dir.join(format!("{from_name}.buckets"));
                    let to_path = cluster_dir.join(format!("{to_name}.buckets"));
                    if from_path.exists() && to_path.exists() {
                        eprintln!("\nTransition consistency audit: {from_name} → {to_name} ({transition_audit_boards} sample boards)...");
                        let from_bf = BucketFile::load(&from_path)?;
                        let to_bf = BucketFile::load(&to_path)?;
                        // Load centroid file for the current (from) street if available.
                        let centroid_path = cluster_dir.join(format!("{from_name}.centroids"));
                        let centroids = if centroid_path.exists() {
                            CentroidFile::load(&centroid_path).ok()
                        } else {
                            None
                        };
                        let report = audit_transition_consistency(
                            &from_bf, &to_bf, transition_audit_boards, 42,
                            centroids.as_ref(),
                        );
                        eprintln!("{}", report.summary());
                    }
                }
            }
```

**Step 2: Build and verify**

Run: `cargo build -p poker-solver-trainer --release`
Expected: Compiles clean.

**Step 3: Run diagnostics on actual data to verify output format**

Run: `cargo run -p poker-solver-trainer --release -- diag-clusters -d local_data/buckets/500bkt --transition-audit --transition-audit-boards 30`
Expected: Output shows normalized EMD values, between-centroid EMD, and separation ratio.

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat: load centroid files in CLI transition audit for separation ratio"
```

---

### Task 4: Verify end-to-end and run full suite

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All pass, under 60 seconds.

**Step 2: Run clippy**

Run: `cargo clippy -p poker-solver-core`
Expected: No new warnings.

**Step 3: Run diagnostics on both bucket directories and verify output**

Run: `cargo run -p poker-solver-trainer --release -- diag-clusters -d local_data/buckets/500bkt --transition-audit --transition-audit-boards 30`

Expected output format:
```
Flop → Turn: NNN buckets, 30 sample boards
  Within-bucket EMD: mean=X.XXXX, max=X.XXXX (normalized)
  Between-centroid EMD: mean=X.XXXX
  Separation ratio: X.XX (higher = better)
```

**Step 4: Commit any fixups**

```bash
git add -u
git commit -m "fix: clean up transition diagnostic output and tests"
```
