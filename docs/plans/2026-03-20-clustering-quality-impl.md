# Clustering Quality Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Improve blueprint v2 clustering quality by persisting centroids, sorting bucket IDs by expected equity, switching exhaustive assignment from L2 to EMD, and weighting EMD by actual centroid equity gaps.

**Architecture:** Four changes layered bottom-up. CentroidFile is a new I/O adapter (binary serialization). Sorting, weighted EMD, and assignment changes are domain logic in the clustering module. Pipeline orchestration wires them together in cluster_pipeline.rs.

**Tech Stack:** Rust, rayon (parallelism), existing clustering.rs EMD functions, existing bucket_file.rs as reference for binary format patterns.

---

### Task 1: CentroidFile — binary format and round-trip

**Files:**
- Create: `crates/core/src/blueprint_v2/centroid_file.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`

**Step 1: Write the failing test**

Add to `crates/core/src/blueprint_v2/centroid_file.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_centroid_file() {
        let centroids = vec![
            vec![0.1, 0.5, 0.4],
            vec![0.3, 0.3, 0.4],
        ];
        let cf = CentroidFile::new(Street::Turn, centroids.clone());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.centroids");
        cf.save(&path).unwrap();
        let loaded = CentroidFile::load(&path).unwrap();
        assert_eq!(loaded.street(), Street::Turn);
        assert_eq!(loaded.centroids().len(), 2);
        assert_eq!(loaded.centroids()[0].len(), 3);
        for (orig, loaded) in centroids.iter().zip(loaded.centroids()) {
            for (a, b) in orig.iter().zip(loaded) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn round_trip_scalar_centroids() {
        let centroids = vec![vec![0.05], vec![0.25], vec![0.75], vec![0.95]];
        let cf = CentroidFile::new(Street::River, centroids.clone());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("river.centroids");
        cf.save(&path).unwrap();
        let loaded = CentroidFile::load(&path).unwrap();
        assert_eq!(loaded.street(), Street::River);
        assert_eq!(loaded.centroids().len(), 4);
        for (orig, loaded) in centroids.iter().zip(loaded.centroids()) {
            assert!((orig[0] - loaded[0]).abs() < 1e-12);
        }
    }

    #[test]
    fn bad_magic_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.centroids");
        std::fs::write(&path, b"XXXX").unwrap();
        assert!(CentroidFile::load(&path).is_err());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core centroid_file -- --nocapture`
Expected: FAIL — module does not exist yet.

**Step 3: Write minimal implementation**

Create `crates/core/src/blueprint_v2/centroid_file.rs`:

```rust
//! Binary format for storing k-means centroids per street.
//!
//! ## Wire format (little-endian)
//!
//! | Offset | Size | Field |
//! |-|-|-|
//! | 0 | 4 | Magic `CEN1` |
//! | 4 | 1 | Street (0-3) |
//! | 5 | 2 | K (number of centroids) |
//! | 7 | 2 | dim (dimension per centroid) |
//! | 9 | K*dim*8 | f64 centroid values |

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::Street;

const MAGIC: &[u8; 4] = b"CEN1";

/// Persistent storage for k-means centroids.
pub struct CentroidFile {
    street: Street,
    centroids: Vec<Vec<f64>>,
}

impl CentroidFile {
    /// Create a new centroid file from computed centroids.
    #[must_use]
    pub fn new(street: Street, centroids: Vec<Vec<f64>>) -> Self {
        Self { street, centroids }
    }

    /// The street these centroids belong to.
    #[must_use]
    pub fn street(&self) -> Street {
        self.street
    }

    /// The learned centroids (one Vec<f64> per cluster).
    #[must_use]
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Save to a binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&[self.street as u8])?;
        let k = self.centroids.len() as u16;
        let dim = self.centroids.first().map_or(0, |c| c.len()) as u16;
        w.write_all(&k.to_le_bytes())?;
        w.write_all(&dim.to_le_bytes())?;
        for centroid in &self.centroids {
            for &val in centroid {
                w.write_all(&val.to_le_bytes())?;
            }
        }
        w.flush()
    }

    /// Load from a binary file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad centroid file magic"));
        }
        let mut street_byte = [0u8; 1];
        r.read_exact(&mut street_byte)?;
        let street = Street::from_u8(street_byte[0])
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad street byte"))?;
        let mut buf2 = [0u8; 2];
        r.read_exact(&mut buf2)?;
        let k = u16::from_le_bytes(buf2) as usize;
        r.read_exact(&mut buf2)?;
        let dim = u16::from_le_bytes(buf2) as usize;
        let mut centroids = Vec::with_capacity(k);
        let mut f64_buf = [0u8; 8];
        for _ in 0..k {
            let mut centroid = Vec::with_capacity(dim);
            for _ in 0..dim {
                r.read_exact(&mut f64_buf)?;
                centroid.push(f64::from_le_bytes(f64_buf));
            }
            centroids.push(centroid);
        }
        Ok(Self { street, centroids })
    }
}
```

Add to `crates/core/src/blueprint_v2/mod.rs`:

```rust
pub mod centroid_file;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core centroid_file -- --nocapture`
Expected: 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/centroid_file.rs crates/core/src/blueprint_v2/mod.rs
git commit -m "feat: add CentroidFile binary format with round-trip serialization"
```

---

### Task 2: Weighted EMD functions

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs`

**Step 1: Write the failing tests**

Add to the existing `#[cfg(test)] mod tests` block in `clustering.rs`:

```rust
#[test]
fn test_emd_u8_vs_f64_weighted_uniform_gaps() {
    // With uniform gaps of 1.0, weighted EMD should equal unweighted EMD.
    let counts: Vec<u8> = vec![20, 15, 10, 1];
    let centroid: Vec<f64> = vec![0.1, 0.4, 0.3, 0.2];
    let gaps = vec![1.0, 1.0, 1.0]; // K-1 = 3 gaps
    let weighted = emd_u8_vs_f64_weighted(&counts, &centroid, &gaps);
    let unweighted = emd_u8_vs_f64(&counts, &centroid);
    assert!((weighted - unweighted).abs() < 1e-10, "w={weighted} uw={unweighted}");
}

#[test]
fn test_emd_u8_vs_f64_weighted_nonuniform_gaps() {
    // Mass shift across a large gap should cost more than across a small gap.
    let a: Vec<u8> = vec![40, 0, 0, 0]; // all mass in bucket 0
    let b_near: Vec<f64> = vec![0.0, 1.0, 0.0, 0.0]; // mass in bucket 1
    let b_far: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0]; // mass in bucket 3
    let gaps = vec![0.1, 0.3, 0.5]; // increasing gaps
    let d_near = emd_u8_vs_f64_weighted(&a, &b_near, &gaps);
    let d_far = emd_u8_vs_f64_weighted(&a, &b_far, &gaps);
    assert!(d_far > d_near, "far={d_far} should > near={d_near}");
}

#[test]
fn test_nearest_centroid_u8_weighted() {
    // Point closer to centroid 0 under weighted EMD.
    let point: Vec<u8> = vec![36, 4, 0, 0];
    let centroids = vec![
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.1, 0.9],
    ];
    let gaps = vec![1.0, 1.0, 1.0];
    let idx = nearest_centroid_u8_weighted(&point, &centroids, &gaps);
    assert_eq!(idx, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_emd_u8_vs_f64_weighted -- --nocapture`
Expected: FAIL — functions don't exist.

**Step 3: Write minimal implementation**

Add to `clustering.rs` after the existing `emd_u8_vs_f64` function:

```rust
/// Weighted EMD between a u8 count histogram and an f64 centroid.
///
/// Like [`emd_u8_vs_f64`] but each CDF step is weighted by the ground
/// distance gap between adjacent centroids. `gaps` has length `K-1` where
/// `K = counts.len()`. With uniform gaps of 1.0 this equals the unweighted
/// variant.
#[must_use]
pub fn emd_u8_vs_f64_weighted(counts: &[u8], centroid: &[f64], gaps: &[f64]) -> f64 {
    debug_assert_eq!(counts.len(), centroid.len());
    debug_assert_eq!(gaps.len(), counts.len() - 1);
    let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    let mut cdf_diff = 0.0_f64;
    let mut distance = 0.0_f64;
    for i in 0..counts.len() - 1 {
        cdf_diff += f64::from(counts[i]) * inv - centroid[i];
        distance += cdf_diff.abs() * gaps[i];
    }
    distance
}

/// Assign a u8 histogram to the nearest centroid using weighted EMD.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn nearest_centroid_u8_weighted(point: &[u8], centroids: &[Vec<f64>], gaps: &[f64]) -> u16 {
    let mut best_idx = 0_u16;
    let mut best_dist = f64::MAX;
    for (ci, centroid) in centroids.iter().enumerate() {
        let d = emd_u8_vs_f64_weighted(point, centroid, gaps);
        if d < best_dist {
            best_dist = d;
            best_idx = ci as u16;
        }
    }
    best_idx
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core test_emd_u8_vs_f64_weighted test_nearest_centroid_u8_weighted -- --nocapture`
Expected: 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add weighted EMD functions for non-uniform centroid spacing"
```

---

### Task 3: Centroid sorting and EV computation helpers

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs`

These are pure domain functions: compute expected equity from a centroid histogram, sort centroids by EV, and derive the gaps vector.

**Step 1: Write the failing tests**

Add to the `tests` module in `clustering.rs`:

```rust
#[test]
fn test_compute_centroid_evs() {
    // 3 river centroids with equity values 0.1, 0.5, 0.9
    let child_evs = vec![0.1, 0.5, 0.9];
    // A turn centroid that puts 50% weight on bucket 0 and 50% on bucket 2
    let centroid = vec![0.5, 0.0, 0.5];
    let ev = compute_centroid_ev(&centroid, &child_evs);
    // EV = 0.5*0.1 + 0.0*0.5 + 0.5*0.9 = 0.5
    assert!((ev - 0.5).abs() < 1e-10);
}

#[test]
fn test_sort_centroids_by_ev() {
    let child_evs = vec![0.1, 0.5, 0.9];
    let centroids = vec![
        vec![0.0, 0.0, 1.0], // EV = 0.9 (highest)
        vec![1.0, 0.0, 0.0], // EV = 0.1 (lowest)
        vec![0.0, 1.0, 0.0], // EV = 0.5 (middle)
    ];
    let (sorted, remap) = sort_centroids_by_ev(&centroids, &child_evs);
    // After sorting: bucket 0 = EV 0.1, bucket 1 = EV 0.5, bucket 2 = EV 0.9
    assert_eq!(remap, vec![2, 0, 1]); // old[0]->new[2], old[1]->new[0], old[2]->new[1]
    assert!((sorted[0][0] - 1.0).abs() < 1e-10); // was centroids[1]
    assert!((sorted[1][1] - 1.0).abs() < 1e-10); // was centroids[2]
    assert!((sorted[2][2] - 1.0).abs() < 1e-10); // was centroids[0]
}

#[test]
fn test_compute_centroid_gaps() {
    let evs = vec![0.1, 0.3, 0.9]; // 3 centroids sorted by EV
    let gaps = compute_centroid_gaps(&evs);
    assert_eq!(gaps.len(), 2);
    assert!((gaps[0] - 0.2).abs() < 1e-10);
    assert!((gaps[1] - 0.6).abs() < 1e-10);
}

#[test]
fn test_remap_labels() {
    let labels: Vec<u16> = vec![0, 1, 2, 0, 2, 1];
    let remap = vec![2, 0, 1]; // old[0]->new[2], old[1]->new[0], old[2]->new[1]
    let remapped = remap_labels(&labels, &remap);
    assert_eq!(remapped, vec![2, 0, 1, 2, 1, 0]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_compute_centroid_evs test_sort_centroids_by_ev test_compute_centroid_gaps test_remap_labels -- --nocapture`
Expected: FAIL — functions don't exist.

**Step 3: Write minimal implementation**

Add to `clustering.rs` (before the `tests` module):

```rust
// ---------------------------------------------------------------------------
// Centroid sorting and EV computation
// ---------------------------------------------------------------------------

/// Compute the expected equity of a centroid histogram given child-street EVs.
///
/// `centroid[i]` is the probability weight on child bucket `i`.
/// `child_evs[i]` is the expected equity of child bucket `i`.
/// Returns `Σ centroid[i] × child_evs[i]`.
#[must_use]
pub fn compute_centroid_ev(centroid: &[f64], child_evs: &[f64]) -> f64 {
    debug_assert_eq!(centroid.len(), child_evs.len());
    centroid.iter().zip(child_evs).map(|(w, ev)| w * ev).sum()
}

/// Sort centroids by their expected equity (ascending) and return a remap vector.
///
/// Returns `(sorted_centroids, remap)` where `remap[old_id] = new_id`.
#[must_use]
pub fn sort_centroids_by_ev(
    centroids: &[Vec<f64>],
    child_evs: &[f64],
) -> (Vec<Vec<f64>>, Vec<u16>) {
    let mut indexed_evs: Vec<(usize, f64)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, compute_centroid_ev(c, child_evs)))
        .collect();
    indexed_evs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // indexed_evs[new_id] = (old_id, ev)
    // We need remap[old_id] = new_id
    let mut remap = vec![0_u16; centroids.len()];
    let mut sorted = Vec::with_capacity(centroids.len());
    #[allow(clippy::cast_possible_truncation)]
    for (new_id, (old_id, _ev)) in indexed_evs.iter().enumerate() {
        remap[*old_id] = new_id as u16;
        sorted.push(centroids[*old_id].clone());
    }
    (sorted, remap)
}

/// Compute the equity gaps between adjacent sorted centroids.
///
/// Given sorted EVs `[ev_0, ev_1, ..., ev_{K-1}]`, returns
/// `[ev_1 - ev_0, ev_2 - ev_1, ..., ev_{K-1} - ev_{K-2}]` (length K-1).
#[must_use]
pub fn compute_centroid_gaps(sorted_evs: &[f64]) -> Vec<f64> {
    sorted_evs.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Apply a remap permutation to bucket labels.
///
/// `remap[old_id] = new_id`. Returns a new label vector with remapped IDs.
#[must_use]
pub fn remap_labels(labels: &[u16], remap: &[u16]) -> Vec<u16> {
    labels.iter().map(|&old| remap[old as usize]).collect()
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core test_compute_centroid_evs test_sort_centroids_by_ev test_compute_centroid_gaps test_remap_labels -- --nocapture`
Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add centroid EV computation, sorting, gap calculation, and label remapping"
```

---

### Task 4: Wire CentroidFile into river clustering

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

River centroids are already sorted (1-D equity k-means). We just need to persist them and return them alongside the BucketFile so downstream streets can use them.

**Step 1: Change `cluster_river_exhaustive` to return centroids**

Change the return type from `BucketFile` to `(BucketFile, CentroidFile)`:

```rust
pub fn cluster_river_exhaustive(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    progress: impl Fn(&str, f64) + Sync,
) -> (BucketFile, CentroidFile) {
```

After the existing `let (_labels, centroids) = fast_kmeans_1d(...)` line, sort centroids and build the CentroidFile:

```rust
    let (_labels, mut centroids) = fast_kmeans_1d(
        &sample_vals,
        bucket_count as usize,
        kmeans_iterations,
        seed,
    );
    // Ensure centroids are sorted ascending (fast_kmeans_1d may not guarantee order).
    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let centroid_file = CentroidFile::new(
        Street::River,
        centroids.iter().map(|&c| vec![c]).collect(),
    );
```

Return `(bucket_file, centroid_file)` at the end of the function (where `bucket_file` is the existing `BucketFile { ... }` expression).

**Step 2: Update `run_clustering_pipeline` to handle the new return type and save centroids**

Add `use super::centroid_file::CentroidFile;` to the imports at the top of `cluster_pipeline.rs`.

In `run_clustering_pipeline`, change the river section:

```rust
    let (river, river_centroids) = if let Some(ref cfvnet_dir) = config.cfvnet_river_data {
        // cfvnet path: build centroids from the bucket file after the fact
        let bf = cluster_river_from_cfvnet(/* ... existing args ... */)?;
        // We don't have centroids from cfvnet — skip centroid file for this path
        let cf = CentroidFile::new(Street::River, vec![]);
        (bf, cf)
    } else {
        let sample = config.river.sample_boards.unwrap_or(DEFAULT_NUM_BOARDS);
        cluster_river_exhaustive(
            config.river.buckets,
            config.kmeans_iterations,
            config.seed,
            sample,
            |phase, p| progress("river", phase, p),
        )
    };
    river.save(&output_dir.join("river.buckets"))?;
    if !river_centroids.centroids().is_empty() {
        river_centroids.save(&output_dir.join("river.centroids"))?;
    }
```

**Step 3: Run the existing e2e test to verify nothing is broken**

Run: `cargo test -p poker-solver-core --test blueprint_v2_e2e -- --nocapture`
Expected: PASS (river now returns a tuple but pipeline handles it).

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: persist river centroids alongside river.buckets"
```

---

### Task 5: Wire sorting + EMD assignment + centroids into turn/flop clustering

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

This is the main integration task. `cluster_histogram_exhaustive` needs to:
1. Accept child-street centroid EVs and gaps
2. Sort centroids by EV after k-means
3. Use EMD (not L2) for exhaustive assignment, weighted by gaps
4. Return a `CentroidFile` alongside the `BucketFile`

**Step 1: Change `cluster_histogram_exhaustive` signature**

```rust
fn cluster_histogram_exhaustive<const N: usize>(
    street: Street,
    prior_buckets: &BucketFile,
    all_canonical: &[WeightedBoard<N>],
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    child_centroid_evs: &[f64],  // NEW: EVs of child-street centroids
    child_centroid_gaps: &[f64], // NEW: gaps between adjacent child centroids
    progress: impl Fn(&str, f64) + Sync,
) -> (BucketFile, CentroidFile) {  // NEW: returns centroids too
```

**Step 2: After `fast_kmeans_histogram`, sort centroids and convert to f64**

Replace the section after `let (_labels, centroids) = fast_kmeans_histogram(...)`:

```rust
    let (_labels, raw_centroids) = fast_kmeans_histogram(
        &all_features,
        bucket_count as usize,
        kmeans_iterations,
        seed,
    );
    progress("k-means", 1.0);

    // Convert f32 centroids to f64 normalized probability distributions.
    let centroids_f64: Vec<Vec<f64>> = raw_centroids
        .iter()
        .map(|c| {
            let total: f64 = c.iter().map(|&v| f64::from(v)).sum();
            let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
            c.iter().map(|&v| f64::from(v) * inv).collect()
        })
        .collect();

    // Sort centroids by expected equity so bucket IDs are ordered.
    let (sorted_centroids, remap) = sort_centroids_by_ev(&centroids_f64, child_centroid_evs);

    // Compute EVs and gaps for this street's centroids (used by parent street).
    let sorted_evs: Vec<f64> = sorted_centroids
        .iter()
        .map(|c| compute_centroid_ev(c, child_centroid_evs))
        .collect();
    let gaps = compute_centroid_gaps(&sorted_evs);

    let centroid_file = CentroidFile::new(street, sorted_centroids.clone());
```

**Step 3: Replace L2 assignment with weighted EMD assignment**

In the Phase 2 assignment block, replace:

```rust
                    nearest_centroid_l2(&hist, &centroids)
```

with:

```rust
                    if child_centroid_gaps.is_empty() {
                        nearest_centroid_u8(&hist, &sorted_centroids)
                    } else {
                        nearest_centroid_u8_weighted(&hist, &sorted_centroids, child_centroid_gaps)
                    }
```

**Step 4: Return the tuple**

Replace the final `BucketFile { ... }` return with `(BucketFile { ... }, centroid_file)`.

**Step 5: Update `cluster_turn_exhaustive` and `cluster_flop_exhaustive` signatures**

Both now need child centroid EVs/gaps and return `(BucketFile, CentroidFile)`:

```rust
pub fn cluster_turn_exhaustive(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    river_centroid_evs: &[f64],  // NEW
    river_centroid_gaps: &[f64], // NEW
    progress: impl Fn(&str, f64) + Sync,
) -> (BucketFile, CentroidFile) {
    let all_canonical = enumerate_canonical_turns();
    cluster_histogram_exhaustive(
        Street::Turn,
        river_buckets,
        &all_canonical,
        bucket_count,
        kmeans_iterations,
        seed,
        sample_boards,
        river_centroid_evs,
        river_centroid_gaps,
        progress,
    )
}
```

```rust
pub fn cluster_flop_exhaustive(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    turn_centroid_evs: &[f64],  // NEW
    turn_centroid_gaps: &[f64], // NEW
    progress: impl Fn(&str, f64) + Sync,
) -> (BucketFile, CentroidFile) {
    let all_canonical = enumerate_canonical_flops();
    cluster_histogram_exhaustive(
        Street::Flop,
        turn_buckets,
        &all_canonical,
        bucket_count,
        kmeans_iterations,
        seed,
        sample_boards,
        turn_centroid_evs,
        turn_centroid_gaps,
        progress,
    )
}
```

**Step 6: Update `run_clustering_pipeline` to thread centroids through**

```rust
    // 2. Turn
    progress("turn", "sampling", 0.0);
    let sample_turn = config.turn.sample_boards.unwrap_or(DEFAULT_TURN_BOARDS);
    // Extract river centroid EVs (scalar centroids = dim 1).
    let river_evs: Vec<f64> = river_centroids.centroids().iter().map(|c| c[0]).collect();
    let river_gaps = compute_centroid_gaps(&river_evs);
    let (turn, turn_centroids) = cluster_turn_exhaustive(
        &river,
        config.turn.buckets,
        config.kmeans_iterations,
        config.seed,
        sample_turn,
        &river_evs,
        &river_gaps,
        |phase, p| progress("turn", phase, p),
    );
    turn.save(&output_dir.join("turn.buckets"))?;
    turn_centroids.save(&output_dir.join("turn.centroids"))?;

    // 3. Flop
    progress("flop", "sampling", 0.0);
    let sample_flop = config.flop.sample_boards.unwrap_or(1755);
    // Compute turn centroid EVs from turn centroids × river EVs.
    let turn_evs: Vec<f64> = turn_centroids.centroids()
        .iter()
        .map(|c| compute_centroid_ev(c, &river_evs))
        .collect();
    let turn_gaps = compute_centroid_gaps(&turn_evs);
    let (flop, flop_centroids) = cluster_flop_exhaustive(
        &turn,
        config.flop.buckets,
        config.kmeans_iterations,
        config.seed,
        sample_flop,
        &turn_evs,
        &turn_gaps,
        |phase, p| progress("flop", phase, p),
    );
    flop.save(&output_dir.join("flop.buckets"))?;
    flop_centroids.save(&output_dir.join("flop.centroids"))?;
```

Add the necessary imports at the top of `cluster_pipeline.rs`:

```rust
use super::centroid_file::CentroidFile;
use super::clustering::{
    compute_centroid_ev, compute_centroid_gaps, sort_centroids_by_ev,
    nearest_centroid_u8_weighted, nearest_centroid_u8,
    // ... existing imports ...
};
```

**Step 7: Run the e2e test**

Run: `cargo test -p poker-solver-core --test blueprint_v2_e2e -- --nocapture`
Expected: PASS — the e2e test exercises the full pipeline.

**Step 8: Run the full test suite**

Run: `cargo test`
Expected: All tests pass. Fix any compilation errors in callers (e.g., per-flop pipeline or diagnostics that call `cluster_turn_exhaustive` / `cluster_flop_exhaustive` — these need the new args or need to pass empty slices).

**Step 9: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: sort bucket IDs by EV, use weighted EMD for exhaustive assignment, persist turn/flop centroids"
```

---

### Task 6: Update callers and fix per-flop compilation

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs` (per-flop pipeline callers, if any directly call `cluster_turn_exhaustive` or `cluster_flop_exhaustive`)
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs` (if it calls clustering functions)

The per-flop pipeline has its own clustering code and does NOT call `cluster_turn_exhaustive`/`cluster_flop_exhaustive`, so it should compile without changes. But check for any other callers.

**Step 1: Search for callers**

Run: `grep -rn "cluster_turn_exhaustive\|cluster_flop_exhaustive\|cluster_river_exhaustive\|cluster_histogram_exhaustive" crates/`

Fix any callers that break. For callers outside the global pipeline, pass empty `&[]` for centroid EVs/gaps and destructure the tuple `let (bf, _cf) = ...` to discard centroids.

**Step 2: Run full test suite**

Run: `cargo test`
Expected: All tests PASS.

**Step 3: Run clippy**

Run: `cargo clippy`
Expected: No new warnings.

**Step 4: Commit (if changes were needed)**

```bash
git add -u
git commit -m "fix: update callers for new clustering return types"
```

---

### Task 7: Verify end-to-end and clean up

**Files:**
- Verify: `crates/core/tests/blueprint_v2_e2e.rs`

**Step 1: Run the full e2e test**

Run: `cargo test -p poker-solver-core --test blueprint_v2_e2e -- --nocapture`
Expected: PASS. Verify that `river.centroids`, `turn.centroids`, and `flop.centroids` files are created in the test output directory.

**Step 2: Run the full test suite under 1 minute**

Run: `time cargo test`
Expected: All pass, under 60 seconds.

**Step 3: Run clippy**

Run: `cargo clippy`
Expected: No warnings.

**Step 4: Verify centroid files exist in test output**

The e2e test uses `tempdir()`. Add a quick assertion to the e2e test (or verify manually) that centroid files were created:

```rust
assert!(cluster_dir.join("river.centroids").exists());
assert!(cluster_dir.join("turn.centroids").exists());
assert!(cluster_dir.join("flop.centroids").exists());
```

**Step 5: Commit**

```bash
git add -u
git commit -m "test: verify centroid files created in e2e pipeline test"
```
