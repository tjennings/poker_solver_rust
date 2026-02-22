# Global K-Means Bucketing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-texture k-means hand bucketing with global pooled k-means so bucket IDs are meaningful across all flop textures.

**Architecture:** Pool all `(hand, texture)` 3D EHS feature vectors into a single k-means run, then map assignments back to `[hand][texture]` shape. Blocked hands assigned to nearest global centroid. Relabel for bucket 0 = strongest.

**Tech Stack:** Rust, existing `kmeans()` and `EhsFeatures` infrastructure in `crates/core/src/preflop/hand_buckets.rs`.

---

### Task 1: Write failing test for `cluster_global`

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs` (tests module, after line 1307)

**Step 1: Write the failing test**

Add this test at the end of the `mod tests` block:

```rust
#[timed_test]
fn cluster_global_assigns_high_ehs_to_strong_buckets_across_textures() {
    // 6 hands across 3 textures, 3 buckets.
    // Hand 0-1: strong EHS ~0.9 on all textures
    // Hand 2-3: medium EHS ~0.5 on all textures
    // Hand 4-5: weak EHS ~0.1 on all textures
    // With per-texture k-means, bucket IDs could differ per texture.
    // With global k-means, same-strength hands must get same bucket everywhere.
    let features = vec![
        // Strong
        vec![[0.90, 0.0, 0.0], [0.91, 0.0, 0.0], [0.89, 0.0, 0.0]],
        vec![[0.88, 0.0, 0.0], [0.92, 0.0, 0.0], [0.87, 0.0, 0.0]],
        // Medium
        vec![[0.50, 0.0, 0.0], [0.52, 0.0, 0.0], [0.48, 0.0, 0.0]],
        vec![[0.48, 0.0, 0.0], [0.51, 0.0, 0.0], [0.49, 0.0, 0.0]],
        // Weak
        vec![[0.10, 0.0, 0.0], [0.12, 0.0, 0.0], [0.11, 0.0, 0.0]],
        vec![[0.08, 0.0, 0.0], [0.09, 0.0, 0.0], [0.13, 0.0, 0.0]],
    ];
    let assignments = cluster_global(&features, 3, 3);

    assert_eq!(assignments.len(), 6, "one row per hand");
    assert_eq!(assignments[0].len(), 3, "one column per texture");

    // After relabeling: bucket 0 = strong, bucket 2 = weak.
    // Strong hands should have bucket 0 on ALL textures.
    assert_eq!(assignments[0][0], 0);
    assert_eq!(assignments[0][1], 0);
    assert_eq!(assignments[0][2], 0);
    assert_eq!(assignments[1][0], 0);

    // Medium hands: bucket 1 on all textures.
    assert_eq!(assignments[2][0], 1);
    assert_eq!(assignments[2][1], 1);
    assert_eq!(assignments[3][0], 1);

    // Weak hands: bucket 2 on all textures.
    assert_eq!(assignments[4][0], 2);
    assert_eq!(assignments[4][1], 2);
    assert_eq!(assignments[5][0], 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core cluster_global_assigns`
Expected: FAIL — `cluster_global` does not exist yet.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "test: add failing test for cluster_global bucket consistency"
```

---

### Task 2: Write failing test for blocked hands in `cluster_global`

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs` (tests module)

**Step 1: Write the failing test**

```rust
#[timed_test]
fn cluster_global_assigns_blocked_hands_to_nearest_centroid() {
    // 5 hands, 2 textures, 2 buckets.
    // Hand 2 is blocked (NaN) on texture 0 but strong on texture 1.
    // Should be assigned to strong bucket on texture 0 via cross-texture average.
    let features = vec![
        vec![[0.90, 0.0, 0.0], [0.92, 0.0, 0.0]],
        vec![[0.88, 0.0, 0.0], [0.91, 0.0, 0.0]],
        vec![[f64::NAN, f64::NAN, f64::NAN], [0.95, 0.0, 0.0]],
        vec![[0.20, 0.0, 0.0], [0.18, 0.0, 0.0]],
        vec![[0.22, 0.0, 0.0], [0.19, 0.0, 0.0]],
    ];
    let assignments = cluster_global(&features, 2, 2);

    let strong_bucket = assignments[0][0];
    let weak_bucket = assignments[3][0];
    assert_ne!(strong_bucket, weak_bucket, "should have 2 distinct clusters");
    assert_eq!(
        assignments[2][0], strong_bucket,
        "blocked strong hand should be assigned to strong bucket"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core cluster_global_assigns_blocked`
Expected: FAIL — `cluster_global` does not exist yet.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "test: add failing test for cluster_global blocked hand assignment"
```

---

### Task 3: Implement `cluster_global`

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs` (after `cluster_per_texture`, around line 484)

**Step 1: Implement the function**

Add `cluster_global` right after `cluster_per_texture`:

```rust
/// Cluster features globally: pool all `(hand, texture)` feature vectors into one
/// k-means run so bucket IDs are consistent across textures.
///
/// Returns `assignments[hand_idx][texture_id]` → bucket ID.
///
/// Unlike `cluster_per_texture` which runs independent k-means per texture,
/// this pools all non-NaN feature vectors, runs k-means once, and maps
/// assignments back. Blocked (NaN) hands are assigned to the nearest global
/// centroid using their cross-texture average.
#[allow(clippy::cast_precision_loss)]
pub fn cluster_global(
    features: &[Vec<EhsFeatures>],
    k: u16,
    num_textures: usize,
) -> Vec<Vec<u16>> {
    let num_hands = features.len();

    // Precompute each hand's average feature vector across non-blocked textures
    let hand_averages: Vec<EhsFeatures> = (0..num_hands)
        .map(|h| {
            let valid: Vec<&EhsFeatures> = features[h]
                .iter()
                .filter(|f| !f[0].is_nan())
                .collect();
            if valid.is_empty() {
                return [0.5, 0.0, 0.0];
            }
            let n = valid.len() as f64;
            [
                valid.iter().map(|f| f[0]).sum::<f64>() / n,
                valid.iter().map(|f| f[1]).sum::<f64>() / n,
                valid.iter().map(|f| f[2]).sum::<f64>() / n,
            ]
        })
        .collect();

    // Pool all non-NaN (hand, texture) feature vectors into one flat list.
    // Track the original (hand, texture) index for each pooled point.
    let mut pooled_points: Vec<EhsFeatures> = Vec::new();
    let mut pooled_origins: Vec<(usize, usize)> = Vec::new(); // (hand_idx, tex_id)
    let mut blocked: Vec<(usize, usize)> = Vec::new();

    for (h, hand_feats) in features.iter().enumerate() {
        for (t, feat) in hand_feats.iter().enumerate() {
            if feat[0].is_nan() {
                blocked.push((h, t));
            } else {
                pooled_points.push(*feat);
                pooled_origins.push((h, t));
            }
        }
    }

    // Run k-means once on all pooled points
    let pooled_assignments = kmeans(&pooled_points, k as usize, 100);

    // Compute global centroids for assigning blocked hands
    let centroids = recompute_centroids(&pooled_points, &pooled_assignments, k as usize);

    // Map assignments back to [hand][texture] shape
    let mut assignments = vec![vec![0u16; num_textures]; num_hands];
    for (i, &(h, t)) in pooled_origins.iter().enumerate() {
        assignments[h][t] = pooled_assignments[i];
    }
    for &(h, t) in &blocked {
        assignments[h][t] = nearest_centroid(&hand_averages[h], &centroids);
    }

    // Relabel so bucket 0 = highest average EHS
    relabel_by_centroid_ehs(&mut assignments, features, k as usize);

    assignments
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core cluster_global`
Expected: Both `cluster_global_assigns_high_ehs_to_strong_buckets_across_textures` and `cluster_global_assigns_blocked_hands_to_nearest_centroid` PASS.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "feat: add cluster_global for globally consistent hand buckets"
```

---

### Task 4: Wire `cluster_global` into `build_flop/turn/river_buckets`

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs` (lines 91, 111, 129)

**Step 1: Replace `cluster_per_texture` calls with `cluster_global`**

Three one-line changes:

Line 91 (in `build_flop_buckets`):
```rust
// OLD:
Ok(cluster_per_texture(&features, num_buckets, board_samples.len()))
// NEW:
Ok(cluster_global(&features, num_buckets, board_samples.len()))
```

Line 111 (in `build_turn_buckets`):
```rust
// OLD:
Ok(cluster_per_texture(&features, num_buckets, turn_board_samples.len()))
// NEW:
Ok(cluster_global(&features, num_buckets, turn_board_samples.len()))
```

Line 129 (in `build_river_buckets`):
```rust
// OLD:
Ok(cluster_per_texture(&features, num_buckets, river_board_samples.len()))
// NEW:
Ok(cluster_global(&features, num_buckets, river_board_samples.len()))
```

**Step 2: Run all tests**

Run: `cargo test -p poker-solver-core`
Expected: All tests pass. The existing `cluster_per_texture_*` tests still pass because `cluster_per_texture` is unchanged — only the `build_*` functions now route through `cluster_global`.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "feat: switch build_flop/turn/river_buckets to cluster_global"
```

---

### Task 5: Wire `cluster_global` into `postflop_abstraction.rs`

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (line 444)

**Step 1: Replace the direct `cluster_per_texture` call**

Line 444:
```rust
// OLD:
let flop_buckets = hand_buckets::cluster_per_texture(&features, num_buckets, num_textures);
// NEW:
let flop_buckets = hand_buckets::cluster_global(&features, num_buckets, num_textures);
```

**Step 2: Run all tests**

Run: `cargo test -p poker-solver-core`
Expected: All pass.

**Step 3: Run clippy**

Run: `cargo clippy -p poker-solver-core`
Expected: No new warnings.

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs
git commit -m "feat: use cluster_global in postflop abstraction builder"
```

---

### Task 6: Verify with trace-hand

**This is a manual validation step, not a code change.**

**Step 1: Clear cache**

Run: `rm -rf cache/postflop`

**Step 2: Run trace-hand**

Run:
```bash
cargo run -p poker-solver-trainer --release -- trace-hand \
  -c sample_configurations/fast_buckets.yaml \
  --cache-dir cache/postflop > local_data/trace_output_3.json
```

**Step 3: Validate bucket quality**

Run the Python diagnostic from the analysis session:

```python
import json, statistics
with open('local_data/trace_output_3.json') as f:
    data = json.load(f)
hands = data['hands']

# Check within-bucket EHS spread (should be much tighter now)
from collections import defaultdict
bucket_ehs = defaultdict(list)
for h in hands:
    for t in h['textures']:
        if not t.get('blocked'):
            bucket_ehs[t['flop_bucket']].append(t['ehs_features'][0])

print("Bucket  Centroid  Min EHS  Max EHS  Std      Range    N")
for b in sorted(bucket_ehs):
    vals = bucket_ehs[b]
    mn, mx = min(vals), max(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f"  {b:2d}    {statistics.mean(vals):.4f}   {mn:.4f}   {mx:.4f}   {std:.4f}   {mx-mn:.4f}   {len(vals)}")

# Check for high-EHS hands in weak buckets (should be 0 or near-0)
mismatches = sum(1 for h in hands for t in h['textures']
                 if not t.get('blocked') and t['ehs_features'][0] >= 0.8 and t['flop_bucket'] >= 15)
print(f"\nHigh-EHS (>=0.8) in weak buckets (>=15): {mismatches} (was 401)")

# Check AA EV
aa = [h for h in hands if h['hand'] == 'AA'][0]
neg = sum(1 for t in aa['textures'] if not t.get('blocked') and t['postflop_ev']['raised']['sb'] < 0)
print(f"AA negative-EV textures: {neg} (was 3)")
```

Expected:
- Within-bucket EHS range < 0.3 for most buckets (was ~1.0)
- High-EHS mismatches near 0 (was 401)
- AA negative-EV textures: 0 (was 3)

**Step 4: Commit trace output (optional)**

No code commit needed — this is validation only.
