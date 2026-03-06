# Blueprint V2: Pluribus-Style Full-Game Solver — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a new full-game MCCFR solver that trains all 4 streets in a single run, with potential-aware clustering, configurable action abstraction, and explorer-compatible snapshots.

**Architecture:** New `blueprint_v2` module in `crates/core/src/` with three CLI subcommands (`cluster`, `train-blueprint`, `diag-clusters`) in the trainer. External-sampling MCCFR with LCFR weighting consumes pre-computed bucket assignment files and produces snapshot bundles loadable by the explorer. Completely independent of the existing preflop/postflop pipeline.

**Tech Stack:** Rust, serde/serde_yaml, bincode, rayon, rs_poker, clap

**Design doc:** `docs/plans/2026-03-06-blueprint-v2-design.md`

---

## Phase 1: Config Types & Bucket File I/O

### Task 1: Blueprint V2 Config Types

Define all YAML-deserializable config types for the new solver.

**Files:**
- Create: `crates/core/src/blueprint_v2/mod.rs`
- Create: `crates/core/src/blueprint_v2/config.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod blueprint_v2;`)
- Test: `crates/core/src/blueprint_v2/config.rs` (inline tests)

**Step 1: Create module structure**

Create `crates/core/src/blueprint_v2/mod.rs`:
```rust
pub mod config;
```

Add `pub mod blueprint_v2;` to `crates/core/src/lib.rs`.

**Step 2: Write the config types**

Create `crates/core/src/blueprint_v2/config.rs` with these types:

```rust
use serde::{Deserialize, Serialize};

/// Top-level config for the full Blueprint V2 pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintV2Config {
    pub game: GameConfig,
    pub clustering: ClusteringConfig,
    pub action_abstraction: ActionAbstractionConfig,
    pub training: TrainingConfig,
    pub snapshots: SnapshotConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub players: u8,           // 2 for HU, future: 3-6
    pub stack_depth: f64,      // in BB
    pub small_blind: f64,      // in BB
    pub big_blind: f64,        // in BB
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    pub algorithm: ClusteringAlgorithm,
    pub preflop: StreetClusterConfig,
    pub flop: StreetClusterConfig,
    pub turn: StreetClusterConfig,
    pub river: StreetClusterConfig,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_kmeans_iterations")]
    pub kmeans_iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringAlgorithm {
    PotentialAwareEmd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreetClusterConfig {
    pub buckets: u16,
}

/// Action abstraction: per street, each entry is a list of sizes per raise depth.
/// Preflop uses "Xbb" (absolute) or "Xx" (multiplier). Postflop uses pot fractions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAbstractionConfig {
    pub preflop: Vec<Vec<String>>,   // e.g. [["2.5bb"], ["3.0x"]]
    pub flop: Vec<Vec<f64>>,         // e.g. [[0.33, 0.67, 1.0], [0.5, 1.0]]
    pub turn: Vec<Vec<f64>>,
    pub river: Vec<Vec<f64>>,
    pub max_raises: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub cluster_path: String,
    #[serde(default)]
    pub iterations: Option<u64>,
    #[serde(default)]
    pub time_limit_minutes: Option<u64>,
    #[serde(default = "default_lcfr_warmup")]
    pub lcfr_warmup_minutes: u64,
    #[serde(default = "default_discount_interval")]
    pub lcfr_discount_interval: u64,
    #[serde(default = "default_prune_after")]
    pub prune_after_minutes: u64,
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i32,
    #[serde(default = "default_prune_explore")]
    pub prune_explore_pct: f64,
    #[serde(default = "default_print_every")]
    pub print_every_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    pub warmup_minutes: u64,
    pub snapshot_every_minutes: u64,
    pub output_dir: String,
}

fn default_seed() -> u64 { 42 }
fn default_kmeans_iterations() -> u32 { 100 }
fn default_lcfr_warmup() -> u64 { 400 }
fn default_discount_interval() -> u64 { 10 }
fn default_prune_after() -> u64 { 200 }
fn default_prune_threshold() -> i32 { -310_000_000 }
fn default_prune_explore() -> f64 { 0.05 }
fn default_print_every() -> u64 { 10 }
```

**Step 3: Write tests for YAML round-trip**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_toy_config() {
        let yaml = r#"
game:
  players: 2
  stack_depth: 10
  small_blind: 0.5
  big_blind: 1.0
clustering:
  algorithm: potential_aware_emd
  preflop: { buckets: 50 }
  flop: { buckets: 50 }
  turn: { buckets: 50 }
  river: { buckets: 50 }
  seed: 42
  kmeans_iterations: 50
action_abstraction:
  preflop:
    - ["2.5bb"]
    - ["3.0x"]
  flop:
    - [0.5, 1.0]
    - [1.0]
  turn:
    - [0.5, 1.0]
  river:
    - [0.5, 1.0]
  max_raises: 2
training:
  cluster_path: clusters/toy/
  time_limit_minutes: 5
  lcfr_warmup_minutes: 2
  lcfr_discount_interval: 1
  prune_after_minutes: 2
  print_every_minutes: 1
snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: runs/toy/
"#;
        let config: BlueprintV2Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.players, 2);
        assert_eq!(config.game.stack_depth, 10.0);
        assert_eq!(config.clustering.flop.buckets, 50);
        assert_eq!(config.action_abstraction.max_raises, 2);
        assert_eq!(config.action_abstraction.flop.len(), 2); // 2 raise depths
        assert_eq!(config.action_abstraction.flop[0], vec![0.5, 1.0]);
    }

    #[test]
    fn test_serialize_round_trip() {
        let config = BlueprintV2Config {
            game: GameConfig { players: 2, stack_depth: 100.0, small_blind: 0.5, big_blind: 1.0 },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig { buckets: 169 },
                flop: StreetClusterConfig { buckets: 200 },
                turn: StreetClusterConfig { buckets: 200 },
                river: StreetClusterConfig { buckets: 200 },
                seed: 42,
                kmeans_iterations: 100,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["2.5bb".into()], vec!["3.0x".into()]],
                flop: vec![vec![0.33, 0.67, 1.0], vec![0.5, 1.0]],
                turn: vec![vec![0.5, 1.0], vec![1.0]],
                river: vec![vec![0.5, 1.0], vec![1.0]],
                max_raises: 3,
            },
            training: TrainingConfig {
                cluster_path: "clusters/".into(),
                iterations: None,
                time_limit_minutes: Some(480),
                lcfr_warmup_minutes: 400,
                lcfr_discount_interval: 10,
                prune_after_minutes: 200,
                prune_threshold: -310_000_000,
                prune_explore_pct: 0.05,
                print_every_minutes: 10,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 60,
                snapshot_every_minutes: 30,
                output_dir: "runs/full/".into(),
            },
        };
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: BlueprintV2Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.game.stack_depth, 100.0);
        assert_eq!(parsed.clustering.river.buckets, 200);
    }
}
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core blueprint_v2::config -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/ crates/core/src/lib.rs
git commit -m "feat(blueprint_v2): add config types with YAML deserialization"
```

---

### Task 2: Bucket File Format (Read/Write)

Binary format for storing bucket assignments per street. Memory-mappable for fast training-time lookup.

**Files:**
- Create: `crates/core/src/blueprint_v2/bucket_file.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs` (add `pub mod bucket_file;`)
- Test: inline in `bucket_file.rs`

**Step 1: Define the bucket file format**

```rust
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;
use serde::{Deserialize, Serialize};

const MAGIC: [u8; 4] = *b"BKT2";
const VERSION: u8 = 1;

/// Street identifier for bucket files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

/// Header for a bucket file
#[derive(Debug, Clone)]
pub struct BucketFileHeader {
    pub street: Street,
    pub bucket_count: u16,
    pub board_count: u32,     // number of board contexts (1 for preflop, ~1755 for flop, etc.)
    pub combos_per_board: u16, // 169 for preflop, 1326 for postflop
}

/// In-memory representation of a bucket file
#[derive(Debug)]
pub struct BucketFile {
    pub header: BucketFileHeader,
    /// Flat array: buckets[board_idx * combos_per_board + combo_idx] = bucket_id
    /// u16 to support >256 buckets
    pub buckets: Vec<u16>,
}
```

**Step 2: Implement write**

```rust
impl BucketFile {
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&MAGIC)?;
        writer.write_all(&[VERSION])?;
        writer.write_all(&[self.header.street as u8])?;
        writer.write_all(&self.header.bucket_count.to_le_bytes())?;
        writer.write_all(&self.header.board_count.to_le_bytes())?;
        writer.write_all(&self.header.combos_per_board.to_le_bytes())?;
        // Write bucket data as u16 LE
        for &b in &self.buckets {
            writer.write_all(&b.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }
}
```

**Step 3: Implement read**

```rust
impl BucketFile {
    pub fn read_from<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != VERSION {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported version"));
        }
        let mut street_byte = [0u8; 1];
        reader.read_exact(&mut street_byte)?;
        let street = match street_byte[0] {
            0 => Street::Preflop,
            1 => Street::Flop,
            2 => Street::Turn,
            3 => Street::River,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad street")),
        };
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf2)?;
        let bucket_count = u16::from_le_bytes(buf2);
        reader.read_exact(&mut buf4)?;
        let board_count = u32::from_le_bytes(buf4);
        reader.read_exact(&mut buf2)?;
        let combos_per_board = u16::from_le_bytes(buf2);

        let total = board_count as usize * combos_per_board as usize;
        let mut buckets = Vec::with_capacity(total);
        for _ in 0..total {
            reader.read_exact(&mut buf2)?;
            buckets.push(u16::from_le_bytes(buf2));
        }

        Ok(BucketFile {
            header: BucketFileHeader { street, bucket_count, board_count, combos_per_board },
            buckets,
        })
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    /// Look up the bucket for a (board_index, combo_index) pair
    pub fn get_bucket(&self, board_idx: u32, combo_idx: u16) -> u16 {
        let idx = board_idx as usize * self.header.combos_per_board as usize + combo_idx as usize;
        self.buckets[idx]
    }
}
```

**Step 4: Write round-trip test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_bucket_file_round_trip() {
        let file = BucketFile {
            header: BucketFileHeader {
                street: Street::Flop,
                bucket_count: 200,
                board_count: 3,
                combos_per_board: 1326,
            },
            buckets: (0..3 * 1326).map(|i| (i % 200) as u16).collect(),
        };
        let mut buf = Vec::new();
        file.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded = BucketFile::read_from(&mut cursor).unwrap();
        assert_eq!(loaded.header.street, Street::Flop);
        assert_eq!(loaded.header.bucket_count, 200);
        assert_eq!(loaded.header.board_count, 3);
        assert_eq!(loaded.buckets.len(), 3 * 1326);
        assert_eq!(loaded.get_bucket(1, 50), file.get_bucket(1, 50));
    }

    #[test]
    fn test_bucket_file_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.buckets");
        let file = BucketFile {
            header: BucketFileHeader {
                street: Street::Preflop,
                bucket_count: 169,
                board_count: 1,
                combos_per_board: 169,
            },
            buckets: (0..169).map(|i| i as u16).collect(),
        };
        file.save(&path).unwrap();
        let loaded = BucketFile::load(&path).unwrap();
        assert_eq!(loaded.header.bucket_count, 169);
        assert_eq!(loaded.buckets, file.buckets);
    }
}
```

**Step 5: Run tests, commit**

```bash
cargo test -p poker-solver-core blueprint_v2::bucket_file -- --nocapture
git add crates/core/src/blueprint_v2/
git commit -m "feat(blueprint_v2): bucket file binary format with read/write"
```

---

## Phase 2: Clustering Pipeline

### Task 3: EMD Distance & K-Means with EMD

Core math: Earth Mover's Distance on ordered histograms, and k-means using EMD.

**Files:**
- Create: `crates/core/src/blueprint_v2/clustering.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Test: inline

**Step 1: Implement EMD for 1D ordered histograms**

```rust
/// Earth Mover's Distance between two probability distributions over ordered buckets.
/// Equivalent to L1 of CDFs. Both `p` and `q` must sum to ~1.0.
/// O(K) where K = len.
pub fn emd(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let mut cdf_diff = 0.0_f64;
    let mut total = 0.0_f64;
    for i in 0..p.len() {
        cdf_diff += p[i] - q[i];
        total += cdf_diff.abs();
    }
    total
}
```

**Step 2: Implement k-means with EMD**

```rust
/// Run k-means clustering on feature vectors using EMD as distance.
/// `data`: slice of feature vectors (each is a probability distribution).
/// Returns: Vec of cluster assignments (one per data point).
pub fn kmeans_emd(
    data: &[Vec<f64>],
    k: usize,
    max_iterations: u32,
    seed: u64,
) -> Vec<u16> { ... }
```

Implementation details:
- Initialize centroids with k-means++ using EMD distance
- Assignment step: each point gets nearest centroid by EMD
- Update step: component-wise mean of assigned points
- Terminate when assignments don't change or max iterations reached

**Step 3: Test EMD correctness**

```rust
#[test]
fn test_emd_identical() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    assert!((emd(&p, &p) - 0.0).abs() < 1e-10);
}

#[test]
fn test_emd_opposite() {
    let p = vec![1.0, 0.0, 0.0, 0.0];
    let q = vec![0.0, 0.0, 0.0, 1.0];
    // CDF diffs: 1.0, 1.0, 1.0, 0.0 -> EMD = 3.0
    assert!((emd(&p, &q) - 3.0).abs() < 1e-10);
}

#[test]
fn test_emd_adjacent() {
    let p = vec![1.0, 0.0, 0.0, 0.0];
    let q = vec![0.0, 1.0, 0.0, 0.0];
    // EMD should be 1.0 (one step)
    assert!((emd(&p, &q) - 1.0).abs() < 1e-10);
}

#[test]
fn test_kmeans_separable_clusters() {
    // Two clear clusters: all-mass-at-0 and all-mass-at-3
    let mut data = Vec::new();
    for _ in 0..50 { data.push(vec![0.9, 0.1, 0.0, 0.0]); }
    for _ in 0..50 { data.push(vec![0.0, 0.0, 0.1, 0.9]); }
    let assignments = kmeans_emd(&data, 2, 100, 42);
    // All first 50 should be same cluster, all last 50 should be same cluster
    assert!(assignments[0..50].iter().all(|&a| a == assignments[0]));
    assert!(assignments[50..100].iter().all(|&a| a == assignments[50]));
    assert_ne!(assignments[0], assignments[50]);
}
```

**Step 4: Run tests, commit**

```bash
cargo test -p poker-solver-core blueprint_v2::clustering -- --nocapture
git commit -m "feat(blueprint_v2): EMD distance and k-means with EMD clustering"
```

---

### Task 4: River Clustering

Cluster river (hand, board) situations by equity.

**Files:**
- Create: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Test: inline

**Key dependencies:** `rs_poker` for card types, existing `rank_hand` / `compute_equity` from `showdown_equity.rs`, existing `CanonicalBoard` / `SuitMapping` from `abstraction/isomorphism.rs`.

**Implementation:**
- Enumerate all possible (hole_pair, full_board) combinations for each canonical flop+turn+river
- Compute equity for each combo against a uniform opponent range
- K-means (1D, which degrades to percentile bucketing for scalar features) into K_river buckets
- Save as `BucketFile` with `Street::River`

**Step 1: Write the river clustering function**

```rust
/// Cluster all river information situations.
/// Returns a BucketFile mapping (board_index, combo_index) -> bucket_id.
pub fn cluster_river(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: &dyn Fn(f64),  // 0.0..1.0 progress callback
) -> BucketFile { ... }
```

The function must:
1. Enumerate all canonical (flop, turn, river) board combinations
2. For each board, compute equity for all 1326 hole card combos (excluding dead cards)
3. Collect all equity values as 1D feature vectors
4. Run k-means (or percentile bucketing since 1D)
5. Assign bucket IDs back to the (board, combo) pairs

**Note:** For 1D features (river equity), k-means degrades to sorted percentile boundaries. Optimize by sorting equity values and dividing into equal-mass buckets, then refine with k-means.

**Step 2: Test with a small set of boards**

```rust
#[test]
fn test_river_clustering_small() {
    let result = cluster_river(10, 50, 42, &|_| {});
    assert!(result.header.bucket_count == 10);
    assert!(result.header.street == Street::River);
    // Every bucket should be in range [0, 10)
    for &b in &result.buckets {
        assert!(b < 10);
    }
}
```

**Step 3: Run tests, commit**

```bash
cargo test -p poker-solver-core blueprint_v2::cluster_pipeline::test_river -- --nocapture
git commit -m "feat(blueprint_v2): river clustering by equity"
```

---

### Task 5: Turn Clustering (Distribution over River Buckets)

For each (hand, flop+turn), enumerate river cards, build histogram over river buckets, cluster with EMD.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Implementation:**

```rust
/// Cluster turn information situations using potential-aware features.
/// Requires river bucket assignments (built in previous step).
pub fn cluster_turn(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: &dyn Fn(f64),
) -> BucketFile { ... }
```

For each (hand, flop, turn):
1. Enumerate ~46 possible river cards
2. For each river card, look up this hand's river bucket
3. Build a K_river-dimensional histogram (probability distribution)
4. Collect all histograms
5. K-means with EMD into K_turn buckets

Parallelized per (flop, turn) board using rayon.

**Test:** Verify that the output bucket file has correct dimensions and all bucket IDs are in range.

**Commit:** `feat(blueprint_v2): turn clustering with potential-aware EMD`

---

### Task 6: Flop Clustering (Distribution over Turn Buckets)

Same pattern as turn, but enumerate ~1,081 (turn, river) completions per (hand, flop).

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

```rust
pub fn cluster_flop(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: &dyn Fn(f64),
) -> BucketFile { ... }
```

This is the most compute-intensive step. For each of ~1,755 canonical flops and ~1,326 combos per flop, enumerate ~1,081 (turn, river) pairs and build histogram over turn buckets. Use rayon parallelism per flop.

**Commit:** `feat(blueprint_v2): flop clustering with potential-aware EMD`

---

### Task 7: Preflop Clustering (Distribution over Flop Buckets)

For each of 169 canonical hands, sample/enumerate flops, build histogram over flop buckets.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

```rust
pub fn cluster_preflop(
    flop_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
) -> BucketFile { ... }
```

For HU with 169 buckets this is a lossless identity mapping. But the code should still go through the clustering path so it generalizes to multi-way (fewer buckets than unique hands).

**Commit:** `feat(blueprint_v2): preflop clustering with potential-aware EMD`

---

### Task 8: Full Clustering Pipeline & `cluster` CLI Command

Orchestrate the bottom-up pipeline and add the CLI subcommand.

**Files:**
- Create: `crates/core/src/blueprint_v2/cluster_pipeline.rs` (orchestrator function)
- Modify: `crates/trainer/src/main.rs` (add `Cluster` subcommand)

**Step 1: Orchestrator function**

```rust
/// Run the full bottom-up clustering pipeline.
/// Saves all bucket files to `output_dir`.
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Cluster river
    let river = cluster_river(config.river.buckets, config.kmeans_iterations, config.seed, ...);
    river.save(&output_dir.join("river.buckets"))?;
    // 2. Cluster turn (uses river)
    let turn = cluster_turn(&river, config.turn.buckets, config.kmeans_iterations, config.seed, ...);
    turn.save(&output_dir.join("turn.buckets"))?;
    // 3. Cluster flop (uses turn)
    let flop = cluster_flop(&turn, config.flop.buckets, config.kmeans_iterations, config.seed, ...);
    flop.save(&output_dir.join("flop.buckets"))?;
    // 4. Cluster preflop (uses flop)
    let preflop = cluster_preflop(&flop, config.preflop.buckets, config.kmeans_iterations, config.seed);
    preflop.save(&output_dir.join("preflop.buckets"))?;
    // 5. Save config + metadata
    save_cluster_metadata(config, output_dir)?;
    Ok(())
}
```

**Step 2: Add CLI subcommand**

In `crates/trainer/src/main.rs`, add to the `Commands` enum:

```rust
/// Compute potential-aware bucket assignments for blueprint V2
Cluster {
    #[arg(short, long)]
    config: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
},
```

Handler: parse `ClusteringConfig` from YAML, call `run_clustering_pipeline`.

**Step 3: Integration test with toy config**

```bash
cargo run -p poker-solver-trainer --release -- cluster \
  -c sample_configurations/blueprint_v2_toy.yaml \
  -o clusters/test/
```

Verify: all 4 `.buckets` files created, `metadata.json` present.

**Step 4: Commit**

```bash
git commit -m "feat(blueprint_v2): full clustering pipeline with CLI command"
```

---

### Task 9: Cluster Diagnostics (`diag-clusters`)

**Files:**
- Create: `crates/core/src/blueprint_v2/cluster_diagnostics.rs`
- Modify: `crates/trainer/src/main.rs` (add `DiagClusters` subcommand)

**Diagnostics to implement:**
1. **Intra-bucket equity variance** — for each bucket on each street, compute the variance of equity values within the bucket. Report mean and max across all buckets.
2. **Bucket size distribution** — count of hands per bucket. Report min/max/mean/std.
3. **EMD between adjacent centroids** — are buckets well-separated?
4. **Sample hands** — for a given (street, board, bucket_id), list representative hands.
5. **Cross-street transitions** — for a given flop, show a matrix of how flop buckets distribute across turn buckets.

**CLI:**
```rust
DiagClusters {
    #[arg(short, long)]
    cluster_dir: PathBuf,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    board: Option<String>,   // optional: focus on a specific board
    #[arg(long)]
    street: Option<String>,  // optional: focus on a specific street
},
```

**Commit:** `feat(blueprint_v2): cluster diagnostics command`

---

## Phase 3: Full-Game Tree

### Task 10: Game Tree Builder

Build a single arena-allocated game tree spanning all 4 streets with configurable action abstraction.

**Files:**
- Create: `crates/core/src/blueprint_v2/game_tree.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Test: inline

**Node types:**

```rust
#[derive(Debug, Clone)]
pub enum GameNode {
    Decision {
        player: u8,            // 0 or 1 for HU
        street: Street,
        actions: Vec<Action>,  // available actions at this node
        children: Vec<u32>,    // arena indices of child nodes (1:1 with actions)
    },
    Chance {
        street: Street,        // the NEW street being dealt
        child: u32,            // single child (street transition)
    },
    Terminal {
        kind: TerminalKind,
        pot: f64,              // total pot in BB
        invested: [f64; 2],    // amount each player has put in (for HU)
    },
}

#[derive(Debug, Clone, Copy)]
pub enum TerminalKind {
    Fold { winner: u8 },
    Showdown,
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(f64),      // amount in BB
    Raise(f64),    // raise TO amount in BB
    AllIn,
}
```

**Tree builder:**

```rust
pub struct GameTree {
    pub nodes: Vec<GameNode>,
    pub root: u32,
    /// For each decision node: (node_idx, num_actions) -> offset into flat regret/strategy buffers
    pub info_set_offsets: Vec<(u32, u32)>,  // (start_offset, num_actions)
}

impl GameTree {
    pub fn build(
        game: &GameConfig,
        action_abstraction: &ActionAbstractionConfig,
    ) -> Self { ... }
}
```

The builder recursively constructs the tree:
- **Preflop:** Parse preflop sizes ("2.5bb", "3.0x") into concrete amounts. Insert fold/check/call + configured sizes + all-in at each depth.
- **Street transitions:** After a betting round completes, insert a Chance node and start the next street.
- **Postflop:** Parse pot-fraction sizes into concrete amounts based on current pot. Insert fold/check/call + sizes + all-in.
- **Max raises:** Enforce `max_raises` cap per street.
- **Terminal detection:** Fold → Terminal(Fold), all players acted and pots equal → if river done: Terminal(Showdown), else Chance node to next street.

**Tests:**

```rust
#[test]
fn test_simple_tree_structure() {
    let game = GameConfig { players: 2, stack_depth: 10.0, small_blind: 0.5, big_blind: 1.0 };
    let aa = ActionAbstractionConfig {
        preflop: vec![vec!["2.5bb".into()]],
        flop: vec![vec![1.0]],
        turn: vec![vec![1.0]],
        river: vec![vec![1.0]],
        max_raises: 1,
    };
    let tree = GameTree::build(&game, &aa);
    // Root should be a Decision node (SB acts first preflop)
    assert!(matches!(tree.nodes[tree.root as usize], GameNode::Decision { .. }));
    // Tree should have terminal nodes
    let terminals = tree.nodes.iter().filter(|n| matches!(n, GameNode::Terminal { .. })).count();
    assert!(terminals > 0);
}

#[test]
fn test_all_in_always_available() {
    let game = GameConfig { players: 2, stack_depth: 10.0, small_blind: 0.5, big_blind: 1.0 };
    let aa = ActionAbstractionConfig {
        preflop: vec![vec!["2.5bb".into()]],
        flop: vec![vec![0.5]],
        turn: vec![vec![0.5]],
        river: vec![vec![0.5]],
        max_raises: 1,
    };
    let tree = GameTree::build(&game, &aa);
    // Every Decision node should have AllIn as an action
    for node in &tree.nodes {
        if let GameNode::Decision { actions, .. } = node {
            assert!(actions.iter().any(|a| matches!(a, Action::AllIn)),
                "Decision node missing AllIn action");
        }
    }
}
```

**Commit:** `feat(blueprint_v2): full-game tree builder with configurable action abstraction`

---

## Phase 4: MCCFR Training Engine

### Task 11: Strategy & Regret Storage

Flat-buffer storage indexed by (node, bucket, action).

**Files:**
- Create: `crates/core/src/blueprint_v2/storage.rs`
- Test: inline

```rust
/// Storage for regrets and strategy sums in the full-game solver.
/// Flat buffer indexed by info_set_offset + action_index.
/// Each info set has `bucket_count * num_actions` entries.
pub struct BlueprintStorage {
    /// Regrets: i32 per (info_set, bucket, action)
    pub regrets: Vec<i32>,
    /// Strategy sums: i64 per (info_set, bucket, action)
    pub strategy_sums: Vec<i64>,
    /// Bucket counts per street
    pub bucket_counts: [u16; 4],
    /// Layout: for each decision node, (offset, num_actions)
    pub layout: Vec<(usize, u16)>,
}
```

Methods:
- `new(tree: &GameTree, bucket_counts: [u16; 4]) -> Self`
- `get_regrets(node_idx: u32, bucket: u16) -> &[i32]` (slice of `num_actions` regrets)
- `get_regrets_mut(node_idx: u32, bucket: u16) -> &mut [i32]`
- `current_strategy(node_idx: u32, bucket: u16) -> Vec<f64>` (regret matching)
- `get_strategy_sums(node_idx: u32, bucket: u16) -> &[i64]`
- `average_strategy(node_idx: u32, bucket: u16) -> Vec<f64>` (normalized strategy sums)
- `save(path: &Path)` / `load(path: &Path)` (bincode serialization)

Regret floor: clamp at `prune_threshold` (default -310M) on update.

**Lazy allocation note:** For MVP, pre-allocate the full buffer. Lazy allocation (Pluribus-style, only allocate when encountered) is a future optimization — for HU the total size is manageable.

**Commit:** `feat(blueprint_v2): strategy and regret storage with flat buffers`

---

### Task 12: External-Sampling MCCFR Traversal

The core training loop.

**Files:**
- Create: `crates/core/src/blueprint_v2/mccfr.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Test: inline

**Key function:**

```rust
/// Run one external-sampling MCCFR iteration for the given traverser.
/// Returns the expected value for the traverser at the root.
pub fn traverse_external(
    tree: &GameTree,
    storage: &mut BlueprintStorage,
    buckets: &AllBuckets,        // bucket lookups for all streets
    deal: &Deal,                 // sampled hole cards + board
    traverser: u8,               // which player is the traverser
    node_idx: u32,
    prune: bool,                 // whether to apply negative-regret pruning
    prune_threshold: i32,
) -> f64 { ... }
```

**Deal struct:**

```rust
pub struct Deal {
    pub hole_cards: [[Card; 2]; 2],  // per player
    pub board: [Card; 5],             // flop(3) + turn(1) + river(1)
}
```

**AllBuckets:** Wrapper that holds all 4 loaded `BucketFile`s and provides `get_bucket(street, board, combo) -> u16`.

**Traversal logic:**
- **Terminal(Fold):** Return ±pot based on who folded
- **Terminal(Showdown):** Use `rank_hand` to compare actual hands, return payoff
- **Chance:** Board cards already sampled — just recurse to child
- **Decision (traverser's node):** Compute regret-matched strategy, traverse ALL actions, update regrets
- **Decision (opponent's node):** Compute strategy, SAMPLE one action according to strategy, recurse

**Negative-regret pruning (traverser's node only):** If `prune` is true, skip actions with regret < `prune_threshold` (except on river and at terminal-adjacent actions). In 5% of iterations, disable pruning.

**Tests:**

```rust
#[test]
fn test_traverse_fold_terminal() {
    // Construct minimal tree: SB folds -> BB wins
    // Verify correct payoff returned
}

#[test]
fn test_traverse_showdown_terminal() {
    // Construct minimal tree: both check to showdown
    // Verify winner gets correct payoff
}
```

**Commit:** `feat(blueprint_v2): external-sampling MCCFR traversal`

---

### Task 13: Training Loop with LCFR & Snapshots

The outer loop that drives iterations, applies LCFR weighting, and saves snapshots.

**Files:**
- Create: `crates/core/src/blueprint_v2/trainer.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Test: inline

```rust
pub struct BlueprintTrainer {
    pub tree: GameTree,
    pub storage: BlueprintStorage,
    pub buckets: AllBuckets,
    pub config: BlueprintV2Config,
    pub rng: StdRng,
    pub start_time: Instant,
    pub iterations: u64,
}

impl BlueprintTrainer {
    pub fn new(config: BlueprintV2Config) -> Result<Self, Box<dyn std::error::Error>> { ... }

    /// Run training until time limit or iteration limit.
    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            // 1. Check stopping criteria
            if self.should_stop() { break; }
            // 2. Sample a deal
            let deal = self.sample_deal();
            // 3. Traverse as P1, then P2
            let prune = self.should_prune();
            traverse_external(&self.tree, &mut self.storage, &self.buckets, &deal, 0, self.tree.root, prune, self.config.training.prune_threshold);
            traverse_external(&self.tree, &mut self.storage, &self.buckets, &deal, 1, self.tree.root, prune, self.config.training.prune_threshold);
            self.iterations += 1;
            // 4. LCFR discounting (time-based)
            if self.should_discount() {
                self.apply_lcfr_discount();
            }
            // 5. Convergence metrics (time-based)
            if self.should_print() {
                self.print_metrics();
            }
            // 6. Snapshot (time-based)
            if self.should_snapshot() {
                self.save_snapshot()?;
            }
        }
        // Save final strategy
        self.save_final()?;
        Ok(())
    }
}
```

**LCFR discount:**
```rust
fn apply_lcfr_discount(&mut self) {
    let elapsed = self.start_time.elapsed().as_secs() / 60;
    let t = elapsed / self.config.training.lcfr_discount_interval;
    let d = t as f64 / (t as f64 + 1.0);
    for r in self.storage.regrets.iter_mut() {
        *r = (*r as f64 * d) as i32;
    }
    for s in self.storage.strategy_sums.iter_mut() {
        *s = (*s as f64 * d) as i64;
    }
}
```

**Snapshot saving:** Calls into bundle format (Task 14).

**Convergence metrics:**
- Strategy L1 delta: compare current regret-matched strategy to previous snapshot
- Mean positive regret across all info sets

**Commit:** `feat(blueprint_v2): training loop with LCFR weighting and time-based snapshots`

---

## Phase 5: Bundle Format & Explorer Integration

### Task 14: Snapshot Bundle Format

Save/load snapshots in a format the explorer can consume.

**Files:**
- Create: `crates/core/src/blueprint_v2/bundle.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Test: inline

**Bundle structure on disk:**
```
training_run/
  config.yaml                    # full BlueprintV2Config
  clusters/                      # full copy of cluster files
    config.yaml
    preflop.buckets
    flop.buckets
    turn.buckets
    river.buckets
    metadata.json
  snapshot_NNNN/                 # time-tagged snapshots
    strategy.bin                 # bincode: Vec<Vec<f32>> indexed by (info_set, bucket, action)
    metadata.json                # { iteration, elapsed_minutes, metrics }
    regrets.bin                  # bincode: raw storage (for resume)
  final/
    strategy.bin
    metadata.json
    regrets.bin
```

**Key types:**

```rust
#[derive(Serialize, Deserialize)]
pub struct BlueprintV2Bundle {
    pub config: BlueprintV2Config,
    pub strategy: BlueprintV2Strategy,
    pub clusters: AllBuckets,
}

#[derive(Serialize, Deserialize)]
pub struct BlueprintV2Strategy {
    /// For each decision node in the tree: Vec of (bucket_count * num_actions) f32 probabilities
    pub strategies: Vec<Vec<f32>>,
    pub iterations: u64,
    pub elapsed_minutes: u64,
}
```

Methods:
- `BlueprintV2Bundle::save(path: &Path)` — write config.yaml, copy clusters, save strategy.bin
- `BlueprintV2Bundle::load(path: &Path) -> Self` — load all components
- `BlueprintV2Strategy::from_storage(storage: &BlueprintStorage, tree: &GameTree)` — extract average strategy

**Commit:** `feat(blueprint_v2): snapshot bundle format with save/load`

---

### Task 15: Explorer Integration

Add `load_blueprint_v2` command to the explorer.

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` (add variant to `StrategySource`, add load command)
- Modify: `crates/devserver/src/main.rs` (add HTTP endpoint)
- Modify: `frontend/src/types.ts` (add type)
- Modify: `frontend/src/Explorer.tsx` (add load option in menu)

**Strategy resolution for 13x13 matrix:**

When the explorer calls `get_strategy_matrix` with a BlueprintV2 source:
1. **Preflop:** For each of 169 canonical hands, look up bucket from `preflop.buckets`, get strategy at current tree node for that bucket. Map to 13x13 cell.
2. **Postflop:** User has entered board cards. For each of 1326 hole card combos:
   - Determine which canonical board this maps to
   - Look up `street.buckets[board_index][combo_index]` → bucket ID
   - Get that bucket's strategy at the current tree node
   - Average across combos within each 13x13 cell

**New StrategySource variant:**
```rust
BlueprintV2 {
    config: BlueprintV2Config,
    strategy: BlueprintV2Strategy,
    clusters: AllBuckets,
    tree: GameTree,
}
```

**New command:**
```rust
#[tauri::command]
pub async fn load_blueprint_v2(path: String, state: State<'_, ExplorerState>) -> Result<String, String> {
    let bundle = BlueprintV2Bundle::load(Path::new(&path)).map_err(|e| e.to_string())?;
    let tree = GameTree::build(&bundle.config.game, &bundle.config.action_abstraction);
    let mut source = state.source.lock().unwrap();
    *source = Some(StrategySource::BlueprintV2 {
        config: bundle.config,
        strategy: bundle.strategy,
        clusters: bundle.clusters,
        tree,
    });
    Ok("Blueprint V2 loaded".to_string())
}
```

**Commit:** `feat(blueprint_v2): explorer integration with 13x13 matrix display`

---

## Phase 6: CLI & Sample Configs

### Task 16: `train-blueprint` CLI Command

**Files:**
- Modify: `crates/trainer/src/main.rs` (add `TrainBlueprint` subcommand)

```rust
/// Train a full-game blueprint strategy using MCCFR
TrainBlueprint {
    #[arg(short, long)]
    config: PathBuf,
},
```

Handler:
1. Parse `BlueprintV2Config` from YAML
2. Construct `BlueprintTrainer`
3. Call `trainer.train()`
4. Print summary (iterations, elapsed time, final metrics)

**Commit:** `feat(blueprint_v2): train-blueprint CLI command`

---

### Task 17: Sample Configs

**Files:**
- Create: `sample_configurations/blueprint_v2_toy.yaml`
- Create: `sample_configurations/blueprint_v2_realistic.yaml`

Toy config (from design doc — 10BB, 50 buckets, trains in minutes).
Realistic config (from design doc — 100BB, 200 buckets, multi-hour training).

**Commit:** `feat(blueprint_v2): sample configs for toy and realistic training`

---

## Phase 7: End-to-End Integration Test

### Task 18: E2E Test

Validate the full pipeline: cluster → train → snapshot → load in explorer.

**Files:**
- Create: `crates/core/tests/blueprint_v2_e2e.rs`

```rust
#[test]
fn test_blueprint_v2_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let cluster_dir = dir.path().join("clusters");
    let training_dir = dir.path().join("training");

    // 1. Run clustering with tiny config
    let config = BlueprintV2Config { /* tiny: 5 buckets, 2 bet sizes, 5BB */ };
    run_clustering_pipeline(&config.clustering, &cluster_dir).unwrap();

    // 2. Verify cluster files exist
    assert!(cluster_dir.join("river.buckets").exists());
    assert!(cluster_dir.join("preflop.buckets").exists());

    // 3. Train for a few iterations
    let mut trainer = BlueprintTrainer::new(config).unwrap();
    // Override to iteration-based for test
    for _ in 0..100 {
        let deal = trainer.sample_deal();
        traverse_external(...);
    }

    // 4. Save snapshot
    trainer.save_snapshot().unwrap();

    // 5. Load bundle
    let bundle = BlueprintV2Bundle::load(&training_dir).unwrap();
    assert!(bundle.strategy.iterations > 0);

    // 6. Verify strategy is valid (probabilities sum to ~1.0)
    // for each decision node, for each bucket, action probs should sum to ~1.0
}
```

**Commit:** `test(blueprint_v2): end-to-end integration test`

---

## Agent Team & Execution Order

| Task | Agent | Parallel Group | Dependencies |
|-|-|-|-|
| 1: Config types | rust-developer | A | none |
| 2: Bucket file I/O | rust-developer | A | none |
| 3: EMD & k-means | rust-developer | A | none |
| 4: River clustering | rust-developer | B | 2, 3 |
| 5: Turn clustering | rust-developer | C | 4 |
| 6: Flop clustering | rust-developer | D | 5 |
| 7: Preflop clustering | rust-developer | E | 6 |
| 8: Cluster CLI + pipeline | rust-developer | F | 1, 7 |
| 9: Cluster diagnostics | rust-developer | F | 8 |
| 10: Game tree builder | rust-developer | A | 1 |
| 11: Storage | rust-developer | B | 10 |
| 12: MCCFR traversal | rust-developer | C | 11, 2 |
| 13: Training loop | rust-developer | D | 12 |
| 14: Bundle format | rust-developer | E | 13 |
| 15: Explorer integration | rust-developer | F | 14 |
| 16: train-blueprint CLI | rust-developer | F | 13 |
| 17: Sample configs | rust-developer | F | 16 |
| 18: E2E test | rust-developer | G | all |

**Two parallel tracks:**
- **Track A (clustering):** Tasks 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
- **Track B (solver):** Tasks 1 → 10 → 11 → 12 → 13 → 14 → 15, 16, 17

Tasks 1, 2, 3, 10 can all start in parallel (Group A).

**Review gates after:**
- Phase 2 complete (clustering pipeline works)
- Phase 4 complete (MCCFR trains and converges)
- Phase 6 complete (explorer loads and displays)

Each review gate: `software-architect` + `idiomatic-rust-enforcer` + `rust-perf-reviewer`
