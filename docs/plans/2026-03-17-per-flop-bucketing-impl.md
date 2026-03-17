# Per-Flop Bucketing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace global clustering with Pluribus-style per-flop bucketing — each canonical flop gets independent turn/river bucket assignments — and add a validation framework comparing blueprint strategy against exact range-solver solutions.

**Architecture:** Bottom-up clustering scoped per-flop: for each of 1,755 canonical flops, cluster river combos by equity, then cluster turn combos by histogram over river buckets. Finally, cluster flops globally by histogram over per-flop turn buckets. MCCFR uses the same shared regret tables (200 buckets), just with better bucket assignments.

**Tech Stack:** Rust (rayon for parallelism, serde for config, existing k-means/EMD clustering)

**Design doc:** `docs/plans/2026-03-17-per-flop-bucketing-design.md`

---

### Task 1: Per-Flop Bucket File Format

Create the `PerFlopBucketFile` struct — a self-contained file holding turn + river bucket assignments for one canonical flop.

**Files:**
- Create: `crates/core/src/blueprint_v2/per_flop_bucket_file.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs` (add module)

**Step 1: Write tests for round-trip serialization**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::core::{Card, Suit, Value};

    fn make_test_flop() -> [Card; 3] {
        [
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Heart),
            Card::new(Value::Two, Suit::Diamond),
        ]
    }

    #[test]
    fn round_trip_write_read() {
        let flop = make_test_flop();
        let turn_cards = vec![
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Club),
        ];
        // 2 turn cards, each with 3 river cards
        let river_cards_per_turn = vec![
            vec![
                Card::new(Value::Ten, Suit::Club),
                Card::new(Value::Nine, Suit::Club),
                Card::new(Value::Eight, Suit::Club),
            ],
            vec![
                Card::new(Value::Seven, Suit::Club),
                Card::new(Value::Six, Suit::Club),
                Card::new(Value::Five, Suit::Club),
            ],
        ];

        let pf = PerFlopBucketFile {
            flop_cards: flop,
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: turn_cards.clone(),
            turn_buckets: vec![0u16; 2 * 1326],        // 2 turn cards × 1326 combos
            river_cards_per_turn: river_cards_per_turn.clone(),
            river_buckets_per_turn: vec![
                vec![0u16; 3 * 1326],   // 3 rivers × 1326 combos
                vec![0u16; 3 * 1326],
            ],
        };

        let mut buf = Vec::new();
        pf.write_to(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = PerFlopBucketFile::read_from(&mut cursor).unwrap();

        assert_eq!(loaded.flop_cards, flop);
        assert_eq!(loaded.turn_bucket_count, 10);
        assert_eq!(loaded.river_bucket_count, 10);
        assert_eq!(loaded.turn_cards, turn_cards);
        assert_eq!(loaded.turn_buckets.len(), 2 * 1326);
        assert_eq!(loaded.river_cards_per_turn.len(), 2);
        assert_eq!(loaded.river_buckets_per_turn[0].len(), 3 * 1326);
    }

    #[test]
    fn get_turn_bucket_lookup() {
        let mut pf = PerFlopBucketFile {
            flop_cards: make_test_flop(),
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![Card::new(Value::Ace, Suit::Club)],
            turn_buckets: vec![0u16; 1326],
            river_cards_per_turn: vec![vec![]],
            river_buckets_per_turn: vec![vec![]],
        };
        pf.turn_buckets[42] = 7;
        assert_eq!(pf.get_turn_bucket(0, 42), 7);
    }

    #[test]
    fn get_river_bucket_lookup() {
        let river_card = Card::new(Value::Ten, Suit::Club);
        let mut pf = PerFlopBucketFile {
            flop_cards: make_test_flop(),
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![Card::new(Value::Ace, Suit::Club)],
            turn_buckets: vec![0u16; 1326],
            river_cards_per_turn: vec![vec![river_card]],
            river_buckets_per_turn: vec![vec![0u16; 1326]],
        };
        pf.river_buckets_per_turn[0][99] = 5;
        assert_eq!(pf.get_river_bucket(0, 0, 99), 5);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core per_flop_bucket_file`
Expected: compilation error (module doesn't exist)

**Step 3: Implement `PerFlopBucketFile`**

```rust
//! Per-flop bucket file: turn + river bucket assignments for one canonical flop.
//!
//! ## Wire format (little-endian)
//!
//! | Field | Size |
//! |-|-|
//! | Magic `PFB1` | 4 |
//! | flop_cards (3 × encoded card) | 3 |
//! | turn_bucket_count | 2 |
//! | river_bucket_count | 2 |
//! | turn_count (u8) | 1 |
//! | turn_cards (turn_count × encoded card) | turn_count |
//! | turn_buckets (turn_count × 1326 × u16) | turn_count × 1326 × 2 |
//! | For each turn card: |  |
//! |   river_count (u8) | 1 |
//! |   river_cards (river_count × encoded card) | river_count |
//! |   river_buckets (river_count × 1326 × u16) | river_count × 1326 × 2 |

use std::io::{self, Read, Write};
use rs_poker::core::Card;
use super::bucket_file::{encode_card, decode_card};  // make these pub(crate)

const MAGIC: [u8; 4] = *b"PFB1";
const COMBOS: usize = 1326;

#[derive(Debug)]
pub struct PerFlopBucketFile {
    pub flop_cards: [Card; 3],
    pub turn_bucket_count: u16,
    pub river_bucket_count: u16,
    pub turn_cards: Vec<Card>,
    /// Flat: turn_buckets[turn_idx * COMBOS + combo_idx]
    pub turn_buckets: Vec<u16>,
    /// Per-turn river cards: river_cards_per_turn[turn_idx] = vec of river cards
    pub river_cards_per_turn: Vec<Vec<Card>>,
    /// Per-turn river buckets: river_buckets_per_turn[turn_idx][river_idx * COMBOS + combo_idx]
    pub river_buckets_per_turn: Vec<Vec<u16>>,
}

impl PerFlopBucketFile {
    pub fn get_turn_bucket(&self, turn_idx: usize, combo_idx: usize) -> u16 {
        self.turn_buckets[turn_idx * COMBOS + combo_idx]
    }

    pub fn get_river_bucket(&self, turn_idx: usize, river_idx: usize, combo_idx: usize) -> u16 {
        self.river_buckets_per_turn[turn_idx][river_idx * COMBOS + combo_idx]
    }

    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&MAGIC)?;
        for &c in &self.flop_cards {
            w.write_all(&[encode_card(c)])?;
        }
        w.write_all(&self.turn_bucket_count.to_le_bytes())?;
        w.write_all(&self.river_bucket_count.to_le_bytes())?;
        w.write_all(&[self.turn_cards.len() as u8])?;
        for &c in &self.turn_cards {
            w.write_all(&[encode_card(c)])?;
        }
        for &b in &self.turn_buckets {
            w.write_all(&b.to_le_bytes())?;
        }
        for (turn_idx, river_cards) in self.river_cards_per_turn.iter().enumerate() {
            w.write_all(&[river_cards.len() as u8])?;
            for &c in river_cards {
                w.write_all(&[encode_card(c)])?;
            }
            for &b in &self.river_buckets_per_turn[turn_idx] {
                w.write_all(&b.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut card_buf = [0u8; 1];
        let mut flop_cards = [Card::default(); 3];
        for c in &mut flop_cards {
            r.read_exact(&mut card_buf)?;
            *c = decode_card(card_buf[0]);
        }
        let mut u16_buf = [0u8; 2];
        r.read_exact(&mut u16_buf)?;
        let turn_bucket_count = u16::from_le_bytes(u16_buf);
        r.read_exact(&mut u16_buf)?;
        let river_bucket_count = u16::from_le_bytes(u16_buf);
        let mut u8_buf = [0u8; 1];
        r.read_exact(&mut u8_buf)?;
        let turn_count = u8_buf[0] as usize;
        let mut turn_cards = Vec::with_capacity(turn_count);
        for _ in 0..turn_count {
            r.read_exact(&mut card_buf)?;
            turn_cards.push(decode_card(card_buf[0]));
        }
        let mut turn_buckets = vec![0u16; turn_count * COMBOS];
        for b in &mut turn_buckets {
            r.read_exact(&mut u16_buf)?;
            *b = u16::from_le_bytes(u16_buf);
        }
        let mut river_cards_per_turn = Vec::with_capacity(turn_count);
        let mut river_buckets_per_turn = Vec::with_capacity(turn_count);
        for _ in 0..turn_count {
            r.read_exact(&mut u8_buf)?;
            let river_count = u8_buf[0] as usize;
            let mut river_cards = Vec::with_capacity(river_count);
            for _ in 0..river_count {
                r.read_exact(&mut card_buf)?;
                river_cards.push(decode_card(card_buf[0]));
            }
            let mut river_buckets = vec![0u16; river_count * COMBOS];
            for b in &mut river_buckets {
                r.read_exact(&mut u16_buf)?;
                *b = u16::from_le_bytes(u16_buf);
            }
            river_cards_per_turn.push(river_cards);
            river_buckets_per_turn.push(river_buckets);
        }
        Ok(Self {
            flop_cards,
            turn_bucket_count,
            river_bucket_count,
            turn_cards,
            turn_buckets,
            river_cards_per_turn,
            river_buckets_per_turn,
        })
    }

    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(file);
        self.write_to(&mut w)
    }

    pub fn load(path: &std::path::Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::new(file);
        Self::read_from(&mut r)
    }
}
```

Note: `encode_card` and `decode_card` in `bucket_file.rs` are currently private. Change them to `pub(crate)`.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core per_flop_bucket_file`
Expected: all 3 tests pass

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/per_flop_bucket_file.rs crates/core/src/blueprint_v2/mod.rs crates/core/src/blueprint_v2/bucket_file.rs
git commit -m "feat: add PerFlopBucketFile format for per-flop turn/river buckets"
```

---

### Task 2: Per-Flop Clustering Pipeline

Implement the core clustering function that processes a single canonical flop: cluster river by equity, cluster turn by histogram over river buckets.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Write test**

```rust
#[test]
#[ignore] // slow: equity enumeration
fn test_cluster_single_flop() {
    use rs_poker::core::{Card, Suit, Value};
    let flop = [
        Card::new(Value::Queen, Suit::Spade),
        Card::new(Value::Jack, Suit::Heart),
        Card::new(Value::Two, Suit::Diamond),
    ];
    let pf = cluster_single_flop(flop, 10, 10, 50, 42, |_, _| {});
    assert_eq!(pf.flop_cards, flop);
    assert_eq!(pf.turn_bucket_count, 10);
    assert_eq!(pf.river_bucket_count, 10);
    // Should have ~48 turn cards (52 - 3 board - 1 overlap with combos isn't removed at this level)
    assert!(pf.turn_cards.len() >= 40 && pf.turn_cards.len() <= 49);
    // Each turn should have river cards
    for (i, river_cards) in pf.river_cards_per_turn.iter().enumerate() {
        assert!(river_cards.len() >= 40, "turn {} has too few rivers", i);
    }
    // Verify bucket range
    for &b in &pf.turn_buckets {
        assert!(b < 10);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_cluster_single_flop -- --ignored`
Expected: FAIL (function doesn't exist)

**Step 3: Implement `cluster_single_flop`**

This function takes a canonical flop and produces a `PerFlopBucketFile`:

1. Enumerate remaining cards (52 - 3 flop = 49)
2. For each possible turn card:
   a. Enumerate remaining river cards (48 - 1 turn = up to 44 after removing combos that overlap)
   b. For each river card, compute equity for all 1326 combos → 1D k-means into `river_bucket_count` buckets
3. For each combo × each turn card, build histogram over that turn's river buckets
4. EMD k-means on the histograms → `turn_bucket_count` turn buckets

The function signature:
```rust
pub fn cluster_single_flop(
    flop: [Card; 3],
    turn_bucket_count: u16,
    river_bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(&str, f64) + Sync,
) -> PerFlopBucketFile
```

Use existing `compute_board_equities`, `kmeans_1d_weighted`, `kmeans_emd_weighted_u8`, and `build_bucket_histogram_u8` functions where possible. Adapt as needed — the river clustering per-(flop,turn) is a smaller version of the current `cluster_river_exhaustive`.

**Step 4: Run test**

Run: `cargo test -p poker-solver-core test_cluster_single_flop -- --ignored --release`
Expected: PASS (may take 10-30 seconds in release mode)

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: implement cluster_single_flop for per-flop bucketing"
```

---

### Task 3: Full Per-Flop Pipeline Orchestrator

Replace `run_clustering_pipeline` with a new function that processes all 1,755 canonical flops in parallel, then clusters flops globally.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Write test**

```rust
#[test]
#[ignore] // slow: processes multiple flops
fn test_per_flop_pipeline_small() {
    // Test with just 3 canonical flops and small bucket counts
    let output_dir = tempfile::TempDir::new().unwrap();
    let config = PerFlopClusteringConfig {
        flop_buckets: 3,
        turn_buckets: 5,
        river_buckets: 5,
        kmeans_iterations: 20,
        seed: 42,
    };
    run_per_flop_pipeline(
        &config,
        output_dir.path(),
        Some(3), // limit to 3 flops for speed
        |_, _, _| {},
    ).unwrap();
    // Check output files exist
    assert!(output_dir.path().join("flop_0000.buckets").exists());
    assert!(output_dir.path().join("flop.buckets").exists());
    assert!(output_dir.path().join("preflop.buckets").exists());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_per_flop_pipeline_small -- --ignored`
Expected: FAIL

**Step 3: Implement `run_per_flop_pipeline`**

```rust
pub struct PerFlopClusteringConfig {
    pub flop_buckets: u16,
    pub turn_buckets: u16,
    pub river_buckets: u16,
    pub kmeans_iterations: u32,
    pub seed: u64,
}

pub fn run_per_flop_pipeline(
    config: &PerFlopClusteringConfig,
    output_dir: &Path,
    flop_limit: Option<usize>,  // for testing: only process N flops
    progress: impl Fn(&str, &str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Enumerate canonical flops
    let all_flops = enumerate_canonical_flops();
    let flops_to_process = match flop_limit {
        Some(n) => &all_flops[..n.min(all_flops.len())],
        None => &all_flops,
    };
    let num_flops = flops_to_process.len();

    // 2. Process each flop in parallel
    let completed = AtomicUsize::new(0);
    std::fs::create_dir_all(output_dir)?;

    flops_to_process.par_iter().enumerate().try_for_each(|(i, wb)| {
        let pf = cluster_single_flop(
            wb.cards,
            config.turn_buckets,
            config.river_buckets,
            config.kmeans_iterations,
            config.seed.wrapping_add(i as u64),
            |_, _| {},
        );
        let path = output_dir.join(format!("flop_{i:04}.buckets"));
        pf.save(&path)?;
        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        progress("per-flop", "clustering", done as f64 / num_flops as f64);
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    // 3. Flop clustering: build per-combo histograms over per-flop turn buckets
    //    For each flop, for each combo, the histogram is the distribution of
    //    turn bucket assignments across all turn cards on that flop.
    progress("flop", "building histograms", 0.0);
    // Load per-flop files, build histograms, run EMD k-means
    // ... (uses existing kmeans_emd_weighted_u8)
    // Save flop.buckets

    // 4. Preflop: 169 lossless buckets (same as before)
    // Save preflop.buckets

    Ok(())
}
```

The flop clustering step (step 3) loads each `flop_NNNN.buckets` file, extracts the turn bucket assignments, builds a histogram per combo over `turn_buckets` bins, and runs EMD k-means to assign each flop's combos into `flop_buckets` global buckets.

**Step 4: Run test**

Run: `cargo test -p poker-solver-core test_per_flop_pipeline_small -- --ignored --release`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: implement full per-flop clustering pipeline orchestrator"
```

---

### Task 4: Update Config for Per-Flop Clustering

Replace the `ClusteringConfig` with the new per-flop structure. Maintain backwards compatibility with old configs by making the `per_flop` section optional.

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs`

**Step 1: Write test**

```rust
#[test]
fn test_deserialize_per_flop_config() {
    let yaml = r#"
clustering:
  flop:
    buckets: 200
  per_flop:
    turn_buckets: 200
    river_buckets: 200
  preflop:
    buckets: 169
"#;
    // Parse just the clustering section
    let cfg: ClusteringConfig = serde_yaml::from_str(yaml).expect("parse failed");
    assert_eq!(cfg.per_flop.as_ref().unwrap().turn_buckets, 200);
    assert_eq!(cfg.per_flop.as_ref().unwrap().river_buckets, 200);
    assert_eq!(cfg.flop.buckets, 200);
    assert_eq!(cfg.preflop.buckets, 169);
}
```

**Step 2: Implement config changes**

Add to `ClusteringConfig`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerFlopConfig {
    #[serde(default = "default_per_flop_buckets")]
    pub turn_buckets: u16,
    #[serde(default = "default_per_flop_buckets")]
    pub river_buckets: u16,
}

fn default_per_flop_buckets() -> u16 { 200 }
```

Add `per_flop: Option<PerFlopConfig>` field to `ClusteringConfig`. Make `algorithm` field optional with a default (for backwards compat). When `per_flop` is present, the pipeline uses per-flop clustering. When absent, falls back to the old global pipeline.

**Step 3: Update existing config tests**

Ensure old YAML configs still parse (backwards compatibility). Update the `test_deserialize_toy_config` test to work with the optional `algorithm` field.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core config`
Expected: all pass

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/config.rs
git commit -m "feat: add per_flop config section for per-flop clustering"
```

---

### Task 5: Update MCCFR Bucket Lookup for Per-Flop Files

Modify `AllBuckets` to load and use per-flop bucket files instead of global turn/river files.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Step 1: Write test**

```rust
#[test]
fn test_per_flop_bucket_lookup() {
    // Create a PerFlopBucketFile with known assignments
    // Load it into AllBuckets
    // Verify get_bucket returns the correct bucket for a given flop/turn/river/combo
}
```

**Step 2: Implement per-flop lookup**

The key change to `AllBuckets`:
- Instead of `bucket_files: [Option<BucketFile>; 4]` (one per street), add a cache of `PerFlopBucketFile` keyed by canonical flop `PackedBoard`.
- `lookup_bucket` for turn/river: canonicalize the board, extract the flop portion, look up the per-flop file (load from disk and cache if needed), then look up the turn/river bucket within that file.
- Add a `per_flop_dir: Option<PathBuf>` field. When set, per-flop lookup is used. When `None`, falls back to global bucket files (backwards compat).

```rust
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    pub bucket_files: [Option<BucketFile>; 4],
    board_maps: [Option<FxHashMap<PackedBoard, u32>>; 4],
    // New: per-flop bucket cache
    per_flop_dir: Option<PathBuf>,
    per_flop_cache: RwLock<FxHashMap<PackedBoard, Arc<PerFlopBucketFile>>>,
    // Map from canonical flop PackedBoard to flop file index (0..1754)
    flop_index_map: Option<FxHashMap<PackedBoard, u16>>,
}
```

The `precompute_buckets` method changes for turn/river to:
1. Canonicalize `board[..3]` → flop key
2. Load or cache the per-flop file
3. Find the turn card index in `pf.turn_cards`
4. Look up turn bucket: `pf.get_turn_bucket(turn_idx, combo_idx)`
5. Find the river card index in `pf.river_cards_per_turn[turn_idx]`
6. Look up river bucket: `pf.get_river_bucket(turn_idx, river_idx, combo_idx)`

Flop bucket lookup still uses the global `flop.buckets` file (same as before — flop bucketing is global).

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core mccfr`
Expected: all existing + new tests pass

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs
git commit -m "feat: MCCFR bucket lookup supports per-flop bucket files"
```

---

### Task 6: Wire Up CLI `cluster` Command

Update the trainer's `cluster` subcommand to use the per-flop pipeline when `per_flop` config is present.

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Update the cluster command handler**

In the `Commands::Cluster` match arm, check if `config.clustering.per_flop` is `Some`. If so, call `run_per_flop_pipeline`. Otherwise, call the existing `run_clustering_pipeline`.

**Step 2: Verify compilation**

Run: `cargo build -p poker-solver-trainer --release`
Expected: compiles

**Step 3: Create test config**

Create `sample_configurations/per_flop_200bkt.yaml` with the new config format.

**Step 4: Smoke test**

Run: `cargo run -p poker-solver-trainer --release -- cluster -c sample_configurations/per_flop_200bkt.yaml -o ./local_data/clusters_per_flop_test`
Expected: processes flops, writes per-flop files + flop.buckets + preflop.buckets

**Step 5: Commit**

```bash
git add crates/trainer/src/main.rs sample_configurations/per_flop_200bkt.yaml
git commit -m "feat: CLI cluster command supports per-flop pipeline"
```

---

### Task 7: Wire Up MCCFR Training with Per-Flop Buckets

Update the trainer's `train-blueprint` command to detect and load per-flop bucket files.

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs` (where `AllBuckets` is constructed)

**Step 1: Update AllBuckets construction**

When loading from a cluster directory, check if `flop_0000.buckets` exists. If so, construct `AllBuckets` with `per_flop_dir` set. Load the global `flop.buckets` and `preflop.buckets` as before.

**Step 2: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: all pass

**Step 3: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: MCCFR training auto-detects per-flop bucket files"
```

---

### Task 8: Validation Framework — Spot Definitions

Create the canonical spot definitions file and the comparison infrastructure.

**Files:**
- Create: `sample_configurations/validation_spots.yaml`
- Modify: `crates/trainer/src/main.rs` (add `validate-blueprint` subcommand)

**Step 1: Define validation spots**

```yaml
# validation_spots.yaml — canonical spots for blueprint validation
spots:
  - name: "SB open, dry flop cbet"
    board: ["Ks", "7d", "2c"]
    oop_range: "22+,A2s+,K2s+,Q2s+,J8s+,T8s+,98s,A2o+,K9o+,QTo+,JTo"
    ip_range: "22+,A2s+,K2s+,Q2s+,J2s+,T6s+,96s+,86s+,76s,65s,54s,A2o+,K2o+,Q5o+,J7o+,T7o+,97o+,87o"
    pot: 6.0
    effective_stack: 97.0
    # ... more spots
```

**Step 2: Implement `validate-blueprint` subcommand**

CLI signature:
```
validate-blueprint --blueprint <path> --spots <yaml> [--cluster-dir <path>]
```

For each spot:
1. Solve with range-solver (exact DCFR)
2. Load blueprint, look up strategy for same spot
3. Compute strategy L2 distance and EV difference
4. Print comparison report

**Step 3: Commit**

```bash
git add sample_configurations/validation_spots.yaml crates/trainer/src/main.rs
git commit -m "feat: add validate-blueprint subcommand for blueprint validation"
```

---

### Task 9: Run Diagnostics on Per-Flop Clusters

Update the `diag-clusters` command to work with per-flop bucket files. Run the transition consistency audit to verify improvement.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_diagnostics.rs`
- Modify: `crates/trainer/src/main.rs`

**Step 1: Update diagnostics**

Add detection of per-flop files in `diag-clusters`. When `flop_0000.buckets` is found, load a sample of per-flop files and run the equity audit on the river/turn buckets within each flop (should show much tighter equity bands since they're per-flop).

**Step 2: Run diagnostics on per-flop clusters**

```bash
cargo run -p poker-solver-trainer --release -- diag-clusters -d local_data/clusters_per_flop_test --audit --audit-boards 50
```

Compare results against the global `clusters_1k_v2` diagnostics.

**Step 3: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_diagnostics.rs crates/trainer/src/main.rs
git commit -m "feat: cluster diagnostics supports per-flop bucket files"
```

---

### Task 10: End-to-End Validation

Train a blueprint with per-flop buckets and validate against range-solver solutions.

**Files:** None (manual testing)

**Step 1: Run per-flop clustering**

```bash
cargo run -p poker-solver-trainer --release -- cluster \
  -c sample_configurations/per_flop_200bkt.yaml \
  -o ./local_data/clusters_per_flop_200
```

**Step 2: Train a short blueprint**

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint \
  -c sample_configurations/per_flop_200bkt.yaml
```

Run for a short time (e.g., 30 min) to get a rough blueprint.

**Step 3: Run validation**

```bash
cargo run -p poker-solver-trainer --release -- validate-blueprint \
  --blueprint ./local_data/blueprints/per_flop_200bkt/snap/latest \
  --spots sample_configurations/validation_spots.yaml
```

Compare against a blueprint trained with global buckets on the same spots.

**Step 4: Document results and commit**

```bash
git commit -m "docs: per-flop bucketing validation results"
```

---

### Task 11: Update Documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/training.md`

**Step 1: Update architecture.md**

Add section on per-flop bucketing: how it works, file format, how it integrates with MCCFR.

**Step 2: Update training.md**

Document the new config format, the `per_flop` section, and the `validate-blueprint` subcommand.

**Step 3: Commit**

```bash
git add docs/architecture.md docs/training.md
git commit -m "docs: document per-flop bucketing and validation framework"
```
