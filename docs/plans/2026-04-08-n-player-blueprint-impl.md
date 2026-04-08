# N-Player Blueprint Trainer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a clean-room `blueprint_mp` module in `crates/core/` that supports 2–8 player MCCFR blueprint training alongside the existing `blueprint_v2`.

**Architecture:** New module `crates/core/src/blueprint_mp/` with strong domain newtypes (`Seat`, `Chips`, `PlayerSet`, `Bucket`, `Street`), N-player game tree with fold-continuation and side pots, 128-bit info set keys, and Pluribus-style external-sampling MCCFR with per-seat traverser cycling. Shares `abstraction/`, `cfr/`, `hand_eval`, `Card`, and `Action` with existing code.

**Tech Stack:** Rust, serde/serde_yaml for config, rayon for parallel training, rs_poker for hand evaluation, AtomicI32/AtomicI64 for lock-free storage.

**Reference implementation:** `crates/core/src/blueprint_v2/` — consult for patterns but do not import from it.

---

## Task 1: Domain Types (`types.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/mod.rs`
- Create: `crates/core/src/blueprint_mp/types.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod blueprint_mp;`)

This is the foundation layer — pure domain types with no dependencies on the game tree or solver.

**Step 1: Write failing tests for `Seat`**

```rust
// In crates/core/src/blueprint_mp/types.rs at bottom
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn seat_new_valid() {
        let s = Seat::new(3, 6);
        assert_eq!(s.index(), 3);
    }

    #[timed_test]
    #[should_panic(expected = "out of range")]
    fn seat_new_invalid() {
        Seat::new(6, 6); // index must be < num_players
    }

    #[timed_test]
    #[should_panic(expected = "out of range")]
    fn seat_exceeds_max_players() {
        Seat::new(0, 9); // max 8 players
    }
}
```

**Step 2: Implement `Seat`**

```rust
use crate::blueprint_mp::MAX_PLAYERS;

/// Player seat index (0-based). Newtype prevents mixing with bucket/action indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Seat(u8);

impl Seat {
    /// Create a new seat, panicking if out of range.
    #[must_use]
    pub fn new(index: u8, num_players: u8) -> Self {
        assert!(
            (index as usize) < num_players as usize && (num_players as usize) <= MAX_PLAYERS,
            "Seat {index} out of range for {num_players} players (max {MAX_PLAYERS})"
        );
        Self(index)
    }

    /// Raw index (0-based).
    #[must_use]
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Construct without validation (for use in hot loops where bounds are guaranteed).
    #[must_use]
    #[inline]
    pub const fn from_raw(index: u8) -> Self {
        Self(index)
    }
}

impl std::fmt::Display for Seat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Seat({})", self.0)
    }
}
```

**Step 3: Write failing tests for `PlayerSet`**

```rust
    #[timed_test]
    fn player_set_empty() {
        let ps = PlayerSet::empty();
        assert_eq!(ps.count(), 0);
        assert!(!ps.contains(Seat::from_raw(0)));
    }

    #[timed_test]
    fn player_set_all() {
        let ps = PlayerSet::all(6);
        assert_eq!(ps.count(), 6);
        for i in 0..6 {
            assert!(ps.contains(Seat::from_raw(i)));
        }
        assert!(!ps.contains(Seat::from_raw(6)));
    }

    #[timed_test]
    fn player_set_insert_remove() {
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(2));
        ps.insert(Seat::from_raw(5));
        assert_eq!(ps.count(), 2);
        assert!(ps.contains(Seat::from_raw(2)));
        ps.remove(Seat::from_raw(2));
        assert_eq!(ps.count(), 1);
        assert!(!ps.contains(Seat::from_raw(2)));
    }

    #[timed_test]
    fn player_set_iter() {
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(1));
        ps.insert(Seat::from_raw(3));
        ps.insert(Seat::from_raw(5));
        let seats: Vec<u8> = ps.iter().map(|s| s.index()).collect();
        assert_eq!(seats, vec![1, 3, 5]);
    }

    #[timed_test]
    fn player_set_next_after_wraps() {
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(1));
        ps.insert(Seat::from_raw(4));
        // next after seat 4 should wrap to seat 1
        assert_eq!(ps.next_after(Seat::from_raw(4), 6).unwrap().index(), 1);
        // next after seat 1 should be seat 4
        assert_eq!(ps.next_after(Seat::from_raw(1), 6).unwrap().index(), 4);
    }

    #[timed_test]
    fn player_set_next_after_single() {
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(3));
        // only one player — next_after returns None
        assert!(ps.next_after(Seat::from_raw(3), 6).is_none());
    }
```

**Step 4: Implement `PlayerSet`**

```rust
/// Compact bitfield for player sets. Supports up to 8 players.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerSet(u8);

impl PlayerSet {
    #[must_use]
    pub const fn empty() -> Self { Self(0) }

    #[must_use]
    pub fn all(num_players: u8) -> Self {
        Self((1u8 << num_players) - 1)
    }

    #[must_use]
    #[inline]
    pub fn contains(self, seat: Seat) -> bool {
        self.0 & (1 << seat.0) != 0
    }

    #[inline]
    pub fn insert(&mut self, seat: Seat) {
        self.0 |= 1 << seat.0;
    }

    #[inline]
    pub fn remove(&mut self, seat: Seat) {
        self.0 &= !(1 << seat.0);
    }

    #[must_use]
    #[inline]
    pub fn count(self) -> u8 {
        self.0.count_ones() as u8
    }

    pub fn iter(self) -> impl Iterator<Item = Seat> {
        (0..8u8).filter(move |&i| self.0 & (1 << i) != 0).map(Seat::from_raw)
    }

    /// Next seat clockwise after `seat` that is in this set.
    /// Returns None if `seat` is the only member (or set is empty).
    #[must_use]
    pub fn next_after(self, seat: Seat, num_players: u8) -> Option<Seat> {
        for offset in 1..num_players {
            let candidate = (seat.0 + offset) % num_players;
            if self.contains(Seat::from_raw(candidate)) {
                return Some(Seat::from_raw(candidate));
            }
        }
        None
    }

    /// Raw bits (for serialization or debug).
    #[must_use]
    pub const fn bits(self) -> u8 { self.0 }

    #[must_use]
    pub const fn from_bits(bits: u8) -> Self { Self(bits) }
}
```

**Step 5: Write failing tests for `Chips`**

```rust
    #[timed_test]
    fn chips_arithmetic() {
        let a = Chips(10.0);
        let b = Chips(3.5);
        assert_eq!((a - b).0, 6.5);
        assert_eq!((a + b).0, 13.5);
    }

    #[timed_test]
    fn chips_zero() {
        assert!(Chips::ZERO.is_zero());
        assert!(!Chips(0.01).is_zero());
    }
```

**Step 6: Implement `Chips`**

```rust
/// Chip amount newtype. All internal values are in chips (1 BB = 2 chips).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Chips(pub f64);

impl Chips {
    pub const ZERO: Self = Self(0.0);
    const EPSILON: f64 = 0.001;

    #[must_use]
    #[inline]
    pub fn is_zero(self) -> bool { self.0.abs() < Self::EPSILON }
}

impl std::ops::Add for Chips {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}

impl std::ops::Sub for Chips {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}

impl std::ops::AddAssign for Chips {
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
}

impl std::ops::SubAssign for Chips {
    fn sub_assign(&mut self, rhs: Self) { self.0 -= rhs.0; }
}

impl std::ops::Mul<f64> for Chips {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

impl std::ops::Mul<u8> for Chips {
    type Output = Self;
    fn mul(self, rhs: u8) -> Self { Self(self.0 * f64::from(rhs)) }
}
```

**Step 7: Implement `Bucket`, `Street`, `MAX_PLAYERS` and wire up `mod.rs`**

```rust
// In types.rs:

/// Bucket index for card abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bucket(pub u16);

/// Poker street.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Preflop),
            1 => Some(Self::Flop),
            2 => Some(Self::Turn),
            3 => Some(Self::River),
            _ => None,
        }
    }

    #[must_use]
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Preflop => Some(Self::Flop),
            Self::Flop => Some(Self::Turn),
            Self::Turn => Some(Self::River),
            Self::River => None,
        }
    }

    #[must_use]
    pub const fn index(self) -> usize { self as usize }
}
```

```rust
// In mod.rs:
pub mod types;
pub const MAX_PLAYERS: usize = 8;
pub use types::*;
```

```rust
// In lib.rs, add:
pub mod blueprint_mp;
```

**Step 8: Run tests**

Run: `cargo test -p poker-solver-core blueprint_mp`
Expected: All tests PASS.

**Step 9: Commit**

```bash
git add crates/core/src/blueprint_mp/ crates/core/src/lib.rs
git commit -m "feat(blueprint_mp): add domain types — Seat, PlayerSet, Chips, Bucket, Street"
```

---

## Task 2: Config (`config.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/config.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod config;`)

**Step 1: Write failing tests for config deserialization**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn deserialize_6max_config() {
        let yaml = r#"
game:
  name: "6-max 100bb ante"
  num_players: 6
  stack_depth: 200
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2
    - seat: 1
      type: bb_ante
      amount: 2

action_abstraction:
  preflop:
    lead: ["5bb", "6bb"]
    raise:
      - ["3.0x"]
  flop:
    lead: [0.33, 0.67, 1.0]
    raise:
      - [0.5, 1.0, 2.0]
  turn:
    lead: [0.5, 1.0]
    raise:
      - [0.67, 1.0]
  river:
    lead: [0.5, 1.0]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

training:
  iterations: 10000

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;
        let cfg: BlueprintMpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.game.num_players, 6);
        assert_eq!(cfg.game.blinds.len(), 3);
        assert!(matches!(cfg.game.blinds[2].kind, ForcedBetKind::BbAnte));
    }

    #[timed_test]
    fn deserialize_heads_up_config() {
        let yaml = r#"
game:
  name: "HU 100bb"
  num_players: 2
  stack_depth: 200
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2

action_abstraction:
  preflop:
    lead: ["5bb"]
    raise:
      - ["3.0x"]
  flop:
    lead: [0.67]
    raise:
      - [1.0]
  turn:
    lead: [1.0]
    raise:
      - [1.0]
  river:
    lead: [1.0]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

training:
  iterations: 100

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;
        let cfg: BlueprintMpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.game.num_players, 2);
        assert_eq!(cfg.game.blinds.len(), 2);
    }

    #[timed_test]
    fn lead_raise_split_config() {
        let yaml = r#"
game:
  name: test
  num_players: 2
  stack_depth: 200
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2

action_abstraction:
  preflop:
    lead: ["5bb"]
    raise:
      - ["3.0x"]
      - ["2.5x"]
  flop:
    lead: [0.33, 0.67]
    raise:
      - [0.5, 1.0]
      - [0.75]
  turn:
    lead: [0.5]
    raise:
      - [1.0]
  river:
    lead: [1.0]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

training:
  iterations: 100

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;
        let cfg: BlueprintMpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.action_abstraction.flop.lead.len(), 2);
        assert_eq!(cfg.action_abstraction.flop.raise.len(), 2);
        assert_eq!(cfg.action_abstraction.preflop.raise.len(), 2);
    }

    #[timed_test]
    fn game_config_validation_rejects_9_players() {
        let cfg = MpGameConfig {
            name: "bad".into(),
            num_players: 9,
            stack_depth: 200.0,
            blinds: vec![],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[timed_test]
    fn game_config_validation_rejects_1_player() {
        let cfg = MpGameConfig {
            name: "bad".into(),
            num_players: 1,
            stack_depth: 200.0,
            blinds: vec![],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[timed_test]
    fn total_forced_bets_computed() {
        let cfg = MpGameConfig {
            name: "test".into(),
            num_players: 6,
            stack_depth: 200.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BbAnte, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        // BB-ante: BB posts 2 chips on behalf of 6 players (12 total ante)
        // But the BB-ante amount means the BB posts that amount as ante
        // SB=1, BB=2, BbAnte=2 → total forced = 5
        assert!((cfg.total_forced_bets() - 5.0).abs() < 0.01);
    }
}
```

**Step 2: Implement config structs**

```rust
use serde::{Deserialize, Serialize};

/// Top-level config for the multiplayer blueprint pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintMpConfig {
    pub game: MpGameConfig,
    pub action_abstraction: MpActionAbstractionConfig,
    pub clustering: MpClusteringConfig,
    pub training: MpTrainingConfig,
    pub snapshots: MpSnapshotConfig,
}

/// Game parameters for N-player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpGameConfig {
    pub name: String,
    pub num_players: u8,
    /// Stack depth in chips (1 BB = 2 chips).
    pub stack_depth: f64,
    /// Ordered list of forced bets (blinds, antes, straddles).
    pub blinds: Vec<ForcedBet>,
    #[serde(default)]
    pub rake_rate: f64,
    #[serde(default)]
    pub rake_cap: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcedBet {
    pub seat: u8,
    #[serde(rename = "type")]
    pub kind: ForcedBetKind,
    pub amount: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForcedBetKind {
    SmallBlind,
    BigBlind,
    Ante,
    BbAnte,
    Straddle,
}

impl MpGameConfig {
    /// Validate game config. Returns Err with a message on invalid config.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_players < 2 || self.num_players > 8 {
            return Err(format!("num_players must be 2-8, got {}", self.num_players));
        }
        if self.stack_depth <= 0.0 {
            return Err("stack_depth must be positive".into());
        }
        for blind in &self.blinds {
            if blind.seat >= self.num_players {
                return Err(format!(
                    "blind seat {} out of range for {} players",
                    blind.seat, self.num_players
                ));
            }
            if blind.amount <= 0.0 {
                return Err(format!("blind amount must be positive, got {}", blind.amount));
            }
        }
        Ok(())
    }

    /// Sum of all forced bets (blinds + antes).
    #[must_use]
    pub fn total_forced_bets(&self) -> f64 {
        self.blinds.iter().map(|b| b.amount).sum()
    }
}

/// Per-street action abstraction with lead/raise split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpActionAbstractionConfig {
    pub preflop: MpStreetSizes,
    pub flop: MpStreetSizes,
    pub turn: MpStreetSizes,
    pub river: MpStreetSizes,
}

/// Sizes for one street, split into lead (opening bet) and raise (by depth).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpStreetSizes {
    /// Opening bet sizes (pot fractions for postflop, string labels for preflop).
    pub lead: Vec<serde_yaml::Value>,
    /// Raise sizes indexed by raise depth. Last entry repeats for deeper raises.
    pub raise: Vec<Vec<serde_yaml::Value>>,
}

/// Per-street bucket counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpClusteringConfig {
    pub preflop: MpStreetCluster,
    pub flop: MpStreetCluster,
    pub turn: MpStreetCluster,
    pub river: MpStreetCluster,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpStreetCluster {
    pub buckets: u16,
}

impl MpClusteringConfig {
    /// Return bucket counts as a [u16; 4] array indexed by street.
    #[must_use]
    pub fn bucket_counts(&self) -> [u16; 4] {
        [
            self.preflop.buckets,
            self.flop.buckets,
            self.turn.buckets,
            self.river.buckets,
        ]
    }
}

/// Training parameters (mirrors blueprint_v2 TrainingConfig).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpTrainingConfig {
    #[serde(default)]
    pub cluster_path: Option<String>,
    #[serde(default)]
    pub iterations: Option<u64>,
    #[serde(default)]
    pub time_limit_minutes: Option<u64>,
    #[serde(default = "default_lcfr_warmup")]
    pub lcfr_warmup_iterations: u64,
    #[serde(default = "default_discount_interval")]
    pub lcfr_discount_interval: u64,
    #[serde(default = "default_prune_after")]
    pub prune_after_iterations: u64,
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u64,
    #[serde(default = "default_dcfr_alpha")]
    pub dcfr_alpha: f64,
    #[serde(default = "default_dcfr_beta")]
    pub dcfr_beta: f64,
    #[serde(default = "default_dcfr_gamma")]
    pub dcfr_gamma: f64,
    #[serde(default = "default_print_every")]
    pub print_every_minutes: u64,
    #[serde(default)]
    pub purify_threshold: f64,
    #[serde(default)]
    pub exploitability_interval_minutes: u64,
    #[serde(default = "default_exploitability_samples")]
    pub exploitability_samples: u64,
}

/// Snapshot settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpSnapshotConfig {
    pub warmup_minutes: u64,
    pub snapshot_every_minutes: u64,
    pub output_dir: String,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub max_snapshots: Option<u32>,
}

// Default value functions
const fn default_lcfr_warmup() -> u64 { 5_000_000 }
const fn default_discount_interval() -> u64 { 500_000 }
const fn default_prune_after() -> u64 { 5_000_000 }
const fn default_prune_threshold() -> i32 { -250 }
const fn default_batch_size() -> u64 { 200 }
fn default_dcfr_alpha() -> f64 { 1.5 }
fn default_dcfr_beta() -> f64 { 0.0 }
fn default_dcfr_gamma() -> f64 { 2.0 }
const fn default_print_every() -> u64 { 10 }
const fn default_exploitability_samples() -> u64 { 100_000 }
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core blueprint_mp::config`
Expected: All PASS.

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_mp/config.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add config types with lead/raise split and N-player blind structure"
```

---

## Task 3: 128-bit InfoKey (`info_key.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/info_key.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod info_key;`)

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn round_trip_components() {
        let key = InfoKey128::new(Seat::from_raw(3), 42, Street::Turn, 15, &[1, 3, 5]);
        assert_eq!(key.seat().index(), 3);
        assert_eq!(key.bucket_bits(), 42);
        assert_eq!(key.street(), Street::Turn);
        assert_eq!(key.spr_bucket(), 15);
    }

    #[timed_test]
    fn different_seats_different_keys() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 42, Street::Flop, 10, &[2]);
        let k2 = InfoKey128::new(Seat::from_raw(3), 42, Street::Flop, 10, &[2]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn max_22_actions() {
        let actions: Vec<u8> = vec![1; 22];
        let key = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &actions);
        assert_eq!(key.seat().index(), 0);
    }

    #[timed_test]
    #[should_panic(expected = "exceeds 22")]
    fn overflow_panics() {
        let actions: Vec<u8> = vec![1; 23];
        InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &actions);
    }

    #[timed_test]
    fn action_order_matters() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 0, Street::Flop, 0, &[1, 2]);
        let k2 = InfoKey128::new(Seat::from_raw(0), 0, Street::Flop, 0, &[2, 1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn hash_equality() {
        use std::collections::HashSet;
        let k1 = InfoKey128::new(Seat::from_raw(2), 100, Street::River, 5, &[3, 7]);
        let k2 = InfoKey128::new(Seat::from_raw(2), 100, Street::River, 5, &[3, 7]);
        let mut set = HashSet::new();
        set.insert(k1);
        assert!(set.contains(&k2));
    }
}
```

**Step 2: Implement 128-bit InfoKey**

```rust
use super::types::{Seat, Street};

/// 128-bit information set key.
///
/// Layout:
///   Bits 127-125: seat (3 bits, 0-7)
///   Bits 124-97:  bucket (28 bits)
///   Bits 96-95:   street (2 bits)
///   Bits 94-90:   spr_bucket (5 bits)
///   Bits 89-0:    action history (90 bits = 22 slots × 4 bits)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InfoKey128 {
    hi: u64,
    lo: u64,
}

const MAX_ACTION_SLOTS: usize = 22;

impl InfoKey128 {
    /// Build a key from components. Panics if action history exceeds 22 slots.
    #[must_use]
    pub fn new(seat: Seat, bucket: u32, street: Street, spr_bucket: u32, actions: &[u8]) -> Self {
        assert!(
            actions.len() <= MAX_ACTION_SLOTS,
            "Action history length {} exceeds 22 slots",
            actions.len()
        );

        // hi contains: seat(3) | bucket(28) | street(2) | spr(5) | top 26 bits of actions
        // lo contains: bottom 64 bits of actions
        //
        // Total action bits = 90 = 26 (in hi) + 64 (in lo)
        let mut hi: u64 = 0;
        hi |= (u64::from(seat.index()) & 0x7) << 61;          // bits 63-61 of hi
        hi |= (u64::from(bucket) & 0x0FFF_FFFF) << 33;         // bits 60-33 of hi
        hi |= (u64::from(street as u8) & 0x3) << 31;           // bits 32-31 of hi
        hi |= (u64::from(spr_bucket) & 0x1F) << 26;            // bits 30-26 of hi

        // Pack actions into 90 bits (MSB-first): actions[0] at bits 89-86, etc.
        // Bits 89-64 go into hi[25:0], bits 63-0 go into lo
        let mut lo: u64 = 0;
        for (i, &code) in actions.iter().enumerate() {
            let bit_pos = 86 - (i * 4); // action 0 starts at bit 86 (within the 90-bit field)
            let code64 = u64::from(code) & 0xF;
            if bit_pos >= 64 {
                hi |= code64 << (bit_pos - 64);
            } else {
                lo |= code64 << bit_pos;
            }
        }

        Self { hi, lo }
    }

    #[must_use]
    pub fn seat(self) -> Seat {
        Seat::from_raw(((self.hi >> 61) & 0x7) as u8)
    }

    #[must_use]
    pub fn bucket_bits(self) -> u32 {
        ((self.hi >> 33) & 0x0FFF_FFFF) as u32
    }

    #[must_use]
    pub fn street(self) -> Street {
        Street::from_u8(((self.hi >> 31) & 0x3) as u8).unwrap()
    }

    #[must_use]
    pub fn spr_bucket(self) -> u32 {
        ((self.hi >> 26) & 0x1F) as u32
    }
}

impl std::fmt::Debug for InfoKey128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InfoKey128(seat={}, bucket={}, street={:?}, spr={})",
            self.seat().index(), self.bucket_bits(), self.street(), self.spr_bucket())
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core blueprint_mp::info_key`
Expected: All PASS.

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_mp/info_key.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add 128-bit InfoKey with seat, 22 action slots"
```

---

## Task 4: Terminal Payoffs (`terminal.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/terminal.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod terminal;`)

This is the most complex domain logic — side pot resolution.

**Step 1: Write failing tests for fold terminal**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::types::*;
    use test_macros::timed_test;

    fn chips_arr(vals: &[f64]) -> [Chips; MAX_PLAYERS] {
        let mut arr = [Chips::ZERO; MAX_PLAYERS];
        for (i, &v) in vals.iter().enumerate() {
            arr[i] = Chips(v);
        }
        arr
    }

    #[timed_test]
    fn fold_last_standing_3_player() {
        // 3 players, each put in 10 chips. Player 2 wins (others folded).
        let contributions = chips_arr(&[10.0, 10.0, 10.0]);
        let payoffs = resolve_fold(contributions, Seat::from_raw(2), 3);
        // Winner: 30 - 10 = +20
        assert!((payoffs[2].0 - 20.0).abs() < 0.01);
        // Losers: 0 - 10 = -10
        assert!((payoffs[0].0 - (-10.0)).abs() < 0.01);
        assert!((payoffs[1].0 - (-10.0)).abs() < 0.01);
    }

    #[timed_test]
    fn fold_payoffs_sum_to_zero() {
        let contributions = chips_arr(&[5.0, 15.0, 10.0, 20.0]);
        let payoffs = resolve_fold(contributions, Seat::from_raw(1), 4);
        let sum: f64 = payoffs.iter().take(4).map(|c| c.0).sum();
        assert!(sum.abs() < 0.01, "payoffs must sum to zero, got {sum}");
    }
}
```

**Step 2: Implement fold resolution**

```rust
use crate::blueprint_mp::{MAX_PLAYERS, types::*};

/// Resolve a fold terminal: last player standing wins the entire pot.
#[must_use]
pub fn resolve_fold(
    contributions: [Chips; MAX_PLAYERS],
    winner: Seat,
    num_players: u8,
) -> [Chips; MAX_PLAYERS] {
    let total_pot: f64 = contributions.iter().take(num_players as usize).map(|c| c.0).sum();
    let mut payoffs = [Chips::ZERO; MAX_PLAYERS];
    for i in 0..num_players as usize {
        payoffs[i] = Chips(-contributions[i].0);
    }
    payoffs[winner.index() as usize].0 += total_pot;
    payoffs
}
```

**Step 3: Write failing tests for showdown with side pots**

```rust
    #[timed_test]
    fn showdown_no_side_pots() {
        // 3 players, all contributed 50. Player 0 wins.
        let contributions = chips_arr(&[50.0, 50.0, 50.0]);
        let mut active = PlayerSet::empty();
        active.insert(Seat::from_raw(0));
        active.insert(Seat::from_raw(1));
        active.insert(Seat::from_raw(2));
        // hand_ranks: higher is better. Player 0 = 100, P1 = 80, P2 = 90
        let hand_ranks = [100u32, 80, 90, 0, 0, 0, 0, 0];
        let payoffs = resolve_showdown(
            &contributions, &hand_ranks, active, 3, 0.0, Chips::ZERO,
        );
        // P0 wins 150, net +100
        assert!((payoffs[0].0 - 100.0).abs() < 0.01);
        assert!((payoffs[1].0 - (-50.0)).abs() < 0.01);
        assert!((payoffs[2].0 - (-50.0)).abs() < 0.01);
    }

    #[timed_test]
    fn showdown_with_side_pot() {
        // P0 all-in for 30, P1 and P2 both put in 100.
        // P0 has best hand (rank 100), P1 has second best (90), P2 worst (80).
        // Main pot: 30 * 3 = 90, contested by all → P0 wins
        // Side pot: (100-30) * 2 = 140, contested by P1 and P2 → P1 wins
        let contributions = chips_arr(&[30.0, 100.0, 100.0]);
        let mut active = PlayerSet::empty();
        active.insert(Seat::from_raw(0));
        active.insert(Seat::from_raw(1));
        active.insert(Seat::from_raw(2));
        let hand_ranks = [100u32, 90, 80, 0, 0, 0, 0, 0];
        let payoffs = resolve_showdown(
            &contributions, &hand_ranks, active, 3, 0.0, Chips::ZERO,
        );
        // P0: wins main pot 90, net = 90 - 30 = +60
        assert!((payoffs[0].0 - 60.0).abs() < 0.01, "P0: {}", payoffs[0].0);
        // P1: wins side pot 140, net = 140 - 100 = +40
        assert!((payoffs[1].0 - 40.0).abs() < 0.01, "P1: {}", payoffs[1].0);
        // P2: loses everything, net = -100
        assert!((payoffs[2].0 - (-100.0)).abs() < 0.01, "P2: {}", payoffs[2].0);
    }

    #[timed_test]
    fn showdown_three_way_side_pots() {
        // P0 all-in 20, P1 all-in 50, P2 put in 100.
        // P2 has best hand, P0 second, P1 worst.
        // Main pot: 20*3 = 60 → P2 wins
        // Side pot 1: (50-20)*2 = 60 → P2 wins
        // Side pot 2: (100-50)*1 = 50 → P2 wins (uncontested)
        let contributions = chips_arr(&[20.0, 50.0, 100.0]);
        let mut active = PlayerSet::all(3);
        let hand_ranks = [80u32, 50, 100, 0, 0, 0, 0, 0];
        let payoffs = resolve_showdown(
            &contributions, &hand_ranks, active, 3, 0.0, Chips::ZERO,
        );
        // P2 wins everything: 170 - 100 = +70
        assert!((payoffs[2].0 - 70.0).abs() < 0.01, "P2: {}", payoffs[2].0);
        assert!((payoffs[0].0 - (-20.0)).abs() < 0.01, "P0: {}", payoffs[0].0);
        assert!((payoffs[1].0 - (-50.0)).abs() < 0.01, "P1: {}", payoffs[1].0);
    }

    #[timed_test]
    fn showdown_with_rake() {
        // 2 players, 50 each. P0 wins. 5% rake, cap 10.
        let contributions = chips_arr(&[50.0, 50.0]);
        let mut active = PlayerSet::empty();
        active.insert(Seat::from_raw(0));
        active.insert(Seat::from_raw(1));
        let hand_ranks = [100u32, 80, 0, 0, 0, 0, 0, 0];
        let payoffs = resolve_showdown(
            &contributions, &hand_ranks, active, 2, 0.05, Chips(10.0),
        );
        // Pot = 100, rake = 5.0, P0 net = 95 - 50 = +45
        assert!((payoffs[0].0 - 45.0).abs() < 0.01);
        assert!((payoffs[1].0 - (-50.0)).abs() < 0.01);
    }

    #[timed_test]
    fn showdown_payoffs_sum_to_negative_rake() {
        let contributions = chips_arr(&[30.0, 100.0, 100.0]);
        let mut active = PlayerSet::all(3);
        let hand_ranks = [100u32, 90, 80, 0, 0, 0, 0, 0];
        let payoffs = resolve_showdown(
            &contributions, &hand_ranks, active, 3, 0.05, Chips(20.0),
        );
        let sum: f64 = payoffs.iter().take(3).map(|c| c.0).sum();
        // Sum should be negative (equal to -rake)
        let total_pot: f64 = contributions.iter().take(3).map(|c| c.0).sum();
        let expected_rake = (total_pot * 0.05).min(20.0);
        assert!((sum + expected_rake).abs() < 0.01, "sum={sum}, rake={expected_rake}");
    }

    #[timed_test]
    fn showdown_tie_splits_pot() {
        // 2 players, equal hands, 50 each.
        let contributions = chips_arr(&[50.0, 50.0]);
        let mut active = PlayerSet::all(2);
        let hand_ranks = [100u32, 100, 0, 0, 0, 0, 0, 0];
        let payoffs = resolve_showdown(
            &contributions, &hand_ranks, active, 2, 0.0, Chips::ZERO,
        );
        assert!((payoffs[0].0).abs() < 0.01, "P0 should break even: {}", payoffs[0].0);
        assert!((payoffs[1].0).abs() < 0.01, "P1 should break even: {}", payoffs[1].0);
    }
```

**Step 4: Implement showdown with side pot resolution**

```rust
/// Resolve a showdown with full side pot support.
///
/// Algorithm:
/// 1. Sort active players by contribution (ascending)
/// 2. Build pots layer by layer
/// 3. Apply rake per pot (capped at rake_cap total)
/// 4. Award each pot to best hand among eligible players
#[must_use]
pub fn resolve_showdown(
    contributions: &[Chips; MAX_PLAYERS],
    hand_ranks: &[u32; MAX_PLAYERS],
    active: PlayerSet,
    num_players: u8,
    rake_rate: f64,
    rake_cap: Chips,
) -> [Chips; MAX_PLAYERS] {
    // Collect (seat, contribution) for active players, sorted by contribution ascending
    let mut players: Vec<(Seat, f64)> = active
        .iter()
        .map(|s| (s, contributions[s.index() as usize].0))
        .collect();
    players.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Build side pots
    let mut pots: Vec<(f64, Vec<Seat>)> = Vec::new();
    let mut prev_level = 0.0;

    for (i, &(_, contribution)) in players.iter().enumerate() {
        let level_amount = contribution - prev_level;
        if level_amount > 0.001 {
            // All players from index i onward contributed at least this level
            let eligible: Vec<Seat> = players[i..].iter().map(|&(s, _)| s).collect();
            // Include contributions from non-active (folded) players at this level
            let mut pot_amount = 0.0;
            for p in 0..num_players as usize {
                let player_contrib = contributions[p].0;
                let contrib_at_level = (player_contrib - prev_level).min(level_amount).max(0.0);
                pot_amount += contrib_at_level;
            }
            pots.push((pot_amount, eligible));
        }
        prev_level = contribution;
    }

    // Award pots
    let mut payoffs = [Chips::ZERO; MAX_PLAYERS];
    let mut total_rake = 0.0;

    for (pot_amount, eligible) in &pots {
        // Apply rake
        let rake = if rake_rate > 0.0 {
            let uncapped = pot_amount * rake_rate;
            let remaining_cap = rake_cap.0 - total_rake;
            if rake_cap.0 > 0.0 {
                uncapped.min(remaining_cap.max(0.0))
            } else {
                uncapped
            }
        } else {
            0.0
        };
        total_rake += rake;
        let net_pot = pot_amount - rake;

        // Find best hand rank among eligible
        let best_rank = eligible
            .iter()
            .map(|s| hand_ranks[s.index() as usize])
            .max()
            .unwrap();

        let winners: Vec<&Seat> = eligible
            .iter()
            .filter(|s| hand_ranks[s.index() as usize] == best_rank)
            .collect();

        let share = net_pot / winners.len() as f64;
        for &&seat in &winners {
            payoffs[seat.index() as usize].0 += share;
        }
    }

    // Convert from winnings to net payoffs (subtract contributions)
    for i in 0..num_players as usize {
        payoffs[i].0 -= contributions[i].0;
    }

    payoffs
}
```

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core blueprint_mp::terminal`
Expected: All PASS.

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_mp/terminal.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add terminal payoff resolution with full side pot support"
```

---

## Task 5: Game Tree Builder (`game_tree.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/game_tree.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod game_tree;`)

This is the largest and most complex task. The developer should reference `crates/core/src/blueprint_v2/game_tree.rs` for patterns but implement from scratch with N-player semantics.

**Step 1: Write failing tests for node types and basic tree structure**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::types::*;
    use test_macros::timed_test;

    #[timed_test]
    fn build_trivial_2_player_tree() {
        // Minimal 2-player: SB/BB, one bet size, preflop only
        let config = build_test_config(2, 200.0);
        let tree = MpGameTree::build(&config);
        assert!(!tree.nodes.is_empty());
        // Root should be a Decision node for the first actor (SB preflop)
        match &tree.nodes[tree.root as usize] {
            MpGameNode::Decision { seat, .. } => {
                // Seat 0 = SB acts first preflop
                assert_eq!(seat.index(), 0);
            }
            _ => panic!("root should be a Decision node"),
        }
    }

    #[timed_test]
    fn fold_produces_terminal_in_2_player() {
        let config = build_test_config(2, 200.0);
        let tree = MpGameTree::build(&config);
        // Find a fold action and verify it leads to Terminal::LastStanding
        let has_fold_terminal = tree_has_fold_terminal(&tree);
        assert!(has_fold_terminal, "2-player tree should have fold terminals");
    }

    #[timed_test]
    fn fold_continues_game_in_3_player() {
        let config = build_test_config(3, 200.0);
        let tree = MpGameTree::build(&config);
        // In 3-player, first fold should NOT be terminal (2 players remain)
        let has_fold_continuation = tree_has_fold_continuation(&tree);
        assert!(has_fold_continuation, "3-player tree should have fold continuations");
    }

    #[timed_test]
    fn all_terminals_accounted() {
        // Every terminal should have valid kind
        let config = build_test_config(3, 200.0);
        let tree = MpGameTree::build(&config);
        for node in &tree.nodes {
            if let MpGameNode::Terminal { kind, .. } = node {
                match kind {
                    TerminalKind::LastStanding { winner } => {
                        assert!(winner.index() < 3);
                    }
                    TerminalKind::Showdown { .. } => {}
                }
            }
        }
    }

    #[timed_test]
    fn tree_has_chance_nodes() {
        let config = build_test_config(2, 200.0);
        let tree = MpGameTree::build(&config);
        let chance_count = tree.nodes.iter().filter(|n| matches!(n, MpGameNode::Chance { .. })).count();
        assert!(chance_count > 0, "tree should have chance nodes between streets");
    }

    // Helper: build a minimal test config
    fn build_test_config(num_players: u8, stack_depth: f64) -> super::TreeBuildConfig {
        // ... fill in with minimal blind/sizing config
        todo!("implement in task")
    }

    fn tree_has_fold_terminal(tree: &MpGameTree) -> bool {
        tree.nodes.iter().any(|n| matches!(n, MpGameNode::Terminal { kind: TerminalKind::LastStanding { .. }, .. }))
    }

    fn tree_has_fold_continuation(tree: &MpGameTree) -> bool {
        // After a fold in 3-player, we should see a Decision node for the remaining players
        // This is detectable if a fold action leads to a Decision node (not Terminal)
        for node in &tree.nodes {
            if let MpGameNode::Decision { actions, children, .. } = node {
                for (i, action) in actions.iter().enumerate() {
                    if matches!(action, TreeAction::Fold) {
                        if matches!(tree.nodes[children[i] as usize], MpGameNode::Decision { .. }) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}
```

**Step 2: Implement node types**

```rust
use crate::blueprint_mp::{MAX_PLAYERS, types::*};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeAction {
    Fold,
    Check,
    Call,
    /// Lead bet (opening bet into an unbet pot). Amount in chips.
    Lead(f64),
    /// Raise TO amount in chips.
    Raise(f64),
    AllIn,
}

#[derive(Debug, Clone)]
pub enum MpGameNode {
    Decision {
        seat: Seat,
        street: Street,
        actions: Vec<TreeAction>,
        children: Vec<u32>,
    },
    Chance {
        next_street: Street,
        child: u32,
    },
    Terminal {
        kind: TerminalKind,
        pot: Chips,
        contributions: [Chips; MAX_PLAYERS],
    },
}

#[derive(Debug, Clone)]
pub enum TerminalKind {
    LastStanding { winner: Seat },
    Showdown { active: PlayerSet },
}

#[derive(Debug, Clone)]
pub struct MpGameTree {
    pub nodes: Vec<MpGameNode>,
    pub root: u32,
    pub num_players: u8,
    pub starting_stack: Chips,
}
```

**Step 3: Implement `BuildState` and recursive tree builder**

The build state tracks N-player stacks, bets, active/all-in sets, and the closing-action logic. Reference `blueprint_v2/game_tree.rs:77-105` for the pattern, but use domain types and N-player logic.

Key differences from `blueprint_v2`:
- `to_act` uses `PlayerSet::next_after()` to find the next active, non-all-in player
- Fold removes a player from `active` and only creates a terminal if `active.count() == 1`
- Closing condition checks that all active non-all-in players have acted since last aggression AND have equal investment
- Side pot information is embedded in Terminal nodes via `contributions`
- Bet/raise sizing uses lead/raise split from config

This is complex enough that the developer should implement and test incrementally:
1. First get 2-player fold/check/call working
2. Then add bet/raise sizing
3. Then add 3+ player fold-continuation
4. Then add all-in and multi-street transitions

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core blueprint_mp::game_tree`
Expected: All PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_mp/game_tree.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add N-player game tree builder with fold-continuation and lead/raise split"
```

---

## Task 6: Storage (`storage.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/storage.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod storage;`)

Storage can largely mirror `blueprint_v2/storage.rs` since the flat-buffer layout (node × bucket × action) is player-count agnostic. Each decision node belongs to one seat, and the bucket count for that node comes from the seat's street.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::game_tree::*;
    use crate::blueprint_mp::types::*;
    use test_macros::timed_test;

    #[timed_test]
    fn storage_regret_round_trip() {
        let tree = make_tiny_tree();
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        // Node 0 is a Decision with 3 actions on preflop (10 buckets)
        storage.add_regret(0, 5, 1, 1000);
        assert_eq!(storage.get_regret(0, 5, 1), 1000);
    }

    #[timed_test]
    fn storage_strategy_sum_round_trip() {
        let tree = make_tiny_tree();
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        storage.add_strategy_sum(0, 5, 2, 500);
        assert_eq!(storage.get_strategy_sum(0, 5, 2), 500);
    }

    fn make_tiny_tree() -> MpGameTree {
        // A minimal tree: one Decision node with 3 actions, all children Terminal
        MpGameTree {
            nodes: vec![
                MpGameNode::Decision {
                    seat: Seat::from_raw(0),
                    street: Street::Preflop,
                    actions: vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn],
                    children: vec![1, 2, 3],
                },
                MpGameNode::Terminal {
                    kind: TerminalKind::LastStanding { winner: Seat::from_raw(1) },
                    pot: Chips(4.0),
                    contributions: [Chips::ZERO; MAX_PLAYERS],
                },
                MpGameNode::Terminal {
                    kind: TerminalKind::Showdown { active: PlayerSet::all(2) },
                    pot: Chips(4.0),
                    contributions: [Chips::ZERO; MAX_PLAYERS],
                },
                MpGameNode::Terminal {
                    kind: TerminalKind::Showdown { active: PlayerSet::all(2) },
                    pot: Chips(400.0),
                    contributions: [Chips::ZERO; MAX_PLAYERS],
                },
            ],
            root: 0,
            num_players: 2,
            starting_stack: Chips(200.0),
        }
    }
}
```

**Step 2: Implement storage (follows blueprint_v2 pattern)**

Reference: `crates/core/src/blueprint_v2/storage.rs:42-132` for the flat-buffer layout and atomic operations. Reuse the same `REGRET_SCALE`, `AtomicI32`/`AtomicI64` pattern. Key difference: `bucket_counts` lookup uses the street from each Decision node.

**Step 3: Run tests and commit**

```bash
git add crates/core/src/blueprint_mp/storage.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add flat-buffer regret/strategy storage"
```

---

## Task 7: MCCFR Traversal (`mccfr.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/mccfr.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod mccfr;`)

**Step 1: Write failing tests for deal sampling**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::types::*;
    use test_macros::timed_test;

    #[timed_test]
    fn deal_has_no_duplicate_cards() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for num_players in 2..=8u8 {
            let deal = sample_deal(num_players, &mut rng);
            let mut all_cards = Vec::new();
            for i in 0..num_players as usize {
                all_cards.push(deal.hole_cards[i][0]);
                all_cards.push(deal.hole_cards[i][1]);
            }
            all_cards.extend_from_slice(&deal.board);
            // Check uniqueness
            let len = all_cards.len();
            all_cards.sort_by_key(|c| (c.value as u8, c.suit as u8));
            all_cards.dedup();
            assert_eq!(all_cards.len(), len, "duplicate cards in {num_players}-player deal");
        }
    }

    #[timed_test]
    fn deal_respects_num_players() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let deal = sample_deal(6, &mut rng);
        assert_eq!(deal.num_players, 6);
    }
}
```

**Step 2: Implement deal sampling**

```rust
use crate::poker::{Card, ALL_SUITS, ALL_VALUES};
use crate::blueprint_mp::{MAX_PLAYERS, types::*};
use rand::Rng;

pub struct MpDeal {
    pub hole_cards: [[Card; 2]; MAX_PLAYERS],
    pub board: [Card; 5],
    pub num_players: u8,
}

pub struct MpDealWithBuckets {
    pub deal: MpDeal,
    pub buckets: [[u16; 4]; MAX_PLAYERS],
}

/// Sample a random deal for N players.
pub fn sample_deal(num_players: u8, rng: &mut impl Rng) -> MpDeal {
    let mut deck = crate::poker::full_deck();
    let needed = 2 * num_players as usize + 5;
    // Partial Fisher-Yates
    for i in 0..needed {
        let j = rng.random_range(i..52);
        deck.swap(i, j);
    }
    let mut hole_cards = [[deck[0]; 2]; MAX_PLAYERS]; // placeholder
    for p in 0..num_players as usize {
        hole_cards[p] = [deck[p * 2], deck[p * 2 + 1]];
    }
    let board_start = 2 * num_players as usize;
    let board = [
        deck[board_start], deck[board_start + 1], deck[board_start + 2],
        deck[board_start + 3], deck[board_start + 4],
    ];
    MpDeal { hole_cards, board, num_players }
}
```

**Step 3: Write failing tests for traversal (using a tiny hand-built tree)**

Build a minimal 2-player tree by hand (a single preflop decision: fold/call/raise, with terminals), run one external-sampling iteration, verify regrets are updated.

**Step 4: Implement `traverse_external`**

Reference: `crates/core/src/blueprint_v2/mccfr.rs:644-730` for the pattern. The structure is identical — the only change is `seat == traverser` comparison instead of `player == traverser`, and terminal_value calls into `terminal.rs` with N-player logic.

**Step 5: Run tests and commit**

```bash
git add crates/core/src/blueprint_mp/mccfr.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add external-sampling MCCFR traversal for N players"
```

---

## Task 8: Training Loop (`trainer.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/trainer.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod trainer;`)

**Step 1: Write a failing integration test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn train_2_player_toy_converges() {
        // Minimal 2-player config: 20bb stacks, 1 bet size per street
        // Run 1000 meta-iterations, verify strategy delta decreases
        let config = build_toy_config(2);
        let result = train_blueprint_mp(&config, None);
        assert!(result.final_strategy_delta < result.initial_strategy_delta);
    }

    #[timed_test]
    fn train_3_player_toy_runs() {
        // Just verify it doesn't panic for 3 players
        let config = build_toy_config(3);
        let result = train_blueprint_mp(&config, Some(100));
        assert!(result.meta_iterations >= 100);
    }
}
```

**Step 2: Implement the training loop**

The training loop cycles through all N players as traverser per meta-iteration, applies DCFR discounting, and logs progress. Reference: `crates/core/src/blueprint_v2/trainer.rs:120+` for the parallel batch pattern.

Key points:
- One meta-iteration = N traversals (one per seat)
- DCFR discount keyed off meta-iterations
- Parallel batches via rayon (same as blueprint_v2)
- Strategy delta computed from storage

**Step 3: Run tests and commit**

```bash
git add crates/core/src/blueprint_mp/trainer.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add N-player training loop with per-seat traverser cycling"
```

---

## Task 9: Exploitability (`exploitability.rs`)

**Files:**
- Create: `crates/core/src/blueprint_mp/exploitability.rs`
- Modify: `crates/core/src/blueprint_mp/mod.rs` (add `pub mod exploitability;`)

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn br_value_decreases_with_training() {
        // Train toy 2-player for 500 iters, measure BR.
        // Train for 2000 more, measure again. Should decrease.
        // ... (uses toy config from trainer tests)
    }

    #[timed_test]
    fn br_values_per_seat_returned() {
        // Verify we get one BR value per player
        let config = build_toy_config(2);
        let (tree, storage) = train_and_return(config, 500);
        let br = compute_exploitability(&tree, &storage, &buckets, 1000);
        assert_eq!(br.per_seat.len(), 2);
    }
}
```

**Step 2: Implement per-player best-response traversal**

For each seat, traverse the tree: at the target seat's nodes, pick the action maximizing value (best response); at all other seats' nodes, sample from the average strategy. Reference: `crates/core/src/blueprint_v2/mccfr.rs:799+` (`traverse_best_response`).

**Step 3: Run tests and commit**

```bash
git add crates/core/src/blueprint_mp/exploitability.rs crates/core/src/blueprint_mp/mod.rs
git commit -m "feat(blueprint_mp): add per-seat best-response exploitability diagnostic"
```

---

## Task 10: CLI Integration (`crates/trainer/`)

**Files:**
- Modify: `crates/trainer/src/main.rs` — add `train-blueprint-mp` subcommand
- Create a sample config: `sample_configurations/blueprint_mp_3player.yaml`

**Step 1: Add CLI subcommand**

Add a `train-blueprint-mp` subcommand that loads a `BlueprintMpConfig` YAML and calls `train_blueprint_mp()`. Reference the existing `train-blueprint` subcommand pattern in `main.rs`.

**Step 2: Create sample config**

```yaml
game:
  name: "3-max 50bb test"
  num_players: 3
  stack_depth: 100
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2

action_abstraction:
  preflop:
    lead: ["5bb"]
    raise:
      - ["3.0x"]
  flop:
    lead: [0.67]
    raise:
      - [1.0]
  turn:
    lead: [0.67]
    raise:
      - [1.0]
  river:
    lead: [0.67]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 50
  turn:
    buckets: 50
  river:
    buckets: 50

training:
  iterations: 10000
  batch_size: 100

snapshots:
  warmup_minutes: 5
  snapshot_every_minutes: 5
  output_dir: "/tmp/blueprint_mp_3p"
```

**Step 3: Verify the CLI works**

Run: `cargo run -p poker-solver-trainer --release -- train-blueprint-mp sample_configurations/blueprint_mp_3player.yaml`
Expected: Training starts, progress logs appear, no panics.

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs sample_configurations/blueprint_mp_3player.yaml
git commit -m "feat(trainer): add train-blueprint-mp CLI subcommand with sample 3-player config"
```

---

## Task 11: Cross-Validation Against `blueprint_v2`

**Files:**
- Create: `crates/core/tests/blueprint_mp_v2_parity.rs` (integration test)

**Step 1: Write parity test**

```rust
//! Verify that blueprint_mp produces equivalent results to blueprint_v2
//! on a 2-player configuration.

use poker_solver_core::blueprint_mp;
use poker_solver_core::blueprint_v2;

#[test]
fn tree_node_count_matches() {
    // Build identical 2-player trees with both modules
    // Verify same number of Decision, Chance, Terminal nodes
}

#[test]
fn strategy_convergence_direction_matches() {
    // Train both for 1000 iterations on identical config
    // Verify strategy delta decreases for both
    // Verify root EV is within reasonable tolerance
}
```

**Step 2: Run and commit**

```bash
git add crates/core/tests/blueprint_mp_v2_parity.rs
git commit -m "test: add cross-validation tests between blueprint_mp and blueprint_v2"
```

---

## Task 12: Documentation Update

**Files:**
- Modify: `docs/architecture.md` — add blueprint_mp section
- Modify: `docs/training.md` — add `train-blueprint-mp` command docs

**Step 1: Update architecture.md**

Add a section describing `blueprint_mp`: purpose, module structure, how it relates to `blueprint_v2`, key domain types.

**Step 2: Update training.md**

Document the `train-blueprint-mp` subcommand, config format differences (lead/raise split, blinds list), and sample usage for 3-player and 6-player configs.

**Step 3: Commit**

```bash
git add docs/architecture.md docs/training.md
git commit -m "docs: add blueprint_mp architecture and training documentation"
```
