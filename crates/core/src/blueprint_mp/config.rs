use serde::{Deserialize, Serialize};

// ── Top-level config ─────────────────────────────────────────────────

/// Top-level config for the multiplayer blueprint pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintMpConfig {
    pub game: MpGameConfig,
    pub action_abstraction: MpActionAbstractionConfig,
    pub clustering: MpClusteringConfig,
    pub training: MpTrainingConfig,
    pub snapshots: MpSnapshotConfig,
}

// ── Game config ──────────────────────────────────────────────────────

/// Core game parameters for N-player poker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpGameConfig {
    pub name: String,
    pub num_players: u8,
    pub stack_depth: f64,
    pub blinds: Vec<ForcedBet>,
    #[serde(default)]
    pub rake_rate: f64,
    #[serde(default)]
    pub rake_cap: f64,
}

impl MpGameConfig {
    /// Validate game config constraints.
    ///
    /// # Errors
    /// Returns `Err` if `num_players` not in 2..=8, `stack_depth` <= 0,
    /// a blind seat >= `num_players`, or a blind amount <= 0.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_players()?;
        self.validate_stack_depth()?;
        self.validate_blinds()
    }

    fn validate_players(&self) -> Result<(), String> {
        if self.num_players < 2 || self.num_players > 8 {
            return Err(format!(
                "num_players must be 2-8, got {}",
                self.num_players
            ));
        }
        Ok(())
    }

    fn validate_stack_depth(&self) -> Result<(), String> {
        if self.stack_depth <= 0.0 {
            return Err(format!(
                "stack_depth must be > 0, got {}",
                self.stack_depth
            ));
        }
        Ok(())
    }

    fn validate_blinds(&self) -> Result<(), String> {
        for blind in &self.blinds {
            if blind.seat >= self.num_players {
                return Err(format!(
                    "blind seat {} >= num_players {}",
                    blind.seat, self.num_players
                ));
            }
            if blind.amount <= 0.0 {
                return Err(format!(
                    "blind amount must be > 0, got {}",
                    blind.amount
                ));
            }
        }
        Ok(())
    }

    /// Sum of all forced bet amounts.
    #[must_use]
    pub fn total_forced_bets(&self) -> f64 {
        self.blinds.iter().map(|b| b.amount).sum()
    }
}

// ── Forced bet types ─────────────────────────────────────────────────

/// A forced bet posted before the hand begins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcedBet {
    pub seat: u8,
    #[serde(rename = "type")]
    pub kind: ForcedBetKind,
    pub amount: f64,
}

/// Kind of forced bet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcedBetKind {
    SmallBlind,
    BigBlind,
    Ante,
    BbAnte,
    Straddle,
}

// ── Action abstraction ───────────────────────────────────────────────

/// Per-street action abstraction with lead/raise split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpActionAbstractionConfig {
    pub preflop: MpStreetSizes,
    pub flop: MpStreetSizes,
    pub turn: MpStreetSizes,
    pub river: MpStreetSizes,
}

/// Lead and raise sizes for a single street.
///
/// `lead` is a flat list of sizes used when opening the action.
/// `raise` is indexed by depth; the last entry repeats for deeper raises.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpStreetSizes {
    pub lead: Vec<serde_yaml::Value>,
    pub raise: Vec<Vec<serde_yaml::Value>>,
}

// ── Clustering config ────────────────────────────────────────────────

/// Card abstraction clustering, one entry per street.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpClusteringConfig {
    pub preflop: MpStreetCluster,
    pub flop: MpStreetCluster,
    pub turn: MpStreetCluster,
    pub river: MpStreetCluster,
}

impl MpClusteringConfig {
    /// Bucket counts as a fixed-size array indexed by street ordinal.
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

/// Bucket count for a single street.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpStreetCluster {
    pub buckets: u16,
}

// ── Training config ──────────────────────────────────────────────────

/// MCCFR training schedule and parameters for multiplayer.
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
    #[serde(default = "default_prune_explore")]
    pub prune_explore_pct: f64,
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

// ── Snapshot config ──────────────────────────────────────────────────

/// Checkpoint output settings for multiplayer training.
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

// ── Default value functions ──────────────────────────────────────────

const fn default_lcfr_warmup() -> u64 {
    5_000_000
}

const fn default_discount_interval() -> u64 {
    500_000
}

const fn default_prune_after() -> u64 {
    5_000_000
}

const fn default_prune_threshold() -> i32 {
    -250
}

fn default_prune_explore() -> f64 {
    0.05
}

const fn default_batch_size() -> u64 {
    200
}

fn default_dcfr_alpha() -> f64 {
    1.5
}

fn default_dcfr_beta() -> f64 {
    0.0
}

fn default_dcfr_gamma() -> f64 {
    2.0
}

const fn default_print_every() -> u64 {
    10
}

const fn default_exploitability_samples() -> u64 {
    100_000
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn deserialize_6max_config() {
        let yaml = r#"
game:
  name: "6max BB-ante"
  num_players: 6
  stack_depth: 200.0
  blinds:
    - seat: 0
      type: small_blind
      amount: 1.0
    - seat: 1
      type: big_blind
      amount: 2.0
    - seat: 1
      type: bb_ante
      amount: 2.0

action_abstraction:
  preflop:
    lead: [0.5, 1.0]
    raise:
      - [0.5, 1.0]
      - [0.67]
  flop:
    lead: [0.33, 0.67, 1.0]
    raise:
      - [0.5, 1.0]
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

        let cfg: BlueprintMpConfig =
            serde_yaml::from_str(yaml).expect("failed to parse 6max config");

        assert_eq!(cfg.game.num_players, 6);
        assert_eq!(cfg.game.blinds.len(), 3);
        assert!(matches!(cfg.game.blinds[2].kind, ForcedBetKind::BbAnte));
    }

    #[timed_test]
    fn deserialize_heads_up_config() {
        let yaml = r#"
game:
  name: "Heads Up"
  num_players: 2
  stack_depth: 100.0
  blinds:
    - seat: 0
      type: small_blind
      amount: 1.0
    - seat: 1
      type: big_blind
      amount: 2.0

action_abstraction:
  preflop:
    lead: [1.0]
    raise:
      - [1.0]
  flop:
    lead: [1.0]
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
  warmup_minutes: 10
  snapshot_every_minutes: 5
  output_dir: "/tmp/hu"
"#;

        let cfg: BlueprintMpConfig =
            serde_yaml::from_str(yaml).expect("failed to parse HU config");

        assert_eq!(cfg.game.name, "Heads Up");
        assert_eq!(cfg.game.num_players, 2);
        assert_eq!(cfg.game.blinds.len(), 2);
        assert!(matches!(cfg.game.blinds[0].kind, ForcedBetKind::SmallBlind));
        assert!(matches!(cfg.game.blinds[1].kind, ForcedBetKind::BigBlind));
        assert!((cfg.game.stack_depth - 100.0).abs() < f64::EPSILON);
    }

    #[timed_test]
    fn lead_raise_split_config() {
        let yaml = r#"
game:
  name: "Split Test"
  num_players: 2
  stack_depth: 100.0
  blinds:
    - seat: 0
      type: small_blind
      amount: 1.0
    - seat: 1
      type: big_blind
      amount: 2.0

action_abstraction:
  preflop:
    lead: [0.5, 1.0]
    raise:
      - [0.5, 1.0]
      - [0.67]
      - [1.0]
  flop:
    lead: [0.33, 0.67, 1.0]
    raise:
      - [0.5, 1.0]
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
  warmup_minutes: 10
  snapshot_every_minutes: 5
  output_dir: "/tmp/split"
"#;

        let cfg: BlueprintMpConfig =
            serde_yaml::from_str(yaml).expect("failed to parse split config");

        // Preflop: 2 lead sizes, 3 raise depths
        assert_eq!(cfg.action_abstraction.preflop.lead.len(), 2);
        assert_eq!(cfg.action_abstraction.preflop.raise.len(), 3);
        // Flop: 3 lead sizes, 1 raise depth
        assert_eq!(cfg.action_abstraction.flop.lead.len(), 3);
        assert_eq!(cfg.action_abstraction.flop.raise.len(), 1);
    }

    #[timed_test]
    fn game_config_validation_rejects_9_players() {
        let game = MpGameConfig {
            name: "Too many".into(),
            num_players: 9,
            stack_depth: 100.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_err());
    }

    #[timed_test]
    fn game_config_validation_rejects_1_player() {
        let game = MpGameConfig {
            name: "Solo".into(),
            num_players: 1,
            stack_depth: 100.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_err());
    }

    #[timed_test]
    fn total_forced_bets_computed() {
        let game = MpGameConfig {
            name: "Forced bets".into(),
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
        assert!((game.total_forced_bets() - 5.0).abs() < f64::EPSILON);
    }

    #[timed_test]
    fn bucket_counts_array() {
        let clustering = MpClusteringConfig {
            preflop: MpStreetCluster { buckets: 169 },
            flop: MpStreetCluster { buckets: 200 },
            turn: MpStreetCluster { buckets: 300 },
            river: MpStreetCluster { buckets: 400 },
        };
        assert_eq!(clustering.bucket_counts(), [169, 200, 300, 400]);
    }

    #[timed_test]
    fn validation_accepts_2_players() {
        let game = MpGameConfig {
            name: "Min".into(),
            num_players: 2,
            stack_depth: 50.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_ok());
    }

    #[timed_test]
    fn validation_accepts_8_players() {
        let game = MpGameConfig {
            name: "Max".into(),
            num_players: 8,
            stack_depth: 200.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_ok());
    }

    #[timed_test]
    fn validation_rejects_zero_stack_depth() {
        let game = MpGameConfig {
            name: "Zero stack".into(),
            num_players: 2,
            stack_depth: 0.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_err());
    }

    #[timed_test]
    fn validation_rejects_blind_seat_out_of_range() {
        let game = MpGameConfig {
            name: "Bad seat".into(),
            num_players: 2,
            stack_depth: 100.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 5, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_err());
    }

    #[timed_test]
    fn validation_rejects_zero_blind_amount() {
        let game = MpGameConfig {
            name: "Zero blind".into(),
            num_players: 2,
            stack_depth: 100.0,
            blinds: vec![
                ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 0.0 },
                ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!(game.validate().is_err());
    }

    #[timed_test]
    fn training_defaults_applied() {
        let yaml = "{}";
        let cfg: MpTrainingConfig =
            serde_yaml::from_str(yaml).expect("failed to parse empty training");

        assert_eq!(cfg.lcfr_warmup_iterations, 5_000_000);
        assert_eq!(cfg.lcfr_discount_interval, 500_000);
        assert_eq!(cfg.prune_after_iterations, 5_000_000);
        assert_eq!(cfg.prune_threshold, -250);
        assert!((cfg.prune_explore_pct - 0.05).abs() < f64::EPSILON);
        assert_eq!(cfg.batch_size, 200);
        assert!((cfg.dcfr_alpha - 1.5).abs() < f64::EPSILON);
        assert!((cfg.dcfr_beta).abs() < f64::EPSILON);
        assert!((cfg.dcfr_gamma - 2.0).abs() < f64::EPSILON);
        assert_eq!(cfg.print_every_minutes, 10);
        assert!((cfg.purify_threshold).abs() < f64::EPSILON);
        assert_eq!(cfg.exploitability_interval_minutes, 0);
        assert_eq!(cfg.exploitability_samples, 100_000);
    }

    #[timed_test]
    fn snapshot_resume_defaults_false() {
        let yaml = r#"
warmup_minutes: 10
snapshot_every_minutes: 5
output_dir: "/tmp/out"
"#;
        let cfg: MpSnapshotConfig =
            serde_yaml::from_str(yaml).expect("failed to parse snapshot config");

        assert!(!cfg.resume);
        assert!(cfg.max_snapshots.is_none());
    }

    #[timed_test]
    fn forced_bet_kind_all_variants_deserialize() {
        for (yaml_kind, expected) in [
            ("small_blind", ForcedBetKind::SmallBlind),
            ("big_blind", ForcedBetKind::BigBlind),
            ("ante", ForcedBetKind::Ante),
            ("bb_ante", ForcedBetKind::BbAnte),
            ("straddle", ForcedBetKind::Straddle),
        ] {
            let yaml = format!(
                "seat: 0\ntype: {yaml_kind}\namount: 1.0"
            );
            let bet: ForcedBet =
                serde_yaml::from_str(&yaml).expect("failed to parse forced bet");
            assert_eq!(bet.kind, expected, "mismatch for {yaml_kind}");
        }
    }

    #[timed_test]
    fn total_forced_bets_empty_blinds() {
        let game = MpGameConfig {
            name: "No blinds".into(),
            num_players: 2,
            stack_depth: 100.0,
            blinds: vec![],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        assert!((game.total_forced_bets()).abs() < f64::EPSILON);
    }

    #[timed_test]
    fn rake_defaults_to_zero() {
        let yaml = r#"
game:
  name: "No rake"
  num_players: 2
  stack_depth: 100.0
  blinds:
    - seat: 0
      type: small_blind
      amount: 1.0
    - seat: 1
      type: big_blind
      amount: 2.0

action_abstraction:
  preflop:
    lead: [1.0]
    raise:
      - [1.0]
  flop:
    lead: [1.0]
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
  warmup_minutes: 10
  snapshot_every_minutes: 5
  output_dir: "/tmp/norake"
"#;
        let cfg: BlueprintMpConfig =
            serde_yaml::from_str(yaml).expect("failed to parse no-rake config");
        assert!((cfg.game.rake_rate).abs() < f64::EPSILON);
        assert!((cfg.game.rake_cap).abs() < f64::EPSILON);
    }
}
