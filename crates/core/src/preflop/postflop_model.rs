/// Configuration for the postflop model used by the preflop solver.
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

fn default_bet_sizes() -> Vec<f32> {
    vec![0.5, 1.0]
}
fn default_max_raises_per_street() -> u8 {
    1
}
fn default_postflop_solve_iterations() -> u32 {
    500
}
fn default_postflop_sprs() -> Vec<f64> {
    vec![3.5]
}
fn default_max_flop_boards() -> usize {
    0
}
fn default_cfr_convergence_threshold() -> f64 {
    0.01
}

fn default_solve_type() -> PostflopSolveType {
    PostflopSolveType::Mccfr
}
fn default_mccfr_sample_pct() -> f64 {
    0.01
}
fn default_value_extraction_samples() -> u32 {
    10_000
}
fn default_ev_convergence_threshold() -> f64 {
    0.001
}

/// Deserialize either a scalar `f64` or a `Vec<f64>` into `Vec<f64>`.
/// Supports backward-compatible YAML: `postflop_spr: 4.0` → `vec![4.0]`.
fn deserialize_sprs<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ScalarOrVec {
        Scalar(f64),
        Vec(Vec<f64>),
    }
    match ScalarOrVec::deserialize(deserializer)? {
        ScalarOrVec::Scalar(v) => Ok(vec![v]),
        ScalarOrVec::Vec(v) => {
            if v.is_empty() {
                Err(de::Error::custom("postflop_sprs must not be empty"))
            } else {
                Ok(v)
            }
        }
    }
}

/// Selects the postflop solve backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PostflopSolveType {
    /// MCCFR with sampled concrete hands and real showdown eval.
    Mccfr,
    /// Exhaustive vanilla CFR with pre-computed equity tables.
    Exhaustive,
}

/// Configuration for the postflop model integrated into the preflop solver.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostflopModelConfig {
    // Postflop tree structure
    #[serde(default = "default_bet_sizes")]
    pub bet_sizes: Vec<f32>,
    #[serde(default = "default_max_raises_per_street", alias = "raises_per_street")]
    pub max_raises_per_street: u8,

    // Postflop solve
    #[serde(default = "default_postflop_solve_iterations")]
    pub postflop_solve_iterations: u32,

    /// SPR values for postflop solves. A tree template is built per SPR.
    /// Accepts both `postflop_sprs: [3.5, 6.0]` and the legacy scalar
    /// `postflop_spr: 3.5` (deserialized as a single-element vec).
    #[serde(
        default = "default_postflop_sprs",
        alias = "postflop_spr",
        deserialize_with = "deserialize_sprs"
    )]
    pub postflop_sprs: Vec<f64>,

    /// Convergence threshold for early per-flop CFR stopping.
    /// MCCFR solver: strategy delta (max probability change).
    /// Exhaustive solver: exploitability (pot fraction).
    #[serde(
        default = "default_cfr_convergence_threshold",
        alias = "cfr_convergence_threshold",
        alias = "cfr_regret_threshold",
        alias = "cfr_delta_threshold"
    )]
    pub cfr_convergence_threshold: f64,

    /// Maximum number of canonical flop boards to solve.
    /// 0 means use all canonical flops (~1,755).
    #[serde(default = "default_max_flop_boards")]
    pub max_flop_boards: usize,

    /// Optional list of explicit flop boards (e.g. `["AhKd2s", "7c8c9c"]`).
    /// When set, overrides `max_flop_boards` entirely.
    #[serde(default)]
    pub fixed_flops: Option<Vec<String>>,

    /// Selects the postflop solve backend.
    #[serde(default = "default_solve_type")]
    pub solve_type: PostflopSolveType,

    /// Fraction of total (hand_pair × turn × river) sample space per flop.
    /// Only used when solve_type is Mccfr. Default: 0.01 (1%).
    #[serde(default = "default_mccfr_sample_pct")]
    pub mccfr_sample_pct: f64,

    /// Number of Monte Carlo samples for post-convergence value extraction.
    /// Only used when solve_type is Mccfr. Default: 10,000.
    #[serde(default = "default_value_extraction_samples")]
    pub value_extraction_samples: u32,

    /// Early-stop threshold for EV estimation (weighted-average delta).
    /// Only used when solve_type is Mccfr. Default: 0.001.
    #[serde(default = "default_ev_convergence_threshold")]
    pub ev_convergence_threshold: f64,
}

impl PostflopModelConfig {
    /// Fast preset: quick MCCFR testing with limited boards.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            solve_type: PostflopSolveType::Mccfr,
            bet_sizes: vec![0.5, 1.0],
            max_raises_per_street: 1,
            postflop_solve_iterations: 100,
            postflop_sprs: vec![3.5],
            cfr_convergence_threshold: 0.01,
            max_flop_boards: 10,
            fixed_flops: None,
            mccfr_sample_pct: 0.05,
            value_extraction_samples: 1_000,
            ev_convergence_threshold: 0.001,
        }
    }

    /// Standard preset: balanced MCCFR accuracy and speed.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            solve_type: PostflopSolveType::Mccfr,
            bet_sizes: vec![0.5, 1.0],
            max_raises_per_street: 1,
            postflop_solve_iterations: 500,
            postflop_sprs: vec![3.5],
            cfr_convergence_threshold: 0.01,
            max_flop_boards: 0,
            fixed_flops: None,
            mccfr_sample_pct: 0.01,
            value_extraction_samples: 10_000,
            ev_convergence_threshold: 0.001,
        }
    }

    /// Exhaustive fast preset: quick exhaustive CFR with limited boards.
    #[must_use]
    pub fn exhaustive_fast() -> Self {
        Self {
            solve_type: PostflopSolveType::Exhaustive,
            postflop_solve_iterations: 200,
            max_flop_boards: 10,
            ..Self::standard()
        }
    }

    /// Exhaustive standard preset: full exhaustive CFR over all boards.
    #[must_use]
    pub fn exhaustive_standard() -> Self {
        Self {
            solve_type: PostflopSolveType::Exhaustive,
            postflop_solve_iterations: 1000,
            ..Self::standard()
        }
    }

    /// Parse a preset name into a config, or `None` for unknown names.
    #[must_use]
    pub fn from_preset(name: &str) -> Option<Self> {
        match name {
            "fast" => Some(Self::fast()),
            "standard" => Some(Self::standard()),
            "exhaustive_fast" => Some(Self::exhaustive_fast()),
            "exhaustive_standard" => Some(Self::exhaustive_standard()),
            _ => None,
        }
    }

    /// The primary (first) SPR value, used as the default for single-SPR solves.
    #[must_use]
    pub fn primary_spr(&self) -> f64 {
        self.postflop_sprs.first().copied().unwrap_or(3.5)
    }
}

impl Default for PostflopModelConfig {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn standard_preset_has_expected_defaults() {
        let cfg = PostflopModelConfig::standard();
        assert_eq!(cfg.solve_type, PostflopSolveType::Mccfr);
        assert_eq!(cfg.bet_sizes, vec![0.5, 1.0]);
        assert_eq!(cfg.max_raises_per_street, 1);
        assert_eq!(cfg.postflop_sprs, vec![3.5]);
    }

    #[timed_test]
    fn fast_preset_uses_mccfr() {
        let cfg = PostflopModelConfig::fast();
        assert_eq!(cfg.solve_type, PostflopSolveType::Mccfr);
        assert_eq!(cfg.max_flop_boards, 10);
        assert!((cfg.mccfr_sample_pct - 0.05).abs() < 1e-9);
    }

    #[timed_test]
    fn exhaustive_fast_preset() {
        let cfg = PostflopModelConfig::exhaustive_fast();
        assert_eq!(cfg.solve_type, PostflopSolveType::Exhaustive);
        assert_eq!(cfg.max_flop_boards, 10);
    }

    #[timed_test]
    fn exhaustive_standard_preset() {
        let cfg = PostflopModelConfig::exhaustive_standard();
        assert_eq!(cfg.solve_type, PostflopSolveType::Exhaustive);
    }

    #[timed_test]
    fn default_impl_equals_standard() {
        let default = PostflopModelConfig::default();
        let standard = PostflopModelConfig::standard();
        assert_eq!(default, standard);
    }

    #[timed_test]
    fn serde_round_trip_standard() {
        let cfg = PostflopModelConfig::standard();
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let restored: PostflopModelConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(cfg, restored);
    }

    #[timed_test]
    fn serde_round_trip_fast() {
        let cfg = PostflopModelConfig::fast();
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let restored: PostflopModelConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(cfg, restored);
    }

    #[timed_test]
    fn solve_type_mccfr_deserializes() {
        let yaml = "solve_type: mccfr\nmccfr_sample_pct: 0.05";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.solve_type, PostflopSolveType::Mccfr);
    }

    #[timed_test]
    fn solve_type_exhaustive_deserializes() {
        let yaml = "solve_type: exhaustive";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.solve_type, PostflopSolveType::Exhaustive);
    }

    #[timed_test]
    fn old_yaml_with_bucket_fields_still_deserializes() {
        let yaml = r"
num_hand_buckets_flop: 15
num_hand_buckets_turn: 300
num_hand_buckets_river: 300
rebucket_rounds: 3
equity_rollout_fraction: 0.5
postflop_solve_samples: 100
";
        // Old unknown fields should be silently ignored by serde_yaml
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        // Should get defaults for all real fields
        assert_eq!(cfg.solve_type, PostflopSolveType::Mccfr);
    }

    #[timed_test]
    fn cfr_convergence_threshold_defaults() {
        let cfg = PostflopModelConfig::standard();
        assert!((cfg.cfr_convergence_threshold - 0.01).abs() < 1e-9);
    }

    #[timed_test]
    fn postflop_sprs_defaults_to_single_value() {
        let cfg = PostflopModelConfig::standard();
        assert_eq!(cfg.postflop_sprs.len(), 1);
    }

    #[timed_test]
    fn postflop_spr_scalar_yaml_deserializes_as_vec() {
        let yaml = "postflop_spr: 4.0";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.postflop_sprs.len(), 1);
        assert!((cfg.postflop_sprs[0] - 4.0).abs() < 1e-9);
    }

    #[timed_test]
    fn postflop_sprs_vec_yaml_deserializes() {
        let yaml = "postflop_sprs: [3.5, 6.0]";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.postflop_sprs.len(), 2);
    }

    #[timed_test]
    fn primary_spr_returns_first_element() {
        let mut cfg = PostflopModelConfig::standard();
        cfg.postflop_sprs = vec![6.0, 3.5];
        assert!((cfg.primary_spr() - 6.0).abs() < 1e-9);
    }

    #[timed_test]
    fn fixed_flops_deserializes_from_yaml() {
        let yaml = r#"
fixed_flops:
  - "AhKd2s"
  - "7c8c9c"
"#;
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        let flops = cfg.fixed_flops.unwrap();
        assert_eq!(flops.len(), 2);
    }

    #[timed_test]
    fn from_preset_fast() {
        let cfg = PostflopModelConfig::from_preset("fast").unwrap();
        assert_eq!(cfg.solve_type, PostflopSolveType::Mccfr);
    }

    #[timed_test]
    fn from_preset_exhaustive_fast() {
        let cfg = PostflopModelConfig::from_preset("exhaustive_fast").unwrap();
        assert_eq!(cfg.solve_type, PostflopSolveType::Exhaustive);
    }

    #[timed_test]
    fn from_preset_unknown_returns_none() {
        assert!(PostflopModelConfig::from_preset("nonexistent").is_none());
    }
}
