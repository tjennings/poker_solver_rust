/// Configuration for the abstracted postflop model used by the preflop solver.
///
/// Controls hand abstraction granularity (EHS k-means buckets), postflop betting structure,
/// and per-iteration sampling counts.
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

fn default_num_hand_buckets_flop() -> u16 {
    20
}
fn default_num_hand_buckets_turn() -> u16 {
    500
}
fn default_num_hand_buckets_river() -> u16 {
    500
}
fn default_bet_sizes() -> Vec<f32> {
    vec![0.5, 1.0]
}
fn default_max_raises_per_street() -> u8 {
    1
}
fn default_flop_samples_per_iter() -> u16 {
    1
}
fn default_postflop_solve_iterations() -> u32 {
    200
}
fn default_postflop_solve_samples() -> u32 {
    0
}
fn default_postflop_sprs() -> Vec<f64> {
    vec![3.5]
}
fn default_max_flop_boards() -> usize {
    0
}
fn default_equity_rollout_fraction() -> f64 {
    1.0
}
fn default_rebucket_rounds() -> u16 {
    1
}
fn default_rebucket_delta_threshold() -> f64 {
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

/// Configuration for the postflop model integrated into the preflop solver.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostflopModelConfig {
    // Hand abstraction (EHS k-means)
    #[serde(default = "default_num_hand_buckets_flop")]
    pub num_hand_buckets_flop: u16,
    #[serde(default = "default_num_hand_buckets_turn")]
    pub num_hand_buckets_turn: u16,
    #[serde(default = "default_num_hand_buckets_river")]
    pub num_hand_buckets_river: u16,

    // Postflop tree structure
    #[serde(default = "default_bet_sizes")]
    pub bet_sizes: Vec<f32>,
    #[serde(default = "default_max_raises_per_street", alias = "raises_per_street")]
    pub max_raises_per_street: u8,

    // Sampling
    #[serde(default = "default_flop_samples_per_iter")]
    pub flop_samples_per_iter: u16,

    // Postflop solve (chance-sampled MCCFR)
    #[serde(default = "default_postflop_solve_iterations")]
    pub postflop_solve_iterations: u32,
    /// Bucket pairs sampled per MCCFR iteration. 0 = use `num_hand_buckets_flop`.
    #[serde(default = "default_postflop_solve_samples")]
    pub postflop_solve_samples: u32,

    /// SPR values for postflop solves. A tree template is built per SPR.
    /// Accepts both `postflop_sprs: [3.5, 6.0]` and the legacy scalar
    /// `postflop_spr: 3.5` (deserialized as a single-element vec).
    #[serde(
        default = "default_postflop_sprs",
        alias = "postflop_spr",
        deserialize_with = "deserialize_sprs"
    )]
    pub postflop_sprs: Vec<f64>,

    /// Number of EV-based rebucketing rounds. 1 = no rebucketing (backward compat).
    #[serde(default = "default_rebucket_rounds")]
    pub rebucket_rounds: u16,

    /// Max-regret-delta threshold for early CFR stopping during rebucketing.
    #[serde(default = "default_rebucket_delta_threshold")]
    pub rebucket_delta_threshold: f64,

    /// Maximum number of canonical flop boards to use for EHS feature computation.
    /// 0 means use all canonical flops (~1,755). Lower values dramatically speed up
    /// the hand bucketing phase at the cost of clustering quality.
    #[serde(default = "default_max_flop_boards")]
    pub max_flop_boards: usize,

    /// Optional list of explicit flop boards (e.g. `["AhKd2s", "7c8c9c"]`).
    /// When set, overrides `max_flop_boards` entirely.
    #[serde(default)]
    pub fixed_flops: Option<Vec<String>>,

    /// Fraction of runouts to evaluate per hand pair for pairwise equity on flop boards.
    /// 1.0 = exhaustive enumeration (exact). 0.1 = sample 10% of runouts (~99 of ~990).
    /// Turn and river equity is always exact regardless of this setting.
    #[serde(default = "default_equity_rollout_fraction", alias = "equity_rollout_samples")]
    pub equity_rollout_fraction: f64,
}

impl PostflopModelConfig {
    /// Fast preset: minimal buckets for quick testing (~2 min build).
    #[must_use]
    pub fn fast() -> Self {
        Self {
            num_hand_buckets_flop: 10,
            num_hand_buckets_turn: 50,
            num_hand_buckets_river: 50,
            max_flop_boards: 10,
            ..Self::standard()
        }
    }

    /// Medium preset: practical quality with reasonable build time (~5 min).
    #[must_use]
    pub fn medium() -> Self {
        Self {
            num_hand_buckets_flop: 15,
            num_hand_buckets_turn: 200,
            num_hand_buckets_river: 200,
            max_flop_boards: 200,
            ..Self::standard()
        }
    }

    /// Standard preset: balanced accuracy and speed (~30 min).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            num_hand_buckets_flop: 20,
            num_hand_buckets_turn: 500,
            num_hand_buckets_river: 500,
            bet_sizes: vec![0.5, 1.0],
            max_raises_per_street: 1,
            flop_samples_per_iter: 1,
            postflop_solve_iterations: 200,
            postflop_solve_samples: 0,
            postflop_sprs: vec![3.5],
            rebucket_rounds: 1,
            rebucket_delta_threshold: 0.001,
            max_flop_boards: 0,
            fixed_flops: None,
            equity_rollout_fraction: 1.0,
        }
    }

    /// Accurate preset: high-fidelity abstraction (~2 hrs).
    #[must_use]
    pub fn accurate() -> Self {
        Self {
            num_hand_buckets_flop: 30,
            num_hand_buckets_turn: 1000,
            num_hand_buckets_river: 1000,
            ..Self::standard()
        }
    }

    /// Parse a preset name into a config, or `None` for unknown names.
    #[must_use]
    pub fn from_preset(name: &str) -> Option<Self> {
        match name {
            "fast" => Some(Self::fast()),
            "medium" => Some(Self::medium()),
            "standard" => Some(Self::standard()),
            "accurate" => Some(Self::accurate()),
            _ => None,
        }
    }

    /// The primary (first) SPR value, used as the default for single-SPR solves.
    #[must_use]
    pub fn primary_spr(&self) -> f64 {
        self.postflop_sprs.first().copied().unwrap_or(3.5)
    }

    /// Total number of hand buckets across all streets.
    #[must_use]
    pub fn total_hand_buckets(&self) -> u32 {
        u32::from(self.num_hand_buckets_flop)
            + u32::from(self.num_hand_buckets_turn)
            + u32::from(self.num_hand_buckets_river)
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
        assert_eq!(cfg.num_hand_buckets_flop, 20);
        assert_eq!(cfg.num_hand_buckets_turn, 500);
        assert_eq!(cfg.num_hand_buckets_river, 500);
        assert_eq!(cfg.bet_sizes, vec![0.5, 1.0]);
        assert_eq!(cfg.max_raises_per_street, 1);
        assert_eq!(cfg.postflop_sprs, vec![3.5]);
    }

    #[timed_test]
    fn fast_preset_has_fewer_buckets() {
        let fast = PostflopModelConfig::fast();
        let std = PostflopModelConfig::standard();
        assert!(fast.num_hand_buckets_flop < std.num_hand_buckets_flop);
        assert!(fast.num_hand_buckets_turn < std.num_hand_buckets_turn);
        assert!(fast.num_hand_buckets_river < std.num_hand_buckets_river);
    }

    #[timed_test]
    fn accurate_preset_has_more_buckets() {
        let acc = PostflopModelConfig::accurate();
        let std = PostflopModelConfig::standard();
        assert!(acc.num_hand_buckets_flop > std.num_hand_buckets_flop);
        assert!(acc.num_hand_buckets_turn > std.num_hand_buckets_turn);
        assert!(acc.num_hand_buckets_river > std.num_hand_buckets_river);
    }

    #[timed_test]
    fn total_hand_buckets_sums_all_streets() {
        let cfg = PostflopModelConfig::standard();
        assert_eq!(cfg.total_hand_buckets(), 20 + 500 + 500);
    }

    #[timed_test]
    fn fast_total_hand_buckets_sums_all_streets() {
        let cfg = PostflopModelConfig::fast();
        assert_eq!(cfg.total_hand_buckets(), 10 + 50 + 50);
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
    fn rebucket_rounds_defaults_to_one() {
        let cfg = PostflopModelConfig::standard();
        assert_eq!(cfg.rebucket_rounds, 1);
    }

    #[timed_test]
    fn rebucket_delta_threshold_defaults() {
        let cfg = PostflopModelConfig::standard();
        assert!((cfg.rebucket_delta_threshold - 0.001).abs() < 1e-9);
    }

    #[timed_test]
    fn postflop_sprs_defaults_to_single_value() {
        let cfg = PostflopModelConfig::standard();
        assert_eq!(cfg.postflop_sprs.len(), 1);
        assert!((cfg.postflop_sprs[0] - 3.5).abs() < 1e-9);
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
        assert!((cfg.postflop_sprs[0] - 3.5).abs() < 1e-9);
        assert!((cfg.postflop_sprs[1] - 6.0).abs() < 1e-9);
    }

    #[timed_test]
    fn rebucket_rounds_round_trip() {
        let mut cfg = PostflopModelConfig::fast();
        cfg.rebucket_rounds = 3;
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let restored: PostflopModelConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.rebucket_rounds, 3);
    }

    #[timed_test]
    fn primary_spr_returns_first_element() {
        let mut cfg = PostflopModelConfig::standard();
        cfg.postflop_sprs = vec![6.0, 3.5];
        assert!((cfg.primary_spr() - 6.0).abs() < 1e-9);
    }

    #[timed_test]
    fn old_yaml_with_removed_fields_still_deserializes() {
        let yaml = r"
num_flop_textures: 100
num_turn_transitions: 5
num_river_transitions: 5
ehs_samples: 500
num_hand_buckets_flop: 15
num_hand_buckets_turn: 300
num_hand_buckets_river: 300
";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.num_hand_buckets_flop, 15);
    }

    #[timed_test]
    fn medium_preset_has_expected_buckets() {
        let cfg = PostflopModelConfig::medium();
        assert_eq!(cfg.num_hand_buckets_flop, 15);
        assert_eq!(cfg.num_hand_buckets_turn, 200);
        assert_eq!(cfg.num_hand_buckets_river, 200);
    }

    #[timed_test]
    fn accurate_preset_has_expected_buckets() {
        let cfg = PostflopModelConfig::accurate();
        assert_eq!(cfg.num_hand_buckets_flop, 30);
        assert_eq!(cfg.num_hand_buckets_turn, 1000);
        assert_eq!(cfg.num_hand_buckets_river, 1000);
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
        assert_eq!(flops[0], "AhKd2s");
        assert_eq!(flops[1], "7c8c9c");
    }

    #[timed_test]
    fn fixed_flops_defaults_to_none() {
        let yaml = "num_hand_buckets_flop: 100";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.fixed_flops.is_none());
    }
}
