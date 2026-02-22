/// Configuration for the abstracted postflop model used by the preflop solver.
///
/// Controls hand abstraction granularity (EHS k-means buckets), postflop betting structure,
/// and per-iteration sampling counts.
use serde::{Deserialize, Serialize};

fn default_num_hand_buckets_flop() -> u16 {
    500
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
fn default_raises_per_street() -> u8 {
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
fn default_canonical_sprs() -> Vec<f64> {
    vec![0.5, 1.0, 1.5, 3.0, 5.0, 10.0, 20.0, 50.0]
}
fn default_max_flop_boards() -> usize {
    0
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
    #[serde(default = "default_raises_per_street")]
    pub raises_per_street: u8,

    // Sampling
    #[serde(default = "default_flop_samples_per_iter")]
    pub flop_samples_per_iter: u16,

    // Postflop solve (chance-sampled MCCFR)
    #[serde(default = "default_postflop_solve_iterations")]
    pub postflop_solve_iterations: u32,
    /// Bucket pairs sampled per MCCFR iteration. 0 = use `num_hand_buckets_flop`.
    #[serde(default = "default_postflop_solve_samples")]
    pub postflop_solve_samples: u32,

    /// Canonical SPR values for postflop solving. One tree is built and solved
    /// per SPR value. At runtime, each pot type maps to the nearest canonical SPR.
    #[serde(default = "default_canonical_sprs")]
    pub canonical_sprs: Vec<f64>,

    /// Maximum number of canonical flop boards to use for EHS feature computation.
    /// 0 means use all canonical flops (~1,755). Lower values dramatically speed up
    /// the hand bucketing phase at the cost of clustering quality.
    #[serde(default = "default_max_flop_boards")]
    pub max_flop_boards: usize,
}

impl PostflopModelConfig {
    /// Fast preset: minimal buckets for quick testing (~30s build).
    #[must_use]
    pub fn fast() -> Self {
        Self {
            num_hand_buckets_flop: 50,
            num_hand_buckets_turn: 50,
            num_hand_buckets_river: 50,
            max_flop_boards: 200,
            ..Self::standard()
        }
    }

    /// Medium preset: practical quality with reasonable build time (~5 min).
    #[must_use]
    pub fn medium() -> Self {
        Self {
            num_hand_buckets_flop: 200,
            num_hand_buckets_turn: 200,
            num_hand_buckets_river: 200,
            max_flop_boards: 500,
            ..Self::standard()
        }
    }

    /// Standard preset: balanced accuracy and speed (~30 min).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            num_hand_buckets_flop: 500,
            num_hand_buckets_turn: 500,
            num_hand_buckets_river: 500,
            bet_sizes: vec![0.5, 1.0],
            raises_per_street: 1,
            flop_samples_per_iter: 1,
            postflop_solve_iterations: 200,
            postflop_solve_samples: 0,
            canonical_sprs: default_canonical_sprs(),
            max_flop_boards: 0,
        }
    }

    /// Accurate preset: high-fidelity abstraction (~2 hrs).
    #[must_use]
    pub fn accurate() -> Self {
        Self {
            num_hand_buckets_flop: 1000,
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
        assert_eq!(cfg.num_hand_buckets_flop, 500);
        assert_eq!(cfg.num_hand_buckets_turn, 500);
        assert_eq!(cfg.num_hand_buckets_river, 500);
        assert_eq!(cfg.bet_sizes, vec![0.5, 1.0]);
        assert_eq!(cfg.raises_per_street, 1);
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
        assert_eq!(cfg.total_hand_buckets(), 500 + 500 + 500);
    }

    #[timed_test]
    fn fast_total_hand_buckets_sums_all_streets() {
        let cfg = PostflopModelConfig::fast();
        assert_eq!(cfg.total_hand_buckets(), 50 + 50 + 50);
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
    fn default_canonical_sprs_has_eight_values() {
        let cfg = PostflopModelConfig::standard();
        assert_eq!(cfg.canonical_sprs.len(), 8);
        assert!((cfg.canonical_sprs[0] - 0.5).abs() < 1e-9);
        assert!((cfg.canonical_sprs[7] - 50.0).abs() < 1e-9);
    }

    #[timed_test]
    fn serde_round_trip_with_canonical_sprs() {
        let mut cfg = PostflopModelConfig::fast();
        cfg.canonical_sprs = vec![1.0, 5.0, 20.0];
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let restored: PostflopModelConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(cfg.canonical_sprs, restored.canonical_sprs);
    }

    #[timed_test]
    fn old_yaml_with_removed_fields_still_deserializes() {
        // Old configs may contain removed texture fields; serde should ignore them.
        let yaml = r"
num_flop_textures: 100
num_turn_transitions: 5
num_river_transitions: 5
ehs_samples: 500
num_hand_buckets_flop: 300
num_hand_buckets_turn: 300
num_hand_buckets_river: 300
";
        let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.num_hand_buckets_flop, 300);
    }

    #[timed_test]
    fn medium_preset_has_expected_buckets() {
        let cfg = PostflopModelConfig::medium();
        assert_eq!(cfg.num_hand_buckets_flop, 200);
        assert_eq!(cfg.num_hand_buckets_turn, 200);
        assert_eq!(cfg.num_hand_buckets_river, 200);
        assert_eq!(cfg.total_hand_buckets(), 600);
    }

    #[timed_test]
    fn accurate_preset_has_expected_buckets() {
        let cfg = PostflopModelConfig::accurate();
        assert_eq!(cfg.num_hand_buckets_flop, 1000);
        assert_eq!(cfg.num_hand_buckets_turn, 1000);
        assert_eq!(cfg.num_hand_buckets_river, 1000);
        assert_eq!(cfg.total_hand_buckets(), 3000);
    }
}
