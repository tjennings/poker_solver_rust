use serde::{Deserialize, Serialize};

use super::postflop_model::PostflopModelConfig;

/// A raise size expressed as either a fixed BB amount or a fraction of pot.
///
/// Serialized as tagged strings: `"2.5bb"` for BB amounts, `"0.75p"` for pot fractions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RaiseSize {
    /// Fixed BB amount — raise TO this many big blinds.
    Bb(f64),
    /// Pot fraction — raise BY `fraction × pot_after_call`.
    PotFraction(f64),
}

const BB_SIZE: u32 = 2;

impl RaiseSize {
    /// Resolve this raise size to a concrete raise-to amount in internal units.
    ///
    /// - `Bb(x)`: raise to `x * bb_size`
    /// - `PotFraction(f)`: raise to `current_bet + f * (pot + to_call)`
    ///
    /// Result is clamped to at least `current_bet + 1` (min-raise).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn resolve(self, current_bet: u32, pot: u32, to_call: u32) -> u32 {
        let raise_to = match self {
            Self::Bb(x) => (x * f64::from(BB_SIZE)) as u32,
            Self::PotFraction(f) => {
                let pot_after_call = pot + to_call;
                let raise_amount = (f * f64::from(pot_after_call)) as u32;
                current_bet + raise_amount
            }
        };
        raise_to.max(current_bet + 1)
    }
}

impl std::fmt::Display for RaiseSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bb(x) => write!(f, "{x}bb"),
            Self::PotFraction(x) => write!(f, "{x}p"),
        }
    }
}

impl Serialize for RaiseSize {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for RaiseSize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RaiseSizeVisitor;

        impl<'de> serde::de::Visitor<'de> for RaiseSizeVisitor {
            type Value = RaiseSize;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str(
                    "a string like \"2.5bb\" or \"0.75p\", or a plain number (pot fraction)",
                )
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Ok(RaiseSize::PotFraction(v))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                self.visit_f64(v as f64)
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                self.visit_f64(v as f64)
            }

            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Self::Value, E> {
                if let Some(num) = s.strip_suffix("bb") {
                    let val: f64 = num.parse().map_err(serde::de::Error::custom)?;
                    Ok(RaiseSize::Bb(val))
                } else if let Some(num) = s.strip_suffix('p') {
                    let val: f64 = num.parse().map_err(serde::de::Error::custom)?;
                    Ok(RaiseSize::PotFraction(val))
                } else if let Ok(val) = s.parse::<f64>() {
                    // Backward compat: plain float string → pot fraction
                    Ok(RaiseSize::PotFraction(val))
                } else {
                    Err(serde::de::Error::custom(format!(
                        "invalid raise size '{s}': must end with 'bb' or 'p', or be a plain number"
                    )))
                }
            }
        }

        deserializer.deserialize_any(RaiseSizeVisitor)
    }
}

/// Which CFR variant the preflop solver should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CfrVariant {
    /// Standard CFR: no discounting, uniform iteration weighting.
    Vanilla,
    /// Discounted CFR with α/β/γ discounting and linear (LCFR) iteration weighting.
    #[default]
    Dcfr,
    /// CFR+ (Tammelin 2014): regrets floored to zero, linear strategy weighting.
    CfrPlus,
    /// DCFR with all exponents = 1.0 (Pluribus linear weighting).
    Linear,
}

fn default_exploration() -> f64 {
    0.05
}

/// Information about a position at the table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    pub name: String,
    pub short_name: String,
}

/// Configuration for a preflop solver instance.
///
/// Defines the game structure: positions, blinds, antes, stacks, and raise sizing.
/// Internal units: SB = 1, BB = 2. A 100 BB stack is 200 internal units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflopConfig {
    pub positions: Vec<PositionInfo>,
    /// (`position_idx`, amount) pairs for blind posts.
    pub blinds: Vec<(usize, u32)>,
    /// (`position_idx`, amount) pairs for antes.
    pub antes: Vec<(usize, u32)>,
    /// Per-position stack sizes in internal units (1 SB = 1).
    pub stacks: Vec<u32>,
    /// Raise sizes indexed by raise depth (fallback).
    ///
    /// Each entry is `"Xbb"` (raise TO X big blinds) or `"Xp"` (raise BY X * pot).
    pub raise_sizes: Vec<Vec<RaiseSize>>,
    /// Per-position raise size overrides indexed as `[position][depth][size_idx]`.
    /// When set, overrides `raise_sizes` for the given position and depth.
    pub position_raise_sizes: Option<Vec<Vec<Vec<RaiseSize>>>>,
    /// Maximum number of raises allowed per round.
    pub raise_cap: u8,
    /// CFR variant: `vanilla`, `dcfr` (default), or `cfrplus`.
    #[serde(default)]
    pub cfr_variant: CfrVariant,
    /// DCFR positive regret discount exponent.
    pub dcfr_alpha: f64,
    /// DCFR negative regret discount exponent.
    pub dcfr_beta: f64,
    /// DCFR strategy sum discount exponent.
    pub dcfr_gamma: f64,
    /// Number of initial iterations without DCFR discounting (warm-up phase).
    ///
    /// During warmup, regrets and strategy sums accumulate without multiplicative
    /// discounting.  This ensures all subtrees receive signal from the initial
    /// uniform-strategy phase before DCFR begins washing out early iterations.
    pub dcfr_warmup: u64,
    /// Exploration factor (ε-greedy). Each action gets at least `exploration / num_actions`
    /// probability during traversal, ensuring off-path subtrees are visited.
    /// The average strategy accumulates the *intended* (non-explored) strategy, so
    /// exploration noise doesn't contaminate the output.
    /// Set to 0.0 for pure regret matching.  Typical value: 0.05.
    #[serde(default = "default_exploration")]
    pub exploration: f64,
    /// Optional postflop model for equity realization.
    ///
    /// When `None`, showdown terminals use raw hand equity (current behaviour).
    /// When `Some`, each preflop showdown terminal is replaced by a sampled
    /// abstracted postflop subtree, allowing the solver to discover equity
    /// realization endogenously.
    #[serde(default)]
    pub postflop_model: Option<PostflopModelConfig>,
}

impl PreflopConfig {
    /// Creates a heads-up configuration with the given stack depth in big blinds.
    #[must_use]
    pub fn heads_up(stack_depth_bb: u32) -> Self {
        let stacks_internal = stack_depth_bb * 2;
        Self {
            positions: vec![
                PositionInfo {
                    name: "Small Blind".into(),
                    short_name: "SB".into(),
                },
                PositionInfo {
                    name: "Big Blind".into(),
                    short_name: "BB".into(),
                },
            ],
            blinds: vec![(0, 1), (1, 2)],
            antes: vec![],
            stacks: vec![stacks_internal, stacks_internal],
            raise_sizes: vec![vec![RaiseSize::Bb(2.5)], vec![RaiseSize::Bb(3.0)]],
            position_raise_sizes: None,
            raise_cap: 4,
            cfr_variant: CfrVariant::Dcfr,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.05,
            postflop_model: None,
        }
    }

    /// Creates a six-max configuration with the given stack depth in big blinds.
    #[must_use]
    pub fn six_max(stack_depth_bb: u32) -> Self {
        let stacks_internal = stack_depth_bb * 2;
        Self {
            positions: vec![
                PositionInfo {
                    name: "Under the Gun".into(),
                    short_name: "UTG".into(),
                },
                PositionInfo {
                    name: "Hijack".into(),
                    short_name: "HJ".into(),
                },
                PositionInfo {
                    name: "Cutoff".into(),
                    short_name: "CO".into(),
                },
                PositionInfo {
                    name: "Button".into(),
                    short_name: "BTN".into(),
                },
                PositionInfo {
                    name: "Small Blind".into(),
                    short_name: "SB".into(),
                },
                PositionInfo {
                    name: "Big Blind".into(),
                    short_name: "BB".into(),
                },
            ],
            blinds: vec![(4, 1), (5, 2)],
            antes: vec![],
            stacks: vec![stacks_internal; 6],
            raise_sizes: vec![vec![RaiseSize::Bb(2.5)], vec![RaiseSize::Bb(3.0)]],
            position_raise_sizes: None,
            raise_cap: 4,
            cfr_variant: CfrVariant::Dcfr,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.05,
            postflop_model: None,
        }
    }

    /// Returns the number of players at the table.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn num_players(&self) -> u8 {
        // Safe: poker tables never exceed 10 players
        self.positions.len() as u8
    }

    /// Returns the raise sizes for a given position and raise depth.
    ///
    /// Uses `position_raise_sizes` if set for that position, otherwise falls
    /// back to `raise_sizes`.
    #[must_use]
    pub fn raise_sizes_for(&self, position: u8, depth: usize) -> &[RaiseSize] {
        if let Some(ref pos_sizes) = self.position_raise_sizes {
            let pos = position as usize;
            if pos < pos_sizes.len() && !pos_sizes[pos].is_empty() {
                let d = depth.min(pos_sizes[pos].len().saturating_sub(1));
                return &pos_sizes[pos][d];
            }
        }
        let d = depth.min(self.raise_sizes.len().saturating_sub(1));
        &self.raise_sizes[d]
    }

    /// Returns the initial pot size (sum of all blinds and antes).
    #[must_use]
    pub fn initial_pot(&self) -> u32 {
        let blind_total: u32 = self.blinds.iter().map(|(_, amt)| amt).sum();
        let ante_total: u32 = self.antes.iter().map(|(_, amt)| amt).sum();
        blind_total + ante_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn hu_config_has_two_positions() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.positions.len(), 2);
        assert_eq!(config.stacks.len(), 2);
        assert_eq!(config.stacks[0], 200);
        assert_eq!(config.stacks[1], 200);
    }

    #[timed_test]
    fn hu_config_blinds_are_sb_bb() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.blinds, vec![(0, 1), (1, 2)]);
    }

    #[timed_test]
    fn six_max_config_has_six_positions() {
        let config = PreflopConfig::six_max(100);
        assert_eq!(config.positions.len(), 6);
        assert_eq!(config.num_players(), 6);
        assert_eq!(config.stacks.len(), 6);
    }

    #[timed_test]
    fn hu_initial_pot_is_three() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.initial_pot(), 3); // SB(1) + BB(2) = 3
    }

    #[timed_test]
    fn six_max_initial_pot_is_three() {
        let config = PreflopConfig::six_max(100);
        assert_eq!(config.initial_pot(), 3);
    }

    #[timed_test]
    fn hu_config_has_dcfr_defaults() {
        let config = PreflopConfig::heads_up(100);
        assert!((config.dcfr_alpha - 1.5).abs() < f64::EPSILON);
        assert!((config.dcfr_beta - 0.5).abs() < f64::EPSILON);
        assert!((config.dcfr_gamma - 2.0).abs() < f64::EPSILON);
    }

    #[timed_test]
    fn hu_config_has_raise_sizes() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.raise_sizes.len(), 2, "should have 2 raise depths");
        assert_eq!(config.raise_sizes[0].len(), 1, "open should have 1 size");
        assert_eq!(config.raise_sizes[1].len(), 1, "3bet should have 1 size");
    }

    #[timed_test]
    fn hu_config_defaults_to_dcfr() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.cfr_variant, CfrVariant::Dcfr);
    }

    #[timed_test]
    fn cfr_variant_deserializes_from_yaml() {
        let yaml = r#"
            positions:
              - name: SB
                short_name: SB
              - name: BB
                short_name: BB
            blinds: [[0, 1], [1, 2]]
            antes: []
            stacks: [200, 200]
            raise_sizes: [["2.5bb"]]
            raise_cap: 4
            cfr_variant: vanilla
            dcfr_alpha: 1.5
            dcfr_beta: 0.5
            dcfr_gamma: 2.0
            dcfr_warmup: 0
        "#;
        let config: PreflopConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.cfr_variant, CfrVariant::Vanilla);
    }

    #[timed_test]
    fn cfr_variant_defaults_to_dcfr_when_omitted() {
        let yaml = r#"
            positions:
              - name: SB
                short_name: SB
              - name: BB
                short_name: BB
            blinds: [[0, 1], [1, 2]]
            antes: []
            stacks: [200, 200]
            raise_sizes: [["2.5bb"]]
            raise_cap: 4
            dcfr_alpha: 1.5
            dcfr_beta: 0.5
            dcfr_gamma: 2.0
            dcfr_warmup: 0
        "#;
        let config: PreflopConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.cfr_variant, CfrVariant::Dcfr);
    }

    #[timed_test]
    fn cfr_variant_cfrplus_deserializes() {
        let yaml = r#"
            positions:
              - name: SB
                short_name: SB
              - name: BB
                short_name: BB
            blinds: [[0, 1], [1, 2]]
            antes: []
            stacks: [200, 200]
            raise_sizes: [["2.5bb"]]
            raise_cap: 4
            cfr_variant: cfrplus
            dcfr_alpha: 1.5
            dcfr_beta: 0.5
            dcfr_gamma: 2.0
            dcfr_warmup: 0
        "#;
        let config: PreflopConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.cfr_variant, CfrVariant::CfrPlus);
    }

    #[timed_test]
    fn config_with_antes() {
        let mut config = PreflopConfig::heads_up(100);
        config.antes = vec![(0, 1), (1, 1)];
        assert_eq!(config.initial_pot(), 5); // SB(1) + BB(2) + ante(1) + ante(1) = 5
    }

    #[timed_test]
    fn raise_size_bb_resolve() {
        // Bb(2.5) at open: raise to 2.5 BB = 5 SB
        let size = RaiseSize::Bb(2.5);
        assert_eq!(size.resolve(2, 3, 1), 5); // current_bet=BB=2, pot=3, to_call=1
    }

    #[timed_test]
    fn raise_size_pot_fraction_resolve() {
        // PotFraction(0.75) at open: pot_after_call=3+1=4, raise_amount=3, raise_to=2+3=5
        let size = RaiseSize::PotFraction(0.75);
        assert_eq!(size.resolve(2, 3, 1), 5);
    }

    #[timed_test]
    fn raise_size_clamps_to_min_raise() {
        // Bb(0.5) resolves to 1 SB, clamped to current_bet+1 = 3
        let size = RaiseSize::Bb(0.5);
        assert_eq!(size.resolve(2, 3, 1), 3);
    }

    #[timed_test]
    fn raise_size_serde_roundtrip() {
        let bb = RaiseSize::Bb(2.5);
        let yaml = serde_yaml::to_string(&bb).unwrap();
        let parsed: RaiseSize = serde_yaml::from_str(yaml.trim()).unwrap();
        assert_eq!(parsed, bb);

        let pf = RaiseSize::PotFraction(0.75);
        let yaml = serde_yaml::to_string(&pf).unwrap();
        let parsed: RaiseSize = serde_yaml::from_str(yaml.trim()).unwrap();
        assert_eq!(parsed, pf);
    }

    #[timed_test]
    fn raise_size_deserialize_plain_float_as_pot_fraction() {
        let parsed: RaiseSize = serde_yaml::from_str("0.75").unwrap();
        assert_eq!(parsed, RaiseSize::PotFraction(0.75));
    }

    #[timed_test]
    fn raise_size_deserialize_plain_integer_as_pot_fraction() {
        let parsed: RaiseSize = serde_yaml::from_str("2.0").unwrap();
        assert_eq!(parsed, RaiseSize::PotFraction(2.0));
    }
}
