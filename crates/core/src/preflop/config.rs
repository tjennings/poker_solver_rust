use serde::{Deserialize, Serialize};

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
    /// Raise size multipliers indexed by raise depth (fallback).
    pub raise_sizes: Vec<Vec<f64>>,
    /// Per-position raise size overrides indexed as `[position][depth][size_idx]`.
    /// When set, overrides `raise_sizes` for the given position and depth.
    pub position_raise_sizes: Option<Vec<Vec<Vec<f64>>>>,
    /// Maximum number of raises allowed per round.
    pub raise_cap: u8,
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
}

impl PreflopConfig {
    /// Creates a heads-up configuration with the given stack depth in big blinds.
    #[must_use]
    pub fn heads_up(stack_depth_bb: u32) -> Self {
        let stacks_internal = stack_depth_bb * 2;
        Self {
            positions: vec![
                PositionInfo { name: "Small Blind".into(), short_name: "SB".into() },
                PositionInfo { name: "Big Blind".into(), short_name: "BB".into() },
            ],
            blinds: vec![(0, 1), (1, 2)],
            antes: vec![],
            stacks: vec![stacks_internal, stacks_internal],
            raise_sizes: vec![vec![2.5], vec![3.0]],
            position_raise_sizes: None,
            raise_cap: 4,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.05,
        }
    }

    /// Creates a six-max configuration with the given stack depth in big blinds.
    #[must_use]
    pub fn six_max(stack_depth_bb: u32) -> Self {
        let stacks_internal = stack_depth_bb * 2;
        Self {
            positions: vec![
                PositionInfo { name: "Under the Gun".into(), short_name: "UTG".into() },
                PositionInfo { name: "Hijack".into(), short_name: "HJ".into() },
                PositionInfo { name: "Cutoff".into(), short_name: "CO".into() },
                PositionInfo { name: "Button".into(), short_name: "BTN".into() },
                PositionInfo { name: "Small Blind".into(), short_name: "SB".into() },
                PositionInfo { name: "Big Blind".into(), short_name: "BB".into() },
            ],
            blinds: vec![(4, 1), (5, 2)],
            antes: vec![],
            stacks: vec![stacks_internal; 6],
            raise_sizes: vec![vec![2.5], vec![3.0]],
            position_raise_sizes: None,
            raise_cap: 4,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.05,
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
    pub fn raise_sizes_for(&self, position: u8, depth: usize) -> &[f64] {
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
    fn config_with_antes() {
        let mut config = PreflopConfig::heads_up(100);
        config.antes = vec![(0, 1), (1, 1)];
        assert_eq!(config.initial_pot(), 5); // SB(1) + BB(2) + ante(1) + ante(1) = 5
    }
}
