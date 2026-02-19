use serde::{Deserialize, Serialize};

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
    /// Raise size multipliers indexed by raise depth.
    pub raise_sizes: Vec<Vec<f64>>,
    /// Maximum number of raises allowed per round.
    pub raise_cap: u8,
    /// DCFR positive regret discount exponent.
    pub dcfr_alpha: f64,
    /// DCFR negative regret discount exponent.
    pub dcfr_beta: f64,
    /// DCFR strategy sum discount exponent.
    pub dcfr_gamma: f64,
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
            raise_sizes: vec![vec![2.0, 3.0], vec![2.5, 3.5], vec![2.5]],
            raise_cap: 4,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
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
            raise_sizes: vec![vec![2.0, 3.0], vec![2.5, 3.5], vec![2.5]],
            raise_cap: 4,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
        }
    }

    /// Returns the number of players at the table.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn num_players(&self) -> u8 {
        // Safe: poker tables never exceed 10 players
        self.positions.len() as u8
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
    fn hu_config_has_multiple_raise_sizes() {
        let config = PreflopConfig::heads_up(100);
        assert_eq!(config.raise_sizes[0].len(), 2, "open should have 2 sizes");
        assert_eq!(config.raise_sizes[1].len(), 2, "3bet should have 2 sizes");
        assert_eq!(config.raise_sizes[2].len(), 1, "4bet+ should have 1 size");
    }

    #[timed_test]
    fn config_with_antes() {
        let mut config = PreflopConfig::heads_up(100);
        config.antes = vec![(0, 1), (1, 1)];
        assert_eq!(config.initial_pot(), 5); // SB(1) + BB(2) + ante(1) + ante(1) = 5
    }
}
