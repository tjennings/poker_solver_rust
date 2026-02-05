//! Hand Strength Metrics for Card Abstraction
//!
//! Provides types for measuring hand strength including Expected Hand Strength (EHS),
//! potential metrics, and the EHS2 formula that combines current strength with
//! hand potential.

/// Hand strength metrics for poker hand evaluation.
///
/// This struct captures both the current hand strength and the potential
/// for improvement or degradation as more cards are dealt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandStrength {
    /// Expected Hand Strength - equity vs random opponent now
    pub ehs: f32,
    /// Positive potential - P(behind now, ahead after runout)
    pub ppot: f32,
    /// Negative potential - P(ahead now, behind after runout)
    pub npot: f32,
    /// `EHS2 = EHS + (1-EHS)*PPot - EHS*NPot`
    pub ehs2: f32,
}

impl HandStrength {
    /// Create a new `HandStrength` with all metrics.
    ///
    /// Computes EHS2 automatically from the provided values using:
    /// `EHS2 = EHS + (1-EHS)*PPot - EHS*NPot`
    ///
    /// # Arguments
    /// * `ehs` - Expected Hand Strength (current equity)
    /// * `ppot` - Positive potential (probability of improving from behind)
    /// * `npot` - Negative potential (probability of falling behind from ahead)
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::HandStrength;
    ///
    /// let hs = HandStrength::new(0.5, 0.2, 0.1);
    /// assert!((hs.ehs2 - 0.55).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn new(ehs: f32, ppot: f32, npot: f32) -> Self {
        let ehs2 = ehs + (1.0 - ehs) * ppot - ehs * npot;
        Self {
            ehs,
            ppot,
            npot,
            ehs2,
        }
    }

    /// Create a river hand strength (no potential, just EHS).
    ///
    /// On the river, there are no more cards to come, so positive and negative
    /// potential are both zero, and EHS2 equals EHS.
    ///
    /// # Arguments
    /// * `ehs` - Expected Hand Strength (current equity)
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::HandStrength;
    ///
    /// let hs = HandStrength::river(0.75);
    /// assert_eq!(hs.ehs, 0.75);
    /// assert_eq!(hs.ehs2, 0.75);
    /// assert_eq!(hs.ppot, 0.0);
    /// assert_eq!(hs.npot, 0.0);
    /// ```
    #[must_use]
    pub fn river(ehs: f32) -> Self {
        Self {
            ehs,
            ppot: 0.0,
            npot: 0.0,
            ehs2: ehs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn river_ehs2_equals_ehs() {
        let hs = HandStrength::river(0.75);
        assert_eq!(hs.ehs2, hs.ehs);
        assert_eq!(hs.ppot, 0.0);
        assert_eq!(hs.npot, 0.0);
    }

    #[test]
    fn ehs2_formula_correct() {
        let hs = HandStrength::new(0.5, 0.2, 0.1);
        // EHS2 = 0.5 + (1-0.5)*0.2 - 0.5*0.1 = 0.5 + 0.1 - 0.05 = 0.55
        assert!((hs.ehs2 - 0.55).abs() < 0.001);
    }

    #[test]
    fn river_with_zero_ehs() {
        let hs = HandStrength::river(0.0);
        assert_eq!(hs.ehs, 0.0);
        assert_eq!(hs.ehs2, 0.0);
    }

    #[test]
    fn river_with_full_ehs() {
        let hs = HandStrength::river(1.0);
        assert_eq!(hs.ehs, 1.0);
        assert_eq!(hs.ehs2, 1.0);
    }

    #[test]
    fn ehs2_with_high_positive_potential() {
        // A drawing hand with high positive potential
        let hs = HandStrength::new(0.3, 0.5, 0.0);
        // EHS2 = 0.3 + (1-0.3)*0.5 - 0.3*0.0 = 0.3 + 0.35 = 0.65
        assert!((hs.ehs2 - 0.65).abs() < 0.001);
    }

    #[test]
    fn ehs2_with_high_negative_potential() {
        // A vulnerable hand with high negative potential
        let hs = HandStrength::new(0.8, 0.0, 0.3);
        // EHS2 = 0.8 + (1-0.8)*0.0 - 0.8*0.3 = 0.8 - 0.24 = 0.56
        assert!((hs.ehs2 - 0.56).abs() < 0.001);
    }

    #[test]
    fn hand_strength_is_copy() {
        let hs1 = HandStrength::new(0.5, 0.2, 0.1);
        let hs2 = hs1;
        assert_eq!(hs1, hs2);
    }

    #[test]
    fn hand_strength_is_clone() {
        let hs1 = HandStrength::new(0.5, 0.2, 0.1);
        let hs2 = hs1.clone();
        assert_eq!(hs1, hs2);
    }

    #[test]
    fn hand_strength_debug() {
        let hs = HandStrength::river(0.5);
        let debug_str = format!("{:?}", hs);
        assert!(debug_str.contains("HandStrength"));
        assert!(debug_str.contains("0.5"));
    }
}
