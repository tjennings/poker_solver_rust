//! Shared DCFR (Discounted Counterfactual Regret) module.
//!
//! Encapsulates the discounting and iteration-weighting logic used by both
//! the preflop LCFR solver and the postflop exhaustive solver. Supports four
//! variants via [`CfrVariant`]:
//!
//! - **Vanilla**: no discounting, uniform iteration weighting.
//! - **DCFR**: α/β/γ discounting with linear (LCFR) iteration weighting.
//! - **CFR+**: regrets floored to zero, linear strategy weighting.
//! - **Linear**: DCFR with all exponents = 1.0 (Pluribus-style).

use serde::{Deserialize, Serialize};

/// Which CFR variant the solver should use.
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

/// Parameters for Discounted CFR variants.
///
/// Encapsulates the DCFR discounting logic shared between preflop and postflop
/// solvers. Supports Vanilla (no-op), DCFR (α/β/γ), CFR+ (floor + linear
/// strategy), and Linear (LCFR).
#[derive(Debug, Clone, Copy)]
pub struct DcfrParams {
    pub variant: CfrVariant,
    /// Positive regret discount exponent.
    pub alpha: f64,
    /// Negative regret discount exponent.
    pub beta: f64,
    /// Strategy sum discount exponent.
    pub gamma: f64,
    /// Number of initial iterations without DCFR discounting (warm-up phase).
    pub warmup: u64,
}

impl Default for DcfrParams {
    fn default() -> Self {
        Self {
            variant: CfrVariant::Dcfr,
            alpha: 1.5,
            beta: 0.5,
            gamma: 2.0,
            warmup: 0,
        }
    }
}

impl DcfrParams {
    /// DCFR with all exponents = 1.0 (Pluribus-style linear weighting).
    #[must_use]
    pub fn linear() -> Self {
        Self {
            variant: CfrVariant::Linear,
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            warmup: 0,
        }
    }

    /// Vanilla CFR: no discounting, uniform iteration weighting.
    #[must_use]
    pub fn vanilla() -> Self {
        Self {
            variant: CfrVariant::Vanilla,
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            warmup: 0,
        }
    }

    /// Construct from explicit config values.
    #[must_use]
    pub fn from_config(
        variant: CfrVariant,
        alpha: f64,
        beta: f64,
        gamma: f64,
        warmup: u64,
    ) -> Self {
        Self {
            variant,
            alpha,
            beta,
            gamma,
            warmup,
        }
    }

    /// Returns `(regret_weight, strategy_weight)` for the given iteration.
    ///
    /// - Vanilla: `(1.0, 1.0)` — uniform weighting.
    /// - DCFR / Linear: `(t, t)` — linear (LCFR) weighting.
    /// - CFR+: `(1.0, t)` — uniform regret, linear strategy.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn iteration_weights(&self, iteration: u64) -> (f64, f64) {
        match self.variant {
            CfrVariant::Vanilla => (1.0, 1.0),
            CfrVariant::Dcfr | CfrVariant::Linear => {
                let w = iteration as f64;
                (w, w)
            }
            CfrVariant::CfrPlus => (1.0, iteration as f64),
        }
    }

    /// Compute the DCFR strategy discount factor: `(t / (t + 1))^γ`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn strategy_discount(&self, iteration: u64) -> f64 {
        let t = iteration as f64;
        (t / (t + 1.0)).powf(self.gamma)
    }

    /// Whether DCFR discounting should be applied at this iteration.
    ///
    /// Returns `true` if the variant is DCFR or Linear **and** the iteration
    /// is past the warmup phase.
    #[must_use]
    pub fn should_discount(&self, iteration: u64) -> bool {
        matches!(self.variant, CfrVariant::Dcfr | CfrVariant::Linear)
            && iteration > self.warmup
    }

    /// Apply DCFR cumulative regret discounting to a raw slice.
    ///
    /// Positive values are multiplied by `t^α / (t^α + 1)`.
    /// Negative values are multiplied by `t^β / (t^β + 1)`.
    /// Zero values are left unchanged.
    #[allow(clippy::cast_precision_loss)]
    pub fn discount_regrets(&self, buf: &mut [f64], iteration: u64) {
        let t = (iteration + 1) as f64;
        let ta = t.powf(self.alpha);
        let pos_factor = ta / (ta + 1.0);
        let tb = t.powf(self.beta);
        let neg_factor = tb / (tb + 1.0);
        for r in buf.iter_mut() {
            if *r > 0.0 {
                *r *= pos_factor;
            } else if *r < 0.0 {
                *r *= neg_factor;
            }
        }
    }

    /// Multiply all strategy sum values by `strategy_discount(iteration)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn discount_strategy_sums(&self, buf: &mut [f64], iteration: u64) {
        let sd = self.strategy_discount(iteration);
        for s in buf.iter_mut() {
            *s *= sd;
        }
    }

    /// Whether regrets should be floored to zero (CFR+ variant).
    #[must_use]
    pub fn should_floor_regrets(&self) -> bool {
        self.variant == CfrVariant::CfrPlus
    }

    /// Floor all negative regret values to zero.
    pub fn floor_regrets(&self, buf: &mut [f64]) {
        for r in buf.iter_mut() {
            if *r < 0.0 {
                *r = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn iteration_weights_vanilla() {
        let p = DcfrParams::vanilla();
        assert_eq!(p.iteration_weights(0), (1.0, 1.0));
        assert_eq!(p.iteration_weights(100), (1.0, 1.0));
    }

    #[timed_test]
    fn iteration_weights_dcfr() {
        let p = DcfrParams::default();
        assert_eq!(p.iteration_weights(0), (0.0, 0.0));
        assert_eq!(p.iteration_weights(10), (10.0, 10.0));
        assert_eq!(p.iteration_weights(100), (100.0, 100.0));
    }

    #[timed_test]
    fn iteration_weights_linear() {
        let p = DcfrParams::linear();
        assert_eq!(p.iteration_weights(5), (5.0, 5.0));
    }

    #[timed_test]
    fn iteration_weights_cfrplus() {
        let p = DcfrParams::from_config(CfrVariant::CfrPlus, 0.0, 0.0, 0.0, 0);
        assert_eq!(p.iteration_weights(0), (1.0, 0.0));
        assert_eq!(p.iteration_weights(10), (1.0, 10.0));
    }

    #[timed_test]
    fn strategy_discount_formula() {
        let p = DcfrParams::default(); // gamma=2.0
        // At iteration 10: (10/11)^2
        let expected = (10.0_f64 / 11.0).powf(2.0);
        let actual = p.strategy_discount(10);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[timed_test]
    fn strategy_discount_iteration_zero() {
        let p = DcfrParams::default();
        // (0/1)^2 = 0
        assert!((p.strategy_discount(0)).abs() < 1e-12);
    }

    #[timed_test]
    fn discount_regrets_applies_factors() {
        let p = DcfrParams::default(); // alpha=1.5, beta=0.5
        let mut buf = vec![10.0, -5.0, 0.0, 3.0, -1.0];
        p.discount_regrets(&mut buf, 9); // iteration 9 → t=10

        let t = 10.0_f64;
        let pos = t.powf(1.5) / (t.powf(1.5) + 1.0);
        let neg = t.powf(0.5) / (t.powf(0.5) + 1.0);

        assert!((buf[0] - 10.0 * pos).abs() < 1e-10);
        assert!((buf[1] - (-5.0 * neg)).abs() < 1e-10);
        assert!((buf[2]).abs() < 1e-10); // zero unchanged
        assert!((buf[3] - 3.0 * pos).abs() < 1e-10);
        assert!((buf[4] - (-1.0 * neg)).abs() < 1e-10);
    }

    #[timed_test]
    fn floor_regrets_zeroes_negatives() {
        let p = DcfrParams::from_config(CfrVariant::CfrPlus, 0.0, 0.0, 0.0, 0);
        let mut buf = vec![5.0, -3.0, 0.0, -0.001, 2.0];
        p.floor_regrets(&mut buf);
        assert_eq!(buf, vec![5.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[timed_test]
    fn should_discount_respects_warmup() {
        let p = DcfrParams::from_config(CfrVariant::Dcfr, 1.5, 0.5, 2.0, 10);
        assert!(!p.should_discount(0));
        assert!(!p.should_discount(10));
        assert!(p.should_discount(11));
        assert!(p.should_discount(100));
    }

    #[timed_test]
    fn should_discount_vanilla_always_false() {
        let p = DcfrParams::vanilla();
        assert!(!p.should_discount(0));
        assert!(!p.should_discount(1000));
    }

    #[timed_test]
    fn should_discount_cfrplus_always_false() {
        let p = DcfrParams::from_config(CfrVariant::CfrPlus, 0.0, 0.0, 0.0, 0);
        assert!(!p.should_discount(100));
    }

    #[timed_test]
    fn should_floor_regrets_only_cfrplus() {
        assert!(DcfrParams::from_config(CfrVariant::CfrPlus, 0.0, 0.0, 0.0, 0)
            .should_floor_regrets());
        assert!(!DcfrParams::default().should_floor_regrets());
        assert!(!DcfrParams::vanilla().should_floor_regrets());
        assert!(!DcfrParams::linear().should_floor_regrets());
    }

    #[timed_test]
    fn linear_constructor_values() {
        let p = DcfrParams::linear();
        assert_eq!(p.variant, CfrVariant::Linear);
        assert!((p.alpha - 1.0).abs() < f64::EPSILON);
        assert!((p.beta - 1.0).abs() < f64::EPSILON);
        assert!((p.gamma - 1.0).abs() < f64::EPSILON);
        assert_eq!(p.warmup, 0);
    }

    #[timed_test]
    fn default_constructor_values() {
        let p = DcfrParams::default();
        assert_eq!(p.variant, CfrVariant::Dcfr);
        assert!((p.alpha - 1.5).abs() < f64::EPSILON);
        assert!((p.beta - 0.5).abs() < f64::EPSILON);
        assert!((p.gamma - 2.0).abs() < f64::EPSILON);
        assert_eq!(p.warmup, 0);
    }

    #[timed_test]
    fn discount_strategy_sums_applies_factor() {
        let p = DcfrParams::default(); // gamma=2.0
        let mut buf = vec![10.0, 20.0, 5.0];
        let sd = p.strategy_discount(10);
        p.discount_strategy_sums(&mut buf, 10);
        assert!((buf[0] - 10.0 * sd).abs() < 1e-10);
        assert!((buf[1] - 20.0 * sd).abs() < 1e-10);
        assert!((buf[2] - 5.0 * sd).abs() < 1e-10);
    }

    #[timed_test]
    fn cfr_variant_serde_roundtrip() {
        for variant in [
            CfrVariant::Vanilla,
            CfrVariant::Dcfr,
            CfrVariant::CfrPlus,
            CfrVariant::Linear,
        ] {
            let yaml = serde_yaml::to_string(&variant).unwrap();
            let parsed: CfrVariant = serde_yaml::from_str(yaml.trim()).unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[timed_test]
    fn cfr_variant_serde_values() {
        let check = |v: CfrVariant, expected: &str| {
            let yaml = serde_yaml::to_string(&v).unwrap();
            assert_eq!(yaml.trim(), expected);
        };
        check(CfrVariant::Vanilla, "vanilla");
        check(CfrVariant::Dcfr, "dcfr");
        check(CfrVariant::CfrPlus, "cfrplus");
        check(CfrVariant::Linear, "linear");
    }
}
