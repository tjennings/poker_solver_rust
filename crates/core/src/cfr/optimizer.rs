//! Pluggable CFR optimizer trait and implementations.
//!
//! Replaces hardcoded DCFR discounting with a trait-based approach,
//! allowing different regret update rules (DCFR, SAPCFR+, etc.)
//! to be selected via config.

// Precision loss on cast to f64 is acceptable for action counts and
// regret values. Truncation from f64 back to i32 is intentional.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use rayon::prelude::*;

/// Trait for CFR regret/strategy update rules.
///
/// The optimizer is called periodically by the trainer to apply discounting
/// and compute strategies. Raw instantaneous regrets are accumulated in
/// `BlueprintStorage.regrets` by the traversal code (unchanged).
pub trait CfrOptimizer: Send + Sync {
    /// Apply discount to accumulated regrets and strategy sums.
    /// Called every `lcfr_discount_interval` iterations.
    ///
    /// `predictions` is the optional prediction buffer (used by SAPCFR+).
    /// DCFR ignores it.
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI32],
        predictions: Option<&[AtomicI32]>,
        iteration: u64,
    );

    /// Compute current strategy at a given offset into the regret buffer.
    ///
    /// Default: standard regret matching (normalize positive regrets).
    /// SAPCFR+ overrides this to use predicted regrets.
    ///
    /// `predictions` is the optional prediction buffer.
    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        predictions: Option<&[AtomicI32]>,
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    );

    /// Name for logging.
    fn name(&self) -> &'static str;

    /// Whether this optimizer requires a prediction buffer in storage.
    fn needs_predictions(&self) -> bool {
        false
    }

    /// Set the decay factor for prediction-based optimizers.
    /// Default: no-op (only `BrcfrPlusOptimizer` overrides this).
    fn set_decay(&self, _decay: f64) {}
}

/// DCFR optimizer: discounted counterfactual regret minimization.
///
/// Ports the exact logic from `BlueprintTrainer::apply_lcfr_discount()`.
/// - Positive regrets discounted by `t^alpha / (t^alpha + 1)`
/// - Negative regrets discounted by `t^beta / (t^beta + 1)`
/// - Strategy sums discounted by `(t / (t + 1))^gamma`
pub struct DcfrOptimizer {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl CfrOptimizer for DcfrOptimizer {
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI32],
        _predictions: Option<&[AtomicI32]>,
        iteration: u64,
    ) {
        let tf = iteration as f64;
        let t_alpha = tf.powf(self.alpha);
        let t_beta = tf.powf(self.beta);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_neg = t_beta / (t_beta + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(self.gamma);

        regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let d = if v >= 0 { d_pos } else { d_neg };
            atom.store((v as f64 * d) as i32, Ordering::Relaxed);
        });

        strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i32, Ordering::Relaxed);
        });
    }

    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        _predictions: Option<&[AtomicI32]>,
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    ) {
        // Standard regret matching: normalize positive regrets.
        let mut sum = 0.0_f64;
        for a in 0..num_actions {
            let r = regrets[offset + a].load(Ordering::Relaxed);
            let v = if r > 0 { r as f64 } else { 0.0 };
            out[a] = v;
            sum += v;
        }
        if sum > 0.0 {
            for v in &mut out[..num_actions] {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / num_actions as f64;
            for v in &mut out[..num_actions] {
                *v = uniform;
            }
        }
    }

    fn name(&self) -> &'static str {
        "dcfr"
    }
}

/// SAPCFR+ optimizer: Sample-and-Predict CFR+ with prediction-based strategy.
///
/// During discounting:
/// - Positive regrets are discounted by `t^alpha / (t^alpha + 1)`
/// - Negative regrets are floored to 0 (RM+ style)
/// - Strategy sums are discounted by `(t / (t + 1))^gamma`
///
/// During strategy computation:
/// - Uses `R_tilde = max(0, R + eta * prediction)` as effective regret
/// - Predictions are the previous iteration's instantaneous regret
pub struct SapcfrPlusOptimizer {
    /// Positive regret discount exponent.
    pub alpha: f64,
    /// Strategy sum discount exponent.
    pub gamma: f64,
    /// Prediction step size (0 = no prediction, 1 = full PCFR+).
    pub eta: f64,
}

impl CfrOptimizer for SapcfrPlusOptimizer {
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI32],
        _predictions: Option<&[AtomicI32]>,
        iteration: u64,
    ) {
        let tf = iteration as f64;
        let t_alpha = tf.powf(self.alpha);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(self.gamma);

        // Discount regrets: floor negative to 0 (RM+ style).
        regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let discounted = (v as f64 * d_pos) as i32;
            atom.store(discounted.max(0), Ordering::Relaxed);
        });

        // Discount strategy sums.
        strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i32, Ordering::Relaxed);
        });
    }

    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        predictions: Option<&[AtomicI32]>,
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    ) {
        // SAPCFR+: strategy from predicted regrets = R + eta * v
        let mut sum = 0.0_f64;
        for a in 0..num_actions {
            let r = regrets[offset + a].load(Ordering::Relaxed);
            let v = predictions.map_or(0, |p| p[offset + a].load(Ordering::Relaxed));
            let predicted = r as f64 + self.eta * v as f64;
            let clamped = predicted.max(0.0);
            out[a] = clamped;
            sum += clamped;
        }
        if sum > 0.0 {
            for v in &mut out[..num_actions] {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / num_actions as f64;
            for v in &mut out[..num_actions] {
                *v = uniform;
            }
        }
    }

    fn name(&self) -> &'static str {
        "sapcfr+"
    }

    fn needs_predictions(&self) -> bool {
        true
    }
}

/// BRCFR+ optimizer: DCFR+ augmented with periodic best-response predictions.
///
/// Discounting is identical to SAPCFR+ (RM+ style: floor negatives to 0).
/// Strategy computation uses BR-derived predictions with configurable decay:
///   `R_tilde = max(0, R + eta * decay * v_br)`
///
/// The `decay` field is set by the trainer based on how many iterations
/// have elapsed since the last BR pass. At `decay = 0`, this is pure DCFR+.
pub struct BrcfrPlusOptimizer {
    /// Positive regret discount exponent.
    pub alpha: f64,
    /// Strategy sum discount exponent.
    pub gamma: f64,
    /// BR prediction weight.
    pub eta: f64,
    /// Current decay factor (0.0 to 1.0), stored as `AtomicU64` bit pattern
    /// for interior mutability through `&self`.
    decay_bits: AtomicU64,
}

impl BrcfrPlusOptimizer {
    /// Create a new BRCFR+ optimizer with decay starting at 0.0.
    #[must_use]
    pub fn new(alpha: f64, gamma: f64, eta: f64) -> Self {
        Self {
            alpha,
            gamma,
            eta,
            decay_bits: AtomicU64::new(0.0_f64.to_bits()),
        }
    }

    /// Read the current decay factor.
    pub fn decay(&self) -> f64 {
        f64::from_bits(self.decay_bits.load(Ordering::Relaxed))
    }
}

impl CfrOptimizer for BrcfrPlusOptimizer {
    fn apply_discount(
        &self,
        regrets: &[AtomicI32],
        strategy_sums: &[AtomicI32],
        _predictions: Option<&[AtomicI32]>,
        iteration: u64,
    ) {
        let tf = iteration as f64;
        let t_alpha = tf.powf(self.alpha);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(self.gamma);

        // Discount regrets: floor negative to 0 (RM+ style).
        regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let discounted = (v as f64 * d_pos) as i32;
            atom.store(discounted.max(0), Ordering::Relaxed);
        });

        // Discount strategy sums.
        strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i32, Ordering::Relaxed);
        });
    }

    fn current_strategy(
        &self,
        regrets: &[AtomicI32],
        predictions: Option<&[AtomicI32]>,
        offset: usize,
        num_actions: usize,
        out: &mut [f64],
    ) {
        let eta_decay = self.eta * self.decay();
        let mut sum = 0.0_f64;
        for a in 0..num_actions {
            let r = regrets[offset + a].load(Ordering::Relaxed);
            let v = predictions.map_or(0, |p| p[offset + a].load(Ordering::Relaxed));
            let predicted = r as f64 + eta_decay * v as f64;
            let clamped = predicted.max(0.0);
            out[a] = clamped;
            sum += clamped;
        }
        if sum > 0.0 {
            for v in &mut out[..num_actions] {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / num_actions as f64;
            for v in &mut out[..num_actions] {
                *v = uniform;
            }
        }
    }

    fn name(&self) -> &'static str {
        "brcfr+"
    }

    fn needs_predictions(&self) -> bool {
        true
    }

    fn set_decay(&self, decay: f64) {
        self.decay_bits.store(decay.to_bits(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    // Tests for DcfrOptimizer

    #[test]
    fn dcfr_discount_positive_regrets() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        let regrets = vec![AtomicI32::new(1000), AtomicI32::new(-500)];
        let strats = vec![AtomicI32::new(2000)];
        opt.apply_discount(&regrets, &strats, None, 10);

        // d_pos = 10^1.5 / (10^1.5 + 1) ~= 31.623 / 32.623 ~= 0.9693
        let r0 = regrets[0].load(Ordering::Relaxed);
        assert!(
            r0 > 960 && r0 < 980,
            "positive regret discounted: got {r0}, expected ~969"
        );

        // d_neg = 10^0.0 / (10^0.0 + 1) = 1/2 = 0.5
        let r1 = regrets[1].load(Ordering::Relaxed);
        assert_eq!(r1, -250, "negative regret halved: got {r1}");
    }

    #[test]
    fn dcfr_discount_strategy_sums() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        let regrets: Vec<AtomicI32> = vec![];
        let strats = vec![AtomicI32::new(2000)];
        opt.apply_discount(&regrets, &strats, None, 10);

        // d_strat = (10/11)^2.0 ~= 0.8264
        let s0 = strats[0].load(Ordering::Relaxed);
        let expected = (2000.0 * (10.0_f64 / 11.0).powf(2.0)) as i32;
        assert_eq!(s0, expected, "strategy sum discounted");
    }

    #[test]
    fn dcfr_discount_ignores_predictions() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        let regrets = vec![AtomicI32::new(1000)];
        let strats = vec![AtomicI32::new(2000)];
        let preds = vec![AtomicI32::new(999)];
        // Passing predictions should not change behavior for DCFR
        opt.apply_discount(&regrets, &strats, Some(&preds), 10);
        let r0 = regrets[0].load(Ordering::Relaxed);
        assert!(r0 > 960 && r0 < 980, "same result with predictions: {r0}");
    }

    #[test]
    fn dcfr_current_strategy_regret_matching() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        let regrets = vec![
            AtomicI32::new(300),  // action 0
            AtomicI32::new(100),  // action 1
            AtomicI32::new(-50),  // action 2 (negative, excluded)
        ];
        let mut out = [0.0; 3];
        opt.current_strategy(&regrets, None, 0, 3, &mut out);
        // 300/(300+100) = 0.75, 100/400 = 0.25, 0/400 = 0
        assert!((out[0] - 0.75).abs() < 0.01);
        assert!((out[1] - 0.25).abs() < 0.01);
        assert!((out[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn dcfr_current_strategy_all_negative_is_uniform() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        let regrets = vec![
            AtomicI32::new(-100),
            AtomicI32::new(-200),
        ];
        let mut out = [0.0; 2];
        opt.current_strategy(&regrets, None, 0, 2, &mut out);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn dcfr_current_strategy_with_offset() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        // Simulate a buffer with multiple nodes. Node at offset 2 has 2 actions.
        let regrets = vec![
            AtomicI32::new(0),    // slot 0 (different node)
            AtomicI32::new(0),    // slot 1 (different node)
            AtomicI32::new(600),  // slot 2 = our node, action 0
            AtomicI32::new(200),  // slot 3 = our node, action 1
        ];
        let mut out = [0.0; 2];
        opt.current_strategy(&regrets, None, 2, 2, &mut out);
        // 600/800 = 0.75, 200/800 = 0.25
        assert!((out[0] - 0.75).abs() < 0.01);
        assert!((out[1] - 0.25).abs() < 0.01);
    }

    #[test]
    fn dcfr_name() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        assert_eq!(opt.name(), "dcfr");
    }

    #[test]
    fn dcfr_needs_predictions_false() {
        use super::*;
        let opt = DcfrOptimizer {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        };
        assert!(!opt.needs_predictions());
    }

    // Tests for SapcfrPlusOptimizer

    #[test]
    fn sapcfr_current_strategy_uses_predictions() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        let regrets = vec![AtomicI32::new(200), AtomicI32::new(100), AtomicI32::new(0)];
        let preds = vec![AtomicI32::new(100), AtomicI32::new(-50), AtomicI32::new(50)];
        let mut out = [0.0; 3];
        opt.current_strategy(&regrets, Some(&preds), 0, 3, &mut out);
        // R_tilde[0] = max(0, 200 + 0.5*100) = 250
        // R_tilde[1] = max(0, 100 + 0.5*(-50)) = 75
        // R_tilde[2] = max(0, 0 + 0.5*50) = 25
        // sum = 350
        assert!(
            (out[0] - 250.0 / 350.0).abs() < 0.01,
            "action 0: got {}, expected {:.4}",
            out[0],
            250.0 / 350.0
        );
        assert!(
            (out[1] - 75.0 / 350.0).abs() < 0.01,
            "action 1: got {}, expected {:.4}",
            out[1],
            75.0 / 350.0
        );
        assert!(
            (out[2] - 25.0 / 350.0).abs() < 0.01,
            "action 2: got {}, expected {:.4}",
            out[2],
            25.0 / 350.0
        );
    }

    #[test]
    fn sapcfr_current_strategy_without_predictions_falls_back() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        // Without predictions, eta * 0 = 0 => standard regret matching
        let regrets = vec![AtomicI32::new(300), AtomicI32::new(100)];
        let mut out = [0.0; 2];
        opt.current_strategy(&regrets, None, 0, 2, &mut out);
        assert!((out[0] - 0.75).abs() < 0.01);
        assert!((out[1] - 0.25).abs() < 0.01);
    }

    #[test]
    fn sapcfr_current_strategy_all_negative_predicted_is_uniform() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        // Even with predictions, if all predicted regrets are negative => uniform
        let regrets = vec![AtomicI32::new(-100), AtomicI32::new(-200)];
        let preds = vec![AtomicI32::new(-100), AtomicI32::new(-100)];
        let mut out = [0.0; 2];
        opt.current_strategy(&regrets, Some(&preds), 0, 2, &mut out);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn sapcfr_discount_floors_negative_regrets() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        let regrets = vec![AtomicI32::new(1000), AtomicI32::new(-500)];
        let strats = vec![AtomicI32::new(2000)];
        opt.apply_discount(&regrets, &strats, None, 10);
        // Negative regrets are floored to 0 (RM+ style).
        let r1 = regrets[1].load(Ordering::Relaxed);
        assert_eq!(r1, 0, "negative regrets floored to 0, got {r1}");
        // Positive regrets are discounted normally.
        let r0 = regrets[0].load(Ordering::Relaxed);
        assert!(r0 > 960 && r0 < 980, "positive regret discounted: got {r0}");
    }

    #[test]
    fn sapcfr_discount_strategy_sums() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        let regrets: Vec<AtomicI32> = vec![];
        let strats = vec![AtomicI32::new(2000)];
        opt.apply_discount(&regrets, &strats, None, 10);
        let s0 = strats[0].load(Ordering::Relaxed);
        let expected = (2000.0 * (10.0_f64 / 11.0).powf(2.0)) as i32;
        assert_eq!(s0, expected, "strategy sum discounted like DCFR");
    }

    #[test]
    fn sapcfr_name() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        assert_eq!(opt.name(), "sapcfr+");
    }

    #[test]
    fn sapcfr_needs_predictions_true() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 0.5,
        };
        assert!(opt.needs_predictions());
    }

    #[test]
    fn sapcfr_current_strategy_with_offset() {
        use super::*;
        let opt = SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 1.0,
        };
        // 4-slot buffer, our node starts at offset 2 with 2 actions
        let regrets = vec![
            AtomicI32::new(0),
            AtomicI32::new(0),
            AtomicI32::new(100), // our action 0
            AtomicI32::new(100), // our action 1
        ];
        let preds = vec![
            AtomicI32::new(0),
            AtomicI32::new(0),
            AtomicI32::new(200), // prediction for action 0
            AtomicI32::new(-200), // prediction for action 1 (negative)
        ];
        let mut out = [0.0; 2];
        opt.current_strategy(&regrets, Some(&preds), 2, 2, &mut out);
        // R_tilde[0] = max(0, 100 + 1.0*200) = 300
        // R_tilde[1] = max(0, 100 + 1.0*(-200)) = max(0, -100) = 0
        // sum = 300 => out = [1.0, 0.0]
        assert!((out[0] - 1.0).abs() < 0.01, "got {}", out[0]);
        assert!((out[1] - 0.0).abs() < 0.01, "got {}", out[1]);
    }

    // Tests for BrcfrPlusOptimizer

    #[test]
    fn brcfr_current_strategy_with_decay() {
        use super::*;
        let opt = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        opt.set_decay(0.5);
        let regrets = vec![AtomicI32::new(200), AtomicI32::new(100), AtomicI32::new(0)];
        let preds = vec![AtomicI32::new(100), AtomicI32::new(-300), AtomicI32::new(50)];
        let mut out = [0.0; 3];
        opt.current_strategy(&regrets, Some(&preds), 0, 3, &mut out);
        // R_tilde[0] = max(0, 200 + 0.6 * 0.5 * 100) = max(0, 230) = 230
        // R_tilde[1] = max(0, 100 + 0.6 * 0.5 * (-300)) = max(0, 10) = 10
        // R_tilde[2] = max(0, 0 + 0.6 * 0.5 * 50) = max(0, 15) = 15
        // sum = 255
        assert!((out[0] - 230.0 / 255.0).abs() < 0.01, "action 0: got {}", out[0]);
        assert!((out[1] - 10.0 / 255.0).abs() < 0.01, "action 1: got {}", out[1]);
        assert!((out[2] - 15.0 / 255.0).abs() < 0.01, "action 2: got {}", out[2]);
    }

    #[test]
    fn brcfr_decay_zero_matches_dcfr_plus() {
        use super::*;
        let brcfr = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        brcfr.set_decay(0.0);
        let regrets = vec![AtomicI32::new(300), AtomicI32::new(100), AtomicI32::new(-50)];
        let preds = vec![AtomicI32::new(999), AtomicI32::new(-999), AtomicI32::new(999)];
        let mut out_brcfr = [0.0; 3];
        brcfr.current_strategy(&regrets, Some(&preds), 0, 3, &mut out_brcfr);
        // With decay=0, predictions are ignored => standard RM+: 300/400, 100/400, 0
        assert!((out_brcfr[0] - 0.75).abs() < 0.01, "got {}", out_brcfr[0]);
        assert!((out_brcfr[1] - 0.25).abs() < 0.01, "got {}", out_brcfr[1]);
        assert!((out_brcfr[2] - 0.0).abs() < 0.01, "got {}", out_brcfr[2]);
    }

    #[test]
    fn brcfr_discount_floors_negative_regrets() {
        use super::*;
        let opt = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        opt.set_decay(1.0);
        let regrets = vec![AtomicI32::new(1000), AtomicI32::new(-500)];
        let strats = vec![AtomicI32::new(2000)];
        opt.apply_discount(&regrets, &strats, None, 10);
        // Negative regrets are floored to 0 (RM+ style).
        let r1 = regrets[1].load(Ordering::Relaxed);
        assert_eq!(r1, 0, "negative regrets floored to 0, got {r1}");
        // Positive regrets are discounted normally.
        let r0 = regrets[0].load(Ordering::Relaxed);
        assert!(r0 > 960 && r0 < 980, "positive regret discounted: got {r0}");
    }

    #[test]
    fn brcfr_name() {
        use super::*;
        let opt = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        assert_eq!(opt.name(), "brcfr+");
    }

    #[test]
    fn brcfr_needs_predictions_true() {
        use super::*;
        let opt = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        assert!(opt.needs_predictions());
    }

    #[test]
    fn brcfr_set_decay_updates_value() {
        use super::*;
        let opt = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        assert!((opt.decay() - 0.0).abs() < f64::EPSILON, "initial decay should be 0.0");
        opt.set_decay(0.75);
        assert!((opt.decay() - 0.75).abs() < f64::EPSILON, "decay should be 0.75 after set");
        opt.set_decay(0.0);
        assert!((opt.decay() - 0.0).abs() < f64::EPSILON, "decay should be 0.0 after reset");
    }

    #[test]
    fn brcfr_set_decay_trait_method() {
        use super::*;
        let opt = BrcfrPlusOptimizer::new(1.5, 2.0, 0.6);
        // Call set_decay via the trait (CfrOptimizer)
        let trait_ref: &dyn CfrOptimizer = &opt;
        trait_ref.set_decay(0.5);
        // Verify the decay updated
        assert!((opt.decay() - 0.5).abs() < f64::EPSILON, "trait set_decay should update value");
    }
}
