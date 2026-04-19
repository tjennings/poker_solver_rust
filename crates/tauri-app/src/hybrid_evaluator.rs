//! Hybrid boundary evaluator: caches Monte Carlo rollout CFVs and refreshes
//! them periodically during MCCFR iterations.
//!
//! The evaluator sits between the MCCFR solver and the rollout sampler,
//! providing a cache layer keyed by boundary ID. Cached values are reused
//! across iterations until the refresh interval triggers a re-sample.

use crate::postflop::BoundaryCfvs;
use poker_solver_core::poker::Card as RsPokerCard;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// BoundarySampler trait — abstraction over rollout sampling
// ---------------------------------------------------------------------------

/// Trait abstracting the boundary CFV sampling operation.
///
/// Production code uses `RolloutLeafEvaluator`; tests inject mock samplers.
pub trait BoundarySampler: Send + Sync {
    fn sample_boundary_cfvs(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        oop_range: &[f64],
        ip_range: &[f64],
        boundary_pot: f64,
        boundary_invested: [f64; 2],
        num_samples: u32,
    ) -> BoundaryCfvs;
}

// ---------------------------------------------------------------------------
// CachedEntry
// ---------------------------------------------------------------------------

/// A single cached boundary CFV result with the iteration it was last refreshed.
#[derive(Clone, Debug)]
pub struct CachedEntry {
    pub cfvs: BoundaryCfvs,
    pub refreshed_at_iter: u32,
}

// ---------------------------------------------------------------------------
// HybridBoundaryEvaluator
// ---------------------------------------------------------------------------

/// Caches rollout-based boundary CFVs and refreshes them at a configurable
/// iteration interval during MCCFR solving.
pub struct HybridBoundaryEvaluator {
    sampler: Box<dyn BoundarySampler>,
    refresh_interval: u32,
    samples_per_refresh: u32,
    cached: RwLock<HashMap<u64, CachedEntry>>,
    current_iter: AtomicU32,
}

impl HybridBoundaryEvaluator {
    /// Create a new hybrid evaluator.
    ///
    /// # Panics
    /// Panics if `samples_per_refresh` is zero.
    pub fn new(
        sampler: Box<dyn BoundarySampler>,
        refresh_interval: u32,
        samples_per_refresh: u32,
    ) -> Self {
        assert!(
            samples_per_refresh > 0,
            "samples_per_refresh must be > 0"
        );
        Self {
            sampler,
            refresh_interval,
            samples_per_refresh,
            cached: RwLock::new(HashMap::new()),
            current_iter: AtomicU32::new(0),
        }
    }

    /// Update the current iteration counter. Called at the start of each
    /// MCCFR iteration.
    pub fn begin_iteration(&self, iter: u32) {
        self.current_iter.store(iter, Ordering::SeqCst);
    }

    /// Returns true if the cache entry should be refreshed given the current
    /// iteration and the iteration at which the entry was last refreshed.
    pub fn should_refresh(&self, iter: u32, last_refresh: u32) -> bool {
        iter.wrapping_sub(last_refresh) >= self.refresh_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal no-op sampler for tests that only exercise refresh logic.
    struct NoOpSampler;

    impl BoundarySampler for NoOpSampler {
        fn sample_boundary_cfvs(
            &self,
            _combos: &[[RsPokerCard; 2]],
            _board: &[RsPokerCard],
            _oop_range: &[f64],
            _ip_range: &[f64],
            _boundary_pot: f64,
            _boundary_invested: [f64; 2],
            _num_samples: u32,
        ) -> BoundaryCfvs {
            BoundaryCfvs {
                oop_cfvs: vec![],
                ip_cfvs: vec![],
            }
        }
    }

    #[test]
    fn should_refresh_triggers_at_interval() {
        let eval = HybridBoundaryEvaluator::new(
            Box::new(NoOpSampler),
            10,
            1,
        );

        // At iter=0, last_refresh=0 → difference is 0 which equals
        // refresh_interval only if interval is 0. With interval=10, diff=0 < 10.
        // BUT iter=0 is the very first call, and there's no cached entry yet,
        // so should_refresh isn't the gate — absence from cache is.
        // For the method itself: 0 - 0 = 0 < 10 → false.
        assert!(!eval.should_refresh(0, 0));

        // At iter=5, last_refresh=0 → 5 < 10 → false.
        assert!(!eval.should_refresh(5, 0));

        // At iter=10, last_refresh=0 → 10 >= 10 → true.
        assert!(eval.should_refresh(10, 0));

        // At iter=15, last_refresh=10 → 5 < 10 → false.
        assert!(!eval.should_refresh(15, 10));

        // At iter=20, last_refresh=10 → 10 >= 10 → true.
        assert!(eval.should_refresh(20, 10));
    }

    #[test]
    #[should_panic(expected = "samples_per_refresh")]
    fn samples_zero_panics_in_new() {
        let _ = HybridBoundaryEvaluator::new(
            Box::new(NoOpSampler),
            10,
            0,
        );
    }
}
