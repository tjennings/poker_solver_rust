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

    /// Look up or compute boundary CFVs for the given boundary.
    ///
    /// Fast path: if a cached entry exists and is still fresh (within the
    /// refresh interval), return it immediately under a read lock.
    ///
    /// Slow path: re-sample via the underlying `BoundarySampler`, then
    /// update the cache under a write lock.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cfvs(
        &self,
        boundary_id: u64,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        oop_range: &[f64],
        ip_range: &[f64],
        boundary_pot: f64,
        boundary_invested: [f64; 2],
    ) -> BoundaryCfvs {
        let iter = self.current_iter.load(Ordering::SeqCst);

        // Fast path: read-lock, return cached if fresh.
        {
            let guard = self.cached.read().unwrap();
            if let Some(entry) = guard.get(&boundary_id) {
                if !self.should_refresh(iter, entry.refreshed_at_iter) {
                    return entry.cfvs.clone();
                }
            }
        }

        // Slow path: re-sample, write-lock, update cache.
        let fresh = self.sampler.sample_boundary_cfvs(
            combos,
            board,
            oop_range,
            ip_range,
            boundary_pot,
            boundary_invested,
            self.samples_per_refresh,
        );
        let mut guard = self.cached.write().unwrap();
        guard.insert(
            boundary_id,
            CachedEntry {
                cfvs: fresh.clone(),
                refreshed_at_iter: iter,
            },
        );
        fresh
    }
}

// ---------------------------------------------------------------------------
// BoundaryEvaluator trait placeholder
// ---------------------------------------------------------------------------

impl range_solver::game::BoundaryEvaluator for HybridBoundaryEvaluator {
    fn compute_cfvs(
        &self,
        _player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        panic!(
            "HybridBoundaryEvaluator: use compute_cfvs with boundary_id, \
             combos, board, ranges, pot, and invested args instead"
        );
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

    /// Mock sampler that returns incrementing values on each call,
    /// allowing tests to distinguish cached vs fresh results.
    struct CountingSampler {
        call_count: std::sync::atomic::AtomicU32,
    }

    impl CountingSampler {
        fn new() -> Self {
            Self {
                call_count: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl BoundarySampler for CountingSampler {
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
            let n = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let val = f64::from(n + 1); // 1.0, 2.0, 3.0, ...
            BoundaryCfvs {
                oop_cfvs: vec![val],
                ip_cfvs: vec![val * 10.0],
            }
        }
    }

    #[test]
    fn compute_cfvs_caches_between_refreshes() {
        let eval = HybridBoundaryEvaluator::new(
            Box::new(CountingSampler::new()),
            10, // refresh every 10 iterations
            1,  // 1 sample per refresh
        );

        let combos: Vec<[RsPokerCard; 2]> = vec![];
        let board: Vec<RsPokerCard> = vec![];
        let empty: Vec<f64> = vec![];
        let boundary_id = 42u64;

        // iter=0: no cache entry → sampler called (returns 1.0 / 10.0)
        eval.begin_iteration(0);
        let result0 = eval.compute_cfvs(
            boundary_id, &combos, &board, &empty, &empty, 100.0, [50.0, 50.0],
        );
        assert_eq!(result0.oop_cfvs, vec![1.0]);
        assert_eq!(result0.ip_cfvs, vec![10.0]);

        // iter=3: cached entry exists, 3 < 10 → cache hit (still 1.0 / 10.0)
        eval.begin_iteration(3);
        let result3 = eval.compute_cfvs(
            boundary_id, &combos, &board, &empty, &empty, 100.0, [50.0, 50.0],
        );
        assert_eq!(result3.oop_cfvs, vec![1.0], "should return cached value at iter=3");
        assert_eq!(result3.ip_cfvs, vec![10.0], "should return cached value at iter=3");

        // iter=10: refresh interval reached → sampler called again (returns 2.0 / 20.0)
        eval.begin_iteration(10);
        let result10 = eval.compute_cfvs(
            boundary_id, &combos, &board, &empty, &empty, 100.0, [50.0, 50.0],
        );
        assert_eq!(result10.oop_cfvs, vec![2.0], "should return fresh value at iter=10");
        assert_eq!(result10.ip_cfvs, vec![20.0], "should return fresh value at iter=10");
    }

    #[test]
    fn compute_cfvs_different_boundary_ids_cached_independently() {
        let eval = HybridBoundaryEvaluator::new(
            Box::new(CountingSampler::new()),
            10,
            1,
        );

        let combos: Vec<[RsPokerCard; 2]> = vec![];
        let board: Vec<RsPokerCard> = vec![];
        let empty: Vec<f64> = vec![];

        eval.begin_iteration(0);

        // First boundary → sampler call #1 (1.0)
        let r1 = eval.compute_cfvs(1, &combos, &board, &empty, &empty, 100.0, [50.0, 50.0]);
        assert_eq!(r1.oop_cfvs, vec![1.0]);

        // Second boundary → sampler call #2 (2.0)
        let r2 = eval.compute_cfvs(2, &combos, &board, &empty, &empty, 100.0, [50.0, 50.0]);
        assert_eq!(r2.oop_cfvs, vec![2.0]);

        // Re-query first boundary → cached (still 1.0)
        let r1_again = eval.compute_cfvs(1, &combos, &board, &empty, &empty, 100.0, [50.0, 50.0]);
        assert_eq!(r1_again.oop_cfvs, vec![1.0]);
    }

    #[test]
    fn hybrid_evaluator_implements_boundary_evaluator() {
        // Compile-time check: HybridBoundaryEvaluator implements BoundaryEvaluator.
        fn assert_impl<T: range_solver::game::BoundaryEvaluator>() {}
        assert_impl::<HybridBoundaryEvaluator>();
    }

    #[test]
    #[should_panic(expected = "use compute_cfvs")]
    fn boundary_evaluator_compute_cfvs_panics_with_message() {
        use range_solver::game::BoundaryEvaluator;
        let eval = HybridBoundaryEvaluator::new(
            Box::new(NoOpSampler),
            10,
            1,
        );
        // The BoundaryEvaluator::compute_cfvs method should panic
        // because the real entry point is HybridBoundaryEvaluator::compute_cfvs
        // with different arguments.
        let _ = BoundaryEvaluator::compute_cfvs(&eval, 0, 100, 50.0, &[], 0, 0);
    }
}
