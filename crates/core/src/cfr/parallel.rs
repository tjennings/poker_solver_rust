//! Shared parallel CFR iteration driver.
//!
//! Provides [`ParallelCfr`] trait and [`parallel_traverse`] function for
//! snapshot + delta-merge parallelism over hand pairs. Used by both
//! preflop and postflop solvers.

use rayon::prelude::*;

/// Context for one CFR iteration. Implementors provide the buffer size
/// and a traversal function that reads strategy from a frozen snapshot
/// and writes regret/strategy deltas to thread-local buffers.
pub trait ParallelCfr: Sync {
    /// Size of the regret/strategy flat buffers.
    fn buffer_size(&self) -> usize;

    /// Traverse one hand pair (both hero positions), accumulating
    /// regret deltas into `regret_delta` and strategy deltas into `strategy_delta`.
    fn traverse_pair(&self, regret_delta: &mut [f64], strategy_delta: &mut [f64], hero: u16, opponent: u16);
}

/// Parallel fold+collect over hand pairs. Returns merged `(regret_delta, strategy_delta)`.
///
/// Each rayon worker thread gets its own zero-initialized delta buffers.
/// After all pairs are traversed, per-thread partitions are collected and
/// merged sequentially. This avoids the O(chunks) identity allocations
/// that `reduce()` requires, cutting allocations to O(threads).
/// Results are mathematically identical to sequential traversal.
pub fn parallel_traverse<T: ParallelCfr>(
    ctx: &T,
    pairs: &[(u16, u16)],
) -> (Vec<f64>, Vec<f64>) {
    let buf_size = ctx.buffer_size();
    let partitions: Vec<(Vec<f64>, Vec<f64>)> = pairs
        .par_iter()
        .fold(
            || (vec![0.0_f64; buf_size], vec![0.0_f64; buf_size]),
            |(mut dr, mut ds), &(h1, h2)| {
                ctx.traverse_pair(&mut dr, &mut ds, h1, h2);
                (dr, ds)
            },
        )
        .collect();

    let mut regret = vec![0.0_f64; buf_size];
    let mut strategy = vec![0.0_f64; buf_size];
    for (dr, ds) in partitions {
        add_into(&mut regret, &dr);
        add_into(&mut strategy, &ds);
    }
    (regret, strategy)
}

/// Like [`parallel_traverse`], but merges results into caller-provided buffers
/// instead of allocating new ones. Zeroes both output slices before accumulating.
///
/// This avoids two `Vec` allocations per call, which matters when the caller
/// invokes traversal in a tight loop (e.g. exhaustive CFR iterations).
pub fn parallel_traverse_into<T: ParallelCfr>(
    ctx: &T,
    pairs: &[(u16, u16)],
    regret_out: &mut [f64],
    strategy_out: &mut [f64],
) {
    let buf_size = ctx.buffer_size();
    debug_assert_eq!(regret_out.len(), buf_size);
    debug_assert_eq!(strategy_out.len(), buf_size);

    regret_out.fill(0.0);
    strategy_out.fill(0.0);
    let partitions: Vec<(Vec<f64>, Vec<f64>)> = pairs
        .par_iter()
        .fold(
            || (vec![0.0_f64; buf_size], vec![0.0_f64; buf_size]),
            |(mut dr, mut ds), &(h1, h2)| {
                ctx.traverse_pair(&mut dr, &mut ds, h1, h2);
                (dr, ds)
            },
        )
        .collect();

    for (dr, ds) in partitions {
        add_into(regret_out, &dr);
        add_into(strategy_out, &ds);
    }
}

/// Element-wise `dst[i] += src[i]`.
#[inline]
pub fn add_into(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    /// Trivial implementor: each pair (h1, h2) adds h1+h2 to dr[0] and h1*h2 to ds[0].
    struct MockCfr;

    impl ParallelCfr for MockCfr {
        fn buffer_size(&self) -> usize {
            2
        }

        fn traverse_pair(&self, regret_delta: &mut [f64], strategy_delta: &mut [f64], hero: u16, opponent: u16) {
            regret_delta[0] += f64::from(hero) + f64::from(opponent);
            strategy_delta[0] += f64::from(hero) * f64::from(opponent);
        }
    }

    #[timed_test]
    fn parallel_traverse_matches_sequential() {
        let pairs: Vec<(u16, u16)> = (0..10_u16)
            .flat_map(|h1| (0..10_u16).map(move |h2| (h1, h2)))
            .collect();

        let mut exp_dr = vec![0.0_f64; 2];
        let mut exp_ds = vec![0.0_f64; 2];
        for &(h1, h2) in &pairs {
            MockCfr.traverse_pair(&mut exp_dr, &mut exp_ds, h1, h2);
        }

        let (dr, ds) = parallel_traverse(&MockCfr, &pairs);

        assert!(
            (dr[0] - exp_dr[0]).abs() < 1e-9,
            "dr mismatch: {} vs {}",
            dr[0],
            exp_dr[0]
        );
        assert!(
            (ds[0] - exp_ds[0]).abs() < 1e-9,
            "ds mismatch: {} vs {}",
            ds[0],
            exp_ds[0]
        );
    }

    #[timed_test]
    fn parallel_traverse_into_matches_parallel_traverse() {
        let pairs: Vec<(u16, u16)> = (0..10_u16)
            .flat_map(|h1| (0..10_u16).map(move |h2| (h1, h2)))
            .collect();

        let (exp_dr, exp_ds) = parallel_traverse(&MockCfr, &pairs);

        let mut dr = vec![0.0_f64; 2];
        let mut ds = vec![0.0_f64; 2];
        parallel_traverse_into(&MockCfr, &pairs, &mut dr, &mut ds);

        assert!(
            (dr[0] - exp_dr[0]).abs() < 1e-9,
            "dr mismatch: {} vs {}",
            dr[0],
            exp_dr[0]
        );
        assert!(
            (ds[0] - exp_ds[0]).abs() < 1e-9,
            "ds mismatch: {} vs {}",
            ds[0],
            exp_ds[0]
        );
    }

    #[timed_test]
    fn parallel_traverse_empty_pairs() {
        let (dr, ds) = parallel_traverse(&MockCfr, &[]);
        assert_eq!(dr.len(), 2);
        assert_eq!(ds.len(), 2);
        assert!(dr[0].abs() < 1e-15);
        assert!(ds[0].abs() < 1e-15);
    }

    #[timed_test]
    fn add_into_works() {
        let mut dst = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0, 30.0];
        add_into(&mut dst, &src);
        assert_eq!(dst, vec![11.0, 22.0, 33.0]);
    }
}
