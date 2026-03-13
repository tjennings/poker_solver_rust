# 2D Stratified Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Make `sample_pot_stack_by_spr` produce uniform distributions over both SPR buckets and pot buckets by picking both dimensions jointly.

**Architecture:** Replace the segment-collection loop in `sample_pot_stack_by_spr` with a simpler single-bucket intersection after picking both SPR and pot buckets uniformly. Pure domain-layer change — no config, no I/O, no new files.

**Tech Stack:** Rust, rand crate

**Design doc:** `docs/plans/2026-03-13-2d-stratified-sampling-design.md`

---

### Task 1: Rewrite `sample_pot_stack_by_spr` and update tests

**Files:**
- Modify: `crates/cfvnet/src/datagen/sampler.rs:97-168` (the function)
- Modify: `crates/cfvnet/src/datagen/sampler.rs:303-330` (test `spr_stratified_covers_all_buckets`)

**Step 1: Write the failing test for 2D cell coverage**

Add this test at the end of the `tests` module in `sampler.rs` (after the `sample_board_5_cards` test, before the closing `}`):

```rust
    #[test]
    fn spr_and_pot_buckets_both_covered() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let pot_intervals = test_pot_intervals();
        let spr_intervals = test_spr_intervals();
        let num_spr = spr_intervals.len();
        let num_pot = pot_intervals.len();
        // 2D grid: cell[spr_idx][pot_idx]
        let mut cell_hits = vec![vec![0u32; num_pot]; num_spr];

        for _ in 0..5000 {
            let (pot, stack) = sample_pot_stack_by_spr(
                &pot_intervals, &spr_intervals, INITIAL_STACK, &mut rng,
            );
            let actual_spr = stack as f64 / pot as f64;
            let spr_idx = spr_intervals.iter().position(|[lo, hi]| actual_spr >= *lo && actual_spr < *hi);
            let pot_idx = pot_intervals.iter().position(|[lo, hi]| pot >= *lo && pot < *hi);
            if let (Some(si), Some(pi)) = (spr_idx, pot_idx) {
                cell_hits[si][pi] += 1;
            }
        }

        // Check pot bucket marginals are roughly uniform (no bucket < 10% of max)
        let mut pot_marginals = vec![0u32; num_pot];
        for si in 0..num_spr {
            for pi in 0..num_pot {
                pot_marginals[pi] += cell_hits[si][pi];
            }
        }
        let max_pot = *pot_marginals.iter().max().unwrap();
        for (pi, &count) in pot_marginals.iter().enumerate() {
            assert!(
                count * 4 >= max_pot,
                "pot bucket {} ({:?}) underrepresented: {} vs max {}",
                pi, pot_intervals[pi], count, max_pot
            );
        }

        // Check SPR bucket marginals are roughly uniform
        let mut spr_marginals = vec![0u32; num_spr];
        for si in 0..num_spr {
            for pi in 0..num_pot {
                spr_marginals[si] += cell_hits[si][pi];
            }
        }
        let max_spr = *spr_marginals.iter().max().unwrap();
        for (si, &count) in spr_marginals.iter().enumerate() {
            assert!(
                count * 4 >= max_spr,
                "SPR bucket {} ({:?}) underrepresented: {} vs max {}",
                si, spr_intervals[si], count, max_spr
            );
        }
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p cfvnet spr_and_pot_buckets_both_covered -- --nocapture`

Expected: FAIL — the pot bucket assertion will fire because the current implementation skews heavily toward small pots.

**Step 3: Rewrite `sample_pot_stack_by_spr`**

Replace lines 92-168 of `sampler.rs` with:

```rust
/// Sample a (pot, stack) pair via 2D-stratified rejection sampling.
///
/// Picks both an SPR bucket and a pot bucket uniformly, then intersects
/// the chosen pot bucket with the feasible pot range for the target SPR.
/// Infeasible (SPR, pot) cells are rejected and retried, producing a
/// distribution that is uniform across both dimensions.
fn sample_pot_stack_by_spr<R: Rng>(
    pot_intervals: &[[i32; 2]],
    spr_intervals: &[[f64; 2]],
    initial_stack: i32,
    rng: &mut R,
) -> (i32, i32) {
    let stack_f = initial_stack as f64;
    loop {
        // 1. Pick BOTH buckets uniformly
        let spr_idx = rng.gen_range(0..spr_intervals.len());
        let [spr_lo, spr_hi] = spr_intervals[spr_idx];
        let target_spr = rng.gen_range(spr_lo..spr_hi);

        let pot_idx = rng.gen_range(0..pot_intervals.len());
        let [pot_lo, pot_hi] = pot_intervals[pot_idx];

        // 2. Compute feasible pot range for this SPR
        let pot_min_f = if target_spr > 0.0 {
            (5.0 / target_spr).ceil()
        } else {
            (2.0 * (stack_f - 5.0)).ceil()
        };
        let pot_max_f = stack_f / (target_spr + 0.5);

        let feasible_lo = pot_min_f.max(1.0) as i32;
        let feasible_hi = pot_max_f as i32;

        // 3. Intersect chosen pot bucket with feasible range
        let lo = pot_lo.max(feasible_lo);
        let hi = pot_hi.min(feasible_hi + 1); // pot_intervals use exclusive upper bound
        if lo >= hi {
            continue;
        }

        // 4. Sample pot, derive stack
        let pot = rng.gen_range(lo..hi);
        let max_stack = initial_stack - pot / 2;
        if max_stack < 5 {
            continue;
        }
        let target_stack = (target_spr * pot as f64).round() as i32;
        let stack = target_stack.clamp(5, max_stack);

        // 5. Verify actual SPR lands in chosen bucket
        let actual_spr = stack as f64 / pot as f64;
        if actual_spr >= spr_lo && actual_spr < spr_hi {
            return (pot, stack);
        }
    }
}
```

**Step 4: Run ALL tests to verify they pass**

Run: `cargo test -p cfvnet`

Expected: ALL tests pass, including the new `spr_and_pot_buckets_both_covered` and all existing SPR tests.

**Step 5: Commit**

```bash
git add crates/cfvnet/src/datagen/sampler.rs
git commit -m "feat(cfvnet): 2D stratified sampling for uniform SPR and pot distribution

Pick both SPR and pot buckets uniformly before sampling, replacing
the old SPR-first approach that skewed heavily toward small pots.
Rejection sampling handles infeasible (SPR, pot) cells."
```
