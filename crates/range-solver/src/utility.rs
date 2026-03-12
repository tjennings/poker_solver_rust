use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use std::cell::Cell;
use std::mem::{self, MaybeUninit};
use std::ptr;

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// UtilityScratch — reusable allocations for exploitability checks
// ---------------------------------------------------------------------------

/// Pre-allocated buffers reused across `compute_mes_ev` / `compute_current_ev`
/// calls to avoid per-call heap allocations. Stored in thread-local storage.
struct UtilityScratch {
    cfvalues_0: Vec<f32>,
    cfvalues_1: Vec<f32>,
}

impl UtilityScratch {
    const fn new() -> Self {
        Self {
            cfvalues_0: Vec::new(),
            cfvalues_1: Vec::new(),
        }
    }
}

thread_local! {
    static UTILITY_SCRATCH: Cell<UtilityScratch> = const { Cell::new(UtilityScratch::new()) };
}

#[inline]
fn take_utility_scratch() -> UtilityScratch {
    UTILITY_SCRATCH.with(|cell| cell.replace(UtilityScratch::new()))
}

#[inline]
fn put_utility_scratch(scratch: UtilityScratch) {
    UTILITY_SCRATCH.with(|cell| cell.set(scratch));
}

// ---------------------------------------------------------------------------
// CfvScratch — reusable allocations for cfvalue/best-cfv recursion
// ---------------------------------------------------------------------------

/// Pre-allocated buffers reused across `compute_cfvalue_recursive` and
/// `compute_best_cfv_recursive` to avoid per-node heap allocations.
struct CfvScratch {
    /// Backing store for the flat cfv_actions matrix.
    cfv_buf: Vec<f32>,
    /// Chance-node cfreach update.
    cfreach_buf: Vec<f32>,
    /// Chance-node f64 accumulator for isomorphic-chance summation.
    result_f64_buf: Vec<f64>,
    /// Strategy vector (or opponent cfreach_actions).
    strategy_buf: Vec<f32>,
}

impl CfvScratch {
    const fn new() -> Self {
        Self {
            cfv_buf: Vec::new(),
            cfreach_buf: Vec::new(),
            result_f64_buf: Vec::new(),
            strategy_buf: Vec::new(),
        }
    }
}

thread_local! {
    static CFV_SCRATCH: Cell<CfvScratch> = const { Cell::new(CfvScratch::new()) };
}

#[inline]
fn take_cfv_scratch() -> CfvScratch {
    CFV_SCRATCH.with(|cell| cell.replace(CfvScratch::new()))
}

#[inline]
fn put_cfv_scratch(scratch: CfvScratch) {
    CFV_SCRATCH.with(|cell| cell.set(scratch));
}

/// Executes `op` for each child, potentially in parallel via rayon.
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    if node.enable_parallelization() {
        node.action_indices().into_par_iter().for_each(op);
    } else {
        node.action_indices().for_each(op);
    }
}

/// Computes the weighted sum of `values` using `weights`.
#[inline]
fn weighted_sum(values: &[f32], weights: &[f32]) -> f32 {
    let f = |sum: f64, (&v, &w): (&f32, &f32)| sum + v as f64 * w as f64;
    values.iter().zip(weights).fold(0.0, f) as f32
}

/// Computes the average of `slice` with the given `weights`.
#[inline]
pub fn compute_average(slice: &[f32], weights: &[f32]) -> f32 {
    let mut weight_sum = 0.0f64;
    let mut value_sum = 0.0f64;
    for (&v, &w) in slice.iter().zip(weights.iter()) {
        weight_sum += w as f64;
        value_sum += v as f64 * w as f64;
    }
    (value_sum / weight_sum) as f32
}

// ---------------------------------------------------------------------------
// Encoding: f32 <-> i16/u16 with scale
// ---------------------------------------------------------------------------

/// Returns the maximum absolute value in the slice.
#[inline]
fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 16 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        let mut tmp: [f32; 8] = slice[..8].try_into().unwrap();
        tmp.iter_mut().for_each(|x| *x = x.abs());
        let mut iter = slice[8..].chunks_exact(8);
        for chunk in iter.by_ref() {
            for i in 0..8 {
                tmp[i] = max(tmp[i], chunk[i].abs());
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
    }
}

/// Returns the maximum value of a non-negative slice.
#[inline]
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 16 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        let mut tmp: [f32; 8] = slice[..8].try_into().unwrap();
        let mut iter = slice[8..].chunks_exact(8);
        for chunk in iter.by_ref() {
            for i in 0..8 {
                tmp[i] = max(tmp[i], chunk[i]);
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
    }
}

/// Encodes an `f32` slice into an `i16` slice, returning the scale factor.
#[inline]
pub(crate) fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;
    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        // SAFETY: The encoder is chosen so that `s * encoder` fits in i32.
        *d = unsafe { (s * encoder).round().to_int_unchecked::<i32>() as i16 }
    });
    scale
}

/// Encodes an `f32` slice into a `u16` slice, returning the scale factor.
#[inline]
pub(crate) fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 {
    let scale = slice_nonnegative_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u16::MAX as f32 / scale_nonzero;
    // note: 0.49999997 + 0.49999997 = 0.99999994 < 1.0 | 0.5 + 0.49999997 = 1.0
    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        // SAFETY: The encoder is chosen so that `s * encoder + 0.49999997` fits in i32.
        *d = unsafe { (s * encoder + 0.49999997).to_int_unchecked::<i32>() as u16 }
    });
    scale
}

// ---------------------------------------------------------------------------
// Swap
// ---------------------------------------------------------------------------

/// Applies a list of index swaps to a slice.
#[inline]
pub(crate) fn apply_swap<T>(slice: &mut [T], swap_list: &[(u16, u16)]) {
    for &(i, j) in swap_list {
        // SAFETY: The swap list is constructed from valid hand indices during
        // isomorphism computation.
        unsafe {
            ptr::swap(
                slice.get_unchecked_mut(i as usize),
                slice.get_unchecked_mut(j as usize),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy normalization
// ---------------------------------------------------------------------------

/// Normalizes a cumulative strategy so each hand's action probabilities sum to 1.
#[inline]
pub(crate) fn normalized_strategy(strategy: &[f32], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), strategy);
    // SAFETY: sum_slices_uninit initializes all `row_size` elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(strategy.chunks_exact(row_size))
        .for_each(|(n, s)| {
            div_slice_uninit(n, s, &denom, default);
        });

    // SAFETY: All `strategy.len()` elements initialized by div_slice_uninit.
    unsafe { normalized.set_len(strategy.len()) };
    normalized
}

/// Normalizes a compressed (u16) cumulative strategy.
#[inline]
pub(crate) fn normalized_strategy_compressed(
    strategy: &[u16],
    num_actions: usize,
) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    // SAFETY: All elements initialized via `write` above.
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    // SAFETY: sum_slices_uninit initializes all `row_size` elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|r| {
        div_slice(r, &denom, default);
    });

    normalized
}

/// Normalizes a cumulative strategy into `buf`, reusing its allocation.
#[inline]
fn normalized_strategy_into(buf: &mut Vec<f32>, strategy: &[f32], num_actions: usize) {
    buf.clear();
    buf.reserve(strategy.len());
    let uninit = &mut buf.spare_capacity_mut()[..strategy.len()];

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), strategy);
    // SAFETY: sum_slices_uninit initializes all `row_size` elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(strategy.chunks_exact(row_size))
        .for_each(|(n, s)| {
            div_slice_uninit(n, s, &denom, default);
        });

    // SAFETY: All `strategy.len()` elements initialized by div_slice_uninit.
    unsafe { buf.set_len(strategy.len()) };
}

/// Normalizes a compressed (u16) cumulative strategy into `buf`, reusing its allocation.
#[inline]
fn normalized_strategy_compressed_into(
    buf: &mut Vec<f32>,
    strategy: &[u16],
    num_actions: usize,
) {
    buf.clear();
    buf.reserve(strategy.len());
    buf.extend(strategy.iter().map(|&s| s as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), buf);
    // SAFETY: sum_slices_uninit initializes all `row_size` elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    buf.chunks_exact_mut(row_size).for_each(|r| {
        div_slice(r, &denom, default);
    });
}

/// Applies a locking strategy: where `locking[i]` is non-negative, override
/// the corresponding strategy entry.
#[inline]
pub(crate) fn apply_locking_strategy(dst: &mut [f32], locking: &[f32]) {
    if !locking.is_empty() {
        dst.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = *s;
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Finalize
// ---------------------------------------------------------------------------

/// Finalizes the solving process: computes EV and normalizes the strategy.
#[inline]
pub fn finalize<T: Game>(game: &mut T) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    // Compute the expected values and save them.
    for player in 0..2 {
        let mut cfvalues = Vec::with_capacity(game.num_private_hands(player));
        compute_cfvalue_recursive(
            cfvalues.spare_capacity_mut(),
            game,
            &mut game.root(),
            player,
            game.initial_weights(player ^ 1),
            true,
        );
    }

    game.set_solved();
}

// ---------------------------------------------------------------------------
// Exploitability
// ---------------------------------------------------------------------------

/// Computes the exploitability of the current strategy.
#[inline]
pub fn compute_exploitability<T: Game>(game: &T) -> f32 {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mes_ev = compute_mes_ev(game);
    if !game.is_raked() {
        (mes_ev[0] + mes_ev[1]) * 0.5
    } else {
        let current_ev = compute_current_ev(game);
        ((mes_ev[0] - current_ev[0]) + (mes_ev[1] - current_ev[1])) * 0.5
    }
}

/// Computes the expected values of each player's current strategy.
///
/// The bias (starting_pot / 2) is already subtracted to increase significant
/// figures. This makes the return value zero-sum when not raked.
#[inline]
pub fn compute_current_ev<T: Game>(game: &T) -> [f32; 2] {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mut scratch = take_utility_scratch();

    scratch.cfvalues_0.clear();
    scratch.cfvalues_0.reserve(game.num_private_hands(0));
    scratch.cfvalues_1.clear();
    scratch.cfvalues_1.reserve(game.num_private_hands(1));

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    compute_cfvalue_recursive(
        scratch.cfvalues_0.spare_capacity_mut(),
        game,
        &mut game.root(),
        0,
        reach[1],
        false,
    );
    // SAFETY: compute_cfvalue_recursive writes all num_private_hands elements.
    unsafe { scratch.cfvalues_0.set_len(game.num_private_hands(0)) };

    compute_cfvalue_recursive(
        scratch.cfvalues_1.spare_capacity_mut(),
        game,
        &mut game.root(),
        1,
        reach[0],
        false,
    );
    // SAFETY: compute_cfvalue_recursive writes all num_private_hands elements.
    unsafe { scratch.cfvalues_1.set_len(game.num_private_hands(1)) };

    let ev0 = weighted_sum(&scratch.cfvalues_0, reach[0]);
    let ev1 = weighted_sum(&scratch.cfvalues_1, reach[1]);

    put_utility_scratch(scratch);

    [ev0, ev1]
}

/// Computes the expected values of the maximally exploitative strategy (MES).
///
/// The bias (starting_pot / 2) is already subtracted. The average of the
/// return values equals the exploitability when not raked.
#[inline]
pub fn compute_mes_ev<T: Game>(game: &T) -> [f32; 2] {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mut scratch = take_utility_scratch();

    scratch.cfvalues_0.clear();
    scratch.cfvalues_0.reserve(game.num_private_hands(0));
    scratch.cfvalues_1.clear();
    scratch.cfvalues_1.reserve(game.num_private_hands(1));

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    compute_best_cfv_recursive(
        scratch.cfvalues_0.spare_capacity_mut(),
        game,
        &game.root(),
        0,
        reach[1],
    );
    // SAFETY: compute_best_cfv_recursive writes all num_private_hands elements.
    unsafe { scratch.cfvalues_0.set_len(game.num_private_hands(0)) };

    compute_best_cfv_recursive(
        scratch.cfvalues_1.spare_capacity_mut(),
        game,
        &game.root(),
        1,
        reach[0],
    );
    // SAFETY: compute_best_cfv_recursive writes all num_private_hands elements.
    unsafe { scratch.cfvalues_1.set_len(game.num_private_hands(1)) };

    let ev0 = weighted_sum(&scratch.cfvalues_0, reach[0]);
    let ev1 = weighted_sum(&scratch.cfvalues_1, reach[1]);

    put_utility_scratch(scratch);

    [ev0, ev1]
}

// ---------------------------------------------------------------------------
// CFValue recursive (current strategy)
// ---------------------------------------------------------------------------

/// Recursively computes the counterfactual values of the current strategy.
///
/// Uses thread-local [`CfvScratch`] to avoid per-node heap allocations.
fn compute_cfvalue_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    save_cfvalues: bool,
) {
    // Terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Opponent node with single action: no scratch needed.
    if !node.is_chance() && node.player() != player && num_actions == 1 {
        compute_cfvalue_recursive(
            result,
            game,
            &mut node.play(0),
            player,
            cfreach,
            save_cfvalues,
        );
        return;
    }

    // Take scratch buffers from TLS.
    let mut scratch = take_cfv_scratch();

    // Prepare cfv_actions from scratch.
    let cfv_needed = num_actions * num_hands;
    let mut cfv_buf = mem::take(&mut scratch.cfv_buf);
    cfv_buf.clear();
    cfv_buf.reserve(cfv_needed);
    let cfv_mutex = MutexLike::new(cfv_buf);

    // -- Chance node --
    if node.is_chance() {
        let mut cfreach_updated = mem::take(&mut scratch.cfreach_buf);
        cfreach_updated.clear();
        cfreach_updated.reserve(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        // SAFETY: mul_slice_scalar_uninit writes all cfreach.len() elements.
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // Return remaining scratch to TLS so children can reuse it.
        put_cfv_scratch(scratch);

        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_mutex.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                save_cfvalues,
            );
        });

        // Re-take scratch for post-recursion work.
        let mut scratch = take_cfv_scratch();
        scratch.result_f64_buf.clear();
        scratch.result_f64_buf.reserve(num_hands);

        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };

        sum_slices_f64_uninit(
            &mut scratch.result_f64_buf.spare_capacity_mut()[..num_hands],
            &cfv_buf,
        );
        // SAFETY: sum_slices_f64_uninit writes all num_hands elements.
        unsafe { scratch.result_f64_buf.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);

        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_buf, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            scratch
                .result_f64_buf
                .iter_mut()
                .zip(&*tmp)
                .for_each(|(r, &v)| *r += v as f64);

            apply_swap(tmp, swap_list);
        }

        result
            .iter_mut()
            .zip(&scratch.result_f64_buf)
            .for_each(|(r, &v)| {
                r.write(v as f32);
            });

        // Save the counterfactual values.
        if save_cfvalues && node.cfvalue_storage_player() == Some(player) {
            // SAFETY: All elements of result have been written above.
            let result = unsafe { &*(result as *const _ as *const [f32]) };
            if game.is_compression_enabled() {
                let cfv_scale =
                    encode_signed_slice(node.cfvalues_chance_compressed_mut(), result);
                node.set_cfvalue_chance_scale(cfv_scale);
            } else {
                node.cfvalues_chance_mut().copy_from_slice(result);
            }
        }

        // Return buffers to scratch.
        scratch.cfv_buf = cfv_buf;
        scratch.cfreach_buf = cfreach_updated;
        put_cfv_scratch(scratch);
    }
    // Player node
    else if node.player() == player {
        // Return scratch to TLS before child recursion.
        put_cfv_scratch(scratch);

        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_mutex.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                save_cfvalues,
            );
        });

        // Re-take scratch for strategy normalization.
        let mut scratch = take_cfv_scratch();

        if game.is_compression_enabled() {
            normalized_strategy_compressed_into(
                &mut scratch.strategy_buf,
                node.strategy_compressed(),
                num_actions,
            );
        } else {
            normalized_strategy_into(
                &mut scratch.strategy_buf,
                node.strategy(),
                num_actions,
            );
        }

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut scratch.strategy_buf, locking);

        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };
        fma_slices_uninit(result, &scratch.strategy_buf, &cfv_buf);

        if save_cfvalues {
            if game.is_compression_enabled() {
                let cfv_scale =
                    encode_signed_slice(node.cfvalues_compressed_mut(), &cfv_buf);
                node.set_cfvalue_scale(cfv_scale);
            } else {
                node.cfvalues_mut().copy_from_slice(&cfv_buf);
            }
        }

        scratch.cfv_buf = cfv_buf;
        put_cfv_scratch(scratch);
    }
    // Opponent node (num_actions >= 2, handled single-action above)
    else {
        if game.is_compression_enabled() {
            normalized_strategy_compressed_into(
                &mut scratch.strategy_buf,
                node.strategy_compressed(),
                num_actions,
            );
        } else {
            normalized_strategy_into(
                &mut scratch.strategy_buf,
                node.strategy(),
                num_actions,
            );
        }

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut scratch.strategy_buf, locking);

        let row_size = cfreach.len();
        scratch
            .strategy_buf
            .chunks_exact_mut(row_size)
            .for_each(|r| mul_slice(r, cfreach));

        // Extract cfreach_actions as local for closure borrowing.
        let cfreach_actions = mem::take(&mut scratch.strategy_buf);
        put_cfv_scratch(scratch);

        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_mutex.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                save_cfvalues,
            );
        });

        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };
        sum_slices_uninit(result, &cfv_buf);

        // Return buffers to scratch.
        let mut scratch = take_cfv_scratch();
        scratch.cfv_buf = cfv_buf;
        scratch.strategy_buf = cfreach_actions;
        put_cfv_scratch(scratch);
    }

    // Save the counterfactual values for IP.
    // Note: scratch has been put back to TLS already in all branches above.
    if save_cfvalues && node.has_cfvalues_ip() && player == 1 {
        // SAFETY: All elements of result have been written by the branches above.
        let result = unsafe { &*(result as *const _ as *const [f32]) };
        if game.is_compression_enabled() {
            let cfv_scale = encode_signed_slice(node.cfvalues_ip_compressed_mut(), result);
            node.set_cfvalue_ip_scale(cfv_scale);
        } else {
            node.cfvalues_ip_mut().copy_from_slice(result);
        }
    }
}

// ---------------------------------------------------------------------------
// Best-response CFValue recursive
// ---------------------------------------------------------------------------

/// Recursively computes the counterfactual values of the best response.
///
/// Uses thread-local [`CfvScratch`] to avoid per-node heap allocations.
fn compute_best_cfv_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &[f32],
) {
    // Terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);

    // Single action (non-chance): just recurse, no scratch needed.
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_recursive(result, game, child, player, cfreach);
        return;
    }

    // Take scratch buffers from TLS.
    let mut scratch = take_cfv_scratch();

    let cfv_needed = num_actions * num_hands;
    let mut cfv_buf = mem::take(&mut scratch.cfv_buf);
    cfv_buf.clear();
    cfv_buf.reserve(cfv_needed);
    let cfv_mutex = MutexLike::new(cfv_buf);

    // -- Chance node --
    if node.is_chance() {
        let mut cfreach_updated = mem::take(&mut scratch.cfreach_buf);
        cfreach_updated.clear();
        cfreach_updated.reserve(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        // SAFETY: mul_slice_scalar_uninit writes all cfreach.len() elements.
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // Return remaining scratch to TLS so children can reuse it.
        put_cfv_scratch(scratch);

        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_mutex.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach_updated,
            );
        });

        // Re-take scratch for post-recursion work.
        let mut scratch = take_cfv_scratch();
        scratch.result_f64_buf.clear();
        scratch.result_f64_buf.reserve(num_hands);

        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };

        sum_slices_f64_uninit(
            &mut scratch.result_f64_buf.spare_capacity_mut()[..num_hands],
            &cfv_buf,
        );
        // SAFETY: sum_slices_f64_uninit writes all num_hands elements.
        unsafe { scratch.result_f64_buf.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);

        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_buf, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            scratch
                .result_f64_buf
                .iter_mut()
                .zip(&*tmp)
                .for_each(|(r, &v)| *r += v as f64);

            apply_swap(tmp, swap_list);
        }

        result
            .iter_mut()
            .zip(&scratch.result_f64_buf)
            .for_each(|(r, &v)| {
                r.write(v as f32);
            });

        // Return buffers to scratch.
        scratch.cfv_buf = cfv_buf;
        scratch.cfreach_buf = cfreach_updated;
        put_cfv_scratch(scratch);
    }
    // Player node: take the best response (max over actions)
    else if node.player() == player {
        // Return scratch to TLS before child recursion.
        put_cfv_scratch(scratch);

        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_mutex.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
            );
        });

        let locking = game.locking_strategy(node);
        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };

        if locking.is_empty() {
            max_slices_uninit(result, &cfv_buf);
        } else {
            max_fma_slices_uninit(result, &cfv_buf, locking);
        }

        // Return buffer to scratch.
        let mut scratch = take_cfv_scratch();
        scratch.cfv_buf = cfv_buf;
        put_cfv_scratch(scratch);
    }
    // Opponent node (num_actions >= 2, single-action handled above)
    else {
        if game.is_compression_enabled() {
            normalized_strategy_compressed_into(
                &mut scratch.strategy_buf,
                node.strategy_compressed(),
                num_actions,
            );
        } else {
            normalized_strategy_into(
                &mut scratch.strategy_buf,
                node.strategy(),
                num_actions,
            );
        }

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut scratch.strategy_buf, locking);

        let row_size = cfreach.len();
        scratch
            .strategy_buf
            .chunks_exact_mut(row_size)
            .for_each(|r| mul_slice(r, cfreach));

        // Extract cfreach_actions as local for closure borrowing.
        let cfreach_actions = mem::take(&mut scratch.strategy_buf);
        put_cfv_scratch(scratch);

        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_mutex.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
            );
        });

        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };
        sum_slices_uninit(result, &cfv_buf);

        // Return buffers to scratch.
        let mut scratch = take_cfv_scratch();
        scratch.cfv_buf = cfv_buf;
        scratch.strategy_buf = cfreach_actions;
        put_cfv_scratch(scratch);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_swap() {
        let mut data = vec![10, 20, 30, 40, 50];
        apply_swap(&mut data, &[(0, 4), (1, 3)]);
        assert_eq!(data, vec![50, 40, 30, 20, 10]);
    }

    #[test]
    fn test_apply_swap_self() {
        let mut data = vec![10, 20, 30];
        apply_swap(&mut data, &[(1, 1)]);
        assert_eq!(data, vec![10, 20, 30]);
    }

    #[test]
    fn test_normalized_strategy_uniform() {
        // All zeros -> uniform
        let strategy = vec![0.0; 6]; // 3 actions, 2 hands
        let result = normalized_strategy(&strategy, 3);
        let expected = 1.0 / 3.0;
        for &v in &result {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalized_strategy_nonuniform() {
        // 2 actions, 2 hands: [1, 3, 3, 1]
        let strategy = vec![1.0, 3.0, 3.0, 1.0];
        let result = normalized_strategy(&strategy, 2);
        // Hand 0: denom = 1+3 = 4, probs = [0.25, 0.75]
        // Hand 1: denom = 3+1 = 4, probs = [0.75, 0.25]
        assert!((result[0] - 0.25).abs() < 1e-6);
        assert!((result[1] - 0.75).abs() < 1e-6);
        assert!((result[2] - 0.75).abs() < 1e-6);
        assert!((result[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_apply_locking_strategy() {
        let mut dst = vec![0.5, 0.3, 0.2];
        let locking = vec![-1.0, 0.8, -1.0];
        apply_locking_strategy(&mut dst, &locking);
        assert_eq!(dst, vec![0.5, 0.8, 0.2]);
    }

    #[test]
    fn test_apply_locking_strategy_empty() {
        let mut dst = vec![0.5, 0.3, 0.2];
        apply_locking_strategy(&mut dst, &[]);
        assert_eq!(dst, vec![0.5, 0.3, 0.2]);
    }

    #[test]
    fn test_compute_average() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0, 1.0];
        let avg = compute_average(&values, &weights);
        assert!((avg - 2.0).abs() < 1e-6);

        let weights2 = vec![0.0, 0.0, 1.0];
        let avg2 = compute_average(&values, &weights2);
        assert!((avg2 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_encode_signed_slice() {
        let src = vec![1.0, -1.0, 0.5, -0.5];
        let mut dst = vec![0i16; 4];
        let scale = encode_signed_slice(&mut dst, &src);
        assert_eq!(scale, 1.0);
        assert_eq!(dst[0], i16::MAX);
        assert_eq!(dst[1], -i16::MAX);
    }

    #[test]
    fn test_encode_unsigned_slice() {
        let src = vec![1.0, 0.0, 0.5];
        let mut dst = vec![0u16; 3];
        let scale = encode_unsigned_slice(&mut dst, &src);
        assert_eq!(scale, 1.0);
        assert_eq!(dst[0], u16::MAX);
        assert_eq!(dst[1], 0);
    }
}
