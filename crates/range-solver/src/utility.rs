use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use std::mem::MaybeUninit;
use std::ptr;

use rayon::prelude::*;

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

    let mut cfvalues = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_cfvalue_recursive(
            cfvalues[player].spare_capacity_mut(),
            game,
            &mut game.root(),
            player,
            reach[player ^ 1],
            false,
        );
        // SAFETY: compute_cfvalue_recursive writes all num_private_hands elements.
        unsafe { cfvalues[player].set_len(game.num_private_hands(player)) };
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    [get_sum(0), get_sum(1)]
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

    let mut cfvalues = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_best_cfv_recursive(
            cfvalues[player].spare_capacity_mut(),
            game,
            &game.root(),
            player,
            reach[player ^ 1],
        );
        // SAFETY: compute_best_cfv_recursive writes all num_private_hands elements.
        unsafe { cfvalues[player].set_len(game.num_private_hands(player)) };
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    [get_sum(0), get_sum(1)]
}

// ---------------------------------------------------------------------------
// CFValue recursive (current strategy)
// ---------------------------------------------------------------------------

/// Recursively computes the counterfactual values of the current strategy.
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

    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // Chance node
    if node.is_chance() {
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        // SAFETY: mul_slice_scalar_uninit writes all cfreach.len() elements.
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                save_cfvalues,
            );
        });

        let mut result_f64 = Vec::with_capacity(num_hands);

        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        // SAFETY: sum_slices_f64_uninit writes all num_hands elements.
        unsafe { result_f64.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);

        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
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
    }
    // Player node
    else if node.player() == player {
        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                save_cfvalues,
            );
        });

        let mut strategy = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        fma_slices_uninit(result, &strategy, &cfv_actions);

        if save_cfvalues {
            if game.is_compression_enabled() {
                let cfv_scale =
                    encode_signed_slice(node.cfvalues_compressed_mut(), &cfv_actions);
                node.set_cfvalue_scale(cfv_scale);
            } else {
                node.cfvalues_mut().copy_from_slice(&cfv_actions);
            }
        }
    }
    // Opponent node with single action
    else if num_actions == 1 {
        compute_cfvalue_recursive(
            result,
            game,
            &mut node.play(0),
            player,
            cfreach,
            save_cfvalues,
        );
    }
    // Opponent node
    else {
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|r| {
            mul_slice(r, cfreach);
        });

        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                save_cfvalues,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }

    // Save the counterfactual values for IP.
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

    // Single action (non-chance): just recurse
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_recursive(result, game, child, player, cfreach);
        return;
    }

    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // Chance node
    if node.is_chance() {
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        // SAFETY: mul_slice_scalar_uninit writes all cfreach.len() elements.
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach_updated,
            );
        });

        let mut result_f64 = Vec::with_capacity(num_hands);

        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        // SAFETY: sum_slices_f64_uninit writes all num_hands elements.
        unsafe { result_f64.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);

        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
    }
    // Player node: take the best response (max over actions)
    else if node.player() == player {
        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
            );
        });

        let locking = game.locking_strategy(node);
        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        if locking.is_empty() {
            max_slices_uninit(result, &cfv_actions);
        } else {
            max_fma_slices_uninit(result, &cfv_actions, locking);
        }
    }
    // Opponent node
    else {
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|r| {
            mul_slice(r, cfreach);
        });

        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
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
