use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::cell::Cell;
use std::io::{self, Write};
use std::mem::{self, MaybeUninit};

// Re-export utility functions that belong to the solver's public API.
pub use crate::utility::{compute_exploitability, compute_average, compute_current_ev, finalize};

// ---------------------------------------------------------------------------
// Discount parameters
// ---------------------------------------------------------------------------

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    pub fn new(current_iteration: u32) -> Self {
        // 0, 1, 4, 16, 64, 256, ...
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;

        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

// ---------------------------------------------------------------------------
// ScratchBuffers — reusable allocations for solve_recursive
// ---------------------------------------------------------------------------

/// Pre-allocated buffers reused across recursive calls to avoid per-call
/// heap allocations. Stored in thread-local storage so each rayon worker
/// thread maintains its own set.
///
/// At each recursion level the buffers are *taken* from TLS, used, and
/// *put back* before/after child recursion. In the parallel (`for_each_child`)
/// case each rayon worker has its own thread-local set, so there is no
/// contention.
struct ScratchBuffers {
    /// Backing store for the flat cfv_actions matrix.
    cfv_buf: Vec<f32>,
    /// Strategy vector (or opponent cfreach_actions).
    strategy_buf: Vec<f32>,
    /// Denominator accumulator for regret-matching normalisation.
    denom_buf: Vec<f32>,
    /// Chance-node cfreach update.
    cfreach_buf: Vec<f32>,
    /// Chance-node f64 accumulator for isomorphic-chance summation.
    result_f64_buf: Vec<f64>,
}

impl ScratchBuffers {
    const fn new() -> Self {
        Self {
            cfv_buf: Vec::new(),
            strategy_buf: Vec::new(),
            denom_buf: Vec::new(),
            cfreach_buf: Vec::new(),
            result_f64_buf: Vec::new(),
        }
    }
}

thread_local! {
    static SCRATCH: Cell<ScratchBuffers> = const { Cell::new(ScratchBuffers::new()) };
}

/// Takes the thread-local scratch buffers, replacing with empty buffers.
#[inline]
fn take_scratch() -> ScratchBuffers {
    SCRATCH.with(|cell| cell.replace(ScratchBuffers::new()))
}

/// Returns scratch buffers to thread-local storage for reuse.
#[inline]
fn put_scratch(scratch: ScratchBuffers) {
    SCRATCH.with(|cell| cell.set(scratch));
}

// ---------------------------------------------------------------------------
// solve() — main entry point
// ---------------------------------------------------------------------------

/// Performs the Discounted CFR algorithm until the given number of iterations
/// or target exploitability is reached.
///
/// Returns the exploitability of the obtained strategy.
pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);

    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        print!("(exploitability = {exploitability:.4e})");
        io::stdout().flush().unwrap();
    }

    let mut result_bufs: [Vec<f32>; 2] = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        // Alternating updates
        for player in 0..2 {
            result_bufs[player].clear();
            result_bufs[player].reserve(game.num_private_hands(player));
            solve_recursive(
                result_bufs[player].spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if (t + 1) % 5 == 0 || t + 1 == max_num_iterations {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            print!("(exploitability = {exploitability:.4e})");
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }

    finalize(game);

    exploitability
}

// ---------------------------------------------------------------------------
// solve_step() — single iteration
// ---------------------------------------------------------------------------

/// Proceeds the Discounted CFR algorithm for one iteration.
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);

    // Alternating updates
    for player in 0..2 {
        let mut result = Vec::with_capacity(game.num_private_hands(player));
        result.reserve(game.num_private_hands(player));
        solve_recursive(
            result.spare_capacity_mut(),
            game,
            &mut root,
            player,
            game.initial_weights(player ^ 1),
            &params,
        );
    }
}

// ---------------------------------------------------------------------------
// solve_recursive() — core CFR traversal
// ---------------------------------------------------------------------------

/// Recursively solves the counterfactual values.
///
/// Uses thread-local [`ScratchBuffers`] to avoid per-call heap allocations.
/// Buffers that must outlive `for_each_child` are extracted as locals before
/// scratch is returned to TLS; children in sequential or parallel branches
/// each acquire their own scratch from TLS.
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // Return the counterfactual values when the node is terminal.
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Simply recurse when there is only one action (non-chance).
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive(result, game, child, player, cfreach, params);
        return;
    }

    // Take scratch buffers from thread-local storage.
    let mut scratch = take_scratch();

    // Prepare cfv_actions: extract cfv_buf, clear, reserve, wrap in MutexLike.
    let cfv_needed = num_actions * num_hands;
    let mut cfv_buf = mem::take(&mut scratch.cfv_buf);
    cfv_buf.clear();
    cfv_buf.reserve(cfv_needed);
    let cfv_mutex = MutexLike::new(cfv_buf);

    // -- Chance node --
    if node.is_chance() {
        // Prepare cfreach_updated, then extract as local so it can be
        // borrowed inside the for_each_child closure.
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
        put_scratch(scratch);

        // Compute the counterfactual values of each action.
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(
                    cfv_mutex.lock().spare_capacity_mut(),
                    action,
                    num_hands,
                ),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
            );
        });

        // Re-take scratch for post-recursion work.
        let mut scratch = take_scratch();
        scratch.result_f64_buf.clear();
        scratch.result_f64_buf.reserve(num_hands);

        // Extract Vec from MutexLike.
        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };

        // Sum up the counterfactual values in f64.
        sum_slices_f64_uninit(
            &mut scratch.result_f64_buf.spare_capacity_mut()[..num_hands],
            &cfv_buf,
        );
        // SAFETY: sum_slices_f64_uninit writes all num_hands elements.
        unsafe { scratch.result_f64_buf.set_len(num_hands) };

        // Process isomorphic chances.
        let isomorphic_chances = game.isomorphic_chances(node);
        for (i, &iso_idx) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_buf, iso_idx as usize, num_hands);

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
            .for_each(|(r, &v)| { r.write(v as f32); });

        // Return buffers to scratch and put back to TLS.
        scratch.cfv_buf = cfv_buf;
        scratch.cfreach_buf = cfreach_updated;
        put_scratch(scratch);
    }
    // -- Current player's decision node --
    else if node.player() == player {
        // Return scratch to TLS before child recursion.
        put_scratch(scratch);

        for_each_child(node, |action| {
            solve_recursive(
                row_mut(
                    cfv_mutex.lock().spare_capacity_mut(),
                    action,
                    num_hands,
                ),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
            );
        });

        // Re-take scratch for regret matching and updates.
        let mut scratch = take_scratch();

        // Compute strategy by regret-matching.
        if game.is_compression_enabled() {
            regret_matching_compressed_into(
                node.regrets_compressed(),
                num_actions,
                &mut scratch.strategy_buf,
                &mut scratch.denom_buf,
            );
        } else {
            regret_matching_into(
                node.regrets(),
                num_actions,
                &mut scratch.strategy_buf,
                &mut scratch.denom_buf,
            );
        }

        // Node-locking.
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut scratch.strategy_buf, locking);

        // Extract Vec from MutexLike.
        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };

        let result =
            fma_slices_uninit(result, &scratch.strategy_buf, &cfv_buf);

        if game.is_compression_enabled() {
            update_compressed_strategy(
                node,
                params,
                &mut scratch.strategy_buf,
                locking,
            );
            update_compressed_regret(
                node,
                params,
                &mut cfv_buf,
                result,
                num_hands,
                locking,
            );
        } else {
            update_uncompressed_strategy(
                node,
                params,
                &scratch.strategy_buf,
            );
            update_uncompressed_regret(
                node,
                params,
                &cfv_buf,
                result,
                num_hands,
            );
        }

        scratch.cfv_buf = cfv_buf;
        put_scratch(scratch);
    }
    // -- Opponent's decision node --
    else {
        // Compute strategy into scratch.strategy_buf.
        if game.is_compression_enabled() {
            regret_matching_compressed_into(
                node.regrets_compressed(),
                num_actions,
                &mut scratch.strategy_buf,
                &mut scratch.denom_buf,
            );
        } else {
            regret_matching_into(
                node.regrets(),
                num_actions,
                &mut scratch.strategy_buf,
                &mut scratch.denom_buf,
            );
        }

        // Node-locking.
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut scratch.strategy_buf, locking);

        // Multiply reach probabilities into strategy to get cfreach_actions.
        let row_size = cfreach.len();
        scratch
            .strategy_buf
            .chunks_exact_mut(row_size)
            .for_each(|r| mul_slice(r, cfreach));

        // Extract cfreach_actions as local for closure borrowing.
        let cfreach_actions = mem::take(&mut scratch.strategy_buf);
        put_scratch(scratch);

        for_each_child(node, |action| {
            solve_recursive(
                row_mut(
                    cfv_mutex.lock().spare_capacity_mut(),
                    action,
                    num_hands,
                ),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });

        // Extract Vec from MutexLike.
        let mut cfv_buf = cfv_mutex.into_inner();
        // SAFETY: All cfv_needed elements initialized by child recursion.
        unsafe { cfv_buf.set_len(cfv_needed) };

        sum_slices_uninit(result, &cfv_buf);

        // Return buffers to scratch.
        let mut scratch = take_scratch();
        scratch.cfv_buf = cfv_buf;
        scratch.strategy_buf = cfreach_actions;
        put_scratch(scratch);
    }
}

// ---------------------------------------------------------------------------
// Strategy / regret update helpers (player branch)
// ---------------------------------------------------------------------------

#[inline]
fn update_compressed_strategy<N: GameNode>(
    node: &mut N,
    params: &DiscountParams,
    strategy: &mut [f32],
    locking: &[f32],
) {
    let scale = node.strategy_scale();
    let decoder = params.gamma_t * scale / u16::MAX as f32;
    let cum_strategy = node.strategy_compressed_mut();

    strategy
        .iter_mut()
        .zip(&*cum_strategy)
        .for_each(|(x, y)| *x += (*y as f32) * decoder);

    if !locking.is_empty() {
        strategy.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        });
    }

    let new_scale = encode_unsigned_slice(cum_strategy, strategy);
    node.set_strategy_scale(new_scale);
}

#[inline]
fn update_compressed_regret<N: GameNode>(
    node: &mut N,
    params: &DiscountParams,
    cfv_buf: &mut [f32],
    result: &[f32],
    num_hands: usize,
    locking: &[f32],
) {
    let scale = node.regret_scale();
    let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
    let beta_decoder = params.beta_t * scale / i16::MAX as f32;
    let cum_regret = node.regrets_compressed_mut();

    cfv_buf
        .iter_mut()
        .zip(&*cum_regret)
        .for_each(|(x, y)| {
            *x += *y as f32
                * if *y >= 0 {
                    alpha_decoder
                } else {
                    beta_decoder
                };
        });

    cfv_buf
        .chunks_exact_mut(num_hands)
        .for_each(|r| sub_slice(r, result));

    if !locking.is_empty() {
        cfv_buf.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        });
    }

    let new_scale = encode_signed_slice(cum_regret, cfv_buf);
    node.set_regret_scale(new_scale);
}

#[inline]
fn update_uncompressed_strategy<N: GameNode>(
    node: &mut N,
    params: &DiscountParams,
    strategy: &[f32],
) {
    let gamma = params.gamma_t;
    let cum_strategy = node.strategy_mut();
    cum_strategy
        .iter_mut()
        .zip(strategy)
        .for_each(|(x, y)| *x = *x * gamma + *y);
}

#[inline]
fn update_uncompressed_regret<N: GameNode>(
    node: &mut N,
    params: &DiscountParams,
    cfv_buf: &[f32],
    result: &[f32],
    num_hands: usize,
) {
    let (alpha, beta) = (params.alpha_t, params.beta_t);
    let cum_regret = node.regrets_mut();
    cum_regret
        .iter_mut()
        .zip(cfv_buf)
        .for_each(|(x, y)| {
            let coef = if x.is_sign_positive() { alpha } else { beta };
            *x = *x * coef + *y;
        });
    cum_regret
        .chunks_exact_mut(num_hands)
        .for_each(|r| sub_slice(r, result));
}

// ---------------------------------------------------------------------------
// Regret matching (buffer-reusing variants)
// ---------------------------------------------------------------------------

/// Computes the strategy by regret-matching into a pre-allocated buffer.
///
/// Writes `max(regret, 0)` normalized per-hand into `strategy`, reusing
/// `denom` as the denominator accumulator.
#[inline]
fn regret_matching_into(
    regret: &[f32],
    num_actions: usize,
    strategy: &mut Vec<f32>,
    denom: &mut Vec<f32>,
) {
    strategy.clear();
    strategy.reserve(regret.len());
    let uninit = &mut strategy.spare_capacity_mut()[..regret.len()];
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    // SAFETY: All regret.len() elements written above.
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    denom.clear();
    denom.reserve(row_size);
    sum_slices_uninit(&mut denom.spare_capacity_mut()[..row_size], strategy);
    // SAFETY: sum_slices_uninit writes all row_size elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|r| {
        div_slice(r, denom, default);
    });
}

/// Computes the strategy by regret-matching from compressed (i16) regrets
/// into a pre-allocated buffer.
#[inline]
fn regret_matching_compressed_into(
    regret: &[i16],
    num_actions: usize,
    strategy: &mut Vec<f32>,
    denom: &mut Vec<f32>,
) {
    strategy.clear();
    strategy.reserve(regret.len());
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    let row_size = strategy.len() / num_actions;
    denom.clear();
    denom.reserve(row_size);
    sum_slices_uninit(&mut denom.spare_capacity_mut()[..row_size], strategy);
    // SAFETY: sum_slices_uninit writes all row_size elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|r| {
        div_slice(r, denom, default);
    });
}

/// Original regret-matching returning a new Vec (used by tests).
#[cfg(test)]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::new();
    let mut denom = Vec::new();
    regret_matching_into(regret, num_actions, &mut strategy, &mut denom);
    strategy
}

/// Original compressed regret-matching returning a new Vec (used by tests).
#[cfg(test)]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::new();
    let mut denom = Vec::new();
    regret_matching_compressed_into(
        regret,
        num_actions,
        &mut strategy,
        &mut denom,
    );
    strategy
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PostFlopGame;

    #[test]
    fn test_discount_params_iteration_0() {
        let p = DiscountParams::new(0);
        assert_eq!(p.beta_t, 0.5);
        assert_eq!(p.gamma_t, 0.0);
        assert_eq!(p.alpha_t, 0.0);
    }

    #[test]
    fn test_discount_params_iteration_1() {
        let p = DiscountParams::new(1);
        assert_eq!(p.alpha_t, 0.0);
        assert_eq!(p.beta_t, 0.5);
    }

    #[test]
    fn test_discount_params_large_iteration() {
        let p = DiscountParams::new(100);
        assert!(
            p.alpha_t > 0.9,
            "alpha_t should be close to 1 at t=100: {}",
            p.alpha_t
        );
        assert_eq!(p.beta_t, 0.5);
        assert!(p.gamma_t > 0.0, "gamma_t should be positive");
    }

    #[test]
    fn test_discount_params_power_of_4() {
        let p = DiscountParams::new(4);
        assert_eq!(p.gamma_t, 0.0);

        let p = DiscountParams::new(5);
        assert!((p.gamma_t - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_regret_matching_all_negative() {
        let regret = vec![-1.0, -2.0, -3.0, -1.0, -2.0, -3.0];
        let strategy = regret_matching(&regret, 3);
        let expected = 1.0 / 3.0;
        for &s in &strategy {
            assert!((s - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_regret_matching_positive() {
        let regret = vec![1.0, 0.0, 0.0, 3.0];
        let strategy = regret_matching(&regret, 2);
        assert!((strategy[0] - 1.0).abs() < 1e-6);
        assert!((strategy[1] - 0.0).abs() < 1e-6);
        assert!((strategy[2] - 0.0).abs() < 1e-6);
        assert!((strategy[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_regret_matching_mixed() {
        let regret = vec![2.0, 1.0];
        let strategy = regret_matching(&regret, 2);
        assert!((strategy[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((strategy[1] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_regret_matching_compressed_all_negative() {
        let regret: Vec<i16> = vec![-100, -200, -300, -100, -200, -300];
        let strategy = regret_matching_compressed(&regret, 3);
        let expected = 1.0 / 3.0;
        for &s in &strategy {
            assert!((s - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_regret_matching_compressed_positive() {
        let regret: Vec<i16> = vec![100, 0, 0, 200];
        let strategy = regret_matching_compressed(&regret, 2);
        assert!((strategy[0] - 1.0).abs() < 1e-6);
        assert!((strategy[1] - 0.0).abs() < 1e-6);
        assert!((strategy[2] - 0.0).abs() < 1e-6);
        assert!((strategy[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_solver_convergence_river() {
        use crate::action_tree::*;
        use crate::bet_size::*;
        use crate::card::*;

        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range =
            "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game =
            PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        let expl = solve(&mut game, 100, 0.0, false);
        assert!(
            expl < 10.0,
            "Exploitability should decrease below 10.0 after 100 iters: {expl}"
        );
    }

    #[test]
    fn test_solver_early_termination() {
        use crate::action_tree::*;
        use crate::bet_size::*;
        use crate::card::*;

        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range =
            "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game =
            PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        let expl = solve(&mut game, 1000, 1e10, false);
        assert!(expl >= 0.0);
    }

    #[test]
    fn test_solve_step_does_not_panic() {
        use crate::action_tree::*;
        use crate::bet_size::*;
        use crate::card::*;

        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range =
            "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game =
            PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        for i in 0..10 {
            solve_step(&game, i);
        }

        let expl = compute_exploitability(&game);
        assert!(expl >= 0.0, "exploitability should be non-negative");
    }

    #[test]
    fn test_solver_convergence_depth_limited_turn() {
        use crate::action_tree::*;
        use crate::bet_size::*;
        use crate::card::*;

        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range =
            "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_bet_sizes: [sizes.clone(), sizes],
            depth_limit: Some(0),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game =
            PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        // Set uniform boundary CFVs (value = 0.0 for all hands)
        let n_boundary = game.num_boundary_nodes();
        for ordinal in 0..n_boundary {
            game.set_boundary_cfvs(ordinal, 0, vec![0.0; game.num_private_hands(0)]);
            game.set_boundary_cfvs(ordinal, 1, vec![0.0; game.num_private_hands(1)]);
        }

        let expl = solve(&mut game, 100, 0.0, false);
        assert!(
            expl < 10.0,
            "Exploitability should decrease below 10.0 after 100 iters: {expl}"
        );
    }

    // -----------------------------------------------------------------
    // BoundaryEvaluator trait extension tests
    // -----------------------------------------------------------------

    #[test]
    fn multi_continuation_solve_step_no_panic() {
        use crate::action_tree::*;
        use crate::bet_size::*;
        use crate::card::*;

        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range =
            "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_bet_sizes: [sizes.clone(), sizes],
            depth_limit: Some(0),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game =
            PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        let k = 4;
        let n_boundary = game.num_boundary_nodes();
        game.init_multi_continuation(k);

        // Set distinct CFVs per continuation
        for ordinal in 0..n_boundary {
            for cont in 0..k {
                let val = (cont as f32 + 1.0) * 0.1;
                game.set_boundary_cfvs_multi(ordinal, 0, cont, vec![val; game.num_private_hands(0)]);
                game.set_boundary_cfvs_multi(ordinal, 1, cont, vec![val; game.num_private_hands(1)]);
            }
        }

        // Run 10 solve steps — should not panic
        for i in 0..10 {
            solve_step(&game, i);
        }

        let expl = compute_exploitability(&game);
        assert!(expl.is_finite(), "exploitability should be finite after multi-continuation solve");
    }

    #[test]
    fn boundary_evaluator_default_num_continuations_is_one() {
        use crate::game::BoundaryEvaluator;

        struct SingleCont;
        impl BoundaryEvaluator for SingleCont {
            fn compute_cfvs(
                &self, _player: usize, _pot: i32, _remaining: f64,
                _opp_reach: &[f32], _num_hands: usize,
                _continuation_index: usize,
            ) -> Vec<f32> {
                vec![0.0]
            }
        }

        let eval = SingleCont;
        assert_eq!(eval.num_continuations(), 1);
    }

    #[test]
    fn boundary_evaluator_supports_continuation_index() {
        use crate::game::BoundaryEvaluator;

        struct MultiBoundary;
        impl BoundaryEvaluator for MultiBoundary {
            fn num_continuations(&self) -> usize { 4 }
            fn compute_cfvs(
                &self, player: usize, _pot: i32, _remaining: f64,
                _opp_reach: &[f32], _num_hands: usize,
                continuation_index: usize,
            ) -> Vec<f32> {
                // Return different values for each continuation
                vec![(continuation_index as f32 + 1.0) * (player as f32 + 1.0)]
            }
        }

        let eval = MultiBoundary;
        assert_eq!(eval.num_continuations(), 4);
        let v0 = eval.compute_cfvs(0, 100, 50.0, &[1.0], 1, 0);
        let v1 = eval.compute_cfvs(0, 100, 50.0, &[1.0], 1, 1);
        assert_ne!(v0[0], v1[0]); // different continuations give different values
    }
}
