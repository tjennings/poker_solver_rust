use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;

// Re-export utility functions that belong to the solver's public API.
pub use crate::utility::{compute_exploitability, compute_average, finalize};

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

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        // Alternating updates
        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_num_iterations {
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

    // Allocate memory for storing the counterfactual values.
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // -- Chance node --
    if node.is_chance() {
        // Update the reach probabilities.
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        // SAFETY: mul_slice_scalar_uninit writes all cfreach.len() elements.
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // Compute the counterfactual values of each action.
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
            );
        });

        // Use 64-bit floating point values.
        let mut result_f64 = Vec::with_capacity(num_hands);

        // Sum up the counterfactual values.
        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        // SAFETY: sum_slices_f64_uninit writes all num_hands elements.
        unsafe { result_f64.set_len(num_hands) };

        // Get information about isomorphic chances.
        let isomorphic_chances = game.isomorphic_chances(node);

        // Process isomorphic chances.
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
    // -- Current player's decision node --
    else if node.player() == player {
        // Compute the counterfactual values of each action.
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
            );
        });

        // Compute the strategy by regret-matching algorithm.
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // Node-locking.
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // Sum up the counterfactual values.
        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        if game.is_compression_enabled() {
            // Update the cumulative strategy.
            let scale = node.strategy_scale();
            let decoder = params.gamma_t * scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();

            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });

            if !locking.is_empty() {
                strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() {
                        *d = 0.0;
                    }
                });
            }

            let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
            node.set_strategy_scale(new_scale);

            // Update the cumulative regret.
            let scale = node.regret_scale();
            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
            let beta_decoder = params.beta_t * scale / i16::MAX as f32;
            let cum_regret = node.regrets_compressed_mut();

            cfv_actions
                .iter_mut()
                .zip(&*cum_regret)
                .for_each(|(x, y)| {
                    *x += *y as f32 * if *y >= 0 { alpha_decoder } else { beta_decoder };
                });

            cfv_actions.chunks_exact_mut(num_hands).for_each(|r| {
                sub_slice(r, result);
            });

            if !locking.is_empty() {
                cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() {
                        *d = 0.0;
                    }
                });
            }

            let new_scale = encode_signed_slice(cum_regret, &cfv_actions);
            node.set_regret_scale(new_scale);
        } else {
            // Update the cumulative strategy.
            let gamma = params.gamma_t;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x = *x * gamma + *y;
            });

            // Update the cumulative regret.
            let (alpha, beta) = (params.alpha_t, params.beta_t);
            let cum_regret = node.regrets_mut();
            cum_regret
                .iter_mut()
                .zip(&*cfv_actions)
                .for_each(|(x, y)| {
                    let coef = if x.is_sign_positive() { alpha } else { beta };
                    *x = *x * coef + *y;
                });
            cum_regret.chunks_exact_mut(num_hands).for_each(|r| {
                sub_slice(r, result);
            });
        }
    }
    // -- Opponent's decision node --
    else {
        // Compute the strategy by regret-matching algorithm.
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // Node-locking.
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // Update the reach probabilities.
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|r| {
            mul_slice(r, cfreach);
        });

        // Compute the counterfactual values of each action.
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });

        // Sum up the counterfactual values.
        let mut cfv_actions = cfv_actions.lock();
        // SAFETY: All num_actions * num_hands elements initialized by recursion.
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

// ---------------------------------------------------------------------------
// Regret matching
// ---------------------------------------------------------------------------

/// Computes the strategy by regret-matching: max(regret, 0) then normalize.
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    // SAFETY: All regret.len() elements written above.
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    // SAFETY: sum_slices_uninit writes all row_size elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|r| {
        div_slice(r, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching from compressed (i16) regrets.
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    // SAFETY: sum_slices_uninit writes all row_size elements.
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|r| {
        div_slice(r, &denom, default);
    });

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
        // t_gamma=0, (0/1)^3=0
        assert_eq!(p.gamma_t, 0.0);
        // t_alpha=(0-1).max(0)=0, pow_alpha=0, alpha_t=0/(0+1)=0
        assert_eq!(p.alpha_t, 0.0);
    }

    #[test]
    fn test_discount_params_iteration_1() {
        let p = DiscountParams::new(1);
        // t_alpha=(1-1).max(0)=0, pow_alpha=0, alpha_t=0
        assert_eq!(p.alpha_t, 0.0);
        assert_eq!(p.beta_t, 0.5);
    }

    #[test]
    fn test_discount_params_large_iteration() {
        let p = DiscountParams::new(100);
        assert!(p.alpha_t > 0.9, "alpha_t should be close to 1 at t=100: {}", p.alpha_t);
        assert_eq!(p.beta_t, 0.5);
        assert!(p.gamma_t > 0.0, "gamma_t should be positive");
    }

    #[test]
    fn test_discount_params_power_of_4() {
        // At iteration 4, nearest_lower_power_of_4 = 4
        // t_gamma = 4 - 4 = 0, pow_gamma = (0/1)^3 = 0, gamma_t = 0
        let p = DiscountParams::new(4);
        assert_eq!(p.gamma_t, 0.0);

        // At iteration 5, nearest_lower_power_of_4 = 4
        // t_gamma = 5 - 4 = 1, pow_gamma = (1/2)^3 = 0.125
        let p = DiscountParams::new(5);
        assert!((p.gamma_t - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_regret_matching_all_negative() {
        // All negative regrets -> uniform strategy
        let regret = vec![-1.0, -2.0, -3.0, -1.0, -2.0, -3.0]; // 3 actions, 2 hands
        let strategy = regret_matching(&regret, 3);
        let expected = 1.0 / 3.0;
        for &s in &strategy {
            assert!((s - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_regret_matching_positive() {
        // 2 actions, 2 hands: regret = [1, 0, 0, 3]
        // hand 0: max(1,0)=1, max(0,0)=0, denom=1, probs=[1, 0]
        // hand 1: max(0,0)=0, max(3,0)=3, denom=3, probs=[0, 1]
        let regret = vec![1.0, 0.0, 0.0, 3.0];
        let strategy = regret_matching(&regret, 2);
        assert!((strategy[0] - 1.0).abs() < 1e-6);
        assert!((strategy[1] - 0.0).abs() < 1e-6);
        assert!((strategy[2] - 0.0).abs() < 1e-6);
        assert!((strategy[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_regret_matching_mixed() {
        // 2 actions, 1 hand: regret = [2, 1]
        // max(2,0)=2, max(1,0)=1, denom=3, probs=[2/3, 1/3]
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
        // 2 actions, 2 hands: [100, 0, 0, 200]
        // hand 0: max(100,0)=100, max(0,0)=0, denom=100, probs=[1.0, 0.0]
        // hand 1: max(0,0)=0, max(200,0)=200, denom=200, probs=[0.0, 1.0]
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
        let ip_range: crate::range::Range = "QQ-JJ,AQs,AJs".parse().unwrap();
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
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
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
        let ip_range: crate::range::Range = "QQ-JJ,AQs,AJs".parse().unwrap();
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
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        // Set a very high target so it terminates immediately
        let expl = solve(&mut game, 1000, 1e10, false);
        // Should have stopped after computing initial exploitability
        assert!(expl >= 0.0);
    }

    #[test]
    fn test_solve_step_does_not_panic() {
        use crate::action_tree::*;
        use crate::bet_size::*;
        use crate::card::*;

        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range = "QQ-JJ,AQs,AJs".parse().unwrap();
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
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        // Run a few steps manually
        for i in 0..10 {
            solve_step(&game, i);
        }

        let expl = compute_exploitability(&game);
        assert!(expl >= 0.0, "exploitability should be non-negative");
    }
}
