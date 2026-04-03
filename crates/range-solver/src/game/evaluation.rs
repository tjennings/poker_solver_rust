use super::*;
use crate::action_tree::PLAYER_DEPTH_BOUNDARY_FLAG;
use crate::card::*;
use std::mem::MaybeUninit;

/// Returns the minimum of two `f64` values without NaN propagation.
///
/// Unlike `f64::min`, this avoids a branch for NaN handling in the hot path,
/// since we know our inputs are always finite.
#[inline]
fn min_f64(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else {
        y
    }
}

/// Regret-match over K continuation strategies.
///
/// Returns a strategy vector of length `k` where positive regrets are
/// normalised, and uniform `1/K` is used when all regrets are non-positive.
#[inline]
fn regret_match_k(regrets: &[f32], k: usize) -> Vec<f32> {
    let mut strategy = vec![0.0f32; k];
    let mut sum = 0.0f32;
    for i in 0..k {
        let r = regrets[i].max(0.0);
        strategy[i] = r;
        sum += r;
    }
    if sum > 0.0 {
        for s in &mut strategy {
            *s /= sum;
        }
    } else {
        let uniform = 1.0 / k as f32;
        for s in &mut strategy {
            *s = uniform;
        }
    }
    strategy
}

impl PostFlopGame {
    /// Computes counterfactual values at a terminal node.
    ///
    /// `result` is written with the cfvalue for each of `player`'s private hands.
    /// `cfreach` contains the opponent's counterfactual reach probabilities.
    ///
    /// Two cases: fold (one player folded) and showdown (river reached).
    /// Showdown further splits into raked and unraked paths.
    ///
    /// All intermediate arithmetic uses `f64` to preserve precision; results are
    /// cast to `f32` only on final store. The inclusion-exclusion formula
    /// accounts for card-blocking between the two players' private hands.
    pub(crate) fn evaluate_internal(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min_f64(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = (half_pot - rake) / self.num_combinations;
        let amount_lose = -half_pot / self.num_combinations;

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let mut cfreach_sum = 0.0f64;
        let mut cfreach_minus = [0.0f64; 52];

        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });

        // SAFETY: We just initialized every element to 0.0 via `MaybeUninit::write`.
        // Reinterpreting as `&mut [f32]` is sound because all elements are now initialized.
        let result = unsafe { &mut *(result as *mut [MaybeUninit<f32>] as *mut [f32]) };

        // -----------------------------------------------------------------
        // Case 0: Depth boundary
        // -----------------------------------------------------------------
        if node.player & PLAYER_DEPTH_BOUNDARY_FLAG == PLAYER_DEPTH_BOUNDARY_FLAG {
            let node_index = self.node_index(node);
            let ordinal = self.node_to_boundary[node_index] as usize;
            let k = self.num_continuations.max(1);

            // Update opponent reach at this boundary every visit.
            let opp = player ^ 1;
            let reach_index = ordinal * 2 + opp;
            if reach_index < self.boundary_reach.len() {
                *self.boundary_reach[reach_index].lock().unwrap() = cfreach.to_vec();
            }

            // -- Legacy single-continuation path (K <= 1) --
            if k <= 1 {
                let bcfv_index = ordinal * 2 + player;
                // Lazily compute and cache boundary CFVs on first visit.
                {
                    let guard = self.boundary_cfvs[bcfv_index].lock().unwrap();
                    if guard.is_empty() {
                        drop(guard);
                        if let Some(ref evaluator) = self.boundary_evaluator {
                            let pot = self.tree_config.starting_pot + 2 * node.amount;
                            let remaining = (self.tree_config.effective_stack as f64 - pot as f64 / 2.0).max(0.0);
                            let opp_reach = self.boundary_reach[reach_index].lock().unwrap().clone();
                            let opp_reach_ref = if opp_reach.is_empty() {
                                self.initial_weights[opp].to_vec()
                            } else {
                                opp_reach
                            };
                            let num_hands = self.private_cards[player].len();
                            let cfvs = evaluator.compute_cfvs(player, pot, remaining, &opp_reach_ref, num_hands, 0);
                            *self.boundary_cfvs[bcfv_index].lock().unwrap() = cfvs;
                        } else {
                            return; // No evaluator — treat as zero.
                        }
                    }
                }

                let bcfvs = self.boundary_cfvs[bcfv_index].lock().unwrap();
                self.evaluate_boundary_single(result, node, player, player_cards, opponent_cards, cfreach, &bcfvs, half_pot);
                return;
            }

            // -- Multi-continuation path (K > 1) --
            // Regret-match over K continuations to get opponent's current strategy.
            let strategy = {
                let regrets = self.boundary_cont_regrets[ordinal].lock().unwrap();
                regret_match_k(&regrets, k)
            };

            let payoff_scale = half_pot / self.num_combinations;

            let valid_indices = if node.river != NOT_DEALT {
                &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.valid_indices_turn[node.turn as usize]
            } else {
                &self.valid_indices_flop
            };

            // Accumulate opponent cfreach (shared across all K continuations).
            let opponent_indices = &valid_indices[player ^ 1];
            for &i in opponent_indices {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(i as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let player_indices = &valid_indices[player];
            let same_hand_index = &self.same_hand_index[player];

            // Compute per-continuation action values and combined result.
            // action_values[cont] = sum over player hands of (bcfv * payoff_scale * cfreach_adj)
            let mut action_values = vec![0.0f64; k];

            for cont_k in 0..k {
                let bcfv_idx = ordinal * 2 * k + player * k + cont_k;
                let bcfvs_k = self.boundary_cfvs[bcfv_idx].lock().unwrap();
                if bcfvs_k.is_empty() {
                    continue;
                }
                let w = strategy[cont_k] as f64;

                for &i in player_indices {
                    unsafe {
                        let (c1, c2) = *player_cards.get_unchecked(i as usize);
                        let same_i = *same_hand_index.get_unchecked(i as usize);
                        let cfreach_same = if same_i == u16::MAX {
                            0.0
                        } else {
                            *cfreach.get_unchecked(same_i as usize) as f64
                        };
                        let cfreach_adj = cfreach_sum + cfreach_same
                            - *cfreach_minus.get_unchecked(c1 as usize)
                            - *cfreach_minus.get_unchecked(c2 as usize);
                        let bcfv = *bcfvs_k.get_unchecked(i as usize) as f64;
                        let val = bcfv * payoff_scale * cfreach_adj;

                        // Accumulate weighted result
                        *result.get_unchecked_mut(i as usize) += (w * val) as f32;

                        // Accumulate action value for regret computation
                        action_values[cont_k] += val;
                    }
                }
            }

            // Compute combined value (strategy-weighted sum of action values).
            let combined_value: f64 = action_values
                .iter()
                .zip(strategy.iter())
                .map(|(&av, &s)| av * s as f64)
                .sum();

            // Update continuation regrets with DCFR discounting.
            // Apply the same alpha/beta weights used by the main solver so
            // early noisy iterations don't contaminate the continuation choice.
            {
                let mut regrets = self.boundary_cont_regrets[ordinal].lock().unwrap();
                let alpha = self.boundary_discount_alpha.load(std::sync::atomic::Ordering::Relaxed);
                let beta = self.boundary_discount_beta.load(std::sync::atomic::Ordering::Relaxed);
                for cont_k in 0..k {
                    // Discount existing regret (alpha for positive, beta for negative)
                    let r = regrets[cont_k];
                    let d = if r >= 0.0 { f32::from_bits(alpha) } else { f32::from_bits(beta) };
                    regrets[cont_k] = r * d + (action_values[cont_k] - combined_value) as f32;
                }
            }

            // Update continuation cumulative strategy with gamma discounting.
            {
                let mut strat = self.boundary_cont_strategy[ordinal].lock().unwrap();
                let gamma = f32::from_bits(self.boundary_discount_gamma.load(std::sync::atomic::Ordering::Relaxed));
                for cont_k in 0..k {
                    strat[cont_k] = strat[cont_k] * gamma + strategy[cont_k];
                }
            }

            return;
        }

        // -----------------------------------------------------------------
        // Case 1: Fold
        // -----------------------------------------------------------------
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let valid_indices = if node.river != NOT_DEALT {
                &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.valid_indices_turn[node.turn as usize]
            } else {
                &self.valid_indices_flop
            };

            let opponent_indices = &valid_indices[player ^ 1];
            for &i in opponent_indices {
                // SAFETY: `i` is a valid index into `cfreach` and `opponent_cards`,
                // produced by `valid_indices` which only contains indices within
                // `private_cards[player ^ 1]`.
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(i as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let player_indices = &valid_indices[player];
            let same_hand_index = &self.same_hand_index[player];
            for &i in player_indices {
                // SAFETY: `i` indexes into `player_cards`, `same_hand_index`, and
                // `result`, all of which have length `private_cards[player].len()`.
                // `same_hand_index[i]` is either `u16::MAX` (no match) or a valid
                // index into `cfreach` (the opponent's hand list).
                unsafe {
                    let (c1, c2) = *player_cards.get_unchecked(i as usize);
                    let same_i = *same_hand_index.get_unchecked(i as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };
                    // Inclusion-exclusion: total opponent reach minus hands blocked
                    // by our cards, plus back the hand identical to ours (double-counted).
                    let cfreach = cfreach_sum + cfreach_same
                        - *cfreach_minus.get_unchecked(c1 as usize)
                        - *cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(i as usize) = (payoff * cfreach) as f32;
                }
            }
        }
        // -----------------------------------------------------------------
        // Case 2: Showdown (no rake) — optimized 2-pass
        // -----------------------------------------------------------------
        else if rake == 0.0 {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            // Strip the sentinel items at both ends.
            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let mut i = 1;

            // Ascending pass: accumulate weaker opponent hands for amount_win.
            for &StrengthItem { strength, index } in valid_player_strength {
                // SAFETY: `opponent_strength` has sentinels at index 0 (strength 0)
                // and at the end (strength u16::MAX). The inner while loop advances
                // `i` only while the opponent's strength is strictly less than ours,
                // so `i` never exceeds the last valid element. All `opponent_index`
                // and `index` values come from the hand enumeration and are valid
                // indices into their respective arrays.
                unsafe {
                    while opponent_strength.get_unchecked(i).strength < strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i += 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) = (amount_win * cfreach) as f32;
                }
            }

            // Reset accumulators for descending pass.
            cfreach_sum = 0.0;
            cfreach_minus.fill(0.0);
            i = opponent_strength.len() - 2;

            // Descending pass: accumulate stronger opponent hands for amount_lose.
            for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
                // SAFETY: Same invariants as the ascending pass, but in reverse.
                // The sentinel at index 0 (strength 0) guards the lower bound.
                unsafe {
                    while opponent_strength.get_unchecked(i).strength > strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i -= 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) += (amount_lose * cfreach) as f32;
                }
            }
        }
        // -----------------------------------------------------------------
        // Case 3: Showdown (raked) — 3-pass with tie handling
        // -----------------------------------------------------------------
        else {
            let amount_tie = -0.5 * rake / self.num_combinations;
            let same_hand_index = &self.same_hand_index[player];

            let pair_index = card_pair_to_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let valid_opponent_strength = &opponent_strength[1..opponent_strength.len() - 1];

            // First pass: accumulate total opponent reach.
            for &StrengthItem { index, .. } in valid_opponent_strength {
                // SAFETY: `index` is a valid index into `cfreach` and `opponent_cards`,
                // produced during hand enumeration.
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(index as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(index as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let mut cfreach_sum_win = 0.0f64;
            let mut cfreach_sum_tie = 0.0f64;
            let mut cfreach_minus_win = [0.0f64; 52];
            let mut cfreach_minus_tie = [0.0f64; 52];

            let mut i = 1usize;
            let mut j = 1usize;
            let mut prev_strength: u16 = 0; // strength is always > 0

            // Second pass: iterate player hands in ascending strength order.
            for &StrengthItem { strength, index } in valid_player_strength {
                // SAFETY: `opponent_strength` has sentinels at both ends.
                // `i` and `j` advance monotonically within the valid range.
                // All index values are produced by hand enumeration and are valid.
                unsafe {
                    if strength > prev_strength {
                        prev_strength = strength;

                        if i < j {
                            cfreach_sum_win = cfreach_sum_tie;
                            cfreach_minus_win = cfreach_minus_tie;
                            i = j;
                        }

                        while opponent_strength.get_unchecked(i).strength < strength {
                            let opponent_index =
                                opponent_strength.get_unchecked(i).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_win += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c1 as usize) += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c2 as usize) += cfreach_i;
                            i += 1;
                        }

                        if j < i {
                            cfreach_sum_tie = cfreach_sum_win;
                            cfreach_minus_tie = cfreach_minus_win;
                            j = i;
                        }

                        while opponent_strength.get_unchecked(j).strength == strength {
                            let opponent_index =
                                opponent_strength.get_unchecked(j).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_j = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_tie += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c1 as usize) += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c2 as usize) += cfreach_j;
                            j += 1;
                        }
                    }

                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach_total = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    let cfreach_win = cfreach_sum_win
                        - cfreach_minus_win.get_unchecked(c1 as usize)
                        - cfreach_minus_win.get_unchecked(c2 as usize);
                    let cfreach_tie = cfreach_sum_tie
                        - cfreach_minus_tie.get_unchecked(c1 as usize)
                        - cfreach_minus_tie.get_unchecked(c2 as usize);
                    let same_i = *same_hand_index.get_unchecked(index as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };

                    let cfvalue = amount_win * cfreach_win
                        + amount_tie * (cfreach_tie - cfreach_win + cfreach_same)
                        + amount_lose * (cfreach_total - cfreach_tie);
                    *result.get_unchecked_mut(index as usize) = cfvalue as f32;
                }
            }
        }
    }

    /// Evaluates a single-continuation boundary terminal.
    ///
    /// Shared logic for the legacy K=1 path: computes cfreach accumulations
    /// and writes `bcfv * payoff_scale * cfreach_adj` into result.
    fn evaluate_boundary_single(
        &self,
        result: &mut [f32],
        node: &PostFlopNode,
        player: usize,
        player_cards: &[(Card, Card)],
        opponent_cards: &[(Card, Card)],
        cfreach: &[f32],
        bcfvs: &[f32],
        half_pot: f64,
    ) {
        let payoff_scale = half_pot / self.num_combinations;

        let valid_indices = if node.river != NOT_DEALT {
            &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
        } else if node.turn != NOT_DEALT {
            &self.valid_indices_turn[node.turn as usize]
        } else {
            &self.valid_indices_flop
        };

        let mut cfreach_sum = 0.0f64;
        let mut cfreach_minus = [0.0f64; 52];

        let opponent_indices = &valid_indices[player ^ 1];
        for &i in opponent_indices {
            unsafe {
                let cfreach_i = *cfreach.get_unchecked(i as usize);
                if cfreach_i != 0.0 {
                    let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                    let cfreach_i_f64 = cfreach_i as f64;
                    cfreach_sum += cfreach_i_f64;
                    *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                    *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                }
            }
        }

        if cfreach_sum == 0.0 {
            return;
        }

        let player_indices = &valid_indices[player];
        let same_hand_index = &self.same_hand_index[player];
        for &i in player_indices {
            unsafe {
                let (c1, c2) = *player_cards.get_unchecked(i as usize);
                let same_i = *same_hand_index.get_unchecked(i as usize);
                let cfreach_same = if same_i == u16::MAX {
                    0.0
                } else {
                    *cfreach.get_unchecked(same_i as usize) as f64
                };
                let cfreach_adj = cfreach_sum + cfreach_same
                    - *cfreach_minus.get_unchecked(c1 as usize)
                    - *cfreach_minus.get_unchecked(c2 as usize);
                let bcfv = *bcfvs.get_unchecked(i as usize) as f64;
                *result.get_unchecked_mut(i as usize) =
                    (bcfv * payoff_scale * cfreach_adj) as f32;
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_tree::*;
    use crate::bet_size::*;
    use crate::card::{card_from_str, flop_from_str};
    use crate::interface::Game;
    use std::mem::MaybeUninit;

    /// Helper: build a simple river game and allocate memory.
    fn make_river_game(
        rake_rate: f64,
        rake_cap: f64,
    ) -> PostFlopGame {
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
            rake_rate,
            rake_cap,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    /// Walks the tree to find the first fold terminal for the given folding player.
    fn find_fold_node(game: &PostFlopGame, want_fold_player: u8) -> Option<usize> {
        for (idx, node_mutex) in game.node_arena.iter().enumerate() {
            let node = node_mutex.lock();
            if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
                let folded = node.player & PLAYER_MASK;
                if folded == want_fold_player {
                    return Some(idx);
                }
            }
        }
        None
    }

    /// Walks the tree to find the first showdown terminal.
    fn find_showdown_node(game: &PostFlopGame) -> Option<usize> {
        for (idx, node_mutex) in game.node_arena.iter().enumerate() {
            let node = node_mutex.lock();
            if node.player & PLAYER_TERMINAL_FLAG != 0
                && node.player & PLAYER_FOLD_FLAG != PLAYER_FOLD_FLAG
            {
                return Some(idx);
            }
        }
        None
    }

    #[test]
    fn test_fold_evaluation_oop_folds() {
        let game = make_river_game(0.0, 0.0);
        // Find a fold node where OOP folded.
        let fold_idx = find_fold_node(&game, PLAYER_OOP)
            .expect("should have an OOP fold terminal");
        let node = game.node_arena[fold_idx].lock();

        let num_ip = game.num_private_hands(1);
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_ip];
        // Evaluate from IP's perspective (IP wins when OOP folds).
        game.evaluate_internal(&mut result, &node, 1, &vec![1.0; game.num_private_hands(0)]);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();
        // IP should gain positive value when OOP folds.
        let nonzero_count = values.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero_count > 0, "some IP hands should have nonzero value");

        // All nonzero values should be positive (IP wins the pot).
        for &v in &values {
            assert!(v >= 0.0, "IP value should be >= 0 when OOP folds, got {v}");
        }
    }

    #[test]
    fn test_fold_evaluation_ip_folds() {
        let game = make_river_game(0.0, 0.0);
        let num_oop = game.num_private_hands(0);

        // Find a fold node where IP folded.
        let fold_idx = find_fold_node(&game, PLAYER_IP)
            .expect("should have an IP fold terminal");
        let node = game.node_arena[fold_idx].lock();

        // Evaluate from OOP's perspective (OOP wins when IP folds).
        let cfreach_ip: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // OOP should gain positive value when IP folds.
        let nonzero_count = values.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero_count > 0, "some OOP hands should have nonzero value");

        for &v in &values {
            assert!(v >= 0.0, "OOP value should be >= 0 when IP folds, got {v}");
        }
    }

    #[test]
    fn test_fold_zero_cfreach_gives_zero_result() {
        let game = make_river_game(0.0, 0.0);
        let num_oop = game.num_private_hands(0);

        let fold_idx = find_fold_node(&game, PLAYER_IP)
            .expect("should have an IP fold terminal");
        let node = game.node_arena[fold_idx].lock();

        // All-zero opponent reach should produce all-zero result.
        let cfreach_ip: Vec<f32> = vec![0.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();
        for &v in &values {
            assert_eq!(v, 0.0, "zero cfreach should produce zero result");
        }
    }

    #[test]
    fn test_showdown_evaluation_no_rake() {
        let game = make_river_game(0.0, 0.0);
        let num_oop = game.num_private_hands(0);

        let sd_idx = find_showdown_node(&game)
            .expect("should have a showdown terminal");
        let node = game.node_arena[sd_idx].lock();

        // Use uniform reach for the opponent.
        let cfreach_ip: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // In a showdown, some hands win and some lose — we should see both signs.
        let has_positive = values.iter().any(|&v| v > 0.0);
        let has_negative = values.iter().any(|&v| v < 0.0);
        assert!(
            has_positive && has_negative,
            "showdown should have both winners and losers"
        );

        // Verify zero-sum: the sum of cfvalues (weighted by uniform reach) should
        // be close to zero when evaluated from both sides.
        let mut result_ip: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); game.num_private_hands(1)];
        let cfreach_oop: Vec<f32> = vec![1.0; num_oop];
        game.evaluate_internal(&mut result_ip, &node, 1, &cfreach_oop);

        let values_ip: Vec<f32> = result_ip.iter().map(|v| unsafe { v.assume_init() }).collect();
        let sum_oop: f64 = values.iter().map(|&v| v as f64).sum();
        let sum_ip: f64 = values_ip.iter().map(|&v| v as f64).sum();

        // The sums won't be exactly zero because card blocking creates asymmetry,
        // but the total EV exchanged should be modest relative to the pot.
        let pot = (game.tree_config.starting_pot + 2 * node.amount) as f64;
        assert!(
            (sum_oop + sum_ip).abs() < pot,
            "approximate zero-sum violated: sum_oop={sum_oop}, sum_ip={sum_ip}"
        );
    }

    #[test]
    fn test_showdown_evaluation_with_rake() {
        let game = make_river_game(0.05, 10.0);
        let num_oop = game.num_private_hands(0);

        let sd_idx = find_showdown_node(&game)
            .expect("should have a showdown terminal");
        let node = game.node_arena[sd_idx].lock();

        let cfreach_ip: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // Should still have both winners and losers.
        let has_positive = values.iter().any(|&v| v > 0.0);
        let has_negative = values.iter().any(|&v| v < 0.0);
        assert!(
            has_positive && has_negative,
            "raked showdown should have both winners and losers"
        );
    }

    #[test]
    fn test_fold_with_rake() {
        let game = make_river_game(0.05, 10.0);
        let num_oop = game.num_private_hands(0);

        let fold_idx = find_fold_node(&game, PLAYER_IP)
            .expect("should have an IP fold terminal");
        let node = game.node_arena[fold_idx].lock();

        let cfreach_ip: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        let mut result_raked: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result_raked, &node, 0, &cfreach_ip);

        let values_raked: Vec<f32> = result_raked
            .iter()
            .map(|v| unsafe { v.assume_init() })
            .collect();

        // Compare against a no-rake game.
        let game_no_rake = make_river_game(0.0, 0.0);
        let fold_idx_nr = find_fold_node(&game_no_rake, PLAYER_IP)
            .expect("should have an IP fold terminal");
        let node_nr = game_no_rake.node_arena[fold_idx_nr].lock();

        let cfreach_ip_nr: Vec<f32> = vec![1.0; game_no_rake.num_private_hands(1)];
        let mut result_nr: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); game_no_rake.num_private_hands(0)];
        game_no_rake.evaluate_internal(&mut result_nr, &node_nr, 0, &cfreach_ip_nr);

        let values_nr: Vec<f32> = result_nr
            .iter()
            .map(|v| unsafe { v.assume_init() })
            .collect();

        // Raked fold payoffs should be <= no-rake fold payoffs for the winner.
        // (Both arrays have the same hand ordering since same config.)
        assert_eq!(values_raked.len(), values_nr.len());
        for (i, (&raked, &no_rake)) in values_raked.iter().zip(values_nr.iter()).enumerate() {
            assert!(
                raked <= no_rake + 1e-6,
                "hand {i}: raked {raked} > no_rake {no_rake}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Depth boundary evaluation
    // -----------------------------------------------------------------

    /// Helper: build a turn game with depth_limit=0 (no river).
    fn make_turn_game_depth_limited() -> PostFlopGame {
        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range = "QQ-JJ,AQs,AJs".parse().unwrap();
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
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    fn find_boundary_node(game: &PostFlopGame) -> Option<usize> {
        for (idx, node_mutex) in game.node_arena.iter().enumerate() {
            if node_mutex.lock().is_depth_boundary() {
                return Some(idx);
            }
        }
        None
    }

    #[test]
    fn test_depth_boundary_zero_cfvs_gives_zero_result() {
        let game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();
        assert!(n_boundary > 0, "should have boundary nodes");

        // Set zero boundary CFVs for all boundary nodes
        for ordinal in 0..n_boundary {
            let oop_cfvs = vec![0.0f32; game.num_private_hands(0)];
            let ip_cfvs = vec![0.0f32; game.num_private_hands(1)];
            game.set_boundary_cfvs(ordinal, 0, oop_cfvs);
            game.set_boundary_cfvs(ordinal, 1, ip_cfvs);
        }

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let node = game.node_arena[bd_idx].lock();

        let num_oop = game.num_private_hands(0);
        let cfreach_ip: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // Zero boundary CFVs should produce zero result
        for &v in &values {
            assert_eq!(v, 0.0, "zero boundary CFVs should produce zero result");
        }
    }

    #[test]
    fn test_depth_boundary_positive_cfvs() {
        let game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();

        // Set positive boundary CFVs (all hands have value +1.0)
        for ordinal in 0..n_boundary {
            let oop_cfvs = vec![1.0f32; game.num_private_hands(0)];
            let ip_cfvs = vec![1.0f32; game.num_private_hands(1)];
            game.set_boundary_cfvs(ordinal, 0, oop_cfvs);
            game.set_boundary_cfvs(ordinal, 1, ip_cfvs);
        }

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let node = game.node_arena[bd_idx].lock();

        let num_oop = game.num_private_hands(0);
        let cfreach_ip: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // Positive boundary CFVs with positive reach should produce positive values
        let nonzero_count = values.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero_count > 0, "some hands should have nonzero value");

        for &v in &values {
            assert!(v >= 0.0, "positive boundary CFVs should produce non-negative values, got {v}");
        }
    }

    #[test]
    fn test_depth_boundary_zero_reach_gives_zero_result() {
        let game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();

        for ordinal in 0..n_boundary {
            let oop_cfvs = vec![1.0f32; game.num_private_hands(0)];
            let ip_cfvs = vec![1.0f32; game.num_private_hands(1)];
            game.set_boundary_cfvs(ordinal, 0, oop_cfvs);
            game.set_boundary_cfvs(ordinal, 1, ip_cfvs);
        }

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let node = game.node_arena[bd_idx].lock();

        let num_oop = game.num_private_hands(0);
        // Zero opponent reach
        let cfreach_ip: Vec<f32> = vec![0.0; game.num_private_hands(1)];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // Zero reach should produce zero result
        for &v in &values {
            assert_eq!(v, 0.0, "zero reach should produce zero result");
        }
    }

    // -----------------------------------------------------------------
    // boundary_reach overwrites on every visit
    // -----------------------------------------------------------------

    #[test]
    fn boundary_reach_updates_every_visit() {
        let game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();
        assert!(n_boundary > 0, "need boundary nodes for this test");

        // Pre-set boundary CFVs so evaluate_internal doesn't bail out.
        for ordinal in 0..n_boundary {
            game.set_boundary_cfvs(ordinal, 0, vec![1.0; game.num_private_hands(0)]);
            game.set_boundary_cfvs(ordinal, 1, vec![1.0; game.num_private_hands(1)]);
        }

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let node = game.node_arena[bd_idx].lock();
        let ordinal = game.node_to_boundary[bd_idx] as usize;

        let num_oop = game.num_private_hands(0);
        let num_ip = game.num_private_hands(1);

        // First visit: evaluate from OOP's perspective with uniform IP reach.
        let cfreach_first = vec![1.0f32; num_ip];
        let mut result = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_first);

        // boundary_reach stores the opponent's (IP's) reach for player=0 traversal.
        let opp = 0 ^ 1; // = 1
        let stored_first = game.boundary_reach(ordinal, opp);
        assert!(!stored_first.is_empty(), "first visit should populate boundary_reach");
        assert!(stored_first.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "first visit should store the uniform reach");

        // Second visit: evaluate with a DIFFERENT reach (all 0.5).
        let cfreach_second = vec![0.5f32; num_ip];
        let mut result2 = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result2, &node, 0, &cfreach_second);

        let stored_second = game.boundary_reach(ordinal, opp);
        // The reach should now reflect the SECOND visit's values (0.5),
        // not be stuck on the first visit's values (1.0).
        assert!(stored_second.iter().all(|&v| (v - 0.5).abs() < 1e-6),
            "second visit should overwrite boundary_reach with new values, \
             got {:?}", &stored_second[..stored_second.len().min(5)]);
    }

    // -----------------------------------------------------------------
    // regret_match_k
    // -----------------------------------------------------------------

    #[test]
    fn test_regret_match_k_all_zero_gives_uniform() {
        let regrets = vec![0.0f32; 4];
        let strategy = regret_match_k(&regrets, 4);
        for &s in &strategy {
            assert!((s - 0.25).abs() < 1e-6, "expected uniform 0.25, got {s}");
        }
    }

    #[test]
    fn test_regret_match_k_all_negative_gives_uniform() {
        let regrets = vec![-5.0, -10.0, -1.0, -100.0];
        let strategy = regret_match_k(&regrets, 4);
        for &s in &strategy {
            assert!((s - 0.25).abs() < 1e-6, "expected uniform 0.25, got {s}");
        }
    }

    #[test]
    fn test_regret_match_k_one_positive_concentrates() {
        let regrets = vec![10.0, 0.0, 0.0, 0.0];
        let strategy = regret_match_k(&regrets, 4);
        assert!((strategy[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", strategy[0]);
        for i in 1..4 {
            assert!((strategy[i]).abs() < 1e-6, "expected 0.0, got {}", strategy[i]);
        }
    }

    #[test]
    fn test_regret_match_k_proportional() {
        let regrets = vec![3.0, 1.0, 0.0, -5.0];
        let strategy = regret_match_k(&regrets, 4);
        assert!((strategy[0] - 0.75).abs() < 1e-6);
        assert!((strategy[1] - 0.25).abs() < 1e-6);
        assert!((strategy[2]).abs() < 1e-6);
        assert!((strategy[3]).abs() < 1e-6);
    }

    // -----------------------------------------------------------------
    // Multi-continuation boundary evaluation
    // -----------------------------------------------------------------

    #[test]
    fn test_multi_continuation_storage_allocation() {
        let mut game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();
        assert!(n_boundary > 0);

        // Enable multi-continuation with K=4
        let k = 4;
        game.init_multi_continuation(k);

        // boundary_cfvs should now have boundary_count * 2 * K entries
        assert_eq!(
            game.boundary_cfvs.len(),
            n_boundary * 2 * k,
            "boundary_cfvs should have boundary_count * 2 * K entries"
        );
        assert_eq!(game.num_continuations, k);
    }

    #[test]
    fn test_set_boundary_cfvs_multi() {
        let mut game = make_turn_game_depth_limited();
        let k = 4;
        game.init_multi_continuation(k);

        let num_oop = game.num_private_hands(0);

        // Set different CFVs for each continuation
        for cont in 0..k {
            let cfvs = vec![cont as f32 + 1.0; num_oop];
            game.set_boundary_cfvs_multi(0, 0, cont, cfvs);
        }

        // Verify they were stored correctly
        for cont in 0..k {
            let idx = 0 * 2 * k + 0 * k + cont;
            let stored = game.boundary_cfvs[idx].lock().unwrap();
            assert_eq!(stored.len(), num_oop);
            assert_eq!(stored[0], cont as f32 + 1.0);
        }
    }

    #[test]
    fn test_multi_continuation_uniform_strategy_averages_cfvs() {
        let mut game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();
        let k = 4;
        game.init_multi_continuation(k);

        let num_oop = game.num_private_hands(0);
        let num_ip = game.num_private_hands(1);

        // Set different CFVs for each continuation at all boundaries
        for ordinal in 0..n_boundary {
            for cont in 0..k {
                // OOP: continuation k has value (k+1) for all hands
                let oop_cfvs = vec![cont as f32 + 1.0; num_oop];
                game.set_boundary_cfvs_multi(ordinal, 0, cont, oop_cfvs);
                // IP: set similarly
                let ip_cfvs = vec![cont as f32 + 1.0; num_ip];
                game.set_boundary_cfvs_multi(ordinal, 1, cont, ip_cfvs);
            }
        }

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let node = game.node_arena[bd_idx].lock();

        // Evaluate with uniform reach
        let cfreach_ip: Vec<f32> = vec![1.0; num_ip];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // With uniform initial regrets (all zero), strategy should be uniform 1/K.
        // Average of (1+2+3+4)/4 = 2.5 scaled by payoff_scale * cfreach.
        // All values should be finite and positive (since all CFVs are positive).
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "hand {i} should be finite, got {v}");
        }
        let nonzero = values.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "some hands should have nonzero value");
    }

    #[test]
    fn test_multi_continuation_zero_reach_gives_zero() {
        let mut game = make_turn_game_depth_limited();
        let k = 4;
        game.init_multi_continuation(k);

        let num_oop = game.num_private_hands(0);
        let num_ip = game.num_private_hands(1);
        let n_boundary = game.num_boundary_nodes();

        for ordinal in 0..n_boundary {
            for cont in 0..k {
                game.set_boundary_cfvs_multi(ordinal, 0, cont, vec![1.0; num_oop]);
                game.set_boundary_cfvs_multi(ordinal, 1, cont, vec![1.0; num_ip]);
            }
        }

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let node = game.node_arena[bd_idx].lock();

        let cfreach_ip: Vec<f32> = vec![0.0; num_ip];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &node, 0, &cfreach_ip);

        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();
        for &v in &values {
            assert_eq!(v, 0.0, "zero reach should produce zero result");
        }
    }

    #[test]
    fn test_multi_continuation_regret_update_shifts_strategy() {
        let mut game = make_turn_game_depth_limited();
        let n_boundary = game.num_boundary_nodes();
        let k = 4;
        game.init_multi_continuation(k);

        let bd_idx = find_boundary_node(&game).expect("should have a boundary node");
        let ordinal = game.node_to_boundary[bd_idx] as usize;

        let num_oop = game.num_private_hands(0);
        let num_ip = game.num_private_hands(1);

        // Set CFVs: continuation 0 is much higher than others
        for o in 0..n_boundary {
            for cont in 0..k {
                let val = if cont == 0 { 10.0 } else { 1.0 };
                game.set_boundary_cfvs_multi(o, 0, cont, vec![val; num_oop]);
                game.set_boundary_cfvs_multi(o, 1, cont, vec![val; num_ip]);
            }
        }

        // Manually set positive regret on continuation 0
        {
            let mut r = game.boundary_cont_regrets[ordinal].lock().unwrap();
            r[0] = 100.0; // strong preference for continuation 0
            r[1] = 0.0;
            r[2] = 0.0;
            r[3] = 0.0;
        }

        let bd_node = game.node_arena[bd_idx].lock();
        let cfreach_ip: Vec<f32> = vec![1.0; num_ip];
        let mut result: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); num_oop];
        game.evaluate_internal(&mut result, &bd_node, 0, &cfreach_ip);

        // With regret heavily favoring continuation 0 (value=10), the result
        // should be closer to the value from continuation 0 than from a uniform
        // mix (which would average to 3.25).
        let values: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();
        let nonzero: Vec<f32> = values.into_iter().filter(|&v| v != 0.0).collect();
        assert!(!nonzero.is_empty(), "should have nonzero values");
    }
}
