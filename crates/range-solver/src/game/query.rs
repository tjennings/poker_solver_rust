use super::*;
use crate::card::*;
use crate::interface::*;
use crate::mutex_like::MutexGuardLike;
use crate::sliceop::*;
use crate::utility::*;

/// Decodes an encoded `i16` slice to an `f32` vec using the given scale.
#[inline]
fn decode_signed_slice(slice: &[i16], scale: f32) -> Vec<f32> {
    let decoder = scale / i16::MAX as f32;
    slice.iter().map(|&x| x as f32 * decoder).collect()
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Returns a lock on the current node.
    #[inline]
    fn node(&self) -> MutexGuardLike<'_, PostFlopNode> {
        self.node_arena[self.node_history.last().copied().unwrap_or(0)].lock()
    }

    /// Resets the interpreter to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.action_history.clear();
        self.node_history.clear();
        self.is_normalized_weight_cached = false;
        self.turn = self.card_config.turn;
        self.river = self.card_config.river;
        self.turn_swapped_suit = None;
        self.turn_swap = None;
        self.river_swap = None;
        self.total_bet_amount = [0, 0];

        self.weights[0].copy_from_slice(&self.initial_weights[0]);
        self.weights[1].copy_from_slice(&self.initial_weights[1]);
        self.assign_zero_weights();
    }

    /// Returns the action history (list of action indices passed to [`play`]).
    #[inline]
    pub fn history(&self) -> &[usize] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        &self.action_history
    }

    /// Replays the given history from the root node.
    #[inline]
    pub fn apply_history(&mut self, history: &[usize]) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        self.back_to_root();
        for &action in history {
            self.play(action);
        }
    }

    /// Returns whether the current node is terminal.
    ///
    /// A chance node after an all-in call is also considered terminal.
    #[inline]
    pub fn is_terminal_node(&self) -> bool {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        let node = self.node();
        node.is_terminal() || node.amount == self.tree_config.effective_stack
    }

    /// Returns whether the current node is a chance node (turn/river deal).
    #[inline]
    pub fn is_chance_node(&self) -> bool {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        self.node().is_chance() && !self.is_terminal_node()
    }

    /// Returns the available actions at the current node.
    ///
    /// Terminal nodes return an empty vec. Chance nodes return one action per
    /// non-isomorphic deal card.
    #[inline]
    pub fn available_actions(&self) -> Vec<Action> {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        if self.is_terminal_node() {
            Vec::new()
        } else {
            self.node()
                .children()
                .iter()
                .map(|c| c.lock().prev_action)
                .collect()
        }
    }

    /// Returns a bitmask of dealable cards at a chance node.
    ///
    /// Bit `i` is set if card `i` can be dealt. Returns `0` if not a chance node.
    pub fn possible_cards(&self) -> u64 {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if !self.is_chance_node() {
            return 0;
        }

        let flop = self.card_config.flop;
        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        let mut dead_mask: u64 = 0;

        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }

        'outer: for card in 0..52u8 {
            let bit_card: u64 = 1 << card;
            let new_board_mask = board_mask | bit_card;

            if new_board_mask != board_mask {
                for &(c1, c2) in &self.private_cards[0] {
                    let oop_mask: u64 = (1 << c1) | (1 << c2);
                    if oop_mask & new_board_mask != 0 {
                        continue;
                    }
                    let combined_mask = oop_mask | new_board_mask;
                    for &(c3, c4) in &self.private_cards[1] {
                        let ip_mask: u64 = (1 << c3) | (1 << c4);
                        if ip_mask & combined_mask == 0 {
                            continue 'outer;
                        }
                    }
                }
            }

            dead_mask |= bit_card;
        }

        ((1u64 << 52) - 1) ^ dead_mask
    }

    /// Returns the current player (0 = OOP, 1 = IP).
    #[inline]
    pub fn current_player(&self) -> usize {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        self.node().player()
    }

    /// Returns the current board cards.
    #[inline]
    pub fn current_board(&self) -> Vec<u8> {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        let mut ret = self.card_config.flop.to_vec();
        if self.turn != NOT_DEALT {
            ret.push(self.turn);
        }
        if self.river != NOT_DEALT {
            ret.push(self.river);
        }
        ret
    }

    /// Navigates to a child node by playing an action.
    ///
    /// For chance nodes, `action` is the card ID to deal (`usize::MAX` picks
    /// the lowest possible card). For player nodes, `action` is the index
    /// into [`available_actions`].
    pub fn play(&mut self, action: usize) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        // -- Chance node --
        if self.is_chance_node() {
            let is_turn = self.turn == NOT_DEALT;
            if self.storage_mode == BoardState::Flop
                || (!is_turn && self.storage_mode == BoardState::Turn)
            {
                panic!("Storage mode is not compatible");
            }

            let actual_card = if action == usize::MAX {
                self.possible_cards().trailing_zeros() as Card
            } else {
                action as Card
            };

            // Swap the suit if swapping was performed on the turn
            let action_card = if let Some((suit1, suit2)) = self.turn_swapped_suit {
                if actual_card & 3 == suit1 {
                    actual_card - suit1 + suit2
                } else if actual_card & 3 == suit2 {
                    actual_card + suit1 - suit2
                } else {
                    actual_card
                }
            } else {
                actual_card
            };

            let actions = self.available_actions();
            let mut action_index = usize::MAX;

            // Find the action index from available actions
            for (i, &a) in actions.iter().enumerate() {
                if a == Action::Chance(action_card) {
                    action_index = i;
                    break;
                }
            }

            // Find the action index from isomorphic chances
            if action_index == usize::MAX {
                let node = self.node();
                let isomorphism = self.isomorphic_chances(&node);
                let isomorphic_cards = if node.turn == NOT_DEALT {
                    &self.isomorphism_card_turn
                } else {
                    &self.isomorphism_card_river[node.turn as usize & 3]
                };
                for (i, &repr_index) in isomorphism.iter().enumerate() {
                    if action_card == isomorphic_cards[i] {
                        action_index = repr_index as usize;
                        if is_turn {
                            if let Action::Chance(repr_card) = actions[repr_index as usize] {
                                self.turn_swapped_suit =
                                    Some((action_card & 3, repr_card & 3));
                            }
                            self.turn_swap = Some(action_card & 3);
                        } else {
                            // `self.turn != self.node().turn` if `self.turn_swap.is_some()`.
                            // Possible only when the flop is monotone.
                            self.river_swap = Some((
                                self.turn & 3,
                                self.isomorphism_card_river[self.turn as usize & 3][i] & 3,
                            ));
                        }
                        break;
                    }
                }
            }

            if action_index == usize::MAX {
                panic!("Invalid action");
            }

            // Update the state
            let node_index = self.node_index(&self.node().play(action_index));
            self.node_history.push(node_index);
            if is_turn {
                self.turn = actual_card;
            } else {
                self.river = actual_card;
            }

            // Update the weights
            self.assign_zero_weights();
        }
        // -- Player node --
        else {
            let node = self.node();
            if action >= node.num_actions() {
                panic!("Invalid action");
            }

            let player = node.player();
            let num_hands = self.num_private_hands(player);

            // Update the weights
            if node.num_actions() > 1 {
                let strategy = self.strategy();
                let weights = row(&strategy, action, num_hands);
                mul_slice(&mut self.weights[player], weights);
            }

            // Cache the counterfactual values
            let node = self.node();
            let vec = if self.is_compression_enabled {
                let slice = row(node.cfvalues_compressed(), action, num_hands);
                let scale = node.cfvalue_scale();
                decode_signed_slice(slice, scale)
            } else {
                row(node.cfvalues(), action, num_hands).to_vec()
            };
            self.cfvalues_cache[player].copy_from_slice(&vec);

            // Update the bet amounts
            let node = self.node();
            match node.play(action).prev_action {
                Action::Call => {
                    self.total_bet_amount[player] = self.total_bet_amount[player ^ 1];
                }
                Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => {
                    let prev_bet_amount = match node.prev_action {
                        Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => a,
                        _ => 0,
                    };
                    let to_call =
                        self.total_bet_amount[player ^ 1] - self.total_bet_amount[player];
                    self.total_bet_amount[player] += amount - prev_bet_amount + to_call;
                }
                _ => {}
            }

            // Update the node
            let node_index = self.node_index(&self.node().play(action));
            self.node_history.push(node_index);
        }

        self.action_history.push(action);
        self.is_normalized_weight_cached = false;
    }
}

// ---------------------------------------------------------------------------
// Strategy queries
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Returns the normalized strategy at the current node.
    ///
    /// The return vec has length `num_actions * num_private_hands(player)`.
    /// Action `i`, hand `j` is at index `i * num_hands + j`.
    pub fn strategy(&self) -> Vec<f32> {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let node = self.node();
        let player = self.current_player();
        let num_actions = node.num_actions();
        let num_hands = self.num_private_hands(player);

        let mut ret = if self.is_compression_enabled {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        let locking = self.locking_strategy(&node);
        apply_locking_strategy(&mut ret, locking);

        ret.chunks_exact_mut(num_hands).for_each(|chunk| {
            self.apply_swap(chunk, player, false);
        });

        ret
    }

    /// Returns the total bet amount for each player `[OOP, IP]`.
    #[inline]
    pub fn total_bet_amount(&self) -> [i32; 2] {
        self.total_bet_amount
    }

    /// Returns the number of hand-pair combinations in the game.
    #[inline]
    pub fn num_combinations(&self) -> f64 {
        self.num_combinations
    }
}

// ---------------------------------------------------------------------------
// Weight caching and value queries
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Computes and caches normalized weights for both players.
    ///
    /// Must be called before [`normalized_weights`], [`equity`], or
    /// [`expected_values`].
    pub fn cache_normalized_weights(&mut self) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.is_normalized_weight_cached {
            return;
        }

        let mut board_mask: u64 = 0;
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }
        if self.river != NOT_DEALT {
            board_mask |= 1 << self.river;
        }

        let mut weight_sum = [0.0f64; 2];
        let mut weight_sum_minus = [[0.0f64; 52]; 2];

        for player in 0..2 {
            let ws = &mut weight_sum[player];
            let wsm = &mut weight_sum_minus[player];
            self.private_cards[player]
                .iter()
                .zip(self.weights[player].iter())
                .for_each(|(&(c1, c2), &w)| {
                    let mask: u64 = (1 << c1) | (1 << c2);
                    if mask & board_mask == 0 {
                        let w = w as f64;
                        *ws += w;
                        wsm[c1 as usize] += w;
                        wsm[c2 as usize] += w;
                    }
                });
        }

        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let same_hand_index = &self.same_hand_index[player];
            let player_weights = &self.weights[player];
            let opponent_weights = &self.weights[player ^ 1];
            let opponent_weight_sum = weight_sum[player ^ 1];
            let opponent_weight_sum_minus = &weight_sum_minus[player ^ 1];

            self.normalized_weights[player]
                .iter_mut()
                .enumerate()
                .for_each(|(i, w)| {
                    let (c1, c2) = player_cards[i];
                    let mask: u64 = (1 << c1) | (1 << c2);
                    if mask & board_mask == 0 {
                        let same_i = same_hand_index[i];
                        let opponent_weight_same = if same_i == u16::MAX {
                            0.0
                        } else {
                            opponent_weights[same_i as usize] as f64
                        };
                        let opponent_weight = opponent_weight_sum + opponent_weight_same
                            - opponent_weight_sum_minus[c1 as usize]
                            - opponent_weight_sum_minus[c2 as usize];
                        *w = player_weights[i] * opponent_weight as f32;
                    } else {
                        *w = 0.0;
                    }
                });
        }

        self.is_normalized_weight_cached = true;
    }

    /// Returns the normalized weights for the given player.
    ///
    /// Call [`cache_normalized_weights`] first after any navigation.
    #[inline]
    pub fn normalized_weights(&self, player: usize) -> &[f32] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }
        &self.normalized_weights[player]
    }

    /// Returns the raw weights for the given player.
    #[inline]
    pub fn weights(&self, player: usize) -> &[f32] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        &self.weights[player]
    }

    /// Returns the per-hand equity for the given player.
    ///
    /// Values are in `[0, 1]`. Call [`cache_normalized_weights`] first.
    pub fn equity(&self, player: usize) -> Vec<f32> {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }
        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let num_hands = self.num_private_hands(player);

        let mut tmp = vec![0.0f64; num_hands];
        if self.river != NOT_DEALT {
            self.equity_internal(&mut tmp, player, self.turn, self.river, 0.5);
        } else if self.turn != NOT_DEALT {
            for river in 0..52u8 {
                if self.turn != river {
                    self.equity_internal(&mut tmp, player, self.turn, river, 0.5 / 44.0);
                }
            }
        } else {
            for turn in 0..52u8 {
                for river in turn + 1..52u8 {
                    self.equity_internal(&mut tmp, player, turn, river, 1.0 / (45.0 * 44.0));
                }
            }
        }
        let tmp: Vec<f32> = tmp.into_iter().map(|v| v as f32).collect();

        tmp.iter()
            .zip(self.weights[player].iter())
            .zip(self.normalized_weights[player].iter())
            .map(|((&v, &w_raw), &w_normalized)| {
                if w_normalized > 0.0 {
                    v * (w_raw / w_normalized) + 0.5
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Returns the per-hand expected values for the given player.
    ///
    /// Call [`cache_normalized_weights`] first. Game must be solved.
    pub fn expected_values(&self, player: usize) -> Vec<f32> {
        if self.state != State::Solved {
            panic!("Game is not solved");
        }
        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let expected_value_detail = self.expected_values_detail(player);

        if self.is_terminal_node() || self.is_chance_node() || self.current_player() != player {
            return expected_value_detail;
        }

        let num_actions = self.node().num_actions();
        let num_hands = self.num_private_hands(player);
        let strategy = self.strategy();

        let mut ret = Vec::with_capacity(num_hands);
        for i in 0..num_hands {
            let mut ev = 0.0;
            for j in 0..num_actions {
                let index = i + j * num_hands;
                ev += expected_value_detail[index] * strategy[index];
            }
            ret.push(ev);
        }

        ret
    }

    /// Returns the per-action, per-hand expected values for the given player.
    ///
    /// If the player is the current player, the return vec has length
    /// `num_actions * num_hands`; otherwise `num_hands`.
    pub fn expected_values_detail(&self, player: usize) -> Vec<f32> {
        if self.state != State::Solved {
            panic!("Game is not solved");
        }
        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let node = self.node();
        let num_hands = self.num_private_hands(player);

        let mut chance_factor: usize = 1;
        if self.card_config.turn == NOT_DEALT && self.turn != NOT_DEALT {
            chance_factor *= 45 - self.bunching_num_dead_cards;
        }
        if self.card_config.river == NOT_DEALT && self.river != NOT_DEALT {
            chance_factor *= 44 - self.bunching_num_dead_cards;
        }

        let num_combinations = self.num_combinations;

        let mut have_actions = false;
        let mut normalizer = (num_combinations * chance_factor as f64) as f32;

        let mut ret = if node.is_terminal() {
            normalizer = num_combinations as f32;
            let mut ret = Vec::with_capacity(num_hands);
            let cfreach = self.weights[player ^ 1].clone();
            self.evaluate(ret.spare_capacity_mut(), &node, player, &cfreach);
            // SAFETY: evaluate writes all num_hands elements.
            unsafe { ret.set_len(num_hands) };
            ret
        } else if node.is_chance() && node.cfvalue_storage_player() == Some(player) {
            if self.is_compression_enabled {
                let slice = node.cfvalues_chance_compressed();
                let scale = node.cfvalue_chance_scale();
                decode_signed_slice(slice, scale)
            } else {
                node.cfvalues_chance().to_vec()
            }
        } else if node.has_cfvalues_ip() && player == PLAYER_IP as usize {
            if self.is_compression_enabled {
                let slice = node.cfvalues_ip_compressed();
                let scale = node.cfvalue_ip_scale();
                decode_signed_slice(slice, scale)
            } else {
                node.cfvalues_ip().to_vec()
            }
        } else if player == self.current_player() {
            have_actions = true;
            if self.is_compression_enabled {
                let slice = node.cfvalues_compressed();
                let scale = node.cfvalue_scale();
                decode_signed_slice(slice, scale)
            } else {
                node.cfvalues().to_vec()
            }
        } else {
            self.cfvalues_cache[player].to_vec()
        };

        let starting_pot = self.tree_config.starting_pot;
        let total_bet_amount = self.total_bet_amount();
        let bias = (total_bet_amount[player] - total_bet_amount[player ^ 1]).max(0);

        ret.chunks_exact_mut(num_hands)
            .enumerate()
            .for_each(|(action, r)| {
                let is_fold =
                    have_actions && self.node().play(action).prev_action == Action::Fold;
                self.apply_swap(r, player, false);
                r.iter_mut()
                    .zip(self.weights[player].iter())
                    .zip(self.normalized_weights[player].iter())
                    .for_each(|((v, &w_raw), &w_normalized)| {
                        if is_fold || w_normalized == 0.0 {
                            *v = 0.0;
                        } else {
                            *v *= normalizer * (w_raw / w_normalized);
                            *v += starting_pot as f32 * 0.5
                                + (self.node().amount + bias) as f32;
                        }
                    });
            });

        ret
    }
}

// ---------------------------------------------------------------------------
// Node locking
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Locks the strategy at the current node.
    ///
    /// The `strategy` slice must have length `num_actions * num_hands`.
    /// Positive values lock that hand; all-zero or all-negative leaves it free.
    pub fn lock_current_strategy(&mut self, strategy: &[f32]) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }
        if self.state == State::Solved {
            panic!("Game is already solved");
        }
        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }
        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let mut node = self.node();
        let player = self.current_player();
        let num_actions = node.num_actions();
        let num_hands = self.num_private_hands(player);

        if strategy.len() != num_actions * num_hands {
            panic!("Invalid strategy length");
        }

        let mut locking = vec![-1.0; num_actions * num_hands];

        for hand in 0..num_hands {
            let mut sum = 0.0f64;
            let mut lock = false;

            for action in 0..num_actions {
                let freq = strategy[action * num_hands + hand];
                if freq > 0.0 {
                    sum += freq as f64;
                    lock = true;
                }
            }

            if lock {
                for action in 0..num_actions {
                    let freq = strategy[action * num_hands + hand].max(0.0) as f64;
                    locking[action * num_hands + hand] = (freq / sum) as f32;
                }
            }
        }

        locking.chunks_exact_mut(num_hands).for_each(|chunk| {
            self.apply_swap(chunk, player, true);
        });

        node.is_locked = true;
        let index = self.node_index(&node);
        self.locking_strategy.insert(index, locking);
    }

    /// Unlocks the strategy at the current node.
    #[inline]
    pub fn unlock_current_strategy(&mut self) {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }
        if self.state == State::Solved {
            panic!("Game is already solved");
        }
        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }
        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        let mut node = self.node();
        if !node.is_locked {
            return;
        }

        node.is_locked = false;
        let index = self.node_index(&node);
        self.locking_strategy.remove(&index);
    }
}

// ---------------------------------------------------------------------------
// Swap helper
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Applies isomorphism swaps to a slice.
    ///
    /// When `reverse` is true, the swaps are applied in reverse order
    /// (river first, then turn) to undo a previous forward swap.
    #[inline]
    pub(crate) fn apply_swap(&self, slice: &mut [f32], player: usize, reverse: bool) {
        let turn_swap = self
            .turn_swap
            .map(|suit| &self.isomorphism_swap_turn[suit as usize][player]);

        let river_swap = self.river_swap.map(|(turn_suit, suit)| {
            &self.isomorphism_swap_river[turn_suit as usize][suit as usize][player]
        });

        let swaps = if !reverse {
            [turn_swap, river_swap]
        } else {
            [river_swap, turn_swap]
        };

        for swap in swaps.into_iter().flatten() {
            for &(i, j) in swap {
                slice.swap(i as usize, j as usize);
            }
        }
    }

    /// Assigns zero weights to hands that conflict with the current board.
    pub(crate) fn assign_zero_weights(&mut self) {
        let mut board_mask: u64 = 0;
        if self.turn != NOT_DEALT {
            board_mask |= 1 << self.turn;
        }
        if self.river != NOT_DEALT {
            board_mask |= 1 << self.river;
        }

        for player in 0..2 {
            let mut dead_mask: u64 = (1u64 << 52) - 1;

            for &(c1, c2) in &self.private_cards[player ^ 1] {
                let mask: u64 = (1 << c1) | (1 << c2);
                if mask & board_mask == 0 {
                    dead_mask &= mask;
                }
                if dead_mask == 0 {
                    break;
                }
            }

            dead_mask |= board_mask;

            self.private_cards[player]
                .iter()
                .zip(self.weights[player].iter_mut())
                .for_each(|(&(c1, c2), w)| {
                    let mask: u64 = (1 << c1) | (1 << c2);
                    if mask & dead_mask != 0 {
                        *w = 0.0;
                    }
                });
        }
    }

    /// Internal equity computation for a specific turn/river runout.
    fn equity_internal(
        &self,
        result: &mut [f64],
        player: usize,
        turn: Card,
        river: Card,
        amount: f64,
    ) {
        let pair_index = card_pair_to_index(turn, river);
        let hand_strength = &self.hand_strength[pair_index];
        let player_strength = &hand_strength[player];
        let opponent_strength = &hand_strength[player ^ 1];

        let player_len = player_strength.len();
        let opponent_len = opponent_strength.len();

        if player_len == 0 || opponent_len == 0 {
            return;
        }

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let opponent_weights = &self.weights[player ^ 1];
        let mut weight_sum = 0.0f64;
        let mut weight_minus = [0.0f64; 52];

        let valid_player_strength = &player_strength[1..player_len - 1];
        let mut i = 1;

        // Ascending pass: accumulate weaker opponent hands
        for &StrengthItem { strength, index } in valid_player_strength {
            // SAFETY: `opponent_strength` has sentinels at both ends. `i` advances
            // monotonically within the valid range.
            unsafe {
                while opponent_strength.get_unchecked(i).strength < strength {
                    let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let weight_i = *opponent_weights.get_unchecked(opponent_index) as f64;
                    weight_sum += weight_i;
                    *weight_minus.get_unchecked_mut(c1 as usize) += weight_i;
                    *weight_minus.get_unchecked_mut(c2 as usize) += weight_i;
                    i += 1;
                }
                let (c1, c2) = *player_cards.get_unchecked(index as usize);
                let opponent_weight = weight_sum
                    - weight_minus.get_unchecked(c1 as usize)
                    - weight_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(index as usize) += amount * opponent_weight;
            }
        }

        // Descending pass: accumulate stronger opponent hands
        weight_sum = 0.0;
        weight_minus.fill(0.0);
        i = opponent_len - 2;

        for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
            // SAFETY: Same invariants as the ascending pass, but in reverse.
            unsafe {
                while opponent_strength.get_unchecked(i).strength > strength {
                    let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let weight_i = *opponent_weights.get_unchecked(opponent_index) as f64;
                    weight_sum += weight_i;
                    *weight_minus.get_unchecked_mut(c1 as usize) += weight_i;
                    *weight_minus.get_unchecked_mut(c2 as usize) += weight_i;
                    i -= 1;
                }
                let (c1, c2) = *player_cards.get_unchecked(index as usize);
                let opponent_weight = weight_sum
                    - weight_minus.get_unchecked(c1 as usize)
                    - weight_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(index as usize) -= amount * opponent_weight;
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
    use crate::solver::solve;

    fn build_river_game() -> PostFlopGame {
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
        PostFlopGame::with_config(card_config, tree).unwrap()
    }

    #[test]
    fn test_solve_and_query() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        solve(&mut game, 100, 0.0, false);

        game.cache_normalized_weights();

        let ev_oop = game.expected_values(0);
        let ev_ip = game.expected_values(1);
        assert_eq!(ev_oop.len(), game.num_private_hands(0));
        assert_eq!(ev_ip.len(), game.num_private_hands(1));

        let equity_oop = game.equity(0);
        for &e in &equity_oop {
            assert!(
                e >= -0.01 && e <= 1.01,
                "equity out of range: {e}"
            );
        }
    }

    #[test]
    fn test_navigate_and_query_strategy() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        solve(&mut game, 100, 0.0, false);

        // Query root strategy
        let strategy = game.strategy();
        let num_hands = game.num_private_hands(0); // OOP acts first
        let num_actions = game.available_actions().len();
        assert_eq!(strategy.len(), num_actions * num_hands);

        // Verify each hand's strategy sums to 1
        for h in 0..num_hands {
            let sum: f32 = (0..num_actions)
                .map(|a| strategy[a * num_hands + h])
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "strategy for hand {h} sums to {sum}"
            );
        }

        // Navigate to a child and back
        game.play(0);
        game.back_to_root();
        assert!(game.history().is_empty());
    }

    #[test]
    fn test_back_to_root() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        solve(&mut game, 10, 0.0, false);

        game.play(0);
        assert_eq!(game.history().len(), 1);
        game.back_to_root();
        assert!(game.history().is_empty());
        assert_eq!(game.total_bet_amount(), [0, 0]);
    }

    #[test]
    fn test_is_terminal_and_chance() {
        let mut game = build_river_game();
        game.allocate_memory(false);

        // Root is not terminal, not chance (it's a river game, OOP decides)
        assert!(!game.is_terminal_node());
        assert!(!game.is_chance_node());
    }

    #[test]
    fn test_available_actions_root() {
        let mut game = build_river_game();
        game.allocate_memory(false);

        let actions = game.available_actions();
        assert!(!actions.is_empty(), "root should have actions");
    }

    #[test]
    fn test_apply_history() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        solve(&mut game, 10, 0.0, false);

        game.play(0);
        let history = game.history().to_vec();
        game.back_to_root();
        game.apply_history(&history);
        assert_eq!(game.history(), &history);
    }

    #[test]
    fn test_normalized_weights() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        solve(&mut game, 10, 0.0, false);

        game.cache_normalized_weights();
        let nw = game.normalized_weights(0);
        assert_eq!(nw.len(), game.num_private_hands(0));

        // All normalized weights should be non-negative
        for &w in nw {
            assert!(w >= 0.0, "normalized weight should be >= 0, got {w}");
        }
    }

    #[test]
    fn test_lock_and_unlock() {
        let mut game = build_river_game();
        game.allocate_memory(false);

        let num_actions = game.available_actions().len();
        let num_hands = game.num_private_hands(game.current_player());

        // Lock all hands to first action
        let mut strategy = vec![0.0; num_actions * num_hands];
        for h in 0..num_hands {
            strategy[h] = 1.0; // action 0
        }
        game.lock_current_strategy(&strategy);

        // Verify the node is locked
        let node = game.node_arena[0].lock();
        assert!(node.is_locked);
        drop(node);

        // Unlock
        game.unlock_current_strategy();
        let node = game.node_arena[0].lock();
        assert!(!node.is_locked);
    }

    #[test]
    fn test_total_bet_amount_after_play() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        solve(&mut game, 10, 0.0, false);

        // At root, bet amounts are zero
        assert_eq!(game.total_bet_amount(), [0, 0]);

        // After OOP bets, the OOP bet amount should increase
        let actions = game.available_actions();
        // Find a bet action
        let bet_idx = actions.iter().position(|a| matches!(a, Action::Bet(_)));
        if let Some(idx) = bet_idx {
            game.play(idx);
            let bets = game.total_bet_amount();
            assert!(bets[0] > 0, "OOP should have bet amount > 0 after betting");
        }
    }

    #[test]
    fn test_current_board() {
        let game = build_river_game();
        let board = game.current_board();
        assert_eq!(board.len(), 5); // flop + turn + river
    }

    #[test]
    fn test_num_combinations() {
        let game = build_river_game();
        assert!(game.num_combinations() > 0.0);
    }

    #[test]
    fn test_possible_cards_not_chance() {
        let mut game = build_river_game();
        game.allocate_memory(false);
        // At a river game root (player node), possible_cards should be 0
        assert_eq!(game.possible_cards(), 0);
    }

    #[test]
    fn test_possible_cards_chance_node() {
        // Build a turn game to get a chance node
        let oop_range: crate::range::Range = "AA,KK".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ".parse().unwrap();
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
            effective_stack: 200,
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        // Navigate to a chance node: OOP check, IP check -> chance node for river
        let actions = game.available_actions();
        let check_idx = actions.iter().position(|a| *a == Action::Check);
        if let Some(idx) = check_idx {
            game.play(idx); // OOP checks
            let actions2 = game.available_actions();
            let check_idx2 = actions2.iter().position(|a| *a == Action::Check);
            if let Some(idx2) = check_idx2 {
                game.play(idx2); // IP checks
                assert!(
                    game.is_chance_node(),
                    "after both check on turn, should be chance node"
                );
                let possible = game.possible_cards();
                assert!(possible != 0, "should have possible cards at chance node");
            }
        }
    }
}
