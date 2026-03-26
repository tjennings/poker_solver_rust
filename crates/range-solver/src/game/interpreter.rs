use super::*;
use crate::interface::*;
use std::mem;

// ---------------------------------------------------------------------------
// BuildTreeInfo — accumulates indices and storage counts during DFS
// ---------------------------------------------------------------------------

#[derive(Default)]
struct BuildTreeInfo {
    flop_index: usize,
    turn_index: usize,
    river_index: usize,
    num_storage: u64,
    num_storage_ip: u64,
    num_storage_chance: u64,
}

// ---------------------------------------------------------------------------
// Game trait (minimal subset for compilation — full impl in Task 15)
// ---------------------------------------------------------------------------

impl Game for PostFlopGame {
    type Node = PostFlopNode;

    #[inline]
    fn root(&self) -> MutexGuardLike<'_, Self::Node> {
        self.node_arena[0].lock()
    }

    #[inline]
    fn num_private_hands(&self, player: usize) -> usize {
        self.private_cards[player].len()
    }

    #[inline]
    fn initial_weights(&self, player: usize) -> &[f32] {
        &self.initial_weights[player]
    }

    #[inline]
    fn evaluate(
        &self,
        result: &mut [mem::MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        self.evaluate_internal(result, node, player, cfreach);
    }

    #[inline]
    fn chance_factor(&self, node: &Self::Node) -> usize {
        if node.turn == NOT_DEALT {
            45 - self.bunching_num_dead_cards
        } else {
            44 - self.bunching_num_dead_cards
        }
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.state == State::Solved
    }

    fn set_solved(&mut self) {
        self.state = State::Solved;
        let history = self.action_history.clone();
        self.apply_history(&history);
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.state == State::MemoryAllocated && self.storage_mode == BoardState::River
    }

    #[inline]
    fn is_raked(&self) -> bool {
        self.tree_config.rake_rate > 0.0 && self.tree_config.rake_cap > 0.0
    }

    #[inline]
    fn isomorphic_chances(&self, node: &Self::Node) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.isomorphism_ref_turn
        } else {
            &self.isomorphism_ref_river[node.turn as usize]
        }
    }

    #[inline]
    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(u16, u16)>; 2] {
        if node.turn == NOT_DEALT {
            &self.isomorphism_swap_turn[self.isomorphism_card_turn[index] as usize & 3]
        } else {
            &self.isomorphism_swap_river[node.turn as usize & 3]
                [self.isomorphism_card_river[node.turn as usize & 3][index] as usize & 3]
        }
    }

    #[inline]
    fn locking_strategy(&self, node: &Self::Node) -> &[f32] {
        if !node.is_locked {
            &[]
        } else {
            let index = self.node_index(node);
            // INVARIANT: locked nodes always have an entry in locking_strategy
            self.locking_strategy.get(&index).unwrap()
        }
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }
}

// ---------------------------------------------------------------------------
// Construction and configuration
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Creates a new [`PostFlopGame`] with the specified configuration.
    ///
    /// Validates the card configuration, enumerates private hands, computes
    /// hand strength tables, isomorphism data, and builds the game tree.
    #[inline]
    pub fn with_config(
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<Self, String> {
        let mut game = Self::default();
        game.update_config(card_config, action_tree)?;
        Ok(game)
    }

    /// Updates the game configuration. Any previous solved result is lost.
    pub fn update_config(
        &mut self,
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<(), String> {
        self.state = State::ConfigError;

        if !action_tree.invalid_terminals().is_empty() {
            return Err("Invalid terminal found in action tree".to_string());
        }

        self.card_config = card_config;
        (
            self.tree_config,
            self.added_lines,
            self.removed_lines,
            self.action_root,
        ) = action_tree.eject();

        self.check_card_config()?;
        self.init_card_fields();
        self.init_root()?;

        self.state = State::TreeBuilt;

        self.init_interpreter();

        Ok(())
    }

    /// Returns the card configuration.
    #[inline]
    pub fn card_config(&self) -> &CardConfig {
        &self.card_config
    }

    /// Returns the tree configuration.
    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Returns the card list of private hands of the given player.
    #[inline]
    pub fn private_cards(&self, player: usize) -> &[(Card, Card)] {
        &self.private_cards[player]
    }

    /// Returns the estimated memory usage in bytes `(uncompressed, compressed)`.
    #[inline]
    pub fn memory_usage(&self) -> (u64, u64) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        let num_elements =
            2 * self.num_storage + self.num_storage_ip + self.num_storage_chance;
        let uncompressed = 4 * num_elements + self.misc_memory_usage;
        let compressed = 2 * num_elements + self.misc_memory_usage;

        (uncompressed, compressed)
    }

    /// Returns whether memory is allocated. If so, returns `Some(is_compression)`.
    #[inline]
    pub fn is_memory_allocated(&self) -> Option<bool> {
        if self.state <= State::TreeBuilt {
            None
        } else {
            Some(self.is_compression_enabled)
        }
    }

    /// Allocates the memory for strategy / regret / cfvalue storage.
    pub fn allocate_memory(&mut self, enable_compression: bool) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.state == State::MemoryAllocated
            && self.storage_mode == BoardState::River
            && self.is_compression_enabled == enable_compression
        {
            return;
        }

        let num_bytes: u64 = if enable_compression { 2 } else { 4 };
        if num_bytes * self.num_storage > isize::MAX as u64
            || num_bytes * self.num_storage_chance > isize::MAX as u64
        {
            panic!("Memory usage exceeds maximum size");
        }

        self.state = State::MemoryAllocated;
        self.is_compression_enabled = enable_compression;

        self.clear_storage();

        let storage_bytes = (num_bytes * self.num_storage) as usize;
        let storage_ip_bytes = (num_bytes * self.num_storage_ip) as usize;
        let storage_chance_bytes = (num_bytes * self.num_storage_chance) as usize;

        self.storage1 = vec![0; storage_bytes];
        self.storage2 = vec![0; storage_bytes];
        self.storage_ip = vec![0; storage_ip_bytes];
        self.storage_chance = vec![0; storage_chance_bytes];

        self.allocate_memory_nodes();

        self.storage_mode = BoardState::River;
        self.target_storage_mode = BoardState::River;
    }

    /// Returns the index of the given node within `node_arena`.
    #[inline]
    pub(crate) fn node_index(&self, node: &PostFlopNode) -> usize {
        let node_ptr = node as *const _ as *const MutexLike<PostFlopNode>;
        // SAFETY: `node` is always a reference into `self.node_arena`, which is
        // a contiguous `Vec`. `MutexLike<T>` is `#[repr(transparent)]`.
        unsafe { node_ptr.offset_from(self.node_arena.as_ptr()) as usize }
    }

    /// Returns references to the four internal storage buffers.
    ///
    /// The buffers hold strategy/regret data (`storage1`, `storage2`),
    /// IP counterfactual values (`storage_ip`), and chance node values
    /// (`storage_chance`).
    pub fn storage_buffers(&self) -> (&[u8], &[u8], &[u8], &[u8]) {
        (
            &self.storage1,
            &self.storage2,
            &self.storage_ip,
            &self.storage_chance,
        )
    }

    /// Overwrites the internal storage buffers with cached data and
    /// re-assigns the raw pointers in each `PostFlopNode`.
    ///
    /// # Panics
    /// Panics if any buffer length does not match the allocated size.
    pub fn set_storage_buffers(
        &mut self,
        s1: Vec<u8>,
        s2: Vec<u8>,
        s_ip: Vec<u8>,
        s_chance: Vec<u8>,
    ) {
        assert_eq!(s1.len(), self.storage1.len(), "storage1 size mismatch");
        assert_eq!(s2.len(), self.storage2.len(), "storage2 size mismatch");
        assert_eq!(s_ip.len(), self.storage_ip.len(), "storage_ip size mismatch");
        assert_eq!(
            s_chance.len(),
            self.storage_chance.len(),
            "storage_chance size mismatch"
        );
        self.storage1 = s1;
        self.storage2 = s2;
        self.storage_ip = s_ip;
        self.storage_chance = s_chance;
        // Re-assign raw pointers in each node to the new buffer locations.
        self.allocate_memory_nodes();
    }
}

// ---------------------------------------------------------------------------
// Depth boundary API
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Returns the number of depth boundary terminals in the tree.
    ///
    /// Each boundary terminal needs CFVs set via [`set_boundary_cfvs`]
    /// before solving.
    #[inline]
    pub fn num_boundary_nodes(&self) -> usize {
        // boundary_cfvs has 2 entries per boundary node (one per player)
        self.boundary_cfvs.len() / 2
    }

    /// Returns the node arena indices of all depth boundary terminals,
    /// ordered by their boundary ordinal.
    pub fn boundary_node_indices(&self) -> Vec<usize> {
        let n = self.num_boundary_nodes();
        let mut result = vec![0usize; n];
        for (idx, &ordinal) in self.node_to_boundary.iter().enumerate() {
            if ordinal != u32::MAX {
                result[ordinal as usize] = idx;
            }
        }
        result
    }

    /// Sets the counterfactual values for a depth boundary terminal.
    ///
    /// `ordinal` identifies the boundary node (0-based, in tree traversal
    /// order). `player` is the player whose CFVs are being set (0 = OOP,
    /// 1 = IP). `cfvs` must have one entry per private hand for `player`.
    ///
    /// CFVs should be in pot-normalised units: a value of 1.0 means the
    /// player wins 1× the half-pot from that point forward.
    pub fn set_boundary_cfvs(&mut self, ordinal: usize, player: usize, cfvs: Vec<f32>) {
        let idx = ordinal * 2 + player;
        if idx >= self.boundary_cfvs.len() {
            self.boundary_cfvs.resize(idx + 1, Vec::new());
        }
        self.boundary_cfvs[idx] = cfvs;
    }

    /// Returns the pot size at a given depth boundary node.
    ///
    /// Useful for callers that need to convert pot-normalised CFVs back to
    /// chip values.
    pub fn boundary_pot(&self, ordinal: usize) -> i32 {
        let indices = self.boundary_node_indices();
        let node = self.node_arena[indices[ordinal]].lock();
        self.tree_config.starting_pot + 2 * node.amount
    }

    /// Returns true if boundary CFVs for the given ordinal and player are empty.
    pub fn boundary_cfvs_empty(&self, ordinal: usize, player: usize) -> bool {
        let idx = ordinal * 2 + player;
        idx >= self.boundary_cfvs.len() || self.boundary_cfvs[idx].is_empty()
    }
}

// ---------------------------------------------------------------------------
// Card config validation
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Validates the card configuration and initialises hand data.
    fn check_card_config(&mut self) -> Result<(), String> {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        if flop.contains(&NOT_DEALT) {
            return Err("Flop cards not initialized".to_string());
        }

        if flop.iter().any(|&c| c >= 52) {
            return Err(format!("Flop cards must be in [0, 52): flop = {flop:?}"));
        }

        if flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2] {
            return Err(format!("Flop cards must be unique: flop = {flop:?}"));
        }

        if turn != NOT_DEALT {
            if turn >= 52 {
                return Err(format!("Turn card must be in [0, 52): turn = {turn}"));
            }
            if flop.contains(&turn) {
                return Err(format!(
                    "Turn card must be different from flop cards: turn = {turn}"
                ));
            }
        }

        if river != NOT_DEALT {
            if river >= 52 {
                return Err(format!("River card must be in [0, 52): river = {river}"));
            }
            if flop.contains(&river) {
                return Err(format!(
                    "River card must be different from flop cards: river = {river}"
                ));
            }
            if turn == river {
                return Err(format!(
                    "River card must be different from turn card: river = {river}"
                ));
            }
            if turn == NOT_DEALT {
                return Err(format!(
                    "River card specified without turn card: river = {river}"
                ));
            }
        }

        let expected_state = match (turn != NOT_DEALT, river != NOT_DEALT) {
            (false, _) => BoardState::Flop,
            (true, false) => BoardState::Turn,
            (true, true) => BoardState::River,
        };

        if self.tree_config.initial_state != expected_state {
            return Err(format!(
                "Invalid initial state of `tree_config`: expected = {expected_state:?}, \
                 actual = {:?}",
                self.tree_config.initial_state
            ));
        }

        if range[0].is_empty() {
            return Err("OOP range is empty".to_string());
        }
        if range[1].is_empty() {
            return Err("IP range is empty".to_string());
        }
        if !range[0].is_valid() {
            return Err("OOP range is invalid (loaded broken data?)".to_string());
        }
        if !range[1].is_valid() {
            return Err("IP range is invalid (loaded broken data?)".to_string());
        }

        self.init_hands();
        self.num_combinations = 0.0;

        for (&(c1, c2), &w1) in self.private_cards[0]
            .iter()
            .zip(self.initial_weights[0].iter())
        {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            for (&(c3, c4), &w2) in self.private_cards[1]
                .iter()
                .zip(self.initial_weights[1].iter())
            {
                let ip_mask: u64 = (1 << c3) | (1 << c4);
                if oop_mask & ip_mask == 0 {
                    self.num_combinations += w1 as f64 * w2 as f64;
                }
            }
        }

        if self.num_combinations == 0.0 {
            return Err("Valid card assignment does not exist".to_string());
        }

        Ok(())
    }

    /// Initialises `initial_weights` and `private_cards` from the ranges.
    fn init_hands(&mut self) {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);

        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if turn != NOT_DEALT {
            board_mask |= 1 << turn;
        }
        if river != NOT_DEALT {
            board_mask |= 1 << river;
        }

        for player in 0..2 {
            let (hands, weights) = config.range[player].get_hands_weights(board_mask);
            self.initial_weights[player] = weights;
            self.private_cards[player] = hands;
        }
    }
}

// ---------------------------------------------------------------------------
// Card fields: same_hand_index, valid_indices, hand_strength, isomorphism
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Initialises all per-card computed fields.
    fn init_card_fields(&mut self) {
        // same_hand_index
        for player in 0..2 {
            let same = &mut self.same_hand_index[player];
            same.clear();

            let player_hands = &self.private_cards[player];
            let opponent_hands = &self.private_cards[player ^ 1];
            for hand in player_hands {
                same.push(
                    opponent_hands
                        .binary_search(hand)
                        .map_or(u16::MAX, |i| i as u16),
                );
            }
        }

        (
            self.valid_indices_flop,
            self.valid_indices_turn,
            self.valid_indices_river,
        ) = self.card_config.valid_indices(&self.private_cards);

        self.hand_strength = self.card_config.hand_strength(&self.private_cards);

        (
            self.isomorphism_ref_turn,
            self.isomorphism_card_turn,
            self.isomorphism_swap_turn,
            self.isomorphism_ref_river,
            self.isomorphism_card_river,
            self.isomorphism_swap_river,
        ) = self.card_config.isomorphism(&self.private_cards);
    }
}

// ---------------------------------------------------------------------------
// Tree building
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Counts the total number of nodes per street.
    fn count_num_nodes(&self) -> [u64; 3] {
        let (turn_coef, river_coef) = match (self.card_config.turn, self.card_config.river) {
            (NOT_DEALT, _) => {
                let mut river_coef = 0;
                let flop = self.card_config.flop;
                let skip_cards = &self.isomorphism_card_turn;
                let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
                let skip_mask: u64 = skip_cards.iter().map(|&card| 1u64 << card).sum();
                for turn in 0..52 {
                    if (1u64 << turn) & (flop_mask | skip_mask) == 0 {
                        river_coef += 48 - self.isomorphism_card_river[turn & 3].len();
                    }
                }
                (49 - self.isomorphism_card_turn.len(), river_coef)
            }
            (turn, NOT_DEALT) => {
                (1, 48 - self.isomorphism_card_river[turn as usize & 3].len())
            }
            _ => (0, 1),
        };

        let num_action_nodes = count_num_action_nodes(
            &self.action_root.lock(),
            self.tree_config.initial_state,
        );

        [
            num_action_nodes[0],
            num_action_nodes[1] * turn_coef as u64,
            num_action_nodes[2] * river_coef as u64,
        ]
    }

    /// Allocates the node arena and builds the game tree via DFS.
    fn init_root(&mut self) -> Result<(), String> {
        let num_nodes = self.count_num_nodes();
        let total_num_nodes = num_nodes[0] + num_nodes[1] + num_nodes[2];

        if total_num_nodes > u32::MAX as u64
            || mem::size_of::<PostFlopNode>() as u64 * total_num_nodes > isize::MAX as u64
        {
            return Err("Too many nodes".to_string());
        }

        self.num_nodes = num_nodes;
        self.node_arena = (0..total_num_nodes)
            .map(|_| MutexLike::new(PostFlopNode::default()))
            .collect::<Vec<_>>();
        self.clear_storage();

        let mut info = BuildTreeInfo {
            turn_index: num_nodes[0] as usize,
            river_index: (num_nodes[0] + num_nodes[1]) as usize,
            ..Default::default()
        };

        match self.tree_config.initial_state {
            BoardState::Flop => info.flop_index += 1,
            BoardState::Turn => info.turn_index += 1,
            BoardState::River => info.river_index += 1,
        }

        {
            let mut root = self.node_arena[0].lock();
            root.turn = self.card_config.turn;
            root.river = self.card_config.river;
        }

        self.build_tree_recursive(0, &self.action_root.lock(), &mut info);

        self.num_storage = info.num_storage;
        self.num_storage_ip = info.num_storage_ip;
        self.num_storage_chance = info.num_storage_chance;

        // Build the boundary ordinal map by scanning the arena for depth
        // boundary terminals. Ordinals are assigned in arena order, which
        // matches the DFS traversal order used during tree construction.
        self.node_to_boundary = vec![u32::MAX; total_num_nodes as usize];
        let mut boundary_count = 0u32;
        for (idx, node_mutex) in self.node_arena.iter().enumerate() {
            if node_mutex.lock().is_depth_boundary() {
                self.node_to_boundary[idx] = boundary_count;
                boundary_count += 1;
            }
        }
        // 2 slots per boundary node: one for each player.
        self.boundary_cfvs = vec![Vec::new(); boundary_count as usize * 2];

        self.misc_memory_usage = self.memory_usage_internal();

        Ok(())
    }

    /// DFS through the action tree, creating `PostFlopNode` entries.
    fn build_tree_recursive(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let is_terminal;
        let is_chance;
        {
            let mut node = self.node_arena[node_index].lock();
            node.player = action_node.player;
            node.amount = action_node.amount;
            is_terminal = node.is_terminal();
            is_chance = node.is_chance();
        }

        if is_terminal {
            return;
        }

        if is_chance {
            self.push_chances(node_index, info);
            let (num_actions, children_offset) = {
                let node = self.node_arena[node_index].lock();
                (node.num_actions(), node.children_offset as usize)
            };
            for action_index in 0..num_actions {
                let child_index = node_index + children_offset + action_index;
                self.build_tree_recursive(
                    child_index,
                    &action_node.children[0].lock(),
                    info,
                );
            }
        } else {
            self.push_actions(node_index, action_node, info);
            let (num_actions, children_offset) = {
                let node = self.node_arena[node_index].lock();
                (node.num_actions(), node.children_offset as usize)
            };
            for action_index in 0..num_actions {
                let child_index = node_index + children_offset + action_index;
                self.build_tree_recursive(
                    child_index,
                    &action_node.children[action_index].lock(),
                    info,
                );
            }
        }
    }

    /// Creates chance-node children (turn or river deals).
    fn push_chances(&self, node_index: usize, info: &mut BuildTreeInfo) {
        let mut node = self.node_arena[node_index].lock();
        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        if node.turn == NOT_DEALT {
            // Deal turn
            let skip_cards = &self.isomorphism_card_turn;
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1u64 << card).sum();

            node.children_offset = (info.turn_index - node_index) as u32;
            for card in 0..52u8 {
                if (1u64 << card) & (flop_mask | skip_mask) == 0 {
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = card;
                }
            }
            info.turn_index += node.num_children as usize;
        } else {
            // Deal river
            let turn_mask = flop_mask | (1 << node.turn);
            let skip_cards = &self.isomorphism_card_river[node.turn as usize & 3];
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1u64 << card).sum();

            node.children_offset = (info.river_index - node_index) as u32;
            for card in 0..52u8 {
                if (1u64 << card) & (turn_mask | skip_mask) == 0 {
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = node.turn;
                    child.river = card;
                }
            }
            info.river_index += node.num_children as usize;
        }

        node.num_elements = node
            .cfvalue_storage_player()
            .map_or(0, |player| self.num_private_hands(player)) as u32;

        info.num_storage_chance += node.num_elements as u64;
    }

    /// Creates action-node children.
    fn push_actions(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let mut node = self.node_arena[node_index].lock();

        let street = match (node.turn, node.river) {
            (NOT_DEALT, _) => BoardState::Flop,
            (_, NOT_DEALT) => BoardState::Turn,
            _ => BoardState::River,
        };

        let base = match street {
            BoardState::Flop => &mut info.flop_index,
            BoardState::Turn => &mut info.turn_index,
            BoardState::River => &mut info.river_index,
        };

        node.children_offset = (*base - node_index) as u32;
        node.num_children = action_node.children.len() as u16;
        *base += node.num_children as usize;

        for (child, action) in node.children().iter().zip(action_node.actions.iter()) {
            let mut child = child.lock();
            child.prev_action = *action;
            child.turn = node.turn;
            child.river = node.river;
        }

        let num_private_hands = self.num_private_hands(node.player as usize);
        node.num_elements = (node.num_actions() * num_private_hands) as u32;
        node.num_elements_ip = match node.prev_action {
            Action::None | Action::Chance(_) => self.num_private_hands(PLAYER_IP as usize) as u16,
            _ => 0,
        };

        info.num_storage += node.num_elements as u64;
        info.num_storage_ip += node.num_elements_ip as u64;
    }
}

// ---------------------------------------------------------------------------
// Memory allocation
// ---------------------------------------------------------------------------

impl PostFlopGame {
    /// Clears all storage buffers.
    fn clear_storage(&mut self) {
        self.storage1 = Vec::new();
        self.storage2 = Vec::new();
        self.storage_ip = Vec::new();
        self.storage_chance = Vec::new();
    }

    /// Assigns raw pointers from the storage buffers to each node.
    fn allocate_memory_nodes(&mut self) {
        let num_bytes = if self.is_compression_enabled { 2 } else { 4 };
        let mut action_counter = 0usize;
        let mut ip_counter = 0usize;
        let mut chance_counter = 0usize;

        for node in &self.node_arena {
            let mut node = node.lock();
            if node.is_terminal() {
                // do nothing
            } else if node.is_chance() {
                // SAFETY: `storage_chance` is allocated with exactly
                // `num_bytes * num_storage_chance` bytes. `chance_counter`
                // is advanced by each chance node's `num_elements * num_bytes`,
                // which sums to `num_storage_chance * num_bytes`.
                unsafe {
                    let ptr = self.storage_chance.as_mut_ptr();
                    node.storage1 = ptr.add(chance_counter);
                }
                chance_counter += num_bytes * node.num_elements as usize;
            } else {
                // SAFETY: `storage1`, `storage2`, `storage_ip` are allocated
                // with exactly the required byte counts. `action_counter` and
                // `ip_counter` advance by each action node's element counts.
                unsafe {
                    let ptr1 = self.storage1.as_mut_ptr();
                    let ptr2 = self.storage2.as_mut_ptr();
                    let ptr3 = self.storage_ip.as_mut_ptr();
                    node.storage1 = ptr1.add(action_counter);
                    node.storage2 = ptr2.add(action_counter);
                    node.storage3 = ptr3.add(ip_counter);
                }
                action_counter += num_bytes * node.num_elements as usize;
                ip_counter += num_bytes * node.num_elements_ip as usize;
            }
        }
    }

    /// Initializes the interpreter state vectors.
    fn init_interpreter(&mut self) {
        let vecs = [
            vec![0.0; self.num_private_hands(0)],
            vec![0.0; self.num_private_hands(1)],
        ];

        self.weights = vecs.clone();
        self.normalized_weights = vecs.clone();
        self.cfvalues_cache = vecs;
    }

    /// Computes the misc memory usage of the struct itself (excluding storage).
    fn memory_usage_internal(&self) -> u64 {
        let mut usage = mem::size_of::<Self>() as u64;

        usage += vec_memory_usage(&self.added_lines);
        usage += vec_memory_usage(&self.removed_lines);
        for line in &self.added_lines {
            usage += vec_memory_usage(line);
        }
        for line in &self.removed_lines {
            usage += vec_memory_usage(line);
        }

        usage += vec_memory_usage(&self.valid_indices_turn);
        usage += vec_memory_usage(&self.valid_indices_river);
        usage += vec_memory_usage(&self.hand_strength);
        usage += vec_memory_usage(&self.isomorphism_ref_turn);
        usage += vec_memory_usage(&self.isomorphism_card_turn);
        usage += vec_memory_usage(&self.isomorphism_ref_river);

        for refs in &self.isomorphism_ref_river {
            usage += vec_memory_usage(refs);
        }
        for cards in &self.isomorphism_card_river {
            usage += vec_memory_usage(cards);
        }

        for player in 0..2 {
            usage += vec_memory_usage(&self.initial_weights[player]);
            usage += vec_memory_usage(&self.private_cards[player]);
            usage += vec_memory_usage(&self.same_hand_index[player]);
            usage += vec_memory_usage(&self.valid_indices_flop[player]);
            for indices in &self.valid_indices_turn {
                usage += vec_memory_usage(&indices[player]);
            }
            for indices in &self.valid_indices_river {
                usage += vec_memory_usage(&indices[player]);
            }
            for strength in &self.hand_strength {
                usage += vec_memory_usage(&strength[player]);
            }
            for swap in &self.isomorphism_swap_turn {
                usage += vec_memory_usage(&swap[player]);
            }
            for swap_list in &self.isomorphism_swap_river {
                for swap in swap_list {
                    usage += vec_memory_usage(&swap[player]);
                }
            }
        }

        usage += vec_memory_usage(&self.node_arena);

        usage
    }
}

/// Returns the heap allocation size of a `Vec` (capacity * element size).
#[inline]
fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * mem::size_of::<T>() as u64
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bet_size::*;
    use crate::card::*;

    fn simple_bet_sizes() -> [BetSizeOptions; 2] {
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        [sizes.clone(), sizes]
    }

    #[test]
    fn test_build_river_game() {
        let oop_range: crate::range::Range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range: crate::range::Range = "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();

        let (mem_uncompressed, mem_compressed) = game.memory_usage();
        assert!(mem_uncompressed > 0, "uncompressed memory should be > 0");
        assert!(mem_compressed > 0, "compressed memory should be > 0");
        assert!(
            mem_uncompressed >= mem_compressed,
            "uncompressed >= compressed"
        );

        game.allocate_memory(false);

        assert!(game.num_private_hands(0) > 0);
        assert!(game.num_private_hands(1) > 0);

        // Verify root node is accessible and not terminal
        let root = game.root();
        assert!(!root.is_terminal());
        assert!(!root.is_chance());
    }

    #[test]
    fn test_build_turn_game() {
        let oop_range: crate::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();

        let (mem, _) = game.memory_usage();
        assert!(mem > 0);

        game.allocate_memory(false);

        assert!(game.num_private_hands(0) > 0);
        assert!(game.num_private_hands(1) > 0);
    }

    #[test]
    fn test_build_flop_game() {
        let oop_range: crate::range::Range = "AA,KK".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 200,
            flop_bet_sizes: simple_bet_sizes(),
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();

        let (mem, _) = game.memory_usage();
        assert!(mem > 0);

        game.allocate_memory(false);

        // Flop game should have nodes across all three streets
        assert!(game.num_nodes[0] > 0 || game.num_nodes[1] > 0 || game.num_nodes[2] > 0);
        assert!(game.num_private_hands(0) > 0);
        assert!(game.num_private_hands(1) > 0);
    }

    #[test]
    fn test_build_flop_compressed() {
        let oop_range: crate::range::Range = "AA,KK".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 200,
            flop_bet_sizes: simple_bet_sizes(),
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();

        game.allocate_memory(true);
        assert!(game.is_compression_enabled);
        assert_eq!(game.state, State::MemoryAllocated);
    }

    #[test]
    fn test_config_validation_empty_range() {
        let oop_range: crate::range::Range = crate::range::Range::default();
        let ip_range: crate::range::Range = "QQ,JJ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let result = PostFlopGame::with_config(card_config, tree);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("OOP range is empty"), "got: {err}");
    }

    #[test]
    fn test_config_validation_duplicate_flop() {
        let oop_range: crate::range::Range = "AA".parse().unwrap();
        let ip_range: crate::range::Range = "KK".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: [0, 0, 5], // duplicate
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 100,
            flop_bet_sizes: simple_bet_sizes(),
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let result = PostFlopGame::with_config(card_config, tree);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_state_mismatch() {
        let oop_range: crate::range::Range = "AA".parse().unwrap();
        let ip_range: crate::range::Range = "KK".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River, // mismatch: no turn/river dealt
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let result = PostFlopGame::with_config(card_config, tree);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Invalid initial state"), "got: {err}");
    }

    #[test]
    fn test_same_hand_index() {
        let oop_range: crate::range::Range = "AA,KK".parse().unwrap();
        let ip_range: crate::range::Range = "AA,QQ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let game = PostFlopGame::with_config(card_config, tree).unwrap();

        // OOP has AA and KK combos; IP has AA and QQ combos.
        // AA combos should match; KK combos should not find a match in IP.
        for (i, &(c1, c2)) in game.private_cards[0].iter().enumerate() {
            let same_idx = game.same_hand_index[0][i];
            if let Ok(j) = game.private_cards[1].binary_search(&(c1, c2)) {
                assert_eq!(same_idx, j as u16);
            } else {
                assert_eq!(same_idx, u16::MAX);
            }
        }
    }

    #[test]
    fn test_memory_idempotent() {
        let oop_range: crate::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();

        game.allocate_memory(false);
        let s1_len = game.storage1.len();
        let s2_len = game.storage2.len();

        // Calling allocate_memory again with same params is a no-op
        game.allocate_memory(false);
        assert_eq!(game.storage1.len(), s1_len);
        assert_eq!(game.storage2.len(), s2_len);
    }

    #[test]
    fn test_node_index_roundtrip() {
        let oop_range: crate::range::Range = "AA,KK".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: simple_bet_sizes(),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let game = PostFlopGame::with_config(card_config, tree).unwrap();

        let root = game.root();
        assert_eq!(game.node_index(&root), 0);
    }

    #[test]
    fn test_build_turn_game_with_depth_limit() {
        let oop_range: crate::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            depth_limit: Some(0),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();

        // Should have boundary nodes
        assert!(game.num_boundary_nodes() > 0, "depth-limited tree should have boundary nodes");

        // Should be able to allocate memory
        game.allocate_memory(false);
        assert_eq!(game.state, State::MemoryAllocated);

        // Private hands should be populated
        assert!(game.num_private_hands(0) > 0);
        assert!(game.num_private_hands(1) > 0);
    }

    #[test]
    fn test_boundary_node_indices() {
        let oop_range: crate::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            depth_limit: Some(0),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let game = PostFlopGame::with_config(card_config, tree).unwrap();

        let indices = game.boundary_node_indices();
        assert_eq!(indices.len(), game.num_boundary_nodes());

        // Each boundary index should point to a depth boundary node
        for &idx in &indices {
            let node = game.node_arena[idx].lock();
            assert!(node.is_depth_boundary());
            assert!(node.is_terminal());
        }
    }

    #[test]
    fn test_set_boundary_cfvs() {
        let oop_range: crate::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            depth_limit: Some(0),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        let n_boundary = game.num_boundary_nodes();
        assert!(n_boundary > 0);

        // Set CFVs for all boundary nodes (both players)
        for ordinal in 0..n_boundary {
            let oop_cfvs = vec![0.0f32; game.num_private_hands(0)];
            let ip_cfvs = vec![0.0f32; game.num_private_hands(1)];
            game.set_boundary_cfvs(ordinal, 0, oop_cfvs);
            game.set_boundary_cfvs(ordinal, 1, ip_cfvs);
        }

        // Verify boundary_cfvs has the right number of entries
        assert_eq!(game.boundary_cfvs.len(), n_boundary * 2);
    }

    #[test]
    fn test_boundary_pot() {
        let oop_range: crate::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: crate::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: simple_bet_sizes(),
            river_bet_sizes: simple_bet_sizes(),
            depth_limit: Some(0),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let game = PostFlopGame::with_config(card_config, tree).unwrap();

        // All boundary pots should be >= starting pot
        for ordinal in 0..game.num_boundary_nodes() {
            let pot = game.boundary_pot(ordinal);
            assert!(pot >= 100, "boundary pot should be >= starting pot, got {pot}");
        }
    }
}
