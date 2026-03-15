use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use cudarc::nvrtc::compile_ptx;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Wrapper around a CUDA device context and stream.
///
/// Provides helpers for uploading/downloading data, allocating GPU memory,
/// compiling CUDA kernels at runtime, and launching solver-specific kernels
/// (regret matching, forward reach propagation, terminal evaluation,
/// backward CFV propagation, and DCFR+ regret updates).
///
/// Kernels are cached after first compilation so repeated launches are fast.
pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// Cache of compiled kernel functions keyed by function name.
    kernel_cache: RefCell<HashMap<String, CudaFunction>>,
}

impl GpuContext {
    /// Create a new GPU context on the given device ordinal (typically 0).
    pub fn new(device_ordinal: usize) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            kernel_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Copy host data to a new GPU buffer.
    pub fn upload<T: cudarc::driver::DeviceRepr + Copy>(
        &self,
        data: &[T],
    ) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.memcpy_stod(data)?)
    }

    /// Copy GPU buffer back to a host Vec.
    pub fn download<T: cudarc::driver::DeviceRepr + Copy + Default>(
        &self,
        buf: &CudaSlice<T>,
    ) -> Result<Vec<T>, GpuError> {
        Ok(self.stream.memcpy_dtov(buf)?)
    }

    /// Allocate a zero-initialized GPU buffer of `len` elements.
    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + Copy + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.alloc_zeros(len)?)
    }

    /// Clone a GPU buffer (device-to-device copy).
    pub fn clone_slice<T: cudarc::driver::DeviceRepr + Copy>(
        &self,
        src: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.clone_dtod(src)?)
    }

    /// Compile a CUDA source string to PTX, load it as a module, and return
    /// the named function. Results are cached so repeated calls with the same
    /// function name return the previously compiled kernel.
    pub fn compile_and_load(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<CudaFunction, GpuError> {
        let cache = self.kernel_cache.borrow();
        if let Some(func) = cache.get(function_name) {
            return Ok(func.clone());
        }
        drop(cache);

        let ptx = compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        let func = module.load_function(function_name)?;
        self.kernel_cache
            .borrow_mut()
            .insert(function_name.to_string(), func.clone());
        Ok(func)
    }

    /// Launch the regret-matching kernel with per-hand layout.
    ///
    /// For each (infoset, hand) pair, converts cumulative regrets into a
    /// current strategy via regret matching (positive-regret normalization,
    /// with uniform fallback when all regrets are non-positive).
    ///
    /// # Layout
    /// - `regrets`: `[num_infosets * max_actions * num_hands]` —
    ///   indexed as `(infoset * max_actions + action) * num_hands + hand`
    /// - `num_actions`: `[num_infosets]` number of valid actions per infoset
    /// - `strategy`: `[num_infosets * max_actions * num_hands]` output, same layout
    pub fn launch_regret_match(
        &self,
        regrets: &CudaSlice<f32>,
        num_actions: &CudaSlice<u32>,
        strategy: &mut CudaSlice<f32>,
        num_infosets: u32,
        max_actions: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/regret_match.cu"),
            "regret_match",
        )?;
        let total_threads = num_infosets * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(regrets)
                .arg(num_actions)
                .arg(strategy)
                .arg(&num_infosets)
                .arg(&max_actions)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the forward-reach propagation kernel with per-hand strategy.
    ///
    /// For each node in the current BFS level, computes:
    ///   `reach[node][hand] = reach[parent][hand] * strategy[parent_infoset][action][hand]`
    ///
    /// This is parallelized over `num_nodes_this_level * num_hands` threads.
    ///
    /// # Layout
    /// - `reach_probs`: `[num_total_nodes * num_hands]` flat array
    /// - `strategy`: `[num_infosets * max_actions * num_hands]` —
    ///   indexed as `(infoset * max_actions + action) * num_hands + hand`
    /// - `level_nodes`: `[num_nodes_this_level]` global node ids for this level
    /// - `parent_nodes`: `[num_nodes_this_level]` parent global node id
    /// - `parent_actions`: `[num_nodes_this_level]` action index from parent
    /// - `parent_infosets`: `[num_nodes_this_level]` infoset id of parent
    pub fn launch_forward_reach(
        &self,
        reach_probs: &mut CudaSlice<f32>,
        strategy: &CudaSlice<f32>,
        level_nodes: &CudaSlice<u32>,
        parent_nodes: &CudaSlice<u32>,
        parent_actions: &CudaSlice<u32>,
        parent_infosets: &CudaSlice<u32>,
        num_nodes_this_level: u32,
        num_hands: u32,
        max_actions: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/forward_reach.cu"),
            "forward_reach",
        )?;
        let total_threads = num_nodes_this_level * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(reach_probs)
                .arg(strategy)
                .arg(level_nodes)
                .arg(parent_nodes)
                .arg(parent_actions)
                .arg(parent_infosets)
                .arg(&num_nodes_this_level)
                .arg(&num_hands)
                .arg(&max_actions)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the fold terminal evaluation kernel.
    ///
    /// Computes counterfactual values at fold terminal nodes. For each
    /// (fold_terminal, hand) pair:
    ///   - If the traverser folded: `cfv[hand] = amount_lose * sum(valid_opp_reach)`
    ///   - If the opponent folded: `cfv[hand] = amount_win * sum(valid_opp_reach)`
    ///
    /// Card blocking: only sums opponent reach for hands that don't share cards
    /// with the traverser's hand.
    ///
    /// # Layout
    /// - `cfvalues`: `[num_nodes * num_hands]` — output (only fold terminals written)
    /// - `opp_reach`: `[num_nodes * num_hands]` — opponent's reach at each node
    /// - `terminal_nodes`: `[num_fold_terminals]` — node indices of fold terminals
    /// - `fold_amount_win`: `[num_fold_terminals]` — positive payoff
    /// - `fold_amount_lose`: `[num_fold_terminals]` — negative payoff
    /// - `fold_player`: `[num_fold_terminals]` — which player folded (0 or 1)
    /// - `valid_matchups`: `[num_hands * num_hands]` — 1.0 if hands don't share cards
    #[allow(clippy::too_many_arguments)]
    pub fn launch_terminal_fold_eval(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        fold_amount_win: &CudaSlice<f32>,
        fold_amount_lose: &CudaSlice<f32>,
        fold_player: &CudaSlice<u32>,
        valid_matchups: &CudaSlice<f32>,
        traverser: u32,
        num_fold_terminals: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/terminal_fold_eval.cu"),
            "terminal_fold_eval",
        )?;
        let total_threads = num_fold_terminals * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(fold_amount_win)
                .arg(fold_amount_lose)
                .arg(fold_player)
                .arg(valid_matchups)
                .arg(&traverser)
                .arg(&num_fold_terminals)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the fold aggregates precomputation kernel.
    ///
    /// For each fold terminal, computes:
    ///   - `total_opp_reach[term]` — sum of all opponent reach
    ///   - `per_card_reach[term * 52 + c]` — sum of opponent reach for hands containing card c
    ///
    /// These aggregates enable O(1) per-hand fold CFV via inclusion-exclusion.
    ///
    /// # Launch config
    /// One block per fold terminal. Threads within the block cooperate via
    /// shared memory atomics to accumulate across all opponent hands.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_precompute_fold_aggregates(
        &self,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        opp_hand_cards: &CudaSlice<u32>,
        total_opp_reach: &mut CudaSlice<f32>,
        per_card_reach: &mut CudaSlice<f32>,
        num_fold_terminals: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/precompute_fold_aggregates.cu"),
            "precompute_fold_aggregates",
        )?;
        // One block per terminal, 256 threads per block
        let block_dim = 256u32.min(num_hands);
        let cfg = LaunchConfig {
            grid_dim: (num_fold_terminals, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0, // shared memory is statically allocated in the kernel
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(opp_hand_cards)
                .arg(total_opp_reach)
                .arg(per_card_reach)
                .arg(&num_fold_terminals)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the O(1) fold evaluation kernel using precomputed aggregates.
    ///
    /// For each (fold_terminal, hand) pair, computes:
    ///   `cfv[h] = payoff * (total_opp_reach - per_card[c1] - per_card[c2] + same_hand_reach)`
    ///
    /// Requires `launch_precompute_fold_aggregates` to have been called first.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fold_eval_from_aggregates(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        fold_amount_win: &CudaSlice<f32>,
        fold_amount_lose: &CudaSlice<f32>,
        fold_player: &CudaSlice<u32>,
        total_opp_reach: &CudaSlice<f32>,
        per_card_reach: &CudaSlice<f32>,
        trav_hand_cards: &CudaSlice<u32>,
        same_hand_index: &CudaSlice<u32>,
        traverser: u32,
        num_fold_terminals: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/fold_eval_from_aggregates.cu"),
            "fold_eval_from_aggregates",
        )?;
        let total_threads = num_fold_terminals * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(fold_amount_win)
                .arg(fold_amount_lose)
                .arg(fold_player)
                .arg(total_opp_reach)
                .arg(per_card_reach)
                .arg(trav_hand_cards)
                .arg(same_hand_index)
                .arg(&traverser)
                .arg(&num_fold_terminals)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the shared-memory showdown evaluation kernel.
    ///
    /// One block per showdown terminal. Opponent reach is loaded into shared
    /// memory once, then each thread computes its traverser hand's CFV by
    /// iterating over opponent hands in fast shared memory.
    ///
    /// Card blocking uses explicit card comparison (no valid_matchups matrix).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_terminal_showdown_eval_shm(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        amount_win: &CudaSlice<f32>,
        amount_lose: &CudaSlice<f32>,
        traverser_strengths: &CudaSlice<u32>,
        opponent_strengths: &CudaSlice<u32>,
        trav_hand_cards: &CudaSlice<u32>,
        opp_hand_cards: &CudaSlice<u32>,
        num_showdown_terminals: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/terminal_showdown_eval_shm.cu"),
            "terminal_showdown_eval_shm",
        )?;
        // Block size must be >= num_hands; round up to next multiple of 32 (warp size)
        let block_size = ((num_hands + 31) / 32) * 32;
        // Cap at 1024 (max block size). For very large hand counts this would
        // need a different approach, but typical poker hands are <= 1081.
        let block_size = block_size.min(1024);
        let shared_mem = num_hands * 4; // sizeof(float) = 4
        let cfg = LaunchConfig {
            grid_dim: (num_showdown_terminals, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(amount_win)
                .arg(amount_lose)
                .arg(traverser_strengths)
                .arg(opponent_strengths)
                .arg(trav_hand_cards)
                .arg(opp_hand_cards)
                .arg(&num_showdown_terminals)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the showdown terminal evaluation kernel.
    ///
    /// Computes counterfactual values at showdown terminal nodes. For each
    /// (showdown_terminal, hand) pair, sums over opponent hands:
    ///   - `cfv[h] += amount_win * opp_reach[h']` if hand h beats h' and not blocked
    ///   - `cfv[h] += amount_lose * opp_reach[h']` if hand h loses to h' and not blocked
    ///   - ties contribute nothing
    ///
    /// # Layout
    /// - `cfvalues`: `[num_nodes * num_hands]` — output (only showdown terminals written)
    /// - `opp_reach`: `[num_nodes * num_hands]` — opponent's reach at each node
    /// - `terminal_nodes`: `[num_showdown_terminals]` — node indices
    /// - `amount_win`: `[num_showdown_terminals]` — positive payoff per combo
    /// - `amount_lose`: `[num_showdown_terminals]` — negative payoff per combo
    /// - `traverser_strengths`: `[num_hands]` — strength ranking for traverser's hands
    /// - `opponent_strengths`: `[num_hands]` — strength ranking for opponent's hands
    /// - `valid_matchups`: `[num_hands * num_hands]` — 1.0 if hands don't share cards
    #[allow(clippy::too_many_arguments)]
    pub fn launch_terminal_showdown_eval(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        amount_win: &CudaSlice<f32>,
        amount_lose: &CudaSlice<f32>,
        traverser_strengths: &CudaSlice<u32>,
        opponent_strengths: &CudaSlice<u32>,
        valid_matchups: &CudaSlice<f32>,
        num_showdown_terminals: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/terminal_showdown_eval.cu"),
            "terminal_showdown_eval",
        )?;
        let total_threads = num_showdown_terminals * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(amount_win)
                .arg(amount_lose)
                .arg(traverser_strengths)
                .arg(opponent_strengths)
                .arg(valid_matchups)
                .arg(&num_showdown_terminals)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the backward CFV propagation kernel.
    ///
    /// For each decision node at the given level, computes:
    ///   - Traverser's nodes: `cfv[node][hand] = sum_a(strategy[a][hand] * cfv[child_a][hand])`
    ///   - Opponent's nodes: `cfv[node][hand] = sum_a(cfv[child_a][hand])`
    ///
    /// Must be called level-by-level from leaves to root (bottom-up).
    ///
    /// # Layout
    /// - `cfvalues`: `[num_nodes * num_hands]` — in/out
    /// - `strategy`: `[num_infosets * max_actions * num_hands]` — per-hand strategy
    /// - `level_nodes`: `[num_nodes_this_level]` — decision nodes at this level
    /// - `child_offsets`: `[num_total_nodes + 1]` — CSR child offsets
    /// - `children_arr`: CSR child node IDs
    /// - `infoset_ids`: `[num_total_nodes]` — infoset id per node (u32::MAX for terminals)
    /// - `node_players`: `[num_nodes_this_level]` — player who acts at each node (0=OOP, 1=IP)
    #[allow(clippy::too_many_arguments)]
    pub fn launch_backward_cfv(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        strategy: &CudaSlice<f32>,
        level_nodes: &CudaSlice<u32>,
        child_offsets: &CudaSlice<u32>,
        children_arr: &CudaSlice<u32>,
        infoset_ids: &CudaSlice<u32>,
        node_players: &CudaSlice<u32>,
        traverser: u32,
        num_nodes_this_level: u32,
        num_hands: u32,
        max_actions: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/backward_cfv.cu"),
            "backward_cfv",
        )?;
        let total_threads = num_nodes_this_level * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(strategy)
                .arg(level_nodes)
                .arg(child_offsets)
                .arg(children_arr)
                .arg(infoset_ids)
                .arg(node_players)
                .arg(&traverser)
                .arg(&num_nodes_this_level)
                .arg(&num_hands)
                .arg(&max_actions)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the combined forward-pass kernel that propagates both players'
    /// reach probabilities in a single pass.
    ///
    /// For each node in the current BFS level:
    /// - The acting player's reach is multiplied by the action probability
    /// - The non-acting player's reach is copied from parent
    ///
    /// # Layout
    /// - `reach_oop`, `reach_ip`: `[num_total_nodes * num_hands]`
    /// - `strategy`: `[num_infosets * max_actions * num_hands]`
    /// - `parent_players`: `[num_nodes_this_level]` — 0=OOP, 1=IP
    #[allow(clippy::too_many_arguments)]
    pub fn launch_forward_pass(
        &self,
        reach_oop: &mut CudaSlice<f32>,
        reach_ip: &mut CudaSlice<f32>,
        strategy: &CudaSlice<f32>,
        level_nodes: &CudaSlice<u32>,
        parent_nodes: &CudaSlice<u32>,
        parent_actions: &CudaSlice<u32>,
        parent_infosets: &CudaSlice<u32>,
        parent_players: &CudaSlice<u32>,
        num_nodes_this_level: u32,
        num_hands: u32,
        max_actions: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/forward_pass.cu"),
            "forward_pass",
        )?;
        let total_threads = num_nodes_this_level * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(reach_oop)
                .arg(reach_ip)
                .arg(strategy)
                .arg(level_nodes)
                .arg(parent_nodes)
                .arg(parent_actions)
                .arg(parent_infosets)
                .arg(parent_players)
                .arg(&num_nodes_this_level)
                .arg(&num_hands)
                .arg(&max_actions)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the extract-strategy kernel.
    ///
    /// Normalizes cumulative strategy sums into final action probabilities
    /// for each (infoset, hand) pair. Falls back to uniform when all
    /// strategy sums are zero.
    ///
    /// # Layout
    /// - `strategy_sum`: `[num_infosets * max_actions * num_hands]` — input
    /// - `num_actions`: `[num_infosets]` — valid action count per infoset
    /// - `output_strategy`: `[num_infosets * max_actions * num_hands]` — output
    pub fn launch_extract_strategy(
        &self,
        strategy_sum: &CudaSlice<f32>,
        num_actions: &CudaSlice<u32>,
        output_strategy: &mut CudaSlice<f32>,
        num_infosets: u32,
        max_actions: u32,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/extract_strategy.cu"),
            "extract_strategy",
        )?;
        let total_threads = num_infosets * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(strategy_sum)
                .arg(num_actions)
                .arg(output_strategy)
                .arg(&num_infosets)
                .arg(&max_actions)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the DCFR+ regret update kernel.
    ///
    /// For each (decision_node, action, hand):
    ///   1. Computes instantaneous regret: `cfv[child_a][hand] - cfv[node][hand]`
    ///   2. Updates cumulative regret with DCFR discounting
    ///   3. Updates strategy sum: `sum = sum * strat_discount + strategy`
    ///
    /// # Layout
    /// - `regrets`: `[num_infosets * max_actions * num_hands]` — in/out
    /// - `strategy_sum`: `[num_infosets * max_actions * num_hands]` — in/out
    /// - `strategy`: `[num_infosets * max_actions * num_hands]` — current strategy
    /// - `cfvalues`: `[num_nodes * num_hands]` — computed CFVs
    /// - `decision_nodes`: `[num_decision_nodes]` — node indices
    #[allow(clippy::too_many_arguments)]
    pub fn launch_update_regrets(
        &self,
        regrets: &mut CudaSlice<f32>,
        strategy_sum: &mut CudaSlice<f32>,
        strategy: &CudaSlice<f32>,
        cfvalues: &CudaSlice<f32>,
        decision_nodes: &CudaSlice<u32>,
        child_offsets: &CudaSlice<u32>,
        children_arr: &CudaSlice<u32>,
        infoset_ids: &CudaSlice<u32>,
        num_actions_arr: &CudaSlice<u32>,
        num_decision_nodes: u32,
        num_hands: u32,
        max_actions: u32,
        pos_discount: f32,
        neg_discount: f32,
        strat_discount: f32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/update_regrets.cu"),
            "update_regrets",
        )?;
        let total_threads = num_decision_nodes * max_actions * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(regrets)
                .arg(strategy_sum)
                .arg(strategy)
                .arg(cfvalues)
                .arg(decision_nodes)
                .arg(child_offsets)
                .arg(children_arr)
                .arg(infoset_ids)
                .arg(num_actions_arr)
                .arg(&num_decision_nodes)
                .arg(&num_hands)
                .arg(&max_actions)
                .arg(&pos_discount)
                .arg(&neg_discount)
                .arg(&strat_discount)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the batch fold terminal evaluation kernel.
    ///
    /// Like `launch_terminal_fold_eval` but with:
    /// - Per-hand payoffs: `fold_amount_win[term_idx * num_hands + hand]`
    /// - Spot-scoped card blocking via `hands_per_spot`
    ///
    /// # Layout
    /// - `fold_amount_win`: `[num_fold_terminals * num_hands]` — per-hand payoff
    /// - `fold_amount_lose`: `[num_fold_terminals * num_hands]` — per-hand payoff
    /// - `valid_matchups`: `[num_spots * hps * hps]` — per-spot blocking matrices
    #[allow(clippy::too_many_arguments)]
    pub fn launch_terminal_fold_eval_batch(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        fold_amount_win: &CudaSlice<f32>,
        fold_amount_lose: &CudaSlice<f32>,
        fold_player: &CudaSlice<u32>,
        valid_matchups: &CudaSlice<f32>,
        traverser: u32,
        num_fold_terminals: u32,
        num_hands: u32,
        hands_per_spot: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/terminal_fold_eval_batch.cu"),
            "terminal_fold_eval_batch",
        )?;
        let total_threads = num_fold_terminals * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(fold_amount_win)
                .arg(fold_amount_lose)
                .arg(fold_player)
                .arg(valid_matchups)
                .arg(&traverser)
                .arg(&num_fold_terminals)
                .arg(&num_hands)
                .arg(&hands_per_spot)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Zero a GPU float buffer in-place (no host allocation).
    pub fn launch_zero_buffer(
        &self,
        buf: &mut CudaSlice<f32>,
        len: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/zero_buffer.cu"),
            "zero_buffer",
        )?;
        let cfg = LaunchConfig::for_num_elems(len);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(buf)
                .arg(&len)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Copy initial reach values into the root node (node 0) of a reach array.
    pub fn launch_set_root_reach(
        &self,
        reach: &mut CudaSlice<f32>,
        initial: &CudaSlice<f32>,
        num_hands: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/set_root_reach.cu"),
            "set_root_reach",
        )?;
        let cfg = LaunchConfig::for_num_elems(num_hands);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(reach)
                .arg(initial)
                .arg(&num_hands)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the batch showdown terminal evaluation kernel.
    ///
    /// Like `launch_terminal_showdown_eval` but with:
    /// - Per-hand payoffs: `amount_win[term_idx * num_hands + hand]`
    /// - Spot-scoped card blocking via `hands_per_spot`
    ///
    /// # Layout
    /// - `amount_win`: `[num_showdown_terminals * num_hands]` — per-hand payoff
    /// - `amount_lose`: `[num_showdown_terminals * num_hands]` — per-hand payoff
    /// - `valid_matchups`: `[num_spots * hps * hps]` — per-spot blocking matrices
    #[allow(clippy::too_many_arguments)]
    pub fn launch_terminal_showdown_eval_batch(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        amount_win: &CudaSlice<f32>,
        amount_lose: &CudaSlice<f32>,
        traverser_strengths: &CudaSlice<u32>,
        opponent_strengths: &CudaSlice<u32>,
        valid_matchups: &CudaSlice<f32>,
        num_showdown_terminals: u32,
        num_hands: u32,
        hands_per_spot: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/terminal_showdown_eval_batch.cu"),
            "terminal_showdown_eval_batch",
        )?;
        let total_threads = num_showdown_terminals * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(amount_win)
                .arg(amount_lose)
                .arg(traverser_strengths)
                .arg(opponent_strengths)
                .arg(valid_matchups)
                .arg(&num_showdown_terminals)
                .arg(&num_hands)
                .arg(&hands_per_spot)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the batch fold aggregates precomputation kernel.
    ///
    /// One block per (fold_terminal, spot) pair. Each block only iterates over
    /// its spot's hands, keeping shared memory atomics contention low.
    ///
    /// Outputs:
    ///   `total_opp_reach[term_idx * num_spots + spot]`
    ///   `per_card_reach[(term_idx * num_spots + spot) * 52 + card]`
    #[allow(clippy::too_many_arguments)]
    pub fn launch_precompute_fold_aggregates_batch(
        &self,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        opp_hand_cards: &CudaSlice<u32>,
        total_opp_reach: &mut CudaSlice<f32>,
        per_card_reach: &mut CudaSlice<f32>,
        num_fold_terminals: u32,
        num_hands: u32,
        hands_per_spot: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/precompute_fold_aggregates_batch.cu"),
            "precompute_fold_aggregates_batch",
        )?;
        let num_spots = num_hands / hands_per_spot;
        let num_blocks = num_fold_terminals * num_spots;
        let block_dim = 256u32.min(hands_per_spot);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0, // shared memory is statically allocated in the kernel
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(opp_hand_cards)
                .arg(total_opp_reach)
                .arg(per_card_reach)
                .arg(&num_fold_terminals)
                .arg(&num_hands)
                .arg(&hands_per_spot)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the batch O(1) fold evaluation kernel using precomputed aggregates.
    ///
    /// Like `launch_fold_eval_from_aggregates` but with per-hand payoffs and
    /// aggregates indexed by `(terminal, spot)`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fold_eval_from_aggregates_batch(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        fold_amount_win: &CudaSlice<f32>,
        fold_amount_lose: &CudaSlice<f32>,
        fold_player: &CudaSlice<u32>,
        total_opp_reach: &CudaSlice<f32>,
        per_card_reach: &CudaSlice<f32>,
        trav_hand_cards: &CudaSlice<u32>,
        same_hand_index: &CudaSlice<u32>,
        traverser: u32,
        num_fold_terminals: u32,
        num_hands: u32,
        hands_per_spot: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/fold_eval_from_aggregates_batch.cu"),
            "fold_eval_from_aggregates_batch",
        )?;
        let total_threads = num_fold_terminals * num_hands;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(fold_amount_win)
                .arg(fold_amount_lose)
                .arg(fold_player)
                .arg(total_opp_reach)
                .arg(per_card_reach)
                .arg(trav_hand_cards)
                .arg(same_hand_index)
                .arg(&traverser)
                .arg(&num_fold_terminals)
                .arg(&num_hands)
                .arg(&hands_per_spot)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the batch shared-memory showdown evaluation kernel.
    ///
    /// One block per (showdown_terminal, spot) pair. Each block loads one
    /// spot's opponent reach into shared memory, then each thread computes
    /// its traverser hand's CFV by iterating over opponent hands within
    /// the same spot only.
    ///
    /// Uses per-hand payoffs and explicit card comparison for blocking.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_terminal_showdown_eval_shm_batch(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        amount_win: &CudaSlice<f32>,
        amount_lose: &CudaSlice<f32>,
        traverser_strengths: &CudaSlice<u32>,
        opponent_strengths: &CudaSlice<u32>,
        trav_hand_cards: &CudaSlice<u32>,
        opp_hand_cards: &CudaSlice<u32>,
        num_showdown_terminals: u32,
        num_hands: u32,
        hands_per_spot: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/terminal_showdown_eval_shm_batch.cu"),
            "terminal_showdown_eval_shm_batch",
        )?;
        let num_spots = num_hands / hands_per_spot;
        let num_blocks = num_showdown_terminals * num_spots;
        // Block size: min(hands_per_spot, 1024), rounded up to warp size
        let block_size = ((hands_per_spot + 31) / 32) * 32;
        let block_size = block_size.min(1024);
        let shared_mem = hands_per_spot * 4; // sizeof(float) = 4
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(amount_win)
                .arg(amount_lose)
                .arg(traverser_strengths)
                .arg(opponent_strengths)
                .arg(trav_hand_cards)
                .arg(opp_hand_cards)
                .arg(&num_showdown_terminals)
                .arg(&num_hands)
                .arg(&hands_per_spot)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the O(n) sorted-prefix-sum showdown evaluation kernel (batch).
    ///
    /// Uses two linear scans (ascending + descending) per (terminal, spot)
    /// instead of the O(n^2) pairwise comparison. One thread per (terminal, spot).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_showdown_eval_sorted_batch(
        &self,
        cfvalues: &mut CudaSlice<f32>,
        opp_reach: &CudaSlice<f32>,
        terminal_nodes: &CudaSlice<u32>,
        amount_win: &CudaSlice<f32>,
        amount_lose: &CudaSlice<f32>,
        trav_sorted: &CudaSlice<u32>,
        opp_sorted: &CudaSlice<u32>,
        trav_strengths: &CudaSlice<u32>,
        opp_strengths: &CudaSlice<u32>,
        trav_hand_cards: &CudaSlice<u32>,
        opp_hand_cards: &CudaSlice<u32>,
        num_showdown_terminals: u32,
        num_hands: u32,
        hands_per_spot: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/showdown_eval_sorted_batch.cu"),
            "showdown_eval_sorted_batch",
        )?;
        let num_spots = num_hands / hands_per_spot;
        let total_threads = num_showdown_terminals * num_spots;
        let block_size = 256u32;
        let num_blocks = (total_threads + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(cfvalues)
                .arg(opp_reach)
                .arg(terminal_nodes)
                .arg(amount_win)
                .arg(amount_lose)
                .arg(trav_sorted)
                .arg(opp_sorted)
                .arg(trav_strengths)
                .arg(opp_strengths)
                .arg(trav_hand_cards)
                .arg(opp_hand_cards)
                .arg(&num_showdown_terminals)
                .arg(&num_hands)
                .arg(&hands_per_spot)
                .launch(cfg)?;
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("NVRTC compilation error: {0}")]
    Compile(#[from] cudarc::nvrtc::CompileError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_round_trip() {
        let gpu = GpuContext::new(0).expect("CUDA device required");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu_buf = gpu.upload(&data).unwrap();
        let result = gpu.download(&gpu_buf).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_regret_match_kernel() {
        let gpu = GpuContext::new(0).unwrap();

        let num_infosets: u32 = 2;
        let max_actions: u32 = 3;
        let num_hands: u32 = 2;

        // Per-hand layout: (infoset * max_actions + action) * num_hands + hand
        // Infoset 0, hand 0: regrets [10, -5, 20] -> strategy [10/30, 0, 20/30]
        // Infoset 0, hand 1: regrets [0, 6, 4]   -> strategy [0, 0.6, 0.4]
        // Infoset 1, hand 0: regrets [-1, -2, -3] -> uniform [1/3, 1/3, 1/3]
        // Infoset 1, hand 1: regrets [3, 0, -1]   -> strategy [1, 0, 0]
        //
        // Layout: for iset=0, action=0: hand0=10, hand1=0
        //         for iset=0, action=1: hand0=-5, hand1=6
        //         for iset=0, action=2: hand0=20, hand1=4
        //         for iset=1, action=0: hand0=-1, hand1=3
        //         for iset=1, action=1: hand0=-2, hand1=0
        //         for iset=1, action=2: hand0=-3, hand1=-1
        let regrets: Vec<f32> = vec![
            10.0, 0.0,   // iset0, act0
            -5.0, 6.0,   // iset0, act1
            20.0, 4.0,   // iset0, act2
            -1.0, 3.0,   // iset1, act0
            -2.0, 0.0,   // iset1, act1
            -3.0, -1.0,  // iset1, act2
        ];
        let num_actions: Vec<u32> = vec![3, 3];

        let gpu_regrets = gpu.upload(&regrets).unwrap();
        let gpu_num_actions = gpu.upload(&num_actions).unwrap();
        let mut gpu_strategy = gpu.alloc_zeros::<f32>(12).unwrap();

        gpu.launch_regret_match(
            &gpu_regrets,
            &gpu_num_actions,
            &mut gpu_strategy,
            num_infosets,
            max_actions,
            num_hands,
        )
        .unwrap();

        let strategy = gpu.download(&gpu_strategy).unwrap();

        let eps = 1e-5;
        // Infoset 0, hand 0: [10/30, 0, 20/30] = [0.333, 0, 0.667]
        assert!((strategy[0] - 1.0 / 3.0).abs() < eps, "i0h0a0: {}", strategy[0]);
        assert!((strategy[2] - 0.0).abs() < eps, "i0h0a1: {}", strategy[2]);
        assert!((strategy[4] - 2.0 / 3.0).abs() < eps, "i0h0a2: {}", strategy[4]);
        // Infoset 0, hand 1: [0, 6/10, 4/10] = [0, 0.6, 0.4]
        assert!((strategy[1] - 0.0).abs() < eps, "i0h1a0: {}", strategy[1]);
        assert!((strategy[3] - 0.6).abs() < eps, "i0h1a1: {}", strategy[3]);
        assert!((strategy[5] - 0.4).abs() < eps, "i0h1a2: {}", strategy[5]);
        // Infoset 1, hand 0: all negative -> uniform [1/3, 1/3, 1/3]
        assert!((strategy[6] - 1.0 / 3.0).abs() < eps, "i1h0a0: {}", strategy[6]);
        assert!((strategy[8] - 1.0 / 3.0).abs() < eps, "i1h0a1: {}", strategy[8]);
        assert!((strategy[10] - 1.0 / 3.0).abs() < eps, "i1h0a2: {}", strategy[10]);
        // Infoset 1, hand 1: [3, 0, 0] -> [1, 0, 0]
        assert!((strategy[7] - 1.0).abs() < eps, "i1h1a0: {}", strategy[7]);
        assert!((strategy[9] - 0.0).abs() < eps, "i1h1a1: {}", strategy[9]);
        assert!((strategy[11] - 0.0).abs() < eps, "i1h1a2: {}", strategy[11]);
    }

    #[test]
    fn test_forward_reach_kernel() {
        let gpu = GpuContext::new(0).unwrap();

        // Tree: Root(0) -> [Child(1), Child(2)]
        // Root is infoset 0 with 2 actions
        // Per-hand strategy: hand0: [0.6, 0.4], hand1: [0.3, 0.7]
        // 2 hands
        // Initial reach: hand0 = 1.0, hand1 = 0.5
        let num_hands: u32 = 2;
        let num_nodes_this_level: u32 = 2;

        let reach = vec![
            1.0f32, 0.5, // node 0: hand 0 = 1.0, hand 1 = 0.5
            0.0, 0.0,    // node 1: to be filled
            0.0, 0.0,    // node 2: to be filled
        ];
        // Per-hand strategy layout: (iset * max_actions + action) * num_hands + hand
        // iset=0, action=0: hand0=0.6, hand1=0.3
        // iset=0, action=1: hand0=0.4, hand1=0.7
        let strategy = vec![0.6f32, 0.3, 0.4, 0.7];

        let level_nodes = vec![1u32, 2];
        let parent_nodes = vec![0u32, 0];
        let parent_actions = vec![0u32, 1];
        let parent_infosets = vec![0u32, 0];

        let mut gpu_reach = gpu.upload(&reach).unwrap();
        let gpu_strategy = gpu.upload(&strategy).unwrap();
        let gpu_level_nodes = gpu.upload(&level_nodes).unwrap();
        let gpu_parent_nodes = gpu.upload(&parent_nodes).unwrap();
        let gpu_parent_actions = gpu.upload(&parent_actions).unwrap();
        let gpu_parent_infosets = gpu.upload(&parent_infosets).unwrap();

        gpu.launch_forward_reach(
            &mut gpu_reach,
            &gpu_strategy,
            &gpu_level_nodes,
            &gpu_parent_nodes,
            &gpu_parent_actions,
            &gpu_parent_infosets,
            num_nodes_this_level,
            num_hands,
            2, // max_actions
        )
        .unwrap();

        let result = gpu.download(&gpu_reach).unwrap();

        let eps = 1e-5;
        // Child 1 (action 0): reach = parent_reach * strategy[action0]
        //   hand0: 1.0 * 0.6 = 0.6
        //   hand1: 0.5 * 0.3 = 0.15
        assert!(
            (result[2] - 0.6).abs() < eps,
            "child1 hand0: got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.15).abs() < eps,
            "child1 hand1: got {}",
            result[3]
        );
        // Child 2 (action 1): reach = parent_reach * strategy[action1]
        //   hand0: 1.0 * 0.4 = 0.4
        //   hand1: 0.5 * 0.7 = 0.35
        assert!(
            (result[4] - 0.4).abs() < eps,
            "child2 hand0: got {}",
            result[4]
        );
        assert!(
            (result[5] - 0.35).abs() < eps,
            "child2 hand1: got {}",
            result[5]
        );
    }

    #[test]
    fn test_terminal_fold_eval() {
        let gpu = GpuContext::new(0).unwrap();

        // 3 nodes: root(0), fold_terminal(1), other(2)
        // 2 hands
        // Opponent folded (fold_player=1, traverser=0 -> traverser wins)
        let num_hands: u32 = 2;
        let num_nodes: u32 = 3;
        let num_fold_terminals: u32 = 1;

        // Opponent reach at fold terminal (node 1)
        let mut opp_reach = vec![0.0f32; (num_nodes * num_hands) as usize];
        opp_reach[2] = 0.8; // node1, hand0
        opp_reach[3] = 0.2; // node1, hand1

        let terminal_nodes = vec![1u32];
        let fold_amount_win = vec![5.0f32]; // win 5 per combo
        let fold_amount_lose = vec![-5.0f32]; // lose 5 per combo
        let fold_player = vec![1u32]; // IP (opponent) folded
        // All matchups valid (no card conflicts in this test)
        let valid_matchups = vec![1.0f32; (num_hands * num_hands) as usize];

        let mut gpu_cfvalues = gpu.alloc_zeros::<f32>((num_nodes * num_hands) as usize).unwrap();
        let gpu_opp_reach = gpu.upload(&opp_reach).unwrap();
        let gpu_terminals = gpu.upload(&terminal_nodes).unwrap();
        let gpu_win = gpu.upload(&fold_amount_win).unwrap();
        let gpu_lose = gpu.upload(&fold_amount_lose).unwrap();
        let gpu_fold_player = gpu.upload(&fold_player).unwrap();
        let gpu_valid = gpu.upload(&valid_matchups).unwrap();

        // Traverser is 0 (OOP). Opponent (IP) folded.
        gpu.launch_terminal_fold_eval(
            &mut gpu_cfvalues,
            &gpu_opp_reach,
            &gpu_terminals,
            &gpu_win,
            &gpu_lose,
            &gpu_fold_player,
            &gpu_valid,
            0, // traverser
            num_fold_terminals,
            num_hands,
        )
        .unwrap();

        let result = gpu.download(&gpu_cfvalues).unwrap();

        let eps = 1e-5;
        let total_opp_reach = 0.8 + 0.2; // = 1.0
        // Traverser wins since opponent folded: cfv = amount_win * total_opp_reach
        assert!(
            (result[2] - 5.0 * total_opp_reach).abs() < eps,
            "fold hand0: got {}",
            result[2]
        );
        assert!(
            (result[3] - 5.0 * total_opp_reach).abs() < eps,
            "fold hand1: got {}",
            result[3]
        );
    }

    #[test]
    fn test_terminal_fold_eval_traverser_folds() {
        let gpu = GpuContext::new(0).unwrap();

        let num_hands: u32 = 2;
        let num_nodes: u32 = 2;

        let mut opp_reach = vec![0.0f32; (num_nodes * num_hands) as usize];
        opp_reach[2] = 0.6; // node1, hand0
        opp_reach[3] = 0.4; // node1, hand1

        let terminal_nodes = vec![1u32];
        let fold_amount_win = vec![5.0f32];
        let fold_amount_lose = vec![-5.0f32];
        let fold_player = vec![0u32]; // OOP (traverser) folded
        // All matchups valid (no card conflicts in this test)
        let valid_matchups = vec![1.0f32; (num_hands * num_hands) as usize];

        let mut gpu_cfvalues = gpu.alloc_zeros::<f32>((num_nodes * num_hands) as usize).unwrap();
        let gpu_opp_reach = gpu.upload(&opp_reach).unwrap();
        let gpu_terminals = gpu.upload(&terminal_nodes).unwrap();
        let gpu_win = gpu.upload(&fold_amount_win).unwrap();
        let gpu_lose = gpu.upload(&fold_amount_lose).unwrap();
        let gpu_fold_player = gpu.upload(&fold_player).unwrap();
        let gpu_valid = gpu.upload(&valid_matchups).unwrap();

        gpu.launch_terminal_fold_eval(
            &mut gpu_cfvalues,
            &gpu_opp_reach,
            &gpu_terminals,
            &gpu_win,
            &gpu_lose,
            &gpu_fold_player,
            &gpu_valid,
            0, // traverser = OOP, who folded
            1,
            num_hands,
        )
        .unwrap();

        let result = gpu.download(&gpu_cfvalues).unwrap();

        let eps = 1e-5;
        let total_opp_reach = 0.6 + 0.4;
        // Traverser folded -> gets amount_lose
        assert!(
            (result[2] - (-5.0) * total_opp_reach).abs() < eps,
            "fold hand0: got {}",
            result[2]
        );
        assert!(
            (result[3] - (-5.0) * total_opp_reach).abs() < eps,
            "fold hand1: got {}",
            result[3]
        );
    }

    #[test]
    fn test_terminal_showdown_eval() {
        let gpu = GpuContext::new(0).unwrap();

        // 2 nodes: root(0), showdown_terminal(1)
        // 3 hands with strengths:
        //   traverser: [100, 50, 100] (hand 0 and 2 tie, hand 1 loses)
        //   opponent:  [100, 50, 100] (same for this test)
        let num_hands: u32 = 3;
        let num_nodes: u32 = 2;

        let mut opp_reach = vec![0.0f32; (num_nodes * num_hands) as usize];
        opp_reach[3] = 1.0; // node1, hand0: reach=1.0
        opp_reach[4] = 1.0; // node1, hand1: reach=1.0
        opp_reach[5] = 1.0; // node1, hand2: reach=1.0

        let terminal_nodes = vec![1u32];
        let amount_win = vec![10.0f32];
        let amount_lose = vec![-10.0f32];
        let traverser_strengths = vec![100u32, 50, 100];
        let opponent_strengths = vec![100u32, 50, 100];
        // All matchups valid (no card conflicts in this test)
        let valid_matchups = vec![1.0f32; (num_hands * num_hands) as usize];

        let mut gpu_cfvalues = gpu.alloc_zeros::<f32>((num_nodes * num_hands) as usize).unwrap();
        let gpu_opp_reach = gpu.upload(&opp_reach).unwrap();
        let gpu_terminals = gpu.upload(&terminal_nodes).unwrap();
        let gpu_win = gpu.upload(&amount_win).unwrap();
        let gpu_lose = gpu.upload(&amount_lose).unwrap();
        let gpu_trav_strengths = gpu.upload(&traverser_strengths).unwrap();
        let gpu_opp_strengths = gpu.upload(&opponent_strengths).unwrap();
        let gpu_valid = gpu.upload(&valid_matchups).unwrap();

        gpu.launch_terminal_showdown_eval(
            &mut gpu_cfvalues,
            &gpu_opp_reach,
            &gpu_terminals,
            &gpu_win,
            &gpu_lose,
            &gpu_trav_strengths,
            &gpu_opp_strengths,
            &gpu_valid,
            1,
            num_hands,
        )
        .unwrap();

        let result = gpu.download(&gpu_cfvalues).unwrap();

        let eps = 1e-5;
        // Hand 0 (strength=100): beats hand1(50), ties hand2(100)
        //   cfv = 10.0 * 1.0 + 0 = 10.0
        assert!(
            (result[3] - 10.0).abs() < eps,
            "showdown hand0: got {}",
            result[3]
        );
        // Hand 1 (strength=50): loses to hand0(100) and hand2(100)
        //   cfv = -10.0 * 1.0 + -10.0 * 1.0 = -20.0
        assert!(
            (result[4] - (-20.0)).abs() < eps,
            "showdown hand1: got {}",
            result[4]
        );
        // Hand 2 (strength=100): beats hand1(50), ties hand0(100)
        //   cfv = 10.0 * 1.0 + 0 = 10.0
        assert!(
            (result[5] - 10.0).abs() < eps,
            "showdown hand2: got {}",
            result[5]
        );
    }

    #[test]
    fn test_backward_cfv() {
        let gpu = GpuContext::new(0).unwrap();

        // Tree: Root(0) -> [Child(1), Child(2)]
        // Root is infoset 0 with 2 actions, player=0 (OOP)
        // 2 hands, per-hand strategy
        let num_hands: u32 = 2;
        let num_nodes: u32 = 3;
        let max_actions: u32 = 2;

        // Set children's CFVs (as if they're terminals)
        let mut cfvalues = vec![0.0f32; (num_nodes * num_hands) as usize];
        // Child 1 (node 1): hand0=10.0, hand1=5.0
        cfvalues[2] = 10.0;
        cfvalues[3] = 5.0;
        // Child 2 (node 2): hand0=-3.0, hand1=8.0
        cfvalues[4] = -3.0;
        cfvalues[5] = 8.0;

        // Per-hand strategy for infoset 0:
        //   hand0: [0.7, 0.3], hand1: [0.4, 0.6]
        // Layout: (iset * max_actions + action) * num_hands + hand
        let strategy = vec![
            0.7f32, 0.4, // action0: hand0=0.7, hand1=0.4
            0.3, 0.6,    // action1: hand0=0.3, hand1=0.6
        ];

        let level_nodes = vec![0u32]; // just the root
        let child_offsets = vec![0u32, 2, 2, 2]; // root has 2 children
        let children_arr = vec![1u32, 2];
        let infoset_ids = vec![0u32, u32::MAX, u32::MAX];
        // Root is player 0 (OOP), traverser=0 -> strategy-weighted
        let node_players = vec![0u32];

        let mut gpu_cfvalues = gpu.upload(&cfvalues).unwrap();
        let gpu_strategy = gpu.upload(&strategy).unwrap();
        let gpu_level_nodes = gpu.upload(&level_nodes).unwrap();
        let gpu_child_offsets = gpu.upload(&child_offsets).unwrap();
        let gpu_children = gpu.upload(&children_arr).unwrap();
        let gpu_infoset_ids = gpu.upload(&infoset_ids).unwrap();
        let gpu_node_players = gpu.upload(&node_players).unwrap();

        gpu.launch_backward_cfv(
            &mut gpu_cfvalues,
            &gpu_strategy,
            &gpu_level_nodes,
            &gpu_child_offsets,
            &gpu_children,
            &gpu_infoset_ids,
            &gpu_node_players,
            0, // traverser = OOP
            1, // 1 node at this level
            num_hands,
            max_actions,
        )
        .unwrap();

        let result = gpu.download(&gpu_cfvalues).unwrap();

        let eps = 1e-5;
        // Root CFV for hand 0: 0.7 * 10.0 + 0.3 * (-3.0) = 7.0 - 0.9 = 6.1
        assert!(
            (result[0] - 6.1).abs() < eps,
            "root hand0: got {}",
            result[0]
        );
        // Root CFV for hand 1: 0.4 * 5.0 + 0.6 * 8.0 = 2.0 + 4.8 = 6.8
        assert!(
            (result[1] - 6.8).abs() < eps,
            "root hand1: got {}",
            result[1]
        );
    }

    #[test]
    fn test_update_regrets() {
        let gpu = GpuContext::new(0).unwrap();

        // 1 decision node (infoset 0) with 2 actions, 2 hands
        // Child 0 has cfv: hand0=10, hand1=5
        // Child 1 has cfv: hand0=-2, hand1=8
        // Node cfv: hand0=6.1, hand1=6.8 (from backward pass)
        let num_hands: u32 = 2;
        let max_actions: u32 = 2;

        let cfvalues = vec![
            6.1f32, 6.8, // node 0 (root)
            10.0, 5.0,   // node 1 (child 0)
            -2.0, 8.0,   // node 2 (child 1)
        ];

        // Initial regrets: all zero (1 infoset * 2 actions * 2 hands = 4)
        let regrets = vec![0.0f32; (max_actions * num_hands) as usize];

        // Strategy: hand0=[0.7, 0.3], hand1=[0.4, 0.6]
        let strategy = vec![0.7f32, 0.4, 0.3, 0.6];

        // Strategy sum: all zero initially
        let strategy_sum = vec![0.0f32; (max_actions * num_hands) as usize];

        let decision_nodes = vec![0u32];
        let child_offsets = vec![0u32, 2, 2, 2];
        let children_arr = vec![1u32, 2];
        let infoset_ids = vec![0u32, u32::MAX, u32::MAX];
        let num_actions_arr = vec![2u32];

        let mut gpu_regrets = gpu.upload(&regrets).unwrap();
        let mut gpu_strategy_sum = gpu.upload(&strategy_sum).unwrap();
        let gpu_strategy = gpu.upload(&strategy).unwrap();
        let gpu_cfvalues = gpu.upload(&cfvalues).unwrap();
        let gpu_decision_nodes = gpu.upload(&decision_nodes).unwrap();
        let gpu_child_offsets = gpu.upload(&child_offsets).unwrap();
        let gpu_children = gpu.upload(&children_arr).unwrap();
        let gpu_infoset_ids = gpu.upload(&infoset_ids).unwrap();
        let gpu_num_actions = gpu.upload(&num_actions_arr).unwrap();

        // No discounting: pos_discount=1, neg_discount=1, strat_discount=1
        gpu.launch_update_regrets(
            &mut gpu_regrets,
            &mut gpu_strategy_sum,
            &gpu_strategy,
            &gpu_cfvalues,
            &gpu_decision_nodes,
            &gpu_child_offsets,
            &gpu_children,
            &gpu_infoset_ids,
            &gpu_num_actions,
            1,   // num_decision_nodes
            num_hands,
            max_actions,
            1.0, // pos_discount
            1.0, // neg_discount
            1.0, // strat_discount
        )
        .unwrap();

        let result_regrets = gpu.download(&gpu_regrets).unwrap();
        let result_strat_sum = gpu.download(&gpu_strategy_sum).unwrap();

        let eps = 1e-4;

        // Instantaneous regret for hand0:
        //   action0: cfv[child0][hand0] - cfv[node][hand0] = 10.0 - 6.1 = 3.9
        //   action1: cfv[child1][hand0] - cfv[node][hand0] = -2.0 - 6.1 = -8.1
        assert!(
            (result_regrets[0] - 3.9).abs() < eps,
            "regret hand0 action0: got {}",
            result_regrets[0]
        );
        assert!(
            (result_regrets[2] - (-8.1)).abs() < eps,
            "regret hand0 action1: got {}",
            result_regrets[2]
        );

        // Instantaneous regret for hand1:
        //   action0: 5.0 - 6.8 = -1.8
        //   action1: 8.0 - 6.8 = 1.2
        assert!(
            (result_regrets[1] - (-1.8)).abs() < eps,
            "regret hand1 action0: got {}",
            result_regrets[1]
        );
        assert!(
            (result_regrets[3] - 1.2).abs() < eps,
            "regret hand1 action1: got {}",
            result_regrets[3]
        );

        // Strategy sum: old * strat_discount + strategy = 0 * 1.0 + strategy = strategy
        assert!(
            (result_strat_sum[0] - 0.7).abs() < eps,
            "strat_sum hand0 action0: got {}",
            result_strat_sum[0]
        );
        assert!(
            (result_strat_sum[1] - 0.4).abs() < eps,
            "strat_sum hand1 action0: got {}",
            result_strat_sum[1]
        );
        assert!(
            (result_strat_sum[2] - 0.3).abs() < eps,
            "strat_sum hand0 action1: got {}",
            result_strat_sum[2]
        );
        assert!(
            (result_strat_sum[3] - 0.6).abs() < eps,
            "strat_sum hand1 action1: got {}",
            result_strat_sum[3]
        );
    }

    #[test]
    fn test_extract_strategy_kernel() {
        let gpu = GpuContext::new(0).unwrap();
        let num_infosets: u32 = 2;
        let max_actions: u32 = 3;
        let num_hands: u32 = 2;

        // Per-hand layout: (infoset * max_actions + action) * num_hands + hand
        // Infoset 0, hand 0: strategy_sum [10, 20, 30] -> [1/6, 2/6, 3/6]
        // Infoset 0, hand 1: strategy_sum [0, 0, 0] -> uniform [1/3, 1/3, 1/3]
        // Infoset 1, hand 0: strategy_sum [5, 0, 15] -> [1/4, 0, 3/4]
        // Infoset 1, hand 1: strategy_sum [3, 3, 3] -> [1/3, 1/3, 1/3]
        let strategy_sum: Vec<f32> = vec![
            10.0, 0.0, // iset0, act0: hand0=10, hand1=0
            20.0, 0.0, // iset0, act1: hand0=20, hand1=0
            30.0, 0.0, // iset0, act2: hand0=30, hand1=0
            5.0, 3.0,  // iset1, act0
            0.0, 3.0,  // iset1, act1
            15.0, 3.0, // iset1, act2
        ];
        let num_actions: Vec<u32> = vec![3, 3];

        let gpu_strategy_sum = gpu.upload(&strategy_sum).unwrap();
        let gpu_num_actions = gpu.upload(&num_actions).unwrap();
        let mut gpu_output = gpu
            .alloc_zeros::<f32>((num_infosets * max_actions * num_hands) as usize)
            .unwrap();

        gpu.launch_extract_strategy(
            &gpu_strategy_sum,
            &gpu_num_actions,
            &mut gpu_output,
            num_infosets,
            max_actions,
            num_hands,
        )
        .unwrap();

        let result = gpu.download(&gpu_output).unwrap();
        let eps = 1e-5;

        // Infoset 0, hand 0: [10/60, 20/60, 30/60] = [1/6, 1/3, 1/2]
        assert!(
            (result[0] - 1.0 / 6.0).abs() < eps,
            "i0h0a0: {}",
            result[0]
        );
        assert!(
            (result[2] - 1.0 / 3.0).abs() < eps,
            "i0h0a1: {}",
            result[2]
        );
        assert!((result[4] - 0.5).abs() < eps, "i0h0a2: {}", result[4]);

        // Infoset 0, hand 1: all zeros -> uniform [1/3, 1/3, 1/3]
        assert!(
            (result[1] - 1.0 / 3.0).abs() < eps,
            "i0h1a0: {}",
            result[1]
        );
        assert!(
            (result[3] - 1.0 / 3.0).abs() < eps,
            "i0h1a1: {}",
            result[3]
        );
        assert!(
            (result[5] - 1.0 / 3.0).abs() < eps,
            "i0h1a2: {}",
            result[5]
        );

        // Infoset 1, hand 0: [5/20, 0/20, 15/20] = [1/4, 0, 3/4]
        assert!(
            (result[6] - 0.25).abs() < eps,
            "i1h0a0: {}",
            result[6]
        );
        assert!((result[8] - 0.0).abs() < eps, "i1h0a1: {}", result[8]);
        assert!(
            (result[10] - 0.75).abs() < eps,
            "i1h0a2: {}",
            result[10]
        );

        // Infoset 1, hand 1: [3/9, 3/9, 3/9] = [1/3, 1/3, 1/3]
        assert!(
            (result[7] - 1.0 / 3.0).abs() < eps,
            "i1h1a0: {}",
            result[7]
        );
        assert!(
            (result[9] - 1.0 / 3.0).abs() < eps,
            "i1h1a1: {}",
            result[9]
        );
        assert!(
            (result[11] - 1.0 / 3.0).abs() < eps,
            "i1h1a2: {}",
            result[11]
        );
    }

    #[test]
    fn test_update_regrets_with_discounting() {
        let gpu = GpuContext::new(0).unwrap();

        let num_hands: u32 = 1;
        let max_actions: u32 = 2;

        let cfvalues = vec![
            5.0f32, // node 0
            10.0,   // node 1 (child 0)
            -2.0,   // node 2 (child 1)
        ];

        // Pre-existing regrets: action0 = 4.0, action1 = -3.0
        let regrets = vec![4.0f32, -3.0];
        let strategy = vec![1.0f32, 0.0]; // all on action0
        let strategy_sum = vec![2.0f32, 0.0];

        let decision_nodes = vec![0u32];
        let child_offsets = vec![0u32, 2, 2, 2];
        let children_arr = vec![1u32, 2];
        let infoset_ids = vec![0u32, u32::MAX, u32::MAX];
        let num_actions_arr = vec![2u32];

        let mut gpu_regrets = gpu.upload(&regrets).unwrap();
        let mut gpu_strategy_sum = gpu.upload(&strategy_sum).unwrap();
        let gpu_strategy = gpu.upload(&strategy).unwrap();
        let gpu_cfvalues = gpu.upload(&cfvalues).unwrap();
        let gpu_decision_nodes = gpu.upload(&decision_nodes).unwrap();
        let gpu_child_offsets = gpu.upload(&child_offsets).unwrap();
        let gpu_children = gpu.upload(&children_arr).unwrap();
        let gpu_infoset_ids = gpu.upload(&infoset_ids).unwrap();
        let gpu_num_actions = gpu.upload(&num_actions_arr).unwrap();

        // DCFR discounting: pos_discount=0.8, neg_discount=0.5, strat_discount=0.7
        gpu.launch_update_regrets(
            &mut gpu_regrets,
            &mut gpu_strategy_sum,
            &gpu_strategy,
            &gpu_cfvalues,
            &gpu_decision_nodes,
            &gpu_child_offsets,
            &gpu_children,
            &gpu_infoset_ids,
            &gpu_num_actions,
            1,
            num_hands,
            max_actions,
            0.8, // pos_discount
            0.5, // neg_discount
            0.7, // strat_discount
        )
        .unwrap();

        let result_regrets = gpu.download(&gpu_regrets).unwrap();
        let result_strat_sum = gpu.download(&gpu_strategy_sum).unwrap();

        let eps = 1e-4;

        // Instant regret action0: 10 - 5 = 5
        // Old regret action0: 4.0 (positive, discount=0.8)
        // New regret action0: 4.0 * 0.8 + 5.0 = 8.2
        assert!(
            (result_regrets[0] - 8.2).abs() < eps,
            "regret action0: got {}",
            result_regrets[0]
        );

        // Instant regret action1: -2 - 5 = -7
        // Old regret action1: -3.0 (negative, discount=0.5)
        // New regret action1: -3.0 * 0.5 + (-7.0) = -8.5
        assert!(
            (result_regrets[1] - (-8.5)).abs() < eps,
            "regret action1: got {}",
            result_regrets[1]
        );

        // Strategy sum: old * strat_discount + strategy
        // action0: 2.0 * 0.7 + 1.0 = 2.4
        assert!(
            (result_strat_sum[0] - 2.4).abs() < eps,
            "strat_sum action0: got {}",
            result_strat_sum[0]
        );
        // action1: 0.0 * 0.7 + 0.0 = 0.0
        assert!(
            (result_strat_sum[1] - 0.0).abs() < eps,
            "strat_sum action1: got {}",
            result_strat_sum[1]
        );
    }
}
