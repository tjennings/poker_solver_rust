// Persistent DCFR+ mega-kernel: runs the entire solve loop in a single kernel launch.
//
// Eliminates all kernel launch overhead (~30 launches per iteration x 4000 iterations
// = 120,000 launches -> 1 launch). Uses a custom atomic grid-wide barrier for
// synchronization between phases (avoids cooperative_groups.h dependency in NVRTC).
//
// All solver state is passed via a SolverContext struct pointer to stay under
// CUDA's 4KB kernel argument limit.
//
// The kernel uses strided loops so any grid size works: each thread handles
// ceil(work_items / grid_size) elements per phase.

// ============================================================
// SolverContext: all pointers and scalars for the solve
// ============================================================
struct SolverContext {
    // Solver state (read/write)
    float* regrets;              // [num_infosets * max_actions * num_hands]
    float* strategy_sum;         // same layout
    float* strategy;             // same layout (current strategy, overwritten each iter)
    float* reach_oop;            // [num_nodes * num_hands]
    float* reach_ip;             // same
    float* cfvalues;             // same

    // Tree topology (read-only)
    const unsigned int* child_offsets;  // [num_nodes + 1] CSR
    const unsigned int* children;       // CSR child array
    const unsigned int* infoset_ids;    // [num_nodes] (0xFFFFFFFF for terminals)
    const unsigned int* num_actions_arr; // [num_infosets]
    const unsigned int* node_types;     // [num_nodes] 0=OOP, 1=IP, 2=fold, 3=showdown

    // Level data (packed flat)
    const unsigned int* level_offsets;       // [num_levels + 1] node index ranges per level
    const unsigned int* parent_nodes;        // [num_nodes] parent of each node
    const unsigned int* parent_actions;      // [num_nodes] action from parent
    const unsigned int* parent_infosets;     // [num_nodes] infoset of parent
    const unsigned int* parent_players;      // [num_nodes] player at parent (0 or 1)

    // Terminal fold data
    const unsigned int* fold_terminal_nodes;  // [num_fold_terminals]
    const float* fold_amount_win;             // [num_fold_terminals * num_hands]
    const float* fold_amount_lose;            // [num_fold_terminals * num_hands]
    const unsigned int* fold_player;          // [num_fold_terminals]

    // Fold aggregates (precomputed hand cards, same-hand, working buffers)
    const unsigned int* hand_cards_oop;       // [num_hands * 2] (c1,c2) pairs
    const unsigned int* hand_cards_ip;        // [num_hands * 2]
    const unsigned int* same_hand_index_oop;  // [num_hands]
    const unsigned int* same_hand_index_ip;   // [num_hands]
    float* fold_total_opp_reach;              // [num_fold_terminals * num_spots] working buf
    float* fold_per_card_reach;               // [num_fold_terminals * num_spots * 52] working buf

    // Showdown data
    const unsigned int* showdown_terminal_nodes; // [num_showdown_terminals]
    const float* showdown_amount_win;            // [num_showdown_terminals * num_hands]
    const float* showdown_amount_lose;           // [num_showdown_terminals * num_hands]

    // 3-kernel showdown data
    const unsigned int* sorted_opp_oop;    // [num_spots * hands_per_spot] sorted IP indices when trav=OOP
    const unsigned int* sorted_opp_ip;     // [num_spots * hands_per_spot] sorted OOP indices when trav=IP
    const unsigned int* rank_win_oop;      // [num_spots * hands_per_spot]
    const unsigned int* rank_next_oop;     // [num_spots * hands_per_spot]
    const unsigned int* rank_win_ip;       // [num_spots * hands_per_spot]
    const unsigned int* rank_next_ip;      // [num_spots * hands_per_spot]
    float* sd_sorted_reach;                // [num_showdown_terminals * num_hands] working buf
    float* sd_prefix_excl;                 // [num_showdown_terminals * num_hands] working buf
    float* sd_totals;                      // [num_showdown_terminals * num_spots] working buf

    // Decision node lists per player
    const unsigned int* decision_nodes_oop;  // [num_oop_decisions]
    const unsigned int* decision_nodes_ip;   // [num_ip_decisions]

    // Initial reach
    const float* initial_reach_oop;  // [num_hands]
    const float* initial_reach_ip;   // [num_hands]

    // Dimensions
    unsigned int num_hands;
    unsigned int max_actions;
    unsigned int num_infosets;
    unsigned int num_nodes;
    unsigned int num_levels;
    unsigned int hands_per_spot;
    unsigned int num_spots;
    unsigned int num_fold_terminals;
    unsigned int num_showdown_terminals;
    unsigned int num_oop_decisions;
    unsigned int num_ip_decisions;
    unsigned int max_iterations;

    // Grid sync state
    int* barrier_counter;
    int* barrier_sense;
};

// ============================================================
// Grid-wide barrier using atomics (Option A)
// ============================================================
__device__ void grid_sync(int* counter, int* sense, unsigned int num_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        int current_sense = *sense;
        int arrived = atomicAdd(counter, 1);
        if (arrived == (int)num_blocks - 1) {
            *counter = 0;
            __threadfence();
            atomicExch(sense, 1 - current_sense);
        } else {
            while (atomicAdd(sense, 0) == current_sense) {
                // spin-wait
            }
        }
    }
    __syncthreads();
}

// ============================================================
// Phase: Regret Matching
// ============================================================
__device__ void phase_regret_match(
    const SolverContext* ctx,
    unsigned int grid_size, unsigned int global_tid
) {
    unsigned int total = ctx->num_infosets * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int iset = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;
        unsigned int n = ctx->num_actions_arr[iset];

        float pos_sum = 0.0f;
        for (unsigned int a = 0; a < n; a++) {
            float r = ctx->regrets[(iset * ctx->max_actions + a) * ctx->num_hands + hand];
            if (r > 0.0f) pos_sum += r;
        }

        if (pos_sum > 0.0f) {
            for (unsigned int a = 0; a < n; a++) {
                unsigned int si = (iset * ctx->max_actions + a) * ctx->num_hands + hand;
                float r = ctx->regrets[si];
                ctx->strategy[si] = (r > 0.0f) ? (r / pos_sum) : 0.0f;
            }
        } else {
            float uniform = 1.0f / (float)n;
            for (unsigned int a = 0; a < n; a++) {
                ctx->strategy[(iset * ctx->max_actions + a) * ctx->num_hands + hand] = uniform;
            }
        }

        for (unsigned int a = n; a < ctx->max_actions; a++) {
            ctx->strategy[(iset * ctx->max_actions + a) * ctx->num_hands + hand] = 0.0f;
        }
    }
}

// ============================================================
// Phase: Initialize Reach
// ============================================================
__device__ void phase_init_reach(
    const SolverContext* ctx,
    unsigned int grid_size, unsigned int global_tid
) {
    unsigned int reach_size = ctx->num_nodes * ctx->num_hands;
    // Zero all reach
    for (unsigned int idx = global_tid; idx < reach_size; idx += grid_size) {
        ctx->reach_oop[idx] = 0.0f;
        ctx->reach_ip[idx] = 0.0f;
    }
}

__device__ void phase_set_root_reach(
    const SolverContext* ctx,
    unsigned int grid_size, unsigned int global_tid
) {
    for (unsigned int idx = global_tid; idx < ctx->num_hands; idx += grid_size) {
        ctx->reach_oop[idx] = ctx->initial_reach_oop[idx];
        ctx->reach_ip[idx] = ctx->initial_reach_ip[idx];
    }
}

// ============================================================
// Phase: Forward Pass (one level)
// ============================================================
__device__ void phase_forward_pass_level(
    const SolverContext* ctx,
    unsigned int level,
    unsigned int grid_size, unsigned int global_tid
) {
    unsigned int start = ctx->level_offsets[level];
    unsigned int end = ctx->level_offsets[level + 1];
    unsigned int n_nodes_level = end - start;
    unsigned int total = n_nodes_level * ctx->num_hands;

    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int node_local = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;
        unsigned int node = start + node_local;

        unsigned int parent = ctx->parent_nodes[node];
        unsigned int action = ctx->parent_actions[node];
        unsigned int infoset = ctx->parent_infosets[node];
        unsigned int player = ctx->parent_players[node];

        float action_prob = ctx->strategy[(infoset * ctx->max_actions + action) * ctx->num_hands + hand];

        if (player == 0) {
            ctx->reach_oop[node * ctx->num_hands + hand] = ctx->reach_oop[parent * ctx->num_hands + hand] * action_prob;
            ctx->reach_ip[node * ctx->num_hands + hand] = ctx->reach_ip[parent * ctx->num_hands + hand];
        } else {
            ctx->reach_ip[node * ctx->num_hands + hand] = ctx->reach_ip[parent * ctx->num_hands + hand] * action_prob;
            ctx->reach_oop[node * ctx->num_hands + hand] = ctx->reach_oop[parent * ctx->num_hands + hand];
        }
    }
}

// ============================================================
// Phase: Zero CFValues
// ============================================================
__device__ void phase_zero_cfvalues(
    const SolverContext* ctx,
    unsigned int grid_size, unsigned int global_tid
) {
    unsigned int total = ctx->num_nodes * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        ctx->cfvalues[idx] = 0.0f;
    }
}

// ============================================================
// Phase: Fold Precompute Aggregates (per terminal, per spot)
// ============================================================
__device__ void phase_fold_precompute(
    const SolverContext* ctx,
    unsigned int traverser,
    unsigned int grid_size, unsigned int global_tid
) {
    if (ctx->num_fold_terminals == 0) return;

    const float* opp_reach = (traverser == 0) ? ctx->reach_ip : ctx->reach_oop;
    const unsigned int* opp_hand_cards = (traverser == 0) ? ctx->hand_cards_ip : ctx->hand_cards_oop;

    // Each thread handles one or more (terminal, spot, hand) triples to compute aggregates.
    // We need atomic accumulation into per-(terminal, spot) aggregates.
    // First, zero the output buffers.
    unsigned int num_agg = ctx->num_fold_terminals * ctx->num_spots;
    for (unsigned int idx = global_tid; idx < num_agg; idx += grid_size) {
        ctx->fold_total_opp_reach[idx] = 0.0f;
    }
    unsigned int num_card_entries = num_agg * 52;
    for (unsigned int idx = global_tid; idx < num_card_entries; idx += grid_size) {
        ctx->fold_per_card_reach[idx] = 0.0f;
    }
    // Need a grid sync here to ensure zeros are visible before atomic adds
    grid_sync(ctx->barrier_counter, ctx->barrier_sense, gridDim.x);

    // Now accumulate: one thread per (terminal, hand)
    unsigned int total = ctx->num_fold_terminals * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int term_idx = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;

        unsigned int node = ctx->fold_terminal_nodes[term_idx];
        unsigned int spot = hand / ctx->hands_per_spot;
        unsigned int agg_idx = term_idx * ctx->num_spots + spot;

        float r = opp_reach[node * ctx->num_hands + hand];
        if (r != 0.0f) {
            atomicAdd(&ctx->fold_total_opp_reach[agg_idx], r);
            unsigned int c1 = opp_hand_cards[hand * 2];
            unsigned int c2 = opp_hand_cards[hand * 2 + 1];
            if (c1 < 52) atomicAdd(&ctx->fold_per_card_reach[agg_idx * 52 + c1], r);
            if (c2 < 52) atomicAdd(&ctx->fold_per_card_reach[agg_idx * 52 + c2], r);
        }
    }
}

// ============================================================
// Phase: Fold Eval (using aggregates, O(1) per hand)
// ============================================================
__device__ void phase_fold_eval(
    const SolverContext* ctx,
    unsigned int traverser,
    unsigned int grid_size, unsigned int global_tid
) {
    if (ctx->num_fold_terminals == 0) return;

    const float* opp_reach = (traverser == 0) ? ctx->reach_ip : ctx->reach_oop;
    const unsigned int* trav_hand_cards = (traverser == 0) ? ctx->hand_cards_oop : ctx->hand_cards_ip;
    const unsigned int* same_hand_index = (traverser == 0) ? ctx->same_hand_index_oop : ctx->same_hand_index_ip;

    unsigned int total = ctx->num_fold_terminals * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int term_idx = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;

        unsigned int node = ctx->fold_terminal_nodes[term_idx];
        unsigned int spot = hand / ctx->hands_per_spot;
        unsigned int agg_idx = term_idx * ctx->num_spots + spot;

        float payoff;
        if (ctx->fold_player[term_idx] == traverser) {
            payoff = ctx->fold_amount_lose[term_idx * ctx->num_hands + hand];
        } else {
            payoff = ctx->fold_amount_win[term_idx * ctx->num_hands + hand];
        }

        float total_r = ctx->fold_total_opp_reach[agg_idx];

        unsigned int c1 = trav_hand_cards[hand * 2];
        unsigned int c2 = trav_hand_cards[hand * 2 + 1];

        unsigned int card_base = agg_idx * 52;
        float blocked = 0.0f;
        if (c1 < 52) blocked += ctx->fold_per_card_reach[card_base + c1];
        if (c2 < 52) blocked += ctx->fold_per_card_reach[card_base + c2];

        unsigned int same_idx = same_hand_index[hand];
        float same_reach = 0.0f;
        if (same_idx != 0xFFFFFFFF) {
            same_reach = opp_reach[node * ctx->num_hands + same_idx];
        }

        ctx->cfvalues[node * ctx->num_hands + hand] = payoff * (total_r - blocked + same_reach);
    }
}

// ============================================================
// Phase: Showdown Scatter (kernel 1 of 3)
// ============================================================
__device__ void phase_showdown_scatter(
    const SolverContext* ctx,
    unsigned int traverser,
    unsigned int grid_size, unsigned int global_tid
) {
    if (ctx->num_showdown_terminals == 0) return;

    const float* opp_reach = (traverser == 0) ? ctx->reach_ip : ctx->reach_oop;
    const unsigned int* opp_sorted = (traverser == 0) ? ctx->sorted_opp_oop : ctx->sorted_opp_ip;

    unsigned int total = ctx->num_showdown_terminals * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int term_idx = idx / ctx->num_hands;
        unsigned int sorted_pos = idx % ctx->num_hands;

        unsigned int node = ctx->showdown_terminal_nodes[term_idx];
        unsigned int spot = sorted_pos / ctx->hands_per_spot;
        unsigned int local_sorted_pos = sorted_pos % ctx->hands_per_spot;

        unsigned int local_hand = opp_sorted[spot * ctx->hands_per_spot + local_sorted_pos];
        unsigned int global_hand = spot * ctx->hands_per_spot + local_hand;

        ctx->sd_sorted_reach[term_idx * ctx->num_hands + sorted_pos] =
            opp_reach[node * ctx->num_hands + global_hand];
    }
}

// ============================================================
// Phase: Showdown Prefix Sum (kernel 2 of 3)
// Serial prefix sum per segment, one thread per segment.
// ============================================================
__device__ void phase_showdown_prefix(
    const SolverContext* ctx,
    unsigned int traverser,
    unsigned int grid_size, unsigned int global_tid
) {
    if (ctx->num_showdown_terminals == 0) return;

    unsigned int num_segments = ctx->num_showdown_terminals * ctx->num_spots;
    unsigned int hps = ctx->hands_per_spot;

    for (unsigned int seg = global_tid; seg < num_segments; seg += grid_size) {
        unsigned int base = seg * hps;
        // Serial exclusive prefix sum within this segment
        // seg = term_idx * num_spots + spot, so we offset into sd_sorted_reach
        // which is laid out [term_idx * num_hands + spot * hps + ...]
        unsigned int term_idx = seg / ctx->num_spots;
        unsigned int spot = seg % ctx->num_spots;
        unsigned int sr_base = term_idx * ctx->num_hands + spot * hps;

        float running = 0.0f;
        for (unsigned int i = 0; i < hps; i++) {
            float val = ctx->sd_sorted_reach[sr_base + i];
            ctx->sd_prefix_excl[sr_base + i] = running;
            running += val;
        }
        ctx->sd_totals[seg] = running;
    }
}

// ============================================================
// Phase: Showdown Lookup CFV (kernel 3 of 3)
// ============================================================
__device__ void phase_showdown_lookup(
    const SolverContext* ctx,
    unsigned int traverser,
    unsigned int grid_size, unsigned int global_tid
) {
    if (ctx->num_showdown_terminals == 0) return;

    const unsigned int* rank_win = (traverser == 0) ? ctx->rank_win_oop : ctx->rank_win_ip;
    const unsigned int* rank_next = (traverser == 0) ? ctx->rank_next_oop : ctx->rank_next_ip;

    unsigned int hps = ctx->hands_per_spot;
    unsigned int total = ctx->num_showdown_terminals * ctx->num_hands;

    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int term_idx = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;

        unsigned int node = ctx->showdown_terminal_nodes[term_idx];
        unsigned int spot = hand / hps;
        unsigned int local_hand = hand % hps;

        unsigned int seg_base = term_idx * ctx->num_hands + spot * hps;
        unsigned int seg_idx = term_idx * ctx->num_spots + spot;
        float total_reach = ctx->sd_totals[seg_idx];

        unsigned int rw = rank_win[spot * hps + local_hand];
        float win_reach;
        if (rw < hps) {
            win_reach = ctx->sd_prefix_excl[seg_base + rw];
        } else {
            win_reach = total_reach;
        }

        unsigned int rn = rank_next[spot * hps + local_hand];
        float prefix_at_next;
        if (rn < hps) {
            prefix_at_next = ctx->sd_prefix_excl[seg_base + rn];
        } else {
            prefix_at_next = total_reach;
        }
        float lose_reach = total_reach - prefix_at_next;

        float win = ctx->showdown_amount_win[term_idx * ctx->num_hands + hand];
        float lose = ctx->showdown_amount_lose[term_idx * ctx->num_hands + hand];

        ctx->cfvalues[node * ctx->num_hands + hand] = win * win_reach + lose * lose_reach;
    }
}

// ============================================================
// Phase: Backward CFV (one level, all decision nodes)
// ============================================================
__device__ void phase_backward_cfv_level(
    const SolverContext* ctx,
    unsigned int level,
    unsigned int traverser,
    unsigned int grid_size, unsigned int global_tid
) {
    unsigned int start = ctx->level_offsets[level];
    unsigned int end = ctx->level_offsets[level + 1];
    unsigned int n_nodes_level = end - start;
    unsigned int total = n_nodes_level * ctx->num_hands;

    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int node_local = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;
        unsigned int node = start + node_local;

        unsigned int infoset = ctx->infoset_ids[node];
        // Skip terminals
        if (infoset == 0xFFFFFFFF) continue;

        unsigned int first_child = ctx->child_offsets[node];
        unsigned int last_child = ctx->child_offsets[node + 1];
        unsigned int n_actions = last_child - first_child;

        // node_types: 0=OOP, 1=IP
        unsigned int player = ctx->node_types[node];
        int is_traverser = (player == traverser);

        float node_cfv = 0.0f;
        for (unsigned int a = 0; a < n_actions; a++) {
            unsigned int child = ctx->children[first_child + a];
            float child_cfv = ctx->cfvalues[child * ctx->num_hands + hand];

            if (is_traverser) {
                float action_prob = ctx->strategy[(infoset * ctx->max_actions + a) * ctx->num_hands + hand];
                node_cfv += action_prob * child_cfv;
            } else {
                node_cfv += child_cfv;
            }
        }

        ctx->cfvalues[node * ctx->num_hands + hand] = node_cfv;
    }
}

// ============================================================
// Phase: Update Regrets
// ============================================================
__device__ void phase_update_regrets(
    const SolverContext* ctx,
    unsigned int traverser,
    float pos_discount, float neg_discount, float strat_discount,
    unsigned int grid_size, unsigned int global_tid
) {
    const unsigned int* decision_nodes = (traverser == 0) ? ctx->decision_nodes_oop : ctx->decision_nodes_ip;
    unsigned int num_decisions = (traverser == 0) ? ctx->num_oop_decisions : ctx->num_ip_decisions;

    if (num_decisions == 0) return;

    unsigned int total = num_decisions * ctx->max_actions * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int total_per_node = ctx->max_actions * ctx->num_hands;
        unsigned int dec_local = idx / total_per_node;
        unsigned int remainder = idx % total_per_node;
        unsigned int action = remainder / ctx->num_hands;
        unsigned int hand = remainder % ctx->num_hands;

        if (dec_local >= num_decisions) continue;

        unsigned int node = decision_nodes[dec_local];
        unsigned int infoset = ctx->infoset_ids[node];
        if (infoset == 0xFFFFFFFF) continue;

        unsigned int n_actions = ctx->num_actions_arr[infoset];
        if (action >= n_actions) continue;

        unsigned int first_child = ctx->child_offsets[node];
        unsigned int child = ctx->children[first_child + action];

        float child_cfv = ctx->cfvalues[child * ctx->num_hands + hand];
        float node_cfv = ctx->cfvalues[node * ctx->num_hands + hand];
        float inst_regret = child_cfv - node_cfv;

        unsigned int reg_idx = (infoset * ctx->max_actions + action) * ctx->num_hands + hand;

        float old_regret = ctx->regrets[reg_idx];
        float discount = (old_regret >= 0.0f) ? pos_discount : neg_discount;
        float new_regret = old_regret * discount + inst_regret;
        ctx->regrets[reg_idx] = new_regret;

        ctx->strategy_sum[reg_idx] = ctx->strategy_sum[reg_idx] * strat_discount + ctx->strategy[reg_idx];
    }
}

// ============================================================
// Phase: Extract Final Strategy
// ============================================================
__device__ void phase_extract_strategy(
    const SolverContext* ctx,
    unsigned int grid_size, unsigned int global_tid
) {
    unsigned int total = ctx->num_infosets * ctx->num_hands;
    for (unsigned int idx = global_tid; idx < total; idx += grid_size) {
        unsigned int iset = idx / ctx->num_hands;
        unsigned int hand = idx % ctx->num_hands;
        unsigned int n = ctx->num_actions_arr[iset];

        float sum = 0.0f;
        for (unsigned int a = 0; a < n; a++) {
            sum += ctx->strategy_sum[(iset * ctx->max_actions + a) * ctx->num_hands + hand];
        }

        if (sum > 0.0f) {
            for (unsigned int a = 0; a < n; a++) {
                unsigned int si = (iset * ctx->max_actions + a) * ctx->num_hands + hand;
                ctx->strategy[si] = ctx->strategy_sum[si] / sum;
            }
        } else {
            float uniform = 1.0f / (float)n;
            for (unsigned int a = 0; a < n; a++) {
                ctx->strategy[(iset * ctx->max_actions + a) * ctx->num_hands + hand] = uniform;
            }
        }

        for (unsigned int a = n; a < ctx->max_actions; a++) {
            ctx->strategy[(iset * ctx->max_actions + a) * ctx->num_hands + hand] = 0.0f;
        }
    }
}

// ============================================================
// Helper: nearest lower power of 4
// ============================================================
__device__ unsigned int nearest_lower_power_of_4(unsigned int x) {
    if (x == 0) return 0;
    // Find highest set bit position
    unsigned int pos = 31 - __clz(x);
    // Round down to even position
    pos = pos & ~1u;
    return 1u << pos;
}

// ============================================================
// Main persistent kernel
// ============================================================
extern "C" __global__ void dcfr_persistent(SolverContext* ctx) {
    unsigned int grid_size = gridDim.x * blockDim.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_blocks = gridDim.x;

    for (unsigned int t = 1; t <= ctx->max_iterations; t++) {
        // DCFR+ discount parameters (computed by every thread, trivial)
        unsigned int current_iteration = t - 1;

        int t_alpha_i = (int)current_iteration - 1;
        float t_alpha = (t_alpha_i > 0) ? (float)t_alpha_i : 0.0f;
        float pow_alpha = t_alpha * sqrtf(t_alpha);
        float pos_discount = pow_alpha / (pow_alpha + 1.0f);
        float neg_discount = 0.5f;

        unsigned int nlp4 = nearest_lower_power_of_4(current_iteration);
        float t_gamma = (float)(current_iteration - nlp4);
        float strat_discount = t_gamma / (t_gamma + 1.0f);
        strat_discount = strat_discount * strat_discount * strat_discount;

        // === PER-TRAVERSER LOOP ===
        for (unsigned int traverser = 0; traverser < 2; traverser++) {
            // 1. Regret match -> current strategy
            phase_regret_match(ctx, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);

            // 2. Init reach (zero all)
            phase_init_reach(ctx, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);

            // 3. Set root reach
            phase_set_root_reach(ctx, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);

            // 4. Forward pass (per level)
            for (unsigned int level = 1; level < ctx->num_levels; level++) {
                phase_forward_pass_level(ctx, level, grid_size, global_tid);
                grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);
            }

            // 5. Zero cfvalues
            phase_zero_cfvalues(ctx, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);

            // 6. Fold eval (precompute + eval)
            phase_fold_precompute(ctx, traverser, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);
            phase_fold_eval(ctx, traverser, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);

            // 7. Showdown eval (scatter + prefix + lookup)
            phase_showdown_scatter(ctx, traverser, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);
            phase_showdown_prefix(ctx, traverser, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);
            phase_showdown_lookup(ctx, traverser, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);

            // 8. Backward CFV (bottom-up, per level)
            for (int level = (int)ctx->num_levels - 1; level >= 0; level--) {
                phase_backward_cfv_level(ctx, (unsigned int)level, traverser, grid_size, global_tid);
                grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);
            }

            // 9. Update regrets
            phase_update_regrets(ctx, traverser, pos_discount, neg_discount, strat_discount, grid_size, global_tid);
            grid_sync(ctx->barrier_counter, ctx->barrier_sense, num_blocks);
        }
    }

    // 10. Extract final strategy
    phase_extract_strategy(ctx, grid_size, global_tid);
}
