// Backward CFV propagation kernel.
//
// For each decision node at the current level, computes the node's
// counterfactual value:
//   - Traverser's nodes: cfv[node][hand] = sum_a(strategy[a][hand] * cfv[child_a][hand])
//   - Opponent's nodes:  cfv[node][hand] = sum_a(cfv[child_a][hand])
//     (opponent's strategy is already encoded in the forward reach)
//
// Must be called level-by-level from leaves toward the root (bottom-up).
// One thread per (node_in_level, hand) pair.

extern "C" __global__ void backward_cfv(
    float* cfvalues,
    const float* strategy,
    const unsigned int* level_nodes,
    const unsigned int* child_offsets,
    const unsigned int* children_arr,
    const unsigned int* infoset_ids,
    const unsigned int* node_players,
    unsigned int traverser,
    unsigned int num_nodes_this_level,
    unsigned int num_hands,
    unsigned int max_actions
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int node_local = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (node_local >= num_nodes_this_level) return;

    unsigned int node = level_nodes[node_local];
    unsigned int infoset = infoset_ids[node];

    // Skip terminals (their cfvalues are already set)
    if (infoset == 0xFFFFFFFF) return;

    unsigned int first_child = child_offsets[node];
    unsigned int last_child = child_offsets[node + 1];
    unsigned int n_actions = last_child - first_child;

    unsigned int player = node_players[node_local];
    int is_traverser = (player == traverser);

    float node_cfv = 0.0f;
    for (unsigned int a = 0; a < n_actions; a++) {
        unsigned int child = children_arr[first_child + a];
        float child_cfv = cfvalues[child * num_hands + hand];

        if (is_traverser) {
            // Traverser's node: weight by strategy
            float action_prob = strategy[(infoset * max_actions + a) * num_hands + hand];
            node_cfv += action_prob * child_cfv;
        } else {
            // Opponent's node: just sum (strategy already in reach)
            node_cfv += child_cfv;
        }
    }

    cfvalues[node * num_hands + hand] = node_cfv;
}
