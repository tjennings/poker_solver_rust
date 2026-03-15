// Combined forward reach propagation kernel for both players.
//
// For each node at the current BFS level, propagates reach for both OOP
// and IP. The acting player's reach is multiplied by the action probability
// from their strategy. The non-acting player's reach is copied from parent.
//
// One thread per (node_in_level, hand) pair.
//
// Layout:
//   reach_oop[node * num_hands + hand]
//   reach_ip[node * num_hands + hand]
//   strategy[(infoset * max_actions + action) * num_hands + hand]

extern "C" __global__ void forward_pass(
    float* reach_oop,
    float* reach_ip,
    const float* strategy,
    const unsigned int* level_nodes,
    const unsigned int* parent_nodes,
    const unsigned int* parent_actions,
    const unsigned int* parent_infosets,
    const unsigned int* parent_players,
    unsigned int num_nodes_this_level,
    unsigned int num_hands,
    unsigned int max_actions
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int node_local = tid / num_hands;
    unsigned int hand = tid % num_hands;

    if (node_local >= num_nodes_this_level) return;

    unsigned int node = level_nodes[node_local];
    unsigned int parent = parent_nodes[node_local];
    unsigned int action = parent_actions[node_local];
    unsigned int infoset = parent_infosets[node_local];
    unsigned int player = parent_players[node_local];

    float action_prob = strategy[(infoset * max_actions + action) * num_hands + hand];

    if (player == 0) {
        // Parent is OOP decision: OOP reach gets strategy-weighted, IP copied
        reach_oop[node * num_hands + hand] = reach_oop[parent * num_hands + hand] * action_prob;
        reach_ip[node * num_hands + hand] = reach_ip[parent * num_hands + hand];
    } else {
        // Parent is IP decision: IP reach gets strategy-weighted, OOP copied
        reach_ip[node * num_hands + hand] = reach_ip[parent * num_hands + hand] * action_prob;
        reach_oop[node * num_hands + hand] = reach_oop[parent * num_hands + hand];
    }
}
