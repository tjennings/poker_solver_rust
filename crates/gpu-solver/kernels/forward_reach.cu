extern "C" __global__ void forward_reach(
    float* reach_probs,
    const float* strategy,
    const unsigned int* level_nodes,
    const unsigned int* parent_nodes,
    const unsigned int* parent_actions,
    const unsigned int* parent_infosets,
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

    float parent_reach = reach_probs[parent * num_hands + hand];
    float action_prob = strategy[infoset * max_actions + action];

    reach_probs[node * num_hands + hand] = parent_reach * action_prob;
}
