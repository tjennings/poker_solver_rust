// Tiled tabular CFR: backward terminal pass for one player direction.
//
// Computes utility via tiled matrix-vector multiply over coupling tile.
// ACCUMULATES (+=) into util_own so multiple opp tiles can be summed.
// Uses workgroup shared memory to cache opponent reach.
//
// Dispatch: (num_terminals_at_level, ceil(tile_size / 256), 1)
//   workgroup_id.x  = terminal node index within level
//   workgroup_id.y * 256 + local_id.x = own trajectory (local index)
//
// For own_player direction:
//   util_own[T][lt] += SUM_lo coupling[lt][lo] * reach_opp[T][lo]
//   where coupling depends on terminal type (fold/showdown) and player.

const SMEM_TILE: u32 = 256u;

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_own: u32,
    n_traj_opp: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    player: u32,
    tile_offset: u32,
    tile_size: u32,
    opp_tile_size: u32,
    n_decision: u32,
    _pad1: u32,
    _pad2: u32,
};

struct GpuNode {
    node_type: u32,
    position_index: u32,
    first_child: u32,
    num_children: u32,
    p1_invested_bb: f32,
    p2_invested_bb: f32,
    street: u32,
    decision_idx: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> nodes: array<GpuNode>;
@group(0) @binding(3) var<storage, read_write> level_nodes: array<u32>;

@group(2) @binding(0) var<storage, read_write> reach_opp: array<f32>;
@group(2) @binding(2) var<storage, read_write> util_own: array<f32>;

@group(3) @binding(0) var<storage, read_write> w_tile: array<f32>;
@group(3) @binding(1) var<storage, read_write> we_tile: array<f32>;
@group(3) @binding(2) var<storage, read_write> w_neg_tile: array<f32>;

var<workgroup> shared_reach: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let node_local = wid.x;
    if node_local >= uniforms.level_count {
        return;
    }
    let local_own = wid.y * SMEM_TILE + lid.x;
    if local_own >= uniforms.tile_size {
        return;
    }
    let node_idx = level_nodes[uniforms.level_start + node_local];
    let node = nodes[node_idx];

    // Skip decision nodes
    if node.node_type < 2u {
        return;
    }

    let ts = uniforms.tile_size;
    let ots = uniforms.opp_tile_size;
    let num_tiles = (ots + SMEM_TILE - 1u) / SMEM_TILE;
    var u: f32 = 0.0;

    for (var tile = 0u; tile < num_tiles; tile++) {
        // Cooperatively load opponent reach into shared memory
        let load_idx = tile * SMEM_TILE + lid.x;
        if load_idx < ots {
            shared_reach[lid.x] = reach_opp[node_idx * ots + load_idx];
        } else {
            shared_reach[lid.x] = 0.0;
        }
        workgroupBarrier();

        // Each thread accumulates its dot product
        let tile_end = min((tile + 1u) * SMEM_TILE, ots);
        for (var lo = tile * SMEM_TILE; lo < tile_end; lo++) {
            let sr = shared_reach[lo - tile * SMEM_TILE];
            let mat_idx = local_own * ots + lo;

            if uniforms.player == 0u {
                // Computing P1 utility
                if node.node_type == 2u {
                    // P1 folded: P1 loses p1_invested
                    u -= node.p1_invested_bb * w_tile[mat_idx] * sr;
                } else if node.node_type == 3u {
                    // P2 folded: P1 wins p2_invested
                    u += node.p2_invested_bb * w_tile[mat_idx] * sr;
                } else {
                    // Showdown: P1 wins p2_inv*equity, loses p1_inv*(1-equity)
                    u += node.p2_invested_bb * we_tile[mat_idx] * sr;
                    u -= node.p1_invested_bb * w_neg_tile[mat_idx] * sr;
                }
            } else {
                // Computing P2 utility
                if node.node_type == 2u {
                    // P1 folded: P2 wins p1_invested
                    u += node.p1_invested_bb * w_tile[mat_idx] * sr;
                } else if node.node_type == 3u {
                    // P2 folded: P2 loses p2_invested
                    u -= node.p2_invested_bb * w_tile[mat_idx] * sr;
                } else {
                    // Showdown: P2 wins p1_inv*(1-equity), loses p2_inv*equity
                    u += node.p1_invested_bb * w_neg_tile[mat_idx] * sr;
                    u -= node.p2_invested_bb * we_tile[mat_idx] * sr;
                }
            }
        }
        workgroupBarrier();
    }

    // Accumulate (not assign) — multiple opp tiles contribute
    util_own[node_idx * ts + local_own] += u;
}
