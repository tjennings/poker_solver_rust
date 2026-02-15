// Tiled tabular CFR: initialize reach probabilities for one player's tile.
//
// Sets reach_own[root * tile_size + lt] = 1.0 for lt in 0..tile_size.
// Called after encoder.clear_buffer() zeros the reach buffer.
//
// Dispatch: ceil(tile_size / 256) workgroups.

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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0) var<storage, read_write> reach_own: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lt = gid.x;
    if lt >= uniforms.tile_size {
        return;
    }

    // Root is always node 0. Set reach = 1.0 for this tile's trajectories.
    reach_own[lt] = 1.0;
}
