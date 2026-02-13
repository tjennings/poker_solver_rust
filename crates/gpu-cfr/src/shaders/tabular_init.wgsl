// Tabular CFR: initialize reach probabilities.
//
// Called after encoder.clear_buffer() zeros all reach/utility buffers.
// Sets reach_p1[root * n_traj_p1 + t] = 1.0 for all P1 trajectories,
// and reach_p2[root * n_traj_p2 + t] = 1.0 for all P2 trajectories.
//
// Dispatch: ceil(max_traj / 256) workgroups.

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_p1: u32,
    n_traj_p2: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(1) var<storage, read_write> reach_p2: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;

    // Root is always node 0. Set reach = 1.0 for all trajectories.
    if t < uniforms.n_traj_p1 {
        reach_p1[t] = 1.0;
    }
    if t < uniforms.n_traj_p2 {
        reach_p2[t] = 1.0;
    }
}
