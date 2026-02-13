// Tabular CFR: backward pass for terminal nodes with shared-memory tiling.
//
// Computes utilities via tiled matrix-vector multiply over coupling matrices.
// Uses workgroup shared memory to cache opponent reach vectors.
//
// Dispatch: (num_terminals_at_level, ceil(max_traj / 256), 1)
//   workgroup_id.x  = terminal node index within level
//   workgroup_id.y * 256 + local_id.x = trajectory ID
//
// P1 utility (traj_id < n_traj_p1):
//   util_p1[T][t1] = Σ_t2 M[t1][t2] * reach_p2[T][t2]
//   where M depends on terminal type (fold/showdown)
//
// P2 utility (traj_id < n_traj_p2):
//   Uses transposed matrices, sums over P1 trajectories.

const TILE_SIZE: u32 = 256u;

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

@group(2) @binding(0) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(1) var<storage, read_write> reach_p2: array<f32>;
@group(2) @binding(2) var<storage, read_write> util_p1: array<f32>;
@group(2) @binding(3) var<storage, read_write> util_p2: array<f32>;

@group(3) @binding(0) var<storage, read_write> w_mat: array<f32>;
@group(3) @binding(1) var<storage, read_write> we_mat: array<f32>;
@group(3) @binding(2) var<storage, read_write> w_neg_mat: array<f32>;
@group(3) @binding(3) var<storage, read_write> w_mat_t: array<f32>;
@group(3) @binding(4) var<storage, read_write> we_mat_t: array<f32>;
@group(3) @binding(5) var<storage, read_write> w_neg_mat_t: array<f32>;

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
    let traj_id = wid.y * TILE_SIZE + lid.x;
    let node_idx = level_nodes[uniforms.level_start + node_local];
    let node = nodes[node_idx];

    if node.node_type < 2u {
        return;
    }

    let n1 = uniforms.n_traj_p1;
    let n2 = uniforms.n_traj_p2;

    // --- P1 utility: tiled dot product over P2 trajectories ---
    if traj_id < n1 {
        var u1: f32 = 0.0;
        let num_tiles = (n2 + TILE_SIZE - 1u) / TILE_SIZE;

        for (var tile = 0u; tile < num_tiles; tile++) {
            // Cooperatively load P2 reach into shared memory
            let load_idx = tile * TILE_SIZE + lid.x;
            if load_idx < n2 {
                shared_reach[lid.x] = reach_p2[node_idx * n2 + load_idx];
            } else {
                shared_reach[lid.x] = 0.0;
            }
            workgroupBarrier();

            // Each thread accumulates its dot product using shared reach
            let tile_end = min((tile + 1u) * TILE_SIZE, n2);
            for (var t2 = tile * TILE_SIZE; t2 < tile_end; t2++) {
                let sr = shared_reach[t2 - tile * TILE_SIZE];
                if node.node_type == 2u {
                    u1 -= node.p1_invested_bb * w_mat[traj_id * n2 + t2] * sr;
                } else if node.node_type == 3u {
                    u1 += node.p2_invested_bb * w_mat[traj_id * n2 + t2] * sr;
                } else {
                    u1 += node.p2_invested_bb * we_mat[traj_id * n2 + t2] * sr;
                    u1 -= node.p1_invested_bb * w_neg_mat[traj_id * n2 + t2] * sr;
                }
            }
            workgroupBarrier();
        }

        util_p1[node_idx * n1 + traj_id] = u1;
    }

    // --- P2 utility: tiled dot product over P1 trajectories ---
    if traj_id < n2 {
        var u2: f32 = 0.0;
        let num_tiles_p1 = (n1 + TILE_SIZE - 1u) / TILE_SIZE;

        for (var tile = 0u; tile < num_tiles_p1; tile++) {
            let load_idx = tile * TILE_SIZE + lid.x;
            if load_idx < n1 {
                shared_reach[lid.x] = reach_p1[node_idx * n1 + load_idx];
            } else {
                shared_reach[lid.x] = 0.0;
            }
            workgroupBarrier();

            let tile_end = min((tile + 1u) * TILE_SIZE, n1);
            for (var t1 = tile * TILE_SIZE; t1 < tile_end; t1++) {
                let sr = shared_reach[t1 - tile * TILE_SIZE];
                if node.node_type == 2u {
                    u2 += node.p1_invested_bb * w_mat_t[traj_id * n1 + t1] * sr;
                } else if node.node_type == 3u {
                    u2 -= node.p2_invested_bb * w_mat_t[traj_id * n1 + t1] * sr;
                } else {
                    u2 += node.p1_invested_bb * w_neg_mat_t[traj_id * n1 + t1] * sr;
                    u2 -= node.p2_invested_bb * we_mat_t[traj_id * n1 + t1] * sr;
                }
            }
            workgroupBarrier();
        }

        util_p2[node_idx * n2 + traj_id] = u2;
    }
}
