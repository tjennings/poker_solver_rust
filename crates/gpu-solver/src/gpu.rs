use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// Wrapper around a CUDA device context and stream.
///
/// Provides helpers for uploading/downloading data, allocating GPU memory,
/// compiling CUDA kernels at runtime, and launching solver-specific kernels
/// (regret matching, forward reach propagation).
pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}

impl GpuContext {
    /// Create a new GPU context on the given device ordinal (typically 0).
    pub fn new(device_ordinal: usize) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
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

    /// Compile a CUDA source string to PTX, load it as a module, and return
    /// the named function.
    pub fn compile_and_load(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<CudaFunction, GpuError> {
        let ptx = compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        Ok(module.load_function(function_name)?)
    }

    /// Launch the regret-matching kernel.
    ///
    /// For each information set, converts cumulative regrets into a current
    /// strategy via regret matching (positive-regret normalization, with
    /// uniform fallback when all regrets are non-positive).
    ///
    /// # Layout
    /// - `regrets`: `[num_infosets * max_actions]` flat array, row-major
    /// - `num_actions`: `[num_infosets]` number of valid actions per infoset
    /// - `strategy`: `[num_infosets * max_actions]` output, same layout
    pub fn launch_regret_match(
        &self,
        regrets: &CudaSlice<f32>,
        num_actions: &CudaSlice<u32>,
        strategy: &mut CudaSlice<f32>,
        num_infosets: u32,
        max_actions: u32,
    ) -> Result<(), GpuError> {
        let kernel = self.compile_and_load(
            include_str!("../kernels/regret_match.cu"),
            "regret_match",
        )?;
        let cfg = LaunchConfig::for_num_elems(num_infosets);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(regrets)
                .arg(num_actions)
                .arg(strategy)
                .arg(&num_infosets)
                .arg(&max_actions)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Launch the forward-reach propagation kernel.
    ///
    /// For each node in the current BFS level, computes:
    ///   `reach[node][hand] = reach[parent][hand] * strategy[parent_infoset][action]`
    ///
    /// This is parallelized over `num_nodes_this_level * num_hands` threads.
    ///
    /// # Layout
    /// - `reach_probs`: `[num_total_nodes * num_hands]` flat array
    /// - `strategy`: `[num_infosets * max_actions]` flat array
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

        // Infoset 0: regrets [10, -5, 20] -> strategy [10/30, 0, 20/30] = [0.333, 0, 0.667]
        // Infoset 1: regrets [-1, -2, -3] -> all negative -> uniform [0.333, 0.333, 0.333]
        let regrets: Vec<f32> = vec![10.0, -5.0, 20.0, -1.0, -2.0, -3.0];
        let num_actions: Vec<u32> = vec![3, 3];

        let gpu_regrets = gpu.upload(&regrets).unwrap();
        let gpu_num_actions = gpu.upload(&num_actions).unwrap();
        let mut gpu_strategy = gpu.alloc_zeros::<f32>(6).unwrap();

        gpu.launch_regret_match(
            &gpu_regrets,
            &gpu_num_actions,
            &mut gpu_strategy,
            num_infosets,
            max_actions,
        )
        .unwrap();

        let strategy = gpu.download(&gpu_strategy).unwrap();

        let eps = 1e-5;
        assert!((strategy[0] - 1.0 / 3.0).abs() < eps);
        assert!((strategy[1] - 0.0).abs() < eps);
        assert!((strategy[2] - 2.0 / 3.0).abs() < eps);
        assert!((strategy[3] - 1.0 / 3.0).abs() < eps);
        assert!((strategy[4] - 1.0 / 3.0).abs() < eps);
        assert!((strategy[5] - 1.0 / 3.0).abs() < eps);
    }

    #[test]
    fn test_forward_reach_kernel() {
        let gpu = GpuContext::new(0).unwrap();

        // Tree: Root(0) -> [Child(1), Child(2)]
        // Root is infoset 0 with strategy [0.6, 0.4]
        // 2 hands
        // Initial reach for hand 0: 1.0, hand 1: 0.5
        let num_hands: u32 = 2;
        let num_nodes_this_level: u32 = 2;

        // reach_probs layout: [num_total_nodes * num_hands]
        // We need at least 3 nodes (root + 2 children)
        let reach = vec![
            1.0f32, 0.5, // node 0 (root): hand 0 = 1.0, hand 1 = 0.5
            0.0, 0.0, // node 1 (child): to be filled
            0.0, 0.0, // node 2 (child): to be filled
        ];
        let strategy = vec![0.6f32, 0.4]; // infoset 0: [action0=0.6, action1=0.4]

        let level_nodes = vec![1u32, 2]; // nodes at this level
        let parent_nodes = vec![0u32, 0]; // both children's parent is node 0
        let parent_actions = vec![0u32, 1]; // child 1 via action 0, child 2 via action 1
        let parent_infosets = vec![0u32, 0]; // parent's infoset for both

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
        // Child 1 (action 0): reach = parent_reach * 0.6 = [0.6, 0.3]
        assert!(
            (result[2] - 0.6).abs() < eps,
            "child1 hand0: got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.3).abs() < eps,
            "child1 hand1: got {}",
            result[3]
        );
        // Child 2 (action 1): reach = parent_reach * 0.4 = [0.4, 0.2]
        assert!(
            (result[4] - 0.4).abs() < eps,
            "child2 hand0: got {}",
            result[4]
        );
        assert!(
            (result[5] - 0.2).abs() < eps,
            "child2 hand1: got {}",
            result[5]
        );
    }
}
