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
}
