use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, ValidAsZeroBits};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// Wrapper around a CUDA device context and stream.
///
/// Provides helpers for uploading/downloading data, allocating GPU memory,
/// and compiling CUDA kernels at runtime.
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
}
