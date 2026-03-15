//! GPU-native river situation sampler.
//!
//! Generates random river boards and uniform ranges entirely on the GPU.
//! Boards are sampled via Fisher-Yates partial shuffle in a CUDA kernel;
//! ranges are set to 1.0 for unblocked combos, 0.0 for blocked combos.
//!
//! The sampler reuses pre-allocated GPU buffers across batches to avoid
//! per-batch allocation overhead.

#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

/// GPU-native river situation sampler.
///
/// Pre-allocates GPU buffers for boards and ranges, then generates random
/// situations entirely on-GPU using CUDA kernels. No CPU allocation or
/// CPU-GPU transfer happens during sampling.
#[cfg(feature = "cuda")]
pub struct GpuSampler {
    /// Pre-allocated board buffer: `[batch_size * 5]`.
    boards: CudaSlice<u32>,
    /// Pre-allocated OOP range buffer: `[batch_size * 1326]`.
    ranges_oop: CudaSlice<f32>,
    /// Pre-allocated IP range buffer: `[batch_size * 1326]`.
    ranges_ip: CudaSlice<f32>,
    /// Combo cards lookup: `[1326 * 2]` (static, uploaded once).
    combo_cards: CudaSlice<u32>,
    /// Batch size (number of situations per sample call).
    batch_size: usize,
    /// Monotonically increasing seed counter.
    seed: u64,
}

#[cfg(feature = "cuda")]
impl GpuSampler {
    /// Create a new GPU sampler with pre-allocated buffers.
    ///
    /// # Arguments
    /// * `gpu` - CUDA context.
    /// * `batch_size` - Number of situations per sample call.
    /// * `seed` - Initial random seed.
    pub fn new(gpu: &GpuContext, batch_size: usize, seed: u64) -> Result<Self, String> {
        let gpu_err = |e: GpuError| format!("GPU sampler init: {e}");

        let boards = gpu.alloc_zeros::<u32>(batch_size * 5).map_err(gpu_err)?;
        let ranges_oop = gpu.alloc_zeros::<f32>(batch_size * 1326).map_err(gpu_err)?;
        let ranges_ip = gpu.alloc_zeros::<f32>(batch_size * 1326).map_err(gpu_err)?;

        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let combo_cards = gpu.upload(&combo_cards_flat).map_err(gpu_err)?;

        Ok(Self {
            boards,
            ranges_oop,
            ranges_ip,
            combo_cards,
            batch_size,
            seed,
        })
    }

    /// Generate a new batch of random river situations entirely on the GPU.
    ///
    /// Overwrites the internal pre-allocated buffers. After this call,
    /// use `boards()`, `ranges_oop()`, and `ranges_ip()` to access the
    /// GPU-resident data.
    pub fn sample(&mut self, gpu: &GpuContext) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU sample: {e}");
        let drv_err = |e: cudarc::driver::DriverError| format!("GPU sample launch: {e}");

        let board_seed = self.seed as u32;
        self.seed += 1;
        let range_seed = self.seed as u32;
        self.seed += 1;

        // Phase 1: Sample boards
        {
            let kernel = gpu.compile_and_load(
                include_str!("../../kernels/sample_river_situations.cu"),
                "sample_boards",
            ).map_err(gpu_err)?;

            let cfg = LaunchConfig::for_num_elems(self.batch_size as u32);
            let batch_size_u32 = self.batch_size as u32;
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&mut self.boards)
                    .arg(&batch_size_u32)
                    .arg(&board_seed)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }
        }

        // Phase 2: Build ranges (blocking-aware)
        {
            let kernel = gpu.compile_and_load(
                include_str!("../../kernels/sample_river_situations.cu"),
                "build_ranges",
            ).map_err(gpu_err)?;

            let total_combos = (self.batch_size * 1326) as u32;
            let cfg = LaunchConfig::for_num_elems(total_combos);
            let batch_size_u32 = self.batch_size as u32;
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&mut self.ranges_oop)
                    .arg(&mut self.ranges_ip)
                    .arg(&self.boards)
                    .arg(&self.combo_cards)
                    .arg(&batch_size_u32)
                    .arg(&range_seed)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }
        }

        Ok(())
    }

    /// Board cards on GPU: `[batch_size * 5]`.
    pub fn boards(&self) -> &CudaSlice<u32> {
        &self.boards
    }

    /// OOP range weights on GPU: `[batch_size * 1326]`.
    pub fn ranges_oop(&self) -> &CudaSlice<f32> {
        &self.ranges_oop
    }

    /// IP range weights on GPU: `[batch_size * 1326]`.
    pub fn ranges_ip(&self) -> &CudaSlice<f32> {
        &self.ranges_ip
    }

    /// Batch size (number of situations).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// GPU-native flop situation sampler.
///
/// Samples 3-card boards (flop only) instead of 4-card (turn) or 5-card (river).
/// Otherwise identical to `GpuSampler`.
#[cfg(feature = "cuda")]
pub struct GpuFlopSampler {
    /// Pre-allocated board buffer: `[batch_size * 3]`.
    boards: CudaSlice<u32>,
    /// Pre-allocated OOP range buffer: `[batch_size * 1326]`.
    ranges_oop: CudaSlice<f32>,
    /// Pre-allocated IP range buffer: `[batch_size * 1326]`.
    ranges_ip: CudaSlice<f32>,
    /// Combo cards lookup: `[1326 * 2]` (static, uploaded once).
    combo_cards: CudaSlice<u32>,
    /// Batch size (number of situations per sample call).
    batch_size: usize,
    /// Monotonically increasing seed counter.
    seed: u64,
}

#[cfg(feature = "cuda")]
impl GpuFlopSampler {
    /// Create a new GPU flop sampler with pre-allocated buffers.
    pub fn new(gpu: &GpuContext, batch_size: usize, seed: u64) -> Result<Self, String> {
        let gpu_err = |e: GpuError| format!("GPU flop sampler init: {e}");

        let boards = gpu.alloc_zeros::<u32>(batch_size * 3).map_err(gpu_err)?;
        let ranges_oop = gpu.alloc_zeros::<f32>(batch_size * 1326).map_err(gpu_err)?;
        let ranges_ip = gpu.alloc_zeros::<f32>(batch_size * 1326).map_err(gpu_err)?;

        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let combo_cards = gpu.upload(&combo_cards_flat).map_err(gpu_err)?;

        Ok(Self {
            boards,
            ranges_oop,
            ranges_ip,
            combo_cards,
            batch_size,
            seed,
        })
    }

    /// Generate a new batch of random flop situations entirely on the GPU.
    pub fn sample(&mut self, gpu: &GpuContext) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU flop sample: {e}");
        let drv_err = |e: cudarc::driver::DriverError| format!("GPU flop sample launch: {e}");

        let board_seed = self.seed as u32;
        self.seed += 1;
        let range_seed = self.seed as u32;
        self.seed += 1;

        // Phase 1: Sample 3-card boards
        {
            let kernel = gpu.compile_and_load(
                include_str!("../../kernels/sample_flop_situations.cu"),
                "sample_flop_boards",
            ).map_err(gpu_err)?;

            let cfg = LaunchConfig::for_num_elems(self.batch_size as u32);
            let batch_size_u32 = self.batch_size as u32;
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&mut self.boards)
                    .arg(&batch_size_u32)
                    .arg(&board_seed)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }
        }

        // Phase 2: Build ranges (blocking-aware for 3-card boards)
        {
            let kernel = gpu.compile_and_load(
                include_str!("../../kernels/sample_flop_situations.cu"),
                "build_flop_ranges",
            ).map_err(gpu_err)?;

            let total_combos = (self.batch_size * 1326) as u32;
            let cfg = LaunchConfig::for_num_elems(total_combos);
            let batch_size_u32 = self.batch_size as u32;
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&mut self.ranges_oop)
                    .arg(&mut self.ranges_ip)
                    .arg(&self.boards)
                    .arg(&self.combo_cards)
                    .arg(&batch_size_u32)
                    .arg(&range_seed)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }
        }

        Ok(())
    }

    /// Board cards on GPU: `[batch_size * 3]`.
    pub fn boards(&self) -> &CudaSlice<u32> {
        &self.boards
    }

    /// OOP range weights on GPU: `[batch_size * 1326]`.
    pub fn ranges_oop(&self) -> &CudaSlice<f32> {
        &self.ranges_oop
    }

    /// IP range weights on GPU: `[batch_size * 1326]`.
    pub fn ranges_ip(&self) -> &CudaSlice<f32> {
        &self.ranges_ip
    }

    /// Batch size (number of situations).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// GPU-native turn situation sampler.
///
/// Samples 4-card boards (flop + turn) instead of 5-card (full river).
/// Otherwise identical to `GpuSampler`.
#[cfg(feature = "cuda")]
pub struct GpuTurnSampler {
    /// Pre-allocated board buffer: `[batch_size * 4]`.
    boards: CudaSlice<u32>,
    /// Pre-allocated OOP range buffer: `[batch_size * 1326]`.
    ranges_oop: CudaSlice<f32>,
    /// Pre-allocated IP range buffer: `[batch_size * 1326]`.
    ranges_ip: CudaSlice<f32>,
    /// Combo cards lookup: `[1326 * 2]` (static, uploaded once).
    combo_cards: CudaSlice<u32>,
    /// Batch size (number of situations per sample call).
    batch_size: usize,
    /// Monotonically increasing seed counter.
    seed: u64,
}

#[cfg(feature = "cuda")]
impl GpuTurnSampler {
    /// Create a new GPU turn sampler with pre-allocated buffers.
    pub fn new(gpu: &GpuContext, batch_size: usize, seed: u64) -> Result<Self, String> {
        let gpu_err = |e: GpuError| format!("GPU turn sampler init: {e}");

        let boards = gpu.alloc_zeros::<u32>(batch_size * 4).map_err(gpu_err)?;
        let ranges_oop = gpu.alloc_zeros::<f32>(batch_size * 1326).map_err(gpu_err)?;
        let ranges_ip = gpu.alloc_zeros::<f32>(batch_size * 1326).map_err(gpu_err)?;

        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let combo_cards = gpu.upload(&combo_cards_flat).map_err(gpu_err)?;

        Ok(Self {
            boards,
            ranges_oop,
            ranges_ip,
            combo_cards,
            batch_size,
            seed,
        })
    }

    /// Generate a new batch of random turn situations entirely on the GPU.
    pub fn sample(&mut self, gpu: &GpuContext) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU turn sample: {e}");
        let drv_err = |e: cudarc::driver::DriverError| format!("GPU turn sample launch: {e}");

        let board_seed = self.seed as u32;
        self.seed += 1;
        let range_seed = self.seed as u32;
        self.seed += 1;

        // Phase 1: Sample 4-card boards
        {
            let kernel = gpu.compile_and_load(
                include_str!("../../kernels/sample_turn_situations.cu"),
                "sample_turn_boards",
            ).map_err(gpu_err)?;

            let cfg = LaunchConfig::for_num_elems(self.batch_size as u32);
            let batch_size_u32 = self.batch_size as u32;
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&mut self.boards)
                    .arg(&batch_size_u32)
                    .arg(&board_seed)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }
        }

        // Phase 2: Build ranges (blocking-aware for 4-card boards)
        {
            let kernel = gpu.compile_and_load(
                include_str!("../../kernels/sample_turn_situations.cu"),
                "build_turn_ranges",
            ).map_err(gpu_err)?;

            let total_combos = (self.batch_size * 1326) as u32;
            let cfg = LaunchConfig::for_num_elems(total_combos);
            let batch_size_u32 = self.batch_size as u32;
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&mut self.ranges_oop)
                    .arg(&mut self.ranges_ip)
                    .arg(&self.boards)
                    .arg(&self.combo_cards)
                    .arg(&batch_size_u32)
                    .arg(&range_seed)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }
        }

        Ok(())
    }

    /// Board cards on GPU: `[batch_size * 4]`.
    pub fn boards(&self) -> &CudaSlice<u32> {
        &self.boards
    }

    /// OOP range weights on GPU: `[batch_size * 1326]`.
    pub fn ranges_oop(&self) -> &CudaSlice<f32> {
        &self.ranges_oop
    }

    /// IP range weights on GPU: `[batch_size * 1326]`.
    pub fn ranges_ip(&self) -> &CudaSlice<f32> {
        &self.ranges_ip
    }

    /// Batch size (number of situations).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}
