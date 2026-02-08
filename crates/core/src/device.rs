//! Device abstraction for selecting compute backends.
//!
//! This module provides a unified way to select between different Burn backends:
//! - CPU (ndarray) - Always available, good for debugging
//! - WGPU - GPU acceleration via WebGPU/Metal/Vulkan

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::wgpu::{Wgpu, WgpuDevice};

/// Available compute devices for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// CPU backend using ndarray
    #[default]
    Cpu,
    /// GPU backend using WGPU (Metal on macOS, Vulkan on Linux/Windows)
    Wgpu,
}

impl Device {
    /// Auto-detect the best available device.
    ///
    /// Priority: WGPU → CPU
    #[must_use]
    pub fn auto() -> Self {
        // For now, default to CPU as it's most reliable
        // In production, would check for GPU availability
        Self::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU (ndarray)"),
            Self::Wgpu => write!(f, "GPU (WGPU)"),
        }
    }
}

/// Trait for types that can be converted to a Burn device.
pub trait IntoBurnDevice<B: burn::tensor::backend::Backend> {
    /// Convert to the backend's device type.
    fn into_burn_device(self) -> B::Device;
}

impl IntoBurnDevice<NdArray> for Device {
    fn into_burn_device(self) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

impl IntoBurnDevice<Wgpu> for Device {
    fn into_burn_device(self) -> WgpuDevice {
        WgpuDevice::default()
    }
}

/// Type alias for the CPU backend.
pub type CpuBackend = NdArray;

/// Type alias for the GPU backend.
pub type GpuBackend = Wgpu;

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn device_display() {
        assert_eq!(format!("{}", Device::Cpu), "CPU (ndarray)");
        assert_eq!(format!("{}", Device::Wgpu), "GPU (WGPU)");
    }

    #[timed_test]
    fn device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[timed_test]
    fn device_auto_returns_cpu() {
        // For now, auto returns CPU as the safe default
        assert_eq!(Device::auto(), Device::Cpu);
    }

    #[timed_test]
    fn device_equality() {
        assert_eq!(Device::Cpu, Device::Cpu);
        assert_eq!(Device::Wgpu, Device::Wgpu);
        assert_ne!(Device::Cpu, Device::Wgpu);
    }
}
