pub mod card_features;
pub mod config;
pub mod eval;
pub mod hunl_encoder;
pub mod memory;
pub mod model_buffer;
pub mod network;
pub mod solver;
pub mod traverse;

pub use config::SdCfrConfig;

/// Select the best available compute device.
///
/// Tries CUDA first (Linux), then Metal (macOS), falls back to CPU.
pub fn best_available_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    if let Ok(device) = candle_core::Device::cuda_if_available(0) {
        if device.is_cuda() {
            return device;
        }
    }
    #[cfg(feature = "metal")]
    if let Ok(device) = candle_core::Device::new_metal(0) {
        return device;
    }
    candle_core::Device::Cpu
}

/// Errors that can occur during SD-CFR training or evaluation.
#[derive(thiserror::Error, Debug)]
pub enum SdCfrError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no samples in advantage buffer")]
    EmptyBuffer,
    #[error("empty model buffer — no trained networks available")]
    EmptyModelBuffer,
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
}
