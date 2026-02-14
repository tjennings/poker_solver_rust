pub mod card_features;
pub mod config;
pub mod eval;
pub mod memory;
pub mod model_buffer;
pub mod network;
pub mod solver;
pub mod traverse;

pub use config::SdCfrConfig;

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
