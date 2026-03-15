//! River model inference wrapper for GPU leaf evaluation at depth boundaries.
//!
//! Loads a trained river CFVNet and runs batched inference. The bridge between
//! cudarc GPU buffers and burn tensors uses a GPU→CPU→GPU bounce:
//!
//! 1. Download encoded inputs from cudarc buffer (GPU→CPU)
//! 2. Create burn tensor from CPU data (CPU→GPU on burn's device)
//! 3. Run model.forward()
//! 4. Download output tensor (GPU→CPU)
//! 5. Upload to cudarc buffer (CPU→GPU)
//!
//! This roundtrip happens once per DCFR+ iteration for the entire batch,
//! so the transfer cost is amortized across all boundaries × rivers × spots.

#[cfg(feature = "training")]
use std::path::Path;

#[cfg(feature = "training")]
use burn::module::Module;
#[cfg(feature = "training")]
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
#[cfg(feature = "training")]
use burn::tensor::backend::Backend;
#[cfg(feature = "training")]
use burn::tensor::{Tensor, TensorData};

#[cfg(feature = "training")]
use cfvnet::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};

#[cfg(feature = "training")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "training")]
use cudarc::driver::CudaSlice;

/// GPU leaf evaluator that runs a trained river CFVNet for inference.
///
/// Wraps a burn `CfvNet` model on a specified device. Provides methods to
/// run batched inference on encoded inputs stored in cudarc GPU buffers.
#[cfg(feature = "training")]
pub struct GpuLeafEvaluator<B: Backend> {
    model: CfvNet<B>,
    device: B::Device,
}

#[cfg(feature = "training")]
impl<B: Backend> GpuLeafEvaluator<B> {
    /// Create a new leaf evaluator with the given model and device.
    pub fn new(model: CfvNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }

    /// Load a trained river model from disk.
    ///
    /// `model_path` should point to the model file WITHOUT the `.mpk.gz`
    /// extension (burn's recorder adds it automatically).
    ///
    /// `hidden_layers` and `hidden_size` must match the architecture used
    /// during training.
    pub fn load(
        model_path: &Path,
        device: &B::Device,
        hidden_layers: usize,
        hidden_size: usize,
    ) -> Result<Self, String> {
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let model = CfvNet::<B>::new(device, hidden_layers, hidden_size, INPUT_SIZE)
            .load_file(model_path, &recorder, device)
            .map_err(|e| format!("Failed to load river model from {}: {e}", model_path.display()))?;

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Run batched inference on encoded inputs.
    ///
    /// Takes encoded input data as a cudarc GPU buffer, transfers it to a
    /// burn tensor, runs the forward pass, and returns the output as a new
    /// cudarc GPU buffer.
    ///
    /// # Arguments
    /// * `gpu` — cudarc GPU context for buffer transfers.
    /// * `inputs` — GPU buffer of shape `[batch_size * INPUT_SIZE]` (2720-dim encoded inputs).
    /// * `batch_size` — Number of input vectors in the batch.
    ///
    /// # Returns
    /// GPU buffer of shape `[batch_size * OUTPUT_SIZE]` (1326-dim CFV predictions).
    pub fn infer_from_cudarc(
        &self,
        gpu: &GpuContext,
        inputs: &CudaSlice<f32>,
        batch_size: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let gpu_err = |e: GpuError| format!("GPU transfer error: {e}");

        // 1. Download encoded inputs from cudarc buffer (GPU→CPU)
        let inputs_host: Vec<f32> = gpu.download(inputs).map_err(gpu_err)?;
        assert_eq!(
            inputs_host.len(),
            batch_size * INPUT_SIZE,
            "Input buffer size mismatch: expected {} got {}",
            batch_size * INPUT_SIZE,
            inputs_host.len()
        );

        // 2. Create burn tensor (CPU→GPU on burn's device)
        let data = TensorData::new(inputs_host, [batch_size, INPUT_SIZE]);
        let input_tensor = Tensor::<B, 2>::from_data(data, &self.device);

        // 3. Run forward pass
        let output_tensor = self.model.forward(input_tensor);

        // 4. Download output tensor (GPU→CPU)
        let out_data = output_tensor.into_data();
        let output_host: Vec<f32> = out_data.to_vec().expect("output tensor conversion");
        assert_eq!(
            output_host.len(),
            batch_size * OUTPUT_SIZE,
            "Output size mismatch: expected {} got {}",
            batch_size * OUTPUT_SIZE,
            output_host.len()
        );

        // 5. Upload to cudarc buffer (CPU→GPU)
        gpu.upload(&output_host).map_err(gpu_err)
    }

    /// Run batched inference directly on a burn tensor.
    ///
    /// Useful when inputs are already in burn tensor format (no cudarc bridge needed).
    pub fn infer(&self, inputs: &Tensor<B, 2>) -> Tensor<B, 2> {
        self.model.forward(inputs.clone())
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &CfvNet<B> {
        &self.model
    }

    /// Reference to the device.
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_leaf_evaluator_output_shape() {
        let device = Default::default();
        // Small model for testing (1 layer, 8 hidden)
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = GpuLeafEvaluator::new(model, device.clone());

        let batch_size = 3;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, INPUT_SIZE], &device);
        let output = evaluator.infer(&input);

        assert_eq!(output.dims(), [batch_size, OUTPUT_SIZE]);
    }

    #[test]
    fn test_leaf_evaluator_finite_output() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = GpuLeafEvaluator::new(model, device.clone());

        let batch_size = 2;
        let input = Tensor::<TestBackend, 2>::ones([batch_size, INPUT_SIZE], &device);
        let output = evaluator.infer(&input);

        let out_data = output.into_data();
        let out_vec: Vec<f32> = out_data.to_vec().expect("tensor conversion");

        for (i, &v) in out_vec.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_leaf_evaluator_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let device: <TestBackend as Backend>::Device = Default::default();

        // Create and save a model
        let model = CfvNet::<TestBackend>::new(&device, 1, 8, INPUT_SIZE);
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let model_path = dir.path().join("test_model");
        model.clone().save_file(&model_path, &recorder).unwrap();

        // Load it back
        let evaluator = GpuLeafEvaluator::<TestBackend>::load(
            &model_path,
            &device,
            1,
            8,
        ).unwrap();

        // Compare outputs
        let input = Tensor::<TestBackend, 2>::ones([1, INPUT_SIZE], &device);
        let out_orig = model.forward(input.clone());
        let out_loaded = evaluator.infer(&input);

        let diff: f32 = (out_orig - out_loaded).abs().sum().into_scalar();
        assert!(diff < 1e-6, "loaded model should match original, diff={diff}");
    }
}
