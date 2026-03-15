//! GPU-integrated CFVNet trainer using burn.
//!
//! Connects the GPU reservoir's mini-batches to burn's training loop.
//! Mini-batch data is transferred from cudarc buffers to burn tensors via
//! a GPU->CPU->GPU bounce (download from cudarc, create burn TensorData,
//! upload to burn's device). This overhead is bounded (~2ms per mini-batch)
//! and acceptable since the solve phase dominates wall-clock time.

#[cfg(feature = "training")]
use burn::module::{AutodiffModule, Module};
#[cfg(feature = "training")]
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
#[cfg(feature = "training")]
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
#[cfg(feature = "training")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "training")]
use burn::tensor::{Tensor, TensorData};

#[cfg(feature = "training")]
use cfvnet::model::loss::cfvnet_loss;
#[cfg(feature = "training")]
use cfvnet::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};

#[cfg(feature = "training")]
use super::reservoir::GpuMiniBatch;
#[cfg(feature = "training")]
use crate::gpu::GpuContext;

/// Internal trait to erase the concrete optimizer type (which uses
/// burn's private `OptimizerAdaptor`).
#[cfg(feature = "training")]
trait GpuTrainerOptim<B: AutodiffBackend>: Send {
    fn step(&mut self, lr: f64, model: CfvNet<B>, grads: GradientsParams) -> CfvNet<B>;
}

/// Blanket impl: any burn Optimizer that works on CfvNet implements our trait.
#[cfg(feature = "training")]
impl<B: AutodiffBackend, O: Optimizer<CfvNet<B>, B> + Send> GpuTrainerOptim<B> for O {
    fn step(&mut self, lr: f64, model: CfvNet<B>, grads: GradientsParams) -> CfvNet<B> {
        Optimizer::step(self, lr, model, grads)
    }
}

/// GPU-integrated CFVNet trainer.
///
/// Wraps a `CfvNet` model, optimizer, and device. Accepts `GpuMiniBatch`
/// data from the reservoir and performs forward/backward/step on the burn
/// backend.
#[cfg(feature = "training")]
pub struct GpuTrainer<B: AutodiffBackend> {
    model: CfvNet<B>,
    optim: Box<dyn GpuTrainerOptim<B>>,
    device: B::Device,
    learning_rate: f64,
    huber_delta: f64,
    aux_loss_weight: f64,
    train_step_count: u64,
}

#[cfg(feature = "training")]
impl<B: AutodiffBackend> GpuTrainer<B> {
    /// Create a new trainer with a fresh `CfvNet` model.
    pub fn new(
        device: &B::Device,
        hidden_layers: usize,
        hidden_size: usize,
        learning_rate: f64,
        huber_delta: f64,
        aux_loss_weight: f64,
    ) -> Self {
        let model = CfvNet::<B>::new(device, hidden_layers, hidden_size, INPUT_SIZE);
        let optim = AdamConfig::new().init::<B, CfvNet<B>>();
        Self {
            model,
            optim: Box::new(optim),
            device: device.clone(),
            learning_rate,
            huber_delta,
            aux_loss_weight,
            train_step_count: 0,
        }
    }

    /// Number of training steps performed so far.
    pub fn train_step_count(&self) -> u64 {
        self.train_step_count
    }

    /// Perform one training step using a mini-batch from the GPU reservoir.
    ///
    /// Downloads the mini-batch from cudarc GPU buffers, creates burn tensors
    /// on the burn device, runs forward + loss + backward + optimizer step.
    /// Returns the scalar loss value.
    pub fn train_step(&mut self, gpu: &GpuContext, batch: &GpuMiniBatch) -> Result<f32, String> {
        let gpu_err = |e: crate::gpu::GpuError| format!("GPU download error: {e}");

        let bs = batch.batch_size;

        // Download mini-batch from cudarc buffers (GPU->CPU)
        let inputs_host = gpu.download(&batch.inputs).map_err(gpu_err)?;
        let targets_host = gpu.download(&batch.targets).map_err(gpu_err)?;
        let masks_host = gpu.download(&batch.masks).map_err(gpu_err)?;
        let ranges_host = gpu.download(&batch.ranges).map_err(gpu_err)?;
        let game_values_host = gpu.download(&batch.game_values).map_err(gpu_err)?;

        // Create burn tensors on the inner (non-autodiff) backend, then lift
        let input_inner = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(inputs_host, [bs, INPUT_SIZE]),
            &self.device,
        );
        let target_inner = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(targets_host, [bs, OUTPUT_SIZE]),
            &self.device,
        );
        let mask_inner = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(masks_host, [bs, OUTPUT_SIZE]),
            &self.device,
        );
        let range_inner = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(ranges_host, [bs, OUTPUT_SIZE]),
            &self.device,
        );
        let game_value_inner = Tensor::<B::InnerBackend, 1>::from_data(
            TensorData::new(game_values_host, [bs]),
            &self.device,
        );

        // Lift to autodiff backend
        let input = Tensor::<B, 2>::from_inner(input_inner);
        let target = Tensor::<B, 2>::from_inner(target_inner);
        let mask = Tensor::<B, 2>::from_inner(mask_inner);
        let range = Tensor::<B, 2>::from_inner(range_inner);
        let game_value = Tensor::<B, 1>::from_inner(game_value_inner);

        // Forward pass
        let pred = self.model.forward(input);

        // Compute loss
        let loss = cfvnet_loss(
            pred,
            target,
            mask,
            range,
            game_value,
            self.huber_delta,
            self.aux_loss_weight,
        );

        // Read loss value
        let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

        // Backward + optimizer step
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        // Temporarily take ownership of the model for the optimizer step,
        // then put the updated model back.
        let model = std::mem::replace(
            &mut self.model,
            CfvNet::<B>::new(&self.device, 1, 1, INPUT_SIZE), // placeholder
        );
        self.model = self.optim.step(self.learning_rate, model, grads_params);

        self.train_step_count += 1;
        Ok(loss_val)
    }

    /// Compute validation loss on a mini-batch WITHOUT updating model weights.
    ///
    /// Uses the inner (non-autodiff) backend for inference only.
    pub fn validation_loss(&self, gpu: &GpuContext, batch: &GpuMiniBatch) -> Result<f32, String> {
        let gpu_err = |e: crate::gpu::GpuError| format!("GPU download error: {e}");

        let bs = batch.batch_size;

        let inputs_host = gpu.download(&batch.inputs).map_err(gpu_err)?;
        let targets_host = gpu.download(&batch.targets).map_err(gpu_err)?;
        let masks_host = gpu.download(&batch.masks).map_err(gpu_err)?;
        let ranges_host = gpu.download(&batch.ranges).map_err(gpu_err)?;
        let game_values_host = gpu.download(&batch.game_values).map_err(gpu_err)?;

        let valid_model = self.model.valid();

        let input = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(inputs_host, [bs, INPUT_SIZE]),
            &self.device,
        );
        let target = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(targets_host, [bs, OUTPUT_SIZE]),
            &self.device,
        );
        let mask = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(masks_host, [bs, OUTPUT_SIZE]),
            &self.device,
        );
        let range = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(ranges_host, [bs, OUTPUT_SIZE]),
            &self.device,
        );
        let game_value = Tensor::<B::InnerBackend, 1>::from_data(
            TensorData::new(game_values_host, [bs]),
            &self.device,
        );

        let pred = valid_model.forward(input);
        let loss = cfvnet_loss(pred, target, mask, range, game_value, self.huber_delta, self.aux_loss_weight);
        let val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        Ok(val)
    }

    /// Save a checkpoint to `dir/checkpoint_N.mpk.gz`.
    pub fn save_checkpoint(&self, dir: &std::path::Path, label: &str) {
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let path = dir.join(label);
        if let Err(e) = self.model.clone().save_file(path, &recorder) {
            eprintln!("Warning: failed to save checkpoint '{}': {}", label, e);
        }
    }

    /// Save the final model to `dir/model.mpk.gz`.
    pub fn save_final(&self, dir: &std::path::Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let path = dir.join("model");
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| format!("save model: {e}"))?;
        Ok(())
    }

    /// Reference to the inner model (for inference in validation).
    pub fn model(&self) -> &CfvNet<B> {
        &self.model
    }
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    type TestB = Autodiff<NdArray>;

    /// Create a trainer and run a few steps with synthetic data to verify
    /// loss is finite and decreasing.
    #[test]
    fn test_trainer_loss_decreases_synthetic() {
        let device = Default::default();
        let mut trainer = GpuTrainer::<TestB>::new(&device, 2, 64, 0.001, 1.0, 0.0);

        // Create synthetic training data directly as burn tensors
        // (No GPU context needed for NdArray backend)
        let bs = 16;
        let mut losses = Vec::new();

        for step in 0..20 {
            // Create deterministic synthetic data
            let inputs = vec![0.1f32 * (step as f32 + 1.0); bs * INPUT_SIZE];
            let targets = vec![0.01f32; bs * OUTPUT_SIZE];
            let masks = vec![1.0f32; bs * OUTPUT_SIZE];
            let ranges = vec![0.5f32; bs * OUTPUT_SIZE];
            let game_values = vec![0.0f32; bs];

            // Create tensors on inner backend and lift
            let input_inner = Tensor::<NdArray, 2>::from_data(
                TensorData::new(inputs, [bs, INPUT_SIZE]),
                &device,
            );
            let target_inner = Tensor::<NdArray, 2>::from_data(
                TensorData::new(targets, [bs, OUTPUT_SIZE]),
                &device,
            );
            let mask_inner = Tensor::<NdArray, 2>::from_data(
                TensorData::new(masks, [bs, OUTPUT_SIZE]),
                &device,
            );
            let range_inner = Tensor::<NdArray, 2>::from_data(
                TensorData::new(ranges, [bs, OUTPUT_SIZE]),
                &device,
            );
            let gv_inner = Tensor::<NdArray, 1>::from_data(
                TensorData::new(game_values, [bs]),
                &device,
            );

            let input = Tensor::<TestB, 2>::from_inner(input_inner);
            let target = Tensor::<TestB, 2>::from_inner(target_inner);
            let mask = Tensor::<TestB, 2>::from_inner(mask_inner);
            let range = Tensor::<TestB, 2>::from_inner(range_inner);
            let game_value = Tensor::<TestB, 1>::from_inner(gv_inner);

            let pred = trainer.model.forward(input);
            let loss = cfvnet_loss(pred, target, mask, range, game_value, 1.0, 0.0);
            let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            losses.push(loss_val);

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &trainer.model);
            trainer.model = trainer.optim.step(0.001, trainer.model, grads_params);

            assert!(loss_val.is_finite(), "loss should be finite at step {step}");
        }

        // Loss should decrease over training
        let first_loss = losses[0];
        let last_loss = losses[losses.len() - 1];
        assert!(
            last_loss < first_loss,
            "loss should decrease: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_trainer_save_load() {
        let device = Default::default();
        let trainer = GpuTrainer::<TestB>::new(&device, 2, 64, 0.001, 1.0, 0.0);

        let dir = tempfile::tempdir().unwrap();
        trainer.save_final(dir.path()).unwrap();

        assert!(
            dir.path().join("model.mpk.gz").exists(),
            "model file should exist"
        );
    }
}
