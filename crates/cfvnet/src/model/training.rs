use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::model::dataset::{CfvDataset, CfvItem};
use crate::model::loss::cfvnet_loss;
use crate::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};

/// Configuration for the CFVnet training loop.
pub struct TrainConfig {
    pub hidden_layers: usize,
    pub hidden_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub lr_min: f64,
    pub huber_delta: f64,
    pub aux_loss_weight: f64,
    pub validation_split: f64,
    pub checkpoint_every_n_epochs: usize,
}

/// Result returned after training completes.
pub struct TrainResult {
    pub final_train_loss: f32,
}

/// All tensors needed for one training batch.
struct Batch<B: AutodiffBackend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
}

/// Collate a batch of `CfvItem`s into stacked tensors for the forward pass.
fn collate_batch<B: AutodiffBackend>(
    items: &[CfvItem],
    device: &B::Device,
) -> Batch<B> {
    let bs = items.len();

    let mut input_data = Vec::with_capacity(bs * INPUT_SIZE);
    let mut target_data = Vec::with_capacity(bs * OUTPUT_SIZE);
    let mut mask_data = Vec::with_capacity(bs * OUTPUT_SIZE);
    let mut range_data = Vec::with_capacity(bs * OUTPUT_SIZE);
    let mut gv_data = Vec::with_capacity(bs);

    for item in items {
        input_data.extend_from_slice(&item.input);
        target_data.extend_from_slice(&item.target);
        mask_data.extend_from_slice(&item.mask);
        range_data.extend_from_slice(&item.range);
        gv_data.push(item.game_value);
    }

    let input = Tensor::from_data(
        TensorData::new(input_data, [bs, INPUT_SIZE]),
        device,
    );
    let target = Tensor::from_data(
        TensorData::new(target_data, [bs, OUTPUT_SIZE]),
        device,
    );
    let mask = Tensor::from_data(
        TensorData::new(mask_data, [bs, OUTPUT_SIZE]),
        device,
    );
    let range_t = Tensor::from_data(
        TensorData::new(range_data, [bs, OUTPUT_SIZE]),
        device,
    );
    let game_value = Tensor::from_data(
        TensorData::new(gv_data, [bs]),
        device,
    );

    Batch { input, target, mask, range: range_t, game_value }
}

/// Train a `CfvNet` on the given dataset using a custom Adam training loop.
///
/// Returns the final training loss. Shuffles indices each epoch, processes
/// mini-batches, and updates the model via backpropagation.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    dataset: &CfvDataset,
    config: &TrainConfig,
    _output_dir: Option<&std::path::Path>,
) -> TrainResult {
    let mut model = CfvNet::<B>::new(device, config.hidden_layers, config.hidden_size);

    let mut optim = AdamConfig::new().init::<B, CfvNet<B>>();

    let n = dataset.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut final_loss = f32::MAX;

    for _epoch in 0..config.epochs {
        indices.shuffle(&mut rng);

        for batch_start in (0..n).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(n);
            let batch_indices = &indices[batch_start..batch_end];

            let items: Vec<CfvItem> = batch_indices
                .iter()
                .filter_map(|&idx| dataset.get(idx))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = collate_batch::<B>(&items, device);

            let pred = model.forward(batch.input);

            let loss = cfvnet_loss(
                pred,
                batch.target,
                batch.mask,
                batch.range,
                batch.game_value,
                config.huber_delta,
                config.aux_loss_weight,
            );

            // Read scalar loss before consuming the tensor for backward.
            // INVARIANT: loss is shape [1], so to_vec always has exactly one element.
            final_loss = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads_params);
        }
    }

    TrainResult {
        final_train_loss: final_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use std::io::Write;
    use tempfile::NamedTempFile;

    use crate::datagen::storage::{write_record, TrainingRecord};

    type B = Autodiff<NdArray>;

    fn write_test_data(n: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..n {
            let mut rec = TrainingRecord {
                board: [0, 4, 8, 12, 16],
                pot: 100.0,
                effective_stack: 50.0,
                player: (i % 2) as u8,
                game_value: 0.1 * i as f32,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            // Set some non-zero values so the target is not all zeros.
            for j in 0..10 {
                rec.cfvs[j] = (i as f32 + j as f32) * 0.01;
                rec.oop_range[j] = 0.1;
                rec.ip_range[j] = 0.1;
            }
            write_record(&mut file, &rec).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn overfit_single_batch() {
        let file = write_test_data(16);
        let dataset = CfvDataset::from_file(file.path()).unwrap();

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,
            hidden_size: 64,
            batch_size: 16,
            epochs: 200,
            learning_rate: 0.001,
            lr_min: 0.001,
            huber_delta: 1.0,
            aux_loss_weight: 0.0,
            validation_split: 0.0,
            checkpoint_every_n_epochs: 0,
        };

        let result = train::<B>(&device, &dataset, &config, None);
        assert!(
            result.final_train_loss < 0.01,
            "should overfit small data, got loss {}",
            result.final_train_loss
        );
    }
}
