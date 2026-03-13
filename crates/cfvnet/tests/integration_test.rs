//! Full pipeline integration test: generate -> train -> evaluate

use tempfile::TempDir;

#[test]
fn full_pipeline_smoke_test() {
    let tmp = TempDir::new().unwrap();
    let data_path = tmp.path().join("training.bin");
    let model_path = tmp.path().join("model");

    // 1. Generate training data (4 situations = up to 8 records)
    let config = cfvnet::config::CfvnetConfig {
        game: cfvnet::config::GameConfig {
            initial_stack: 200,
            bet_sizes: vec!["50%".into(), "a".into()],
            ..Default::default()
        },
        datagen: cfvnet::config::DatagenConfig {
            num_samples: 4,
            solver_iterations: 100,
            target_exploitability: 0.05,
            threads: 1,
            seed: 42,
            ..Default::default()
        },
        training: Default::default(),
        evaluation: Default::default(),
    };

    cfvnet::datagen::generate::generate_training_data(&config, &data_path)
        .expect("data generation failed");
    assert!(data_path.exists(), "training data file should exist");

    // Verify record count
    let mut file = std::fs::File::open(&data_path).unwrap();
    let count = cfvnet::datagen::storage::count_records(&mut file, 5).unwrap();
    // Some situations with effective_stack=0 may be skipped, so count may be < 8
    assert!(count >= 2, "should have at least 2 records, got {count}");

    // 2. Train a tiny model
    use burn::backend::{Autodiff, NdArray};
    type B = Autodiff<NdArray>;
    let device = Default::default();

    let train_config = cfvnet::model::training::TrainConfig {
        hidden_layers: 2,
        hidden_size: 32,
        batch_size: 8,
        epochs: 10,
        learning_rate: 0.001,
        lr_min: 0.001,
        huber_delta: 1.0,
        aux_loss_weight: 0.0,
        validation_split: 0.0,
        checkpoint_every_n_epochs: 0,
        gpu_chunk_size: 100,
        epochs_per_chunk: 1,
        prefetch_chunks: 1,
        pot_weighted_loss: false,
    };

    let result = cfvnet::model::training::train::<B>(
        &device,
        &data_path,
        5,
        &train_config,
        Some(model_path.as_path()),
    );
    assert!(
        result.final_train_loss.is_finite(),
        "training loss should be finite"
    );
    println!(
        "Integration test: final train loss = {}",
        result.final_train_loss
    );
}
