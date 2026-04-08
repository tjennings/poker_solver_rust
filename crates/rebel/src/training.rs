//! Training integration — convert ReBeL buffer records to cfvnet format.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::Path;

use cfvnet::datagen::storage::{self, TrainingRecord};

use crate::data_buffer::{BufferRecord, DiskBuffer};

/// Convert a `BufferRecord` to a cfvnet `TrainingRecord`.
///
/// Maps buffer fields to training fields:
/// - `board` is sliced to `board[..board_card_count]`
/// - `oop_reach` becomes `oop_range` (reach probs serve as range weights)
/// - `ip_reach` becomes `ip_range`
/// - `cfvs`, `pot`, `effective_stack`, `player`, `game_value`, `valid_mask` copy directly
pub fn to_training_record(rec: &BufferRecord) -> TrainingRecord {
    let board_len = rec.board_card_count as usize;

    // Normalize reach probabilities to sum to 1.0.
    // cfvnet expects probability distributions, not raw reach probs.
    let oop_range = normalize_range(&rec.oop_reach);
    let ip_range = normalize_range(&rec.ip_reach);

    // Recompute game_value with normalized ranges.
    let acting_range = if rec.player == 0 { &oop_range } else { &ip_range };
    let game_value: f32 = acting_range
        .iter()
        .zip(rec.cfvs.iter())
        .map(|(&r, &c)| r * c)
        .sum();

    TrainingRecord {
        board: rec.board[..board_len].to_vec(),
        pot: rec.pot,
        effective_stack: rec.effective_stack,
        player: rec.player,
        game_value,
        oop_range,
        ip_range,
        cfvs: rec.cfvs,
        valid_mask: rec.valid_mask,
    }
}

/// Normalize a reach probability vector to sum to 1.0.
fn normalize_range(reach: &[f32; 1326]) -> [f32; 1326] {
    let sum: f32 = reach.iter().sum();
    if sum <= 0.0 {
        return *reach;
    }
    let mut out = *reach;
    let inv = 1.0 / sum;
    for v in &mut out {
        *v *= inv;
    }
    out
}

/// Export buffer records to cfvnet `TrainingRecord` binary files.
///
/// Reads all records from the buffer and writes them to `output_path`
/// in cfvnet binary format, compatible with the cfvnet training pipeline.
///
/// Returns the number of records exported.
pub fn export_training_data(
    buffer: &DiskBuffer,
    output_path: &Path,
) -> io::Result<usize> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    let count = buffer.len();
    for i in 0..count {
        let buf_rec = buffer.read_record(i)?;
        let train_rec = to_training_record(&buf_rec);
        storage::write_record(&mut writer, &train_rec)?;
    }

    Ok(count)
}

/// Build a cfvnet `TrainConfig` from rebel training settings.
///
/// Fields not present in rebel's `TrainingConfig` are set to sensible defaults
/// matching cfvnet's own defaults.
pub fn build_train_config(
    config: &crate::config::TrainingConfig,
) -> cfvnet::model::training::TrainConfig {
    cfvnet::model::training::TrainConfig {
        hidden_layers: config.hidden_layers,
        hidden_size: config.hidden_size,
        batch_size: config.batch_size,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        lr_min: 0.00001,
        huber_delta: config.huber_delta,
        aux_loss_weight: 1.0,
        validation_split: 0.05,
        checkpoint_every_n_epochs: 1000,
        shuffle_buffer_size: 262_144,
        prefetch_depth: 4,
        encoder_threads: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .saturating_sub(2)
            .max(1),
        gpu_prefetch: 1,
        grad_clip_norm: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_buffer::BufferRecord;
    use std::io::BufReader;

    fn make_test_record(pot: f32, board_card_count: u8) -> BufferRecord {
        let mut rec = BufferRecord {
            board: [10, 20, 30, 40, 50],
            board_card_count,
            pot,
            effective_stack: 200.0,
            player: 1,
            game_value: 0.42,
            oop_reach: [0.0; 1326],
            ip_reach: [0.0; 1326],
            cfvs: [0.0; 1326],
            valid_mask: [0; 1326],
        };
        rec.oop_reach[0] = 0.8;
        rec.oop_reach[100] = 0.3;
        rec.ip_reach[0] = 0.6;
        rec.ip_reach[50] = 0.9;
        rec.cfvs[0] = 1.5;
        rec.cfvs[1] = -2.3;
        rec.valid_mask[0] = 1;
        rec.valid_mask[1] = 1;
        rec.valid_mask[100] = 1;
        rec
    }

    #[test]
    fn test_to_training_record_basic() {
        let buf_rec = make_test_record(150.0, 5);
        let train_rec = to_training_record(&buf_rec);

        // Board length matches board_card_count
        assert_eq!(train_rec.board.len(), 5);
        assert_eq!(train_rec.board, vec![10, 20, 30, 40, 50]);

        // Scalar fields match
        assert_eq!(train_rec.pot, 150.0);
        assert_eq!(train_rec.effective_stack, 200.0);
        assert_eq!(train_rec.player, 1);

        // game_value is recomputed with normalized IP range (player=1)
        // ip_reach = [0.6, 0, ..., 0.9 @50, ...], sum = 1.5
        // normalized: ip_range[0] = 0.4, ip_range[50] = 0.6
        // cfvs[0] = 1.5, cfvs[50] = 0.0
        // game_value = 0.4 * 1.5 + 0.6 * 0.0 = 0.6
        assert!((train_rec.game_value - 0.6).abs() < 1e-5,
            "game_value should be ~0.6 after normalization, got {}", train_rec.game_value);

        // CFVs match (unchanged by normalization)
        assert_eq!(train_rec.cfvs[0], 1.5);
        assert_eq!(train_rec.cfvs[1], -2.3);

        // Valid mask matches
        assert_eq!(train_rec.valid_mask[0], 1);
        assert_eq!(train_rec.valid_mask[1], 1);
        assert_eq!(train_rec.valid_mask[100], 1);
        assert_eq!(train_rec.valid_mask[2], 0);

        // Ranges are normalized (sum to 1.0)
        let oop_sum: f32 = train_rec.oop_range.iter().sum();
        assert!((oop_sum - 1.0).abs() < 1e-5, "oop_range should sum to 1.0, got {oop_sum}");
        let ip_sum: f32 = train_rec.ip_range.iter().sum();
        assert!((ip_sum - 1.0).abs() < 1e-5, "ip_range should sum to 1.0, got {ip_sum}");
        // Normalized values preserve relative proportions
        // oop: [0.8, 0, ..., 0.3 @100, ...] sum=1.1 → [0.727, 0, ..., 0.273, ...]
        assert!(train_rec.oop_range[0] > train_rec.oop_range[100]);
        assert_eq!(train_rec.oop_range[1], 0.0);
        // ip: [0.6, ..., 0.9 @50, ...] sum=1.5 → [0.4, ..., 0.6, ...]
        assert!(train_rec.ip_range[50] > train_rec.ip_range[0]);
        assert_eq!(train_rec.ip_range[1], 0.0);
    }

    #[test]
    fn test_to_training_record_short_board() {
        let buf_rec = make_test_record(100.0, 3);
        let train_rec = to_training_record(&buf_rec);

        // Board should only have 3 elements (flop)
        assert_eq!(train_rec.board.len(), 3);
        assert_eq!(train_rec.board, vec![10, 20, 30]);

        // Other fields still correct
        assert_eq!(train_rec.pot, 100.0);
        assert_eq!(train_rec.player, 1);
    }

    #[test]
    fn test_export_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let buffer_path = dir.path().join("buffer.bin");
        let export_path = dir.path().join("export.bin");

        // Create a buffer with known records
        let mut buffer = DiskBuffer::create(&buffer_path, 100).unwrap();
        let rec1 = make_test_record(100.0, 5);
        let rec2 = make_test_record(200.0, 4);
        buffer.append(&rec1).unwrap();
        buffer.append(&rec2).unwrap();
        assert_eq!(buffer.len(), 2);

        // Export
        let exported = export_training_data(&buffer, &export_path).unwrap();
        assert_eq!(exported, 2);

        // Read back with cfvnet's read_record
        let file = File::open(&export_path).unwrap();
        let mut reader = BufReader::new(file);

        let loaded1 = storage::read_record(&mut reader).unwrap();
        assert_eq!(loaded1.board, vec![10, 20, 30, 40, 50]);
        assert_eq!(loaded1.pot, 100.0);
        assert_eq!(loaded1.effective_stack, 200.0);
        assert_eq!(loaded1.player, 1);
        // game_value recomputed with normalized ranges
        assert!((loaded1.game_value - 0.6).abs() < 1e-5,
            "game_value should be ~0.6, got {}", loaded1.game_value);
        // Ranges are normalized
        let oop_sum: f32 = loaded1.oop_range.iter().sum();
        assert!((oop_sum - 1.0).abs() < 1e-5);
        let ip_sum: f32 = loaded1.ip_range.iter().sum();
        assert!((ip_sum - 1.0).abs() < 1e-5);
        assert_eq!(loaded1.cfvs[0], 1.5);
        assert_eq!(loaded1.cfvs[1], -2.3);
        assert_eq!(loaded1.valid_mask[0], 1);
        assert_eq!(loaded1.valid_mask[1], 1);
        assert_eq!(loaded1.valid_mask[100], 1);
        assert_eq!(loaded1.valid_mask[2], 0);

        let loaded2 = storage::read_record(&mut reader).unwrap();
        assert_eq!(loaded2.board, vec![10, 20, 30, 40]); // board_card_count=4
        assert_eq!(loaded2.pot, 200.0);
    }

    #[test]
    fn test_build_train_config() {
        let rebel_config = crate::config::TrainingConfig {
            hidden_layers: 5,
            hidden_size: 256,
            batch_size: 2048,
            epochs: 100,
            learning_rate: 0.001,
            huber_delta: 0.5,
        };

        let train_config = build_train_config(&rebel_config);

        // Direct mappings
        assert_eq!(train_config.hidden_layers, 5);
        assert_eq!(train_config.hidden_size, 256);
        assert_eq!(train_config.batch_size, 2048);
        assert_eq!(train_config.epochs, 100);
        assert!((train_config.learning_rate - 0.001).abs() < 1e-9);
        assert!((train_config.huber_delta - 0.5).abs() < 1e-9);

        // Defaults for fields not in rebel config
        assert!((train_config.lr_min - 0.00001).abs() < 1e-9);
        assert!((train_config.aux_loss_weight - 1.0).abs() < 1e-9);
        assert!((train_config.validation_split - 0.05).abs() < 1e-9);
        assert_eq!(train_config.checkpoint_every_n_epochs, 1000);
        assert_eq!(train_config.shuffle_buffer_size, 262_144);
        assert_eq!(train_config.prefetch_depth, 4);
        assert!(train_config.encoder_threads >= 1);
    }
}
