//! SD-CFR training loop: alternating updates with neural network advantage estimation.
//!
//! Each iteration trains one player at a time:
//! 1. Run K traversals to collect advantage samples into the reservoir buffer
//! 2. Train a fresh value network from the accumulated samples
//! 3. Store the trained network in the model buffer for later averaging

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use poker_solver_core::game::{Game, Player};

use crate::SdCfrError;
use crate::config::SdCfrConfig;
use crate::memory::ReservoirBuffer;
use crate::model_buffer::ModelBuffer;
use crate::network::AdvantageNet;
use crate::traverse::{AdvantageSample, StateEncoder, traverse};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The fully-trained output of an SD-CFR run: model buffers for both players.
pub struct TrainedSdCfr {
    pub model_buffers: [ModelBuffer; 2],
    pub config: SdCfrConfig,
}

/// SD-CFR training loop coordinator.
///
/// Manages per-player advantage buffers, value networks, and model buffers.
/// Each call to [`step`](Self::step) runs one full CFR iteration (both players).
pub struct SdCfrSolver<G: Game, E: StateEncoder<G::State>> {
    game: G,
    encoder: E,
    config: SdCfrConfig,
    advantage_buffers: [ReservoirBuffer<AdvantageSample>; 2],
    model_buffers: [ModelBuffer; 2],
    value_nets: [Option<(AdvantageNet, VarMap)>; 2],
    current_iteration: u32,
    device: Device,
    rng: StdRng,
}

impl<G: Game, E: StateEncoder<G::State>> SdCfrSolver<G, E> {
    /// Create a new solver, validating the config.
    pub fn new(game: G, encoder: E, config: SdCfrConfig) -> Result<Self, SdCfrError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        let cap = config.advantage_memory_cap;
        Ok(Self {
            game,
            encoder,
            config,
            advantage_buffers: [ReservoirBuffer::new(cap), ReservoirBuffer::new(cap)],
            model_buffers: [ModelBuffer::new(), ModelBuffer::new()],
            value_nets: [None, None],
            current_iteration: 0,
            device: Device::Cpu,
            rng,
        })
    }

    /// Run one full CFR iteration: update both players sequentially.
    pub fn step(&mut self) -> Result<(), SdCfrError> {
        self.current_iteration += 1;
        let t = self.current_iteration;

        self.update_player(Player::Player1, t)?;
        self.update_player(Player::Player2, t)?;
        Ok(())
    }

    /// Run all T iterations and return the trained model.
    pub fn train(&mut self) -> Result<TrainedSdCfr, SdCfrError> {
        let total = self.config.cfr_iterations;
        for _ in 0..total {
            self.step()?;
        }
        Ok(TrainedSdCfr {
            model_buffers: [
                std::mem::take(&mut self.model_buffers[0]),
                std::mem::take(&mut self.model_buffers[1]),
            ],
            config: self.config.clone(),
        })
    }

    /// Current iteration number (0 before any step).
    pub fn iteration(&self) -> u32 {
        self.current_iteration
    }
}

// ---------------------------------------------------------------------------
// Private: per-player update
// ---------------------------------------------------------------------------

impl<G: Game, E: StateEncoder<G::State>> SdCfrSolver<G, E> {
    /// Run K traversals for one player, then train and store the value net.
    fn update_player(&mut self, player: Player, iteration: u32) -> Result<(), SdCfrError> {
        let pi = player_index(player);
        let value_net = self.get_or_init_value_net(pi)?;

        self.run_traversals(player, iteration, &value_net)?;
        self.train_and_store(pi, iteration)
    }

    /// Obtain the current value net for player `pi`, creating a read-only copy.
    ///
    /// On the first iteration the net has random weights (freshly initialized).
    fn get_or_init_value_net(&mut self, pi: usize) -> Result<AdvantageNet, SdCfrError> {
        if self.value_nets[pi].is_none() {
            let (vm, net) = create_fresh_net(&self.config, &self.device)?;
            self.value_nets[pi] = Some((net, vm));
        }
        // Re-create net from the stored VarMap for a read-only copy during traversal
        let (_, vm) = self.value_nets[pi]
            .as_ref()
            .ok_or_else(|| SdCfrError::Config("value net missing after init".into()))?;
        let vs = VarBuilder::from_varmap(vm, DType::F32, &self.device);
        let net = AdvantageNet::new(self.config.num_actions, self.config.hidden_dim, &vs)?;
        Ok(net)
    }

    /// Run K traversals for `player`, accumulating samples in the advantage buffer.
    fn run_traversals(
        &mut self,
        player: Player,
        iteration: u32,
        value_net: &AdvantageNet,
    ) -> Result<(), SdCfrError> {
        let initial_states = self.game.initial_states();
        let pi = player_index(player);

        for _ in 0..self.config.traversals_per_iter {
            let idx = self.rng.random_range(0..initial_states.len());
            let state = &initial_states[idx];
            traverse(
                &self.game,
                state,
                player,
                iteration,
                value_net,
                &self.encoder,
                &mut self.advantage_buffers[pi],
                &mut self.rng,
                &self.device,
            )?;
        }
        Ok(())
    }

    /// Train a fresh value net from accumulated samples and store it.
    fn train_and_store(&mut self, pi: usize, iteration: u32) -> Result<(), SdCfrError> {
        let (net, vm) = train_value_net(
            &self.advantage_buffers[pi],
            &self.config,
            iteration,
            &self.device,
            &mut self.rng,
        )?;
        self.model_buffers[pi].push(&vm, iteration)?;
        self.value_nets[pi] = Some((net, vm));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Value network training
// ---------------------------------------------------------------------------

/// Train a fresh advantage network from scratch on the reservoir buffer.
///
/// Uses weighted MSE loss with linear CFR weighting (weight = sample.iteration / max_iteration).
fn train_value_net(
    buffer: &ReservoirBuffer<AdvantageSample>,
    config: &SdCfrConfig,
    max_iteration: u32,
    device: &Device,
    rng: &mut impl Rng,
) -> Result<(AdvantageNet, VarMap), SdCfrError> {
    if buffer.is_empty() {
        return Err(SdCfrError::EmptyBuffer);
    }

    let (varmap, net) = create_fresh_net(config, device)?;
    let vars = varmap.all_vars();
    let mut opt = candle_nn::AdamW::new_lr(vars.clone(), config.learning_rate)?;

    for _ in 0..config.sgd_steps {
        let batch = buffer.sample_batch(config.batch_size, rng);
        if batch.is_empty() {
            continue;
        }
        let loss = compute_batch_loss(&net, &batch, max_iteration, config.num_actions, device)?;
        let mut grads = loss.backward()?;
        clip_grad_store(&mut grads, &vars, config.grad_clip_norm)?;
        opt.step(&grads)?;
    }

    Ok((net, varmap))
}

/// Create a fresh AdvantageNet with random weights.
fn create_fresh_net(
    config: &SdCfrConfig,
    device: &Device,
) -> Result<(VarMap, AdvantageNet), SdCfrError> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let net = AdvantageNet::new(config.num_actions, config.hidden_dim, &vs)?;
    Ok((varmap, net))
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

/// Compute weighted MSE loss for a mini-batch of advantage samples.
///
/// Loss = sum(w_i * ||pred_i - target_i||^2) / sum(w_i)
fn compute_batch_loss(
    net: &AdvantageNet,
    batch: &[&AdvantageSample],
    max_iteration: u32,
    num_actions: usize,
    device: &Device,
) -> Result<Tensor, SdCfrError> {
    let (cards, bets, targets, weights) =
        samples_to_tensors(batch, max_iteration, num_actions, device)?;
    let predictions = net.forward(&cards, &bets)?;
    weighted_mse(&predictions, &targets, &weights)
}

/// Convert advantage samples into batched tensors for training.
///
/// Returns (card_indices, bet_features, target_advantages, weights).
fn samples_to_tensors(
    batch: &[&AdvantageSample],
    max_iteration: u32,
    num_actions: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor), SdCfrError> {
    let b = batch.len();
    let max_iter_f = f64::from(max_iteration).max(1.0);

    let card_data = collect_card_data(batch);
    let bet_data = collect_bet_data(batch);
    let (target_data, weight_data) = collect_targets_and_weights(batch, num_actions, max_iter_f);

    let cards = Tensor::from_vec(card_data, &[b, 7], device)?;
    let bets = Tensor::from_vec(bet_data, &[b, 48], device)?;
    let targets = Tensor::from_vec(target_data, &[b, num_actions], device)?;
    let weights = Tensor::from_vec(weight_data, &[b, 1], device)?;

    Ok((cards, bets, targets, weights))
}

/// Flatten card features from all samples into a contiguous i64 vector.
fn collect_card_data(batch: &[&AdvantageSample]) -> Vec<i64> {
    batch
        .iter()
        .flat_map(|s| s.features.cards.iter().map(|&c| i64::from(c)))
        .collect()
}

/// Flatten bet features from all samples into a contiguous f32 vector.
fn collect_bet_data(batch: &[&AdvantageSample]) -> Vec<f32> {
    batch
        .iter()
        .flat_map(|s| s.features.bets.iter().copied())
        .collect()
}

/// Build padded target advantages and linear CFR weights for each sample.
fn collect_targets_and_weights(
    batch: &[&AdvantageSample],
    num_actions: usize,
    max_iter_f: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut targets = Vec::with_capacity(batch.len() * num_actions);
    let mut weights = Vec::with_capacity(batch.len());

    for sample in batch {
        let w = f64::from(sample.iteration) / max_iter_f;
        weights.push(w as f32);
        targets.extend(pad_advantages(&sample.advantages, num_actions));
    }
    (targets, weights)
}

/// Pad advantage values to `num_actions` width, filling extra slots with 0.
fn pad_advantages(advantages: &[f32], num_actions: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; num_actions];
    let n = advantages.len().min(num_actions);
    padded[..n].copy_from_slice(&advantages[..n]);
    padded
}

// ---------------------------------------------------------------------------
// Weighted MSE
// ---------------------------------------------------------------------------

/// Compute weighted MSE: sum(w * (pred - target)^2) / sum(w).
fn weighted_mse(
    predictions: &Tensor,
    targets: &Tensor,
    weights: &Tensor,
) -> Result<Tensor, SdCfrError> {
    let diff = predictions.sub(targets)?;
    let sq = diff.sqr()?;
    let weighted = sq.broadcast_mul(weights)?;
    let loss = weighted
        .sum_all()?
        .div(&weights.sum_all()?.clamp(1e-8, f64::MAX)?)?;
    Ok(loss)
}

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

/// Clip gradients in a `GradStore` to `max_norm` using global L2 norm scaling.
///
/// If the total gradient norm exceeds `max_norm`, all gradients are scaled
/// down proportionally. Modifies the `GradStore` in place via remove+insert.
fn clip_grad_store(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
    max_norm: f64,
) -> Result<(), SdCfrError> {
    let total_norm = compute_grad_norm(grads, vars)?;
    if total_norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / total_norm;
    scale_grads_in_place(grads, vars, scale)
}

/// Compute the L2 norm of all gradients across all variables.
fn compute_grad_norm(
    grads: &candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
) -> Result<f64, SdCfrError> {
    let mut total_sq = 0.0f64;
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            total_sq += f64::from(norm_sq);
        }
    }
    Ok(total_sq.sqrt())
}

/// Scale all gradients in place by a constant factor.
fn scale_grads_in_place(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
    scale: f64,
) -> Result<(), SdCfrError> {
    for var in vars {
        if let Some(grad) = grads.remove(var.as_tensor()) {
            let scaled = (&grad * scale)?;
            grads.insert(var.as_tensor(), scaled);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map Player enum to array index: Player1 -> 0, Player2 -> 1.
const fn player_index(player: Player) -> usize {
    match player {
        Player::Player1 => 0,
        Player::Player2 => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_features::{BET_FEATURES, InfoSetFeatures};
    use crate::traverse::StateEncoder;
    use candle_core::Device;
    use poker_solver_core::game::KuhnPoker;
    use poker_solver_core::info_key::InfoKey;
    use rand::SeedableRng;

    // -----------------------------------------------------------------------
    // Kuhn encoder (same as in traverse tests)
    // -----------------------------------------------------------------------

    struct KuhnEncoder {
        game: KuhnPoker,
    }

    impl KuhnEncoder {
        fn new() -> Self {
            Self {
                game: KuhnPoker::new(),
            }
        }
    }

    impl StateEncoder<<KuhnPoker as Game>::State> for KuhnEncoder {
        fn encode(&self, state: &<KuhnPoker as Game>::State, _player: Player) -> InfoSetFeatures {
            let key = self.game.info_set_key(state);
            let hand_bits = InfoKey::from_raw(key).hand_bits();
            let card_value = hand_bits as i8;

            let mut cards = [-1i8; 7];
            cards[0] = card_value;

            InfoSetFeatures {
                cards,
                bets: [0.0f32; BET_FEATURES],
            }
        }
    }

    /// Small config for fast testing.
    fn test_config() -> SdCfrConfig {
        SdCfrConfig {
            cfr_iterations: 3,
            traversals_per_iter: 20,
            advantage_memory_cap: 5_000,
            hidden_dim: 16,
            num_actions: 2,
            sgd_steps: 10,
            batch_size: 32,
            learning_rate: 0.001,
            grad_clip_norm: 1.0,
            seed: 42,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Solver constructs with Kuhn poker
    // -----------------------------------------------------------------------

    #[test]
    fn solver_constructs_with_kuhn() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = test_config();
        let solver = SdCfrSolver::new(game, encoder, config);
        assert!(
            solver.is_ok(),
            "SdCfrSolver::new should succeed with valid config"
        );
        let solver = solver.unwrap();
        assert_eq!(solver.iteration(), 0);
    }

    // -----------------------------------------------------------------------
    // 2. Single step produces model entries
    // -----------------------------------------------------------------------

    #[test]
    fn single_step_produces_model_entries() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        solver.step().unwrap();

        assert_eq!(solver.iteration(), 1);
        assert_eq!(
            solver.model_buffers[0].len(),
            1,
            "Player 1 model buffer should have 1 entry after 1 step"
        );
        assert_eq!(
            solver.model_buffers[1].len(),
            1,
            "Player 2 model buffer should have 1 entry after 1 step"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Train value net reduces loss
    // -----------------------------------------------------------------------

    #[test]
    fn train_value_net_reduces_loss() {
        let config = SdCfrConfig {
            num_actions: 2,
            hidden_dim: 16,
            sgd_steps: 50,
            batch_size: 16,
            learning_rate: 0.001,
            grad_clip_norm: 10.0,
            ..test_config()
        };

        let mut rng = StdRng::seed_from_u64(42);
        let mut buffer = ReservoirBuffer::new(1_000);

        // Populate buffer with synthetic samples that have a learnable pattern
        for i in 0..200 {
            let card_val = (i % 3) as i8;
            let mut cards = [-1i8; 7];
            cards[0] = card_val;
            let target = if card_val == 2 { 1.0 } else { -1.0 };

            let sample = AdvantageSample {
                features: InfoSetFeatures {
                    cards,
                    bets: [0.0f32; BET_FEATURES],
                },
                iteration: 1,
                advantages: vec![target, -target],
                num_actions: 2,
            };
            buffer.push(sample, &mut rng);
        }

        // Compute loss before training (random net)
        let device = Device::Cpu;
        let (_, initial_net) = create_fresh_net(&config, &device).unwrap();
        let batch = buffer.sample_batch(config.batch_size, &mut rng);
        let initial_loss = compute_batch_loss(&initial_net, &batch, 1, config.num_actions, &device)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        // Train
        let mut rng2 = StdRng::seed_from_u64(99);
        let (trained_net, _) = train_value_net(&buffer, &config, 1, &device, &mut rng2).unwrap();

        // Compute loss after training
        let mut rng3 = StdRng::seed_from_u64(42);
        let batch = buffer.sample_batch(config.batch_size, &mut rng3);
        let final_loss = compute_batch_loss(&trained_net, &batch, 1, config.num_actions, &device)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            final_loss < initial_loss,
            "Training should reduce loss: initial={initial_loss}, final={final_loss}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Multiple steps accumulate models
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_steps_accumulate_models() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        let n = 3;
        for _ in 0..n {
            solver.step().unwrap();
        }

        assert_eq!(solver.iteration(), n);
        assert_eq!(
            solver.model_buffers[0].len(),
            n as usize,
            "Player 1 model buffer should have {n} entries after {n} steps"
        );
        assert_eq!(
            solver.model_buffers[1].len(),
            n as usize,
            "Player 2 model buffer should have {n} entries after {n} steps"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Full train completes
    // -----------------------------------------------------------------------

    #[test]
    fn full_train_completes() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = SdCfrConfig {
            cfr_iterations: 5,
            traversals_per_iter: 50,
            advantage_memory_cap: 5_000,
            hidden_dim: 16,
            num_actions: 2,
            sgd_steps: 10,
            batch_size: 32,
            learning_rate: 0.001,
            grad_clip_norm: 1.0,
            seed: 123,
        };
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        let result = solver.train();
        assert!(result.is_ok(), "train() should complete without error");

        let trained = result.unwrap();
        assert_eq!(trained.model_buffers[0].len(), 5);
        assert_eq!(trained.model_buffers[1].len(), 5);
    }

    // -----------------------------------------------------------------------
    // Unit tests for helper functions
    // -----------------------------------------------------------------------

    #[test]
    fn pad_advantages_shorter_than_num_actions() {
        let advs = vec![1.0, 2.0];
        let padded = pad_advantages(&advs, 5);
        assert_eq!(padded, vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_advantages_exact_length() {
        let advs = vec![1.0, 2.0, 3.0];
        let padded = pad_advantages(&advs, 3);
        assert_eq!(padded, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn player_index_mapping() {
        assert_eq!(player_index(Player::Player1), 0);
        assert_eq!(player_index(Player::Player2), 1);
    }

    #[test]
    fn weighted_mse_computation() {
        let pred = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let target = Tensor::new(&[[1.0f32, 3.0], [2.0, 4.0]], &Device::Cpu).unwrap();
        let weights = Tensor::new(&[[1.0f32], [2.0]], &Device::Cpu).unwrap();

        let loss = weighted_mse(&pred, &target, &weights).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // diff = [[0, -1], [1, 0]], sq = [[0, 1], [1, 0]]
        // weighted = [[0, 1], [2, 0]]
        // sum_weighted = 3, sum_weights = 3
        // loss = 3/3 = 1.0
        assert!(
            (loss_val - 1.0).abs() < 1e-5,
            "Expected loss ~1.0, got {loss_val}"
        );
    }

    #[test]
    fn train_value_net_errors_on_empty_buffer() {
        let config = test_config();
        let buffer = ReservoirBuffer::<AdvantageSample>::new(100);
        let device = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(42);

        let result = train_value_net(&buffer, &config, 1, &device, &mut rng);
        assert!(result.is_err(), "should error on empty buffer");
    }
}
