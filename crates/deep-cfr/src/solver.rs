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
use crate::best_available_device;
use crate::config::SdCfrConfig;
use crate::memory::ReservoirBuffer;
use crate::model_buffer::ModelBuffer;
use crate::network::AdvantageNet;
use crate::batched_traverse::traverse_batched;
use crate::traverse::{AdvantageSample, StateEncoder, traverse};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Callback type for checkpoint notifications during training.
///
/// Called with `(iteration, snapshot)` every `checkpoint_interval` iterations.
pub type CheckpointFn<'a> = &'a mut dyn FnMut(u32, &TrainedSdCfr) -> Result<(), SdCfrError>;

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
            device: best_available_device(),
            rng,
        })
    }

    /// Run one full CFR iteration: update both players sequentially.
    ///
    /// Delegates to [`step_with_deals`](Self::step_with_deals) with no deal pool.
    pub fn step(&mut self) -> Result<(), SdCfrError> {
        self.step_with_deals(None)
    }

    /// Run one full CFR iteration using an optional external deal pool.
    ///
    /// If `deal_pool` is `Some`, samples from it instead of calling
    /// `game.initial_states()` each iteration.
    pub fn step_with_deals(&mut self, deal_pool: Option<&[G::State]>) -> Result<(), SdCfrError> {
        self.current_iteration += 1;
        let t = self.current_iteration;
        self.update_player_with_deals(Player::Player1, t, deal_pool)?;
        self.update_player_with_deals(Player::Player2, t, deal_pool)?;
        Ok(())
    }

    /// Run all T iterations and return the trained model.
    ///
    /// Delegates to [`train_with_deals`](Self::train_with_deals) with no deal pool
    /// and no checkpoint callback.
    pub fn train(&mut self) -> Result<TrainedSdCfr, SdCfrError> {
        self.train_with_deals(None, None)
    }

    /// Run all T iterations with optional pre-generated deal pool and checkpoint callback.
    ///
    /// - `deal_pool`: If `Some`, samples from this pool instead of calling
    ///   `game.initial_states()` each iteration.
    /// - `checkpoint_cb`: If `Some`, called every `config.checkpoint_interval` iterations
    ///   with the current iteration number and a snapshot of the model buffers.
    pub fn train_with_deals(
        &mut self,
        deal_pool: Option<&[G::State]>,
        mut checkpoint_cb: Option<CheckpointFn<'_>>,
    ) -> Result<TrainedSdCfr, SdCfrError> {
        let total = self.config.cfr_iterations;
        let interval = self.config.checkpoint_interval;

        for _ in 0..total {
            self.step_with_deals(deal_pool)?;
            self.maybe_checkpoint(interval, &mut checkpoint_cb)?;
        }
        Ok(self.take_result())
    }

    /// Clone the current model buffers and config into a `TrainedSdCfr` snapshot.
    ///
    /// Does not consume or modify the solver state.
    pub fn snapshot(&self) -> TrainedSdCfr {
        TrainedSdCfr {
            model_buffers: [self.model_buffers[0].clone(), self.model_buffers[1].clone()],
            config: self.config.clone(),
        }
    }

    /// Take the model buffers out of the solver, leaving empty defaults behind.
    pub fn take_result(&mut self) -> TrainedSdCfr {
        TrainedSdCfr {
            model_buffers: [
                std::mem::take(&mut self.model_buffers[0]),
                std::mem::take(&mut self.model_buffers[1]),
            ],
            config: self.config.clone(),
        }
    }

    /// Current iteration number (0 before any step).
    pub fn iteration(&self) -> u32 {
        self.current_iteration
    }

    /// Return buffer statistics: `[(stored, total_seen); 2]` for each player.
    pub fn buffer_stats(&self) -> [(usize, u64); 2] {
        [
            (self.advantage_buffers[0].len(), self.advantage_buffers[0].total_seen()),
            (self.advantage_buffers[1].len(), self.advantage_buffers[1].total_seen()),
        ]
    }
}

// ---------------------------------------------------------------------------
// Private: per-player update
// ---------------------------------------------------------------------------

impl<G: Game, E: StateEncoder<G::State>> SdCfrSolver<G, E> {
    /// Run K traversals for one player, then train and store the value net.
    ///
    /// If `deal_pool` is `Some`, samples initial states from it; otherwise
    /// calls `game.initial_states()`.
    fn update_player_with_deals(
        &mut self,
        player: Player,
        iteration: u32,
        deal_pool: Option<&[G::State]>,
    ) -> Result<(), SdCfrError> {
        let pi = player_index(player);
        eprintln!("    P{}: traversing {} deals...", pi + 1, self.config.traversals_per_iter);
        let value_net = self.get_or_init_value_net(pi)?;
        self.run_traversals_with_deals(player, iteration, &value_net, deal_pool)?;
        eprintln!(
            "    P{}: training value net ({} SGD steps, {} samples)...",
            pi + 1,
            self.config.sgd_steps,
            self.advantage_buffers[pi].len(),
        );
        self.train_and_store(pi, iteration)
    }

    /// Invoke the checkpoint callback if the interval is reached.
    fn maybe_checkpoint(
        &self,
        interval: u32,
        cb: &mut Option<CheckpointFn<'_>>,
    ) -> Result<(), SdCfrError> {
        if interval == 0 || !self.current_iteration.is_multiple_of(interval) {
            return Ok(());
        }
        if let Some(f) = cb {
            let snap = self.snapshot();
            f(self.current_iteration, &snap)?;
        }
        Ok(())
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
    ///
    /// If `deal_pool` is `Some`, samples initial states from the provided slice;
    /// otherwise falls back to calling `game.initial_states()`.
    ///
    /// When `parallel_traversals > 1`, uses the batched traversal engine for
    /// GPU-efficient NN inference. Otherwise falls back to sequential traversal.
    fn run_traversals_with_deals(
        &mut self,
        player: Player,
        iteration: u32,
        value_net: &AdvantageNet,
        deal_pool: Option<&[G::State]>,
    ) -> Result<(), SdCfrError> {
        let fallback;
        let states: &[G::State] = match deal_pool {
            Some(pool) => pool,
            None => {
                fallback = self.game.initial_states();
                &fallback
            }
        };

        if self.config.parallel_traversals > 1 {
            self.run_traversals_batched(player, iteration, value_net, states)
        } else {
            self.run_traversals_sequential(player, iteration, value_net, states)
        }
    }

    /// Sequential traversal path: one traversal at a time (original behavior).
    fn run_traversals_sequential(
        &mut self,
        player: Player,
        iteration: u32,
        value_net: &AdvantageNet,
        states: &[G::State],
    ) -> Result<(), SdCfrError> {
        let pi = player_index(player);
        let total = self.config.traversals_per_iter;
        let log_interval = traversal_log_interval(total);

        for k in 0..total {
            if log_interval > 0 && k > 0 && k % log_interval == 0 {
                eprintln!(
                    "    [iter {iteration}] P{} traversal {k}/{total} (buf: {} stored, {} seen)",
                    pi + 1,
                    self.advantage_buffers[pi].len(),
                    self.advantage_buffers[pi].total_seen(),
                );
            }
            let idx = self.rng.random_range(0..states.len());
            let state = &states[idx];
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

    /// Batched traversal path: run B traversals concurrently with batched NN inference.
    fn run_traversals_batched(
        &mut self,
        player: Player,
        iteration: u32,
        value_net: &AdvantageNet,
        states: &[G::State],
    ) -> Result<(), SdCfrError> {
        let pi = player_index(player);
        let total = self.config.traversals_per_iter;
        let batch_size = self.config.parallel_traversals;

        let samples = traverse_batched(
            &self.game,
            states,
            player,
            iteration,
            value_net,
            &self.encoder,
            &self.device,
            &mut self.rng,
            total,
            batch_size,
        )?;

        for sample in samples {
            self.advantage_buffers[pi].push(sample, &mut self.rng);
        }

        eprintln!(
            "    [iter {iteration}] P{} batched {} traversals (batch_size={batch_size}, buf: {} stored, {} seen)",
            pi + 1,
            total,
            self.advantage_buffers[pi].len(),
            self.advantage_buffers[pi].total_seen(),
        );

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
/// Pre-uploads the entire buffer to GPU once, then samples mini-batches via
/// `index_select` to avoid per-step CPU allocation and CPU→GPU transfers.
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
    let log_interval = sgd_log_interval(config.sgd_steps);

    let gpu = buffer_to_gpu_tensors(buffer, max_iteration, config.num_actions, device)?;
    let n = gpu.len;

    for step in 0..config.sgd_steps {
        let indices = random_index_tensor(config.batch_size.min(n), n, rng, device)?;
        let loss = compute_batch_loss_gpu(&net, &gpu, &indices)?;
        if log_interval > 0 && step > 0 && step % log_interval == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            eprintln!(
                "      SGD step {step}/{} loss={loss_val:.6}",
                config.sgd_steps,
            );
        }
        let mut grads = loss.backward()?;
        clip_grad_store(&mut grads, &vars, config.grad_clip_norm)?;
        opt.step(&grads)?;
    }

    Ok((net, varmap))
}

// ---------------------------------------------------------------------------
// GPU-resident training data
// ---------------------------------------------------------------------------

/// All training data pre-uploaded to the compute device.
struct GpuTrainingData {
    cards: Tensor,   // [N, 7] f32 (cast to i64 before forward pass; f32 for Metal compat)
    bets: Tensor,    // [N, 48] f32
    targets: Tensor, // [N, num_actions] f32
    weights: Tensor, // [N, 1] f32
    len: usize,
}

/// Convert the entire reservoir buffer into GPU-resident tensors.
fn buffer_to_gpu_tensors(
    buffer: &ReservoirBuffer<AdvantageSample>,
    max_iteration: u32,
    num_actions: usize,
    device: &Device,
) -> Result<GpuTrainingData, SdCfrError> {
    let data = buffer.data();
    let n = data.len();
    let max_iter_f = f64::from(max_iteration).max(1.0);

    let card_data = collect_card_data_slice(data);
    let bet_data = collect_bet_data_slice(data);
    let (target_data, weight_data) = collect_targets_and_weights_slice(data, num_actions, max_iter_f);

    let cards = Tensor::from_vec(card_data, &[n, 7], device)?;
    let bets = Tensor::from_vec(bet_data, &[n, 48], device)?;
    let targets = Tensor::from_vec(target_data, &[n, num_actions], device)?;
    let weights = Tensor::from_vec(weight_data, &[n, 1], device)?;

    Ok(GpuTrainingData { cards, bets, targets, weights, len: n })
}

/// Generate a tensor of `batch_size` random indices in `[0, n)`.
fn random_index_tensor(
    batch_size: usize,
    n: usize,
    rng: &mut impl Rng,
    device: &Device,
) -> Result<Tensor, SdCfrError> {
    let indices: Vec<u32> = (0..batch_size)
        .map(|_| rng.random_range(0..n as u32))
        .collect();
    Ok(Tensor::from_vec(indices, &[batch_size], device)?)
}

/// Compute weighted MSE loss from pre-uploaded GPU tensors using index_select.
fn compute_batch_loss_gpu(
    net: &AdvantageNet,
    gpu: &GpuTrainingData,
    indices: &Tensor,
) -> Result<Tensor, SdCfrError> {
    let cards_f32 = gpu.cards.index_select(indices, 0)?;
    let cards = cards_f32.to_dtype(DType::I64)?; // -1.0 → -1, 0-51 exact
    let bets = gpu.bets.index_select(indices, 0)?;
    let targets = gpu.targets.index_select(indices, 0)?;
    let weights = gpu.weights.index_select(indices, 0)?;
    let predictions = net.forward(&cards, &bets)?;
    weighted_mse(&predictions, &targets, &weights)
}

/// Flatten card features from a slice of samples as f32.
/// Metal lacks i64 `index_select`, so cards are stored as f32 on GPU
/// and cast to i64 before the forward pass. Values -1..51 are exact in f32.
fn collect_card_data_slice(data: &[AdvantageSample]) -> Vec<f32> {
    data.iter()
        .flat_map(|s| s.features.cards.iter().map(|&c| f32::from(c)))
        .collect()
}

/// Flatten bet features from a slice of samples.
fn collect_bet_data_slice(data: &[AdvantageSample]) -> Vec<f32> {
    data.iter()
        .flat_map(|s| s.features.bets.iter().copied())
        .collect()
}

/// Build padded targets and weights from a slice of samples.
fn collect_targets_and_weights_slice(
    data: &[AdvantageSample],
    num_actions: usize,
    max_iter_f: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut targets = Vec::with_capacity(data.len() * num_actions);
    let mut weights = Vec::with_capacity(data.len());
    for sample in data {
        let w = f64::from(sample.iteration) / max_iter_f;
        weights.push(w as f32);
        targets.extend(pad_advantages(&sample.advantages, num_actions));
    }
    (targets, weights)
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

/// Compute a log interval that yields ~4 log lines for a traversal loop of `total` steps.
/// Returns 0 if total is too small to warrant logging.
fn traversal_log_interval(total: u32) -> u32 {
    if total < 100 {
        return 0;
    }
    (total / 4).max(1)
}

/// Compute a log interval that yields ~4 log lines for an SGD loop of `total` steps.
/// Returns 0 if total is too small to warrant logging.
fn sgd_log_interval(total: usize) -> usize {
    if total < 20 {
        return 0;
    }
    (total / 4).max(1)
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
            checkpoint_interval: 0,
            parallel_traversals: 1,
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

        let device = Device::Cpu;
        let gpu = buffer_to_gpu_tensors(&buffer, 1, config.num_actions, &device).unwrap();
        let all_indices = random_index_tensor(config.batch_size, gpu.len, &mut rng, &device).unwrap();

        // Compute loss before training (random net)
        let (_, initial_net) = create_fresh_net(&config, &device).unwrap();
        let initial_loss = compute_batch_loss_gpu(&initial_net, &gpu, &all_indices)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        // Train
        let mut rng2 = StdRng::seed_from_u64(99);
        let (trained_net, _) = train_value_net(&buffer, &config, 1, &device, &mut rng2).unwrap();

        // Compute loss after training
        let final_loss = compute_batch_loss_gpu(&trained_net, &gpu, &all_indices)
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
            checkpoint_interval: 0,
            parallel_traversals: 1,
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

    // -----------------------------------------------------------------------
    // Deal pool + checkpoint callback tests
    // -----------------------------------------------------------------------

    #[test]
    fn train_with_deal_pool_uses_provided_states() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let deals = game.initial_states();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        let result = solver.train_with_deals(Some(&deals), None);
        assert!(
            result.is_ok(),
            "train_with_deals(Some, None) should succeed"
        );

        let trained = result.unwrap();
        assert_eq!(
            trained.model_buffers[0].len(),
            3,
            "P1 model buffer should have 3 entries (cfr_iterations=3)"
        );
        assert_eq!(
            trained.model_buffers[1].len(),
            3,
            "P2 model buffer should have 3 entries (cfr_iterations=3)"
        );
    }

    #[test]
    fn train_with_checkpoint_callback_is_called() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = SdCfrConfig {
            cfr_iterations: 4,
            checkpoint_interval: 2,
            ..test_config()
        };
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        let mut call_count = 0u32;
        let mut seen_iterations = Vec::new();
        let result = solver.train_with_deals(
            None,
            Some(&mut |iter, trained: &TrainedSdCfr| {
                call_count += 1;
                seen_iterations.push(iter);
                // Verify snapshot has correct number of entries at checkpoint time
                assert_eq!(
                    trained.model_buffers[0].len(),
                    iter as usize,
                    "snapshot at iter {iter} should have {iter} P1 entries"
                );
                Ok(())
            }),
        );
        assert!(
            result.is_ok(),
            "train_with_deals with callback should succeed"
        );
        assert_eq!(
            call_count, 2,
            "callback should fire exactly twice (iters 2 and 4)"
        );
        assert_eq!(seen_iterations, vec![2, 4]);
    }

    #[test]
    fn train_with_deals_none_falls_back_to_game() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        let result = solver.train_with_deals(None, None);
        assert!(
            result.is_ok(),
            "train_with_deals(None, None) should behave like train()"
        );

        let trained = result.unwrap();
        assert_eq!(trained.model_buffers[0].len(), 3);
        assert_eq!(trained.model_buffers[1].len(), 3);
    }

    #[test]
    fn existing_train_still_works() {
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
            checkpoint_interval: 0,
            parallel_traversals: 1,
        };
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        let result = solver.train();
        assert!(result.is_ok(), "delegating train() should still work");

        let trained = result.unwrap();
        assert_eq!(trained.model_buffers[0].len(), 5);
        assert_eq!(trained.model_buffers[1].len(), 5);
    }

    #[test]
    fn step_with_deals_uses_provided_pool() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let deals = game.initial_states();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        solver.step_with_deals(Some(&deals)).unwrap();

        assert_eq!(solver.iteration(), 1);
        assert_eq!(solver.model_buffers[0].len(), 1);
        assert_eq!(solver.model_buffers[1].len(), 1);
    }

    #[test]
    fn snapshot_clones_current_state() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        solver.step().unwrap();
        let snap = solver.snapshot();

        assert_eq!(snap.model_buffers[0].len(), 1);
        assert_eq!(snap.model_buffers[1].len(), 1);

        // Original solver still has its buffers (snapshot is independent)
        solver.step().unwrap();
        assert_eq!(solver.model_buffers[0].len(), 2);
        assert_eq!(
            snap.model_buffers[0].len(),
            1,
            "snapshot should be independent"
        );
    }

    #[test]
    fn take_result_empties_solver_buffers() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let config = test_config();
        let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

        solver.step().unwrap();
        solver.step().unwrap();

        let result = solver.take_result();
        assert_eq!(result.model_buffers[0].len(), 2);
        assert_eq!(result.model_buffers[1].len(), 2);

        // Solver buffers should be empty after take
        assert_eq!(solver.model_buffers[0].len(), 0);
        assert_eq!(solver.model_buffers[1].len(), 0);
    }
}
