//! Batched traversal engine for GPU-efficient SD-CFR.
//!
//! Runs B traversals concurrently using an explicit-stack state machine,
//! batching all NN inference requests into single GPU forward passes.
//!
//! Engine loop:
//! 1. Advance all B traversals until each needs an NN inference (or completes)
//! 2. Collect pending inference requests into one batch
//! 3. Run ONE batched GPU forward pass
//! 4. Distribute strategy results back to each traversal
//! 5. Repeat until all traversals complete

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use poker_solver_core::game::{Game, Player};
use poker_solver_core::Action;

use crate::SdCfrError;
use crate::card_features::{BET_FEATURES, InfoSetFeatures};
use crate::network::AdvantageNet;
use crate::traverse::{
    AdvantageSample, StateEncoder, compute_advantages, sample_action, weighted_sum,
};

// ---------------------------------------------------------------------------
// Strategy cache types
// ---------------------------------------------------------------------------

type StrategyCache = HashMap<CacheKey, Vec<f32>>;

/// Cache key for info set features. Converts f32 bets to bit-exact u32
/// so we can derive Hash + Eq.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct CacheKey {
    cards: [i8; 7],
    bet_bits: [u32; BET_FEATURES],
}

impl CacheKey {
    fn from_features(f: &InfoSetFeatures) -> Self {
        let mut bet_bits = [0u32; BET_FEATURES];
        for (i, &v) in f.bets.iter().enumerate() {
            bet_bits[i] = v.to_bits();
        }
        Self {
            cards: f.cards,
            bet_bits,
        }
    }
}

/// Tracks cache hit/miss statistics.
struct CacheStats {
    hits: u64,
    misses: u64,
}

impl CacheStats {
    fn new() -> Self {
        Self { hits: 0, misses: 0 }
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Frame & phase types
// ---------------------------------------------------------------------------

/// Phase of a single stack frame during iterative traversal.
enum FramePhase {
    /// Just entered this node — needs strategy computation via NN.
    Enter,
    /// NN inference submitted, waiting for strategy result.
    AwaitingStrategy {
        features: InfoSetFeatures,
        actions: Vec<Action>,
        player: Player,
    },
    /// Traverser node: exploring children one by one.
    TraverserExploring {
        strategy: Vec<f32>,
        features: InfoSetFeatures,
        actions: Vec<Action>,
        cf_values: Vec<f64>,
        next_child: usize,
    },
    /// Opponent node: child has been pushed, waiting for its value.
    OpponentWaiting,
}

/// One stack frame representing a decision node in the game tree.
struct Frame<S> {
    state: S,
    phase: FramePhase,
}

// ---------------------------------------------------------------------------
// Traversal state machine
// ---------------------------------------------------------------------------

/// State machine for a single concurrent traversal.
struct Traversal<S> {
    stack: Vec<Frame<S>>,
    samples: Vec<AdvantageSample>,
    rng: StdRng,
    result: Option<f64>,
    traverser: Player,
    iteration: u32,
}

/// What a traversal needs next after being advanced.
enum TraversalNeed {
    /// Needs NN inference for the given features and action count.
    Inference {
        features: InfoSetFeatures,
        num_actions: usize,
    },
    /// Still advancing (terminal node handled, or child pushed).
    Advancing,
    /// Traversal is complete — result is stored in `traversal.result`.
    Complete,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run `count` traversals in batches of `batch_size`, collecting advantage samples.
///
/// Returns all collected [`AdvantageSample`]s from every traversal.
///
/// Each batch of up to `batch_size` traversals runs concurrently, sharing
/// batched NN inference calls for GPU efficiency.
pub fn traverse_batched<G: Game, E: StateEncoder<G::State>>(
    game: &G,
    initial_states: &[G::State],
    traverser: Player,
    iteration: u32,
    value_net: &AdvantageNet,
    encoder: &E,
    device: &Device,
    main_rng: &mut impl Rng,
    count: u32,
    batch_size: usize,
) -> Result<Vec<AdvantageSample>, SdCfrError> {
    let mut all_samples = Vec::new();
    let mut remaining = count;
    let mut cache: StrategyCache = HashMap::new();
    let mut stats = CacheStats::new();

    while remaining > 0 {
        let b = (remaining as usize).min(batch_size);
        let batch_samples = run_batch(
            game,
            initial_states,
            traverser,
            iteration,
            value_net,
            encoder,
            device,
            main_rng,
            b,
            &mut cache,
            &mut stats,
        )?;
        all_samples.extend(batch_samples);
        remaining -= b as u32;
    }

    let total = stats.hits + stats.misses;
    if total > 0 {
        log::debug!(
            "Strategy cache: {total} lookups, {:.1}% hit rate ({} entries)",
            stats.hit_rate() * 100.0,
            cache.len(),
        );
    }

    Ok(all_samples)
}

// ---------------------------------------------------------------------------
// Batch execution
// ---------------------------------------------------------------------------

/// Run one batch of `b` concurrent traversals to completion.
fn run_batch<G: Game, E: StateEncoder<G::State>>(
    game: &G,
    initial_states: &[G::State],
    traverser: Player,
    iteration: u32,
    value_net: &AdvantageNet,
    encoder: &E,
    device: &Device,
    main_rng: &mut impl Rng,
    b: usize,
    cache: &mut StrategyCache,
    stats: &mut CacheStats,
) -> Result<Vec<AdvantageSample>, SdCfrError> {
    let mut traversals = init_traversals(game, initial_states, traverser, iteration, main_rng, b);

    loop {
        // Advance all non-complete traversals until they need inference
        let pending = collect_pending_inferences(game, encoder, &mut traversals)?;

        if pending.is_empty() {
            break;
        }

        // Cached batched NN inference
        let strategies = resolve_strategies(value_net, &pending, device, cache, stats)?;

        // Distribute results — opponent nodes need the game for next_state
        apply_strategies(game, &mut traversals, &pending, &strategies);
    }

    Ok(drain_samples(&mut traversals))
}

/// Create `b` traversals, each starting from a random initial state.
fn init_traversals<G: Game>(
    game: &G,
    initial_states: &[G::State],
    traverser: Player,
    iteration: u32,
    main_rng: &mut impl Rng,
    b: usize,
) -> Vec<Traversal<G::State>> {
    let _ = game; // used only for type inference
    (0..b)
        .map(|_| {
            let mut rng = StdRng::seed_from_u64(main_rng.random());
            let idx = rng.random_range(0..initial_states.len());
            let state = initial_states[idx].clone();
            Traversal {
                stack: vec![Frame {
                    state,
                    phase: FramePhase::Enter,
                }],
                samples: Vec::new(),
                rng,
                result: None,
                traverser,
                iteration,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Advancing traversals
// ---------------------------------------------------------------------------

/// A pending inference request: which traversal needs it, plus the features.
struct PendingInference {
    traversal_idx: usize,
    features: InfoSetFeatures,
    num_actions: usize,
}

/// Advance all incomplete traversals until each needs NN inference or completes.
fn collect_pending_inferences<G: Game, E: StateEncoder<G::State>>(
    game: &G,
    encoder: &E,
    traversals: &mut [Traversal<G::State>],
) -> Result<Vec<PendingInference>, SdCfrError> {
    let mut pending = Vec::new();

    for (idx, trav) in traversals.iter_mut().enumerate() {
        if trav.result.is_some() {
            continue;
        }

        loop {
            match advance_one_step(game, encoder, trav)? {
                TraversalNeed::Inference {
                    features,
                    num_actions,
                } => {
                    pending.push(PendingInference {
                        traversal_idx: idx,
                        features,
                        num_actions,
                    });
                    break;
                }
                TraversalNeed::Complete => break,
                TraversalNeed::Advancing => continue,
            }
        }
    }

    Ok(pending)
}

/// Advance a single traversal by one step.
fn advance_one_step<G: Game, E: StateEncoder<G::State>>(
    game: &G,
    encoder: &E,
    trav: &mut Traversal<G::State>,
) -> Result<TraversalNeed, SdCfrError> {
    if trav.stack.is_empty() {
        return Ok(TraversalNeed::Complete);
    }

    // Determine what to do based on the top frame's phase
    let phase_tag = match &trav.stack.last().expect("checked non-empty").phase {
        FramePhase::Enter => 0,
        FramePhase::AwaitingStrategy { .. } => 1,
        FramePhase::TraverserExploring { .. } => 2,
        FramePhase::OpponentWaiting => 3,
    };

    match phase_tag {
        0 => handle_enter(game, encoder, trav),
        1 => handle_awaiting_strategy(trav),
        2 => handle_traverser_exploring(game, trav),
        3 => Ok(TraversalNeed::Advancing), // shouldn't be reached normally
        _ => unreachable!(),
    }
}

/// Handle the `Enter` phase: check terminal or request strategy.
fn handle_enter<G: Game, E: StateEncoder<G::State>>(
    game: &G,
    encoder: &E,
    trav: &mut Traversal<G::State>,
) -> Result<TraversalNeed, SdCfrError> {
    let frame = trav.stack.last().expect("stack non-empty");

    if game.is_terminal(&frame.state) {
        let utility = game.utility(&frame.state, trav.traverser);
        trav.stack.pop();
        return Ok(complete_or_propagate(trav, utility));
    }

    let current_player = game.player(&frame.state);
    let actions: Vec<Action> = game.actions(&frame.state).to_vec();
    let features = encoder.encode(&frame.state, current_player);
    let num_actions = actions.len();

    let frame = trav.stack.last_mut().expect("stack non-empty");
    frame.phase = FramePhase::AwaitingStrategy {
        features: features.clone(),
        actions,
        player: current_player,
    };

    Ok(TraversalNeed::Inference {
        features,
        num_actions,
    })
}

/// Handle `AwaitingStrategy`: return the inference need so it gets batched.
fn handle_awaiting_strategy<S>(trav: &Traversal<S>) -> Result<TraversalNeed, SdCfrError> {
    let frame = trav.stack.last().expect("stack non-empty");
    if let FramePhase::AwaitingStrategy {
        ref features,
        ref actions,
        ..
    } = frame.phase
    {
        Ok(TraversalNeed::Inference {
            features: features.clone(),
            num_actions: actions.len(),
        })
    } else {
        unreachable!()
    }
}

/// Handle `TraverserExploring`: push the next child or finish this node.
fn handle_traverser_exploring<G: Game>(
    game: &G,
    trav: &mut Traversal<G::State>,
) -> Result<TraversalNeed, SdCfrError> {
    let frame = trav.stack.last_mut().expect("stack non-empty");

    if let FramePhase::TraverserExploring {
        ref strategy,
        ref features,
        ref actions,
        ref cf_values,
        ref mut next_child,
    } = frame.phase
    {
        if *next_child < actions.len() {
            let child_idx = *next_child;
            *next_child += 1;
            let child_state = game.next_state(&frame.state, actions[child_idx]);
            trav.stack.push(Frame {
                state: child_state,
                phase: FramePhase::Enter,
            });
            return Ok(TraversalNeed::Advancing);
        }

        // All children explored — compute advantages and pop
        let node_value = weighted_sum(cf_values, strategy);
        let advantages = compute_advantages(cf_values, node_value);
        let sample = AdvantageSample {
            features: features.clone(),
            iteration: trav.iteration,
            advantages,
            num_actions: actions.len() as u8,
        };
        trav.samples.push(sample);
        trav.stack.pop();
        Ok(complete_or_propagate(trav, node_value))
    } else {
        unreachable!("handle_traverser_exploring called on wrong phase")
    }
}

// ---------------------------------------------------------------------------
// Strategy application (post-inference)
// ---------------------------------------------------------------------------

/// Distribute batched NN results to the traversals that requested them.
fn apply_strategies<G: Game>(
    game: &G,
    traversals: &mut [Traversal<G::State>],
    pending: &[PendingInference],
    strategies: &[Vec<f32>],
) {
    for (pi, strategy) in pending.iter().zip(strategies.iter()) {
        let trav = &mut traversals[pi.traversal_idx];
        apply_single_strategy(game, trav, strategy);
    }
}

/// Apply a strategy result to a traversal in `AwaitingStrategy` phase.
fn apply_single_strategy<G: Game>(
    game: &G,
    trav: &mut Traversal<G::State>,
    strategy: &[f32],
) {
    let frame = trav.stack.last_mut().expect("stack non-empty during apply");

    // Take ownership of the AwaitingStrategy fields
    let old_phase = std::mem::replace(&mut frame.phase, FramePhase::Enter);
    let (features, actions, player) = match old_phase {
        FramePhase::AwaitingStrategy {
            features,
            actions,
            player,
        } => (features, actions, player),
        _ => unreachable!("apply_single_strategy called on non-AwaitingStrategy"),
    };

    if player == trav.traverser {
        frame.phase = FramePhase::TraverserExploring {
            strategy: strategy.to_vec(),
            features,
            actions,
            cf_values: Vec::new(),
            next_child: 0,
        };
    } else {
        // Opponent node: sample one action, push child, set parent to OpponentWaiting
        let chosen_idx = sample_action(strategy, &mut trav.rng);
        let child_state = game.next_state(&frame.state, actions[chosen_idx]);
        frame.phase = FramePhase::OpponentWaiting;
        trav.stack.push(Frame {
            state: child_state,
            phase: FramePhase::Enter,
        });
    }
}

// ---------------------------------------------------------------------------
// Cached batched NN inference
// ---------------------------------------------------------------------------

/// Resolve strategies for all pending requests, using the cache when possible.
///
/// Cache hits return immediately; misses are batched into a single GPU forward
/// pass. New results are inserted into the cache for future lookups.
fn resolve_strategies(
    value_net: &AdvantageNet,
    pending: &[PendingInference],
    device: &Device,
    cache: &mut StrategyCache,
    stats: &mut CacheStats,
) -> Result<Vec<Vec<f32>>, SdCfrError> {
    let mut strategies: Vec<Option<Vec<f32>>> = vec![None; pending.len()];
    let mut miss_indices: Vec<usize> = Vec::new();

    // Check cache for each request
    for (i, p) in pending.iter().enumerate() {
        let key = CacheKey::from_features(&p.features);
        if let Some(cached) = cache.get(&key) {
            strategies[i] = Some(cached[..p.num_actions].to_vec());
            stats.hits += 1;
        } else {
            miss_indices.push(i);
            stats.misses += 1;
        }
    }

    // Run batched inference on misses only
    if !miss_indices.is_empty() {
        let miss_pending: Vec<&PendingInference> =
            miss_indices.iter().map(|&i| &pending[i]).collect();
        let miss_strategies = batched_inference_subset(value_net, &miss_pending, device)?;

        for (&idx, strategy) in miss_indices.iter().zip(miss_strategies) {
            let key = CacheKey::from_features(&pending[idx].features);
            cache.insert(key, strategy.clone());
            strategies[idx] = Some(strategy[..pending[idx].num_actions].to_vec());
        }
    }

    Ok(strategies.into_iter().map(|s| s.expect("all resolved")).collect())
}

/// Run a single batched NN forward pass for a subset of pending requests.
fn batched_inference_subset(
    value_net: &AdvantageNet,
    pending: &[&PendingInference],
    device: &Device,
) -> Result<Vec<Vec<f32>>, SdCfrError> {
    let b = pending.len();
    let card_data: Vec<i64> = pending
        .iter()
        .flat_map(|p| p.features.cards.iter().map(|&c| i64::from(c)))
        .collect();
    let bet_data: Vec<f32> = pending
        .iter()
        .flat_map(|p| p.features.bets.iter().copied())
        .collect();
    let cards = Tensor::from_vec(card_data, &[b, 7], device)?;
    let bets = Tensor::from_vec(bet_data, &[b, BET_FEATURES], device)?;

    let raw_advantages = value_net.forward(&cards, &bets)?;
    let strategy_tensor = AdvantageNet::advantages_to_strategy(&raw_advantages)?;
    let all_probs = strategy_tensor.to_vec2::<f32>()?;

    let strategies = pending
        .iter()
        .zip(all_probs)
        .map(|(p, row)| row[..p.num_actions].to_vec())
        .collect();

    Ok(strategies)
}

// ---------------------------------------------------------------------------
// Value propagation
// ---------------------------------------------------------------------------

/// Pop a completed node and propagate its value to the parent, or mark complete.
fn complete_or_propagate<S>(trav: &mut Traversal<S>, value: f64) -> TraversalNeed {
    if trav.stack.is_empty() {
        trav.result = Some(value);
        return TraversalNeed::Complete;
    }
    propagate_value(trav, value);
    TraversalNeed::Advancing
}

/// Propagate a child's returned value to its parent frame.
fn propagate_value<S>(trav: &mut Traversal<S>, value: f64) {
    loop {
        if trav.stack.is_empty() {
            trav.result = Some(value);
            return;
        }

        let parent = trav.stack.last_mut().expect("stack non-empty");
        match &mut parent.phase {
            FramePhase::TraverserExploring { cf_values, .. } => {
                cf_values.push(value);
                return;
            }
            FramePhase::OpponentWaiting => {
                // Opponent node complete — pop and propagate value upward
                trav.stack.pop();
            }
            _ => return,
        }
    }
}

/// Collect all samples from completed traversals.
fn drain_samples<S>(traversals: &mut [Traversal<S>]) -> Vec<AdvantageSample> {
    traversals
        .iter_mut()
        .flat_map(|t| t.samples.drain(..))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_features::BET_FEATURES;
    use crate::memory::ReservoirBuffer;
    use crate::traverse;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use poker_solver_core::game::KuhnPoker;
    use poker_solver_core::info_key::InfoKey;
    use rand::SeedableRng;

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

    fn make_kuhn_net(hidden_dim: usize) -> (AdvantageNet, VarMap) {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let net = AdvantageNet::new(2, hidden_dim, &vs).unwrap();
        (net, varmap)
    }

    #[test]
    fn batched_traversal_produces_samples() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();
        let mut rng = StdRng::seed_from_u64(42);

        let samples = traverse_batched(
            &game,
            &states,
            Player::Player1,
            1,
            &net,
            &encoder,
            &device,
            &mut rng,
            20,
            8,
        )
        .unwrap();

        assert!(
            !samples.is_empty(),
            "Batched traversal should produce advantage samples"
        );

        for sample in &samples {
            assert_eq!(sample.num_actions, 2, "Kuhn poker always has 2 actions");
            assert_eq!(
                sample.advantages.len(),
                2,
                "advantages length should match num_actions"
            );
        }
    }

    #[test]
    fn batched_matches_sequential_sample_count() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();

        // Run sequential traversals
        let mut seq_rng = StdRng::seed_from_u64(99);
        let mut buffer = ReservoirBuffer::new(10_000);
        for _ in 0..50 {
            let idx = seq_rng.random_range(0..states.len());
            traverse::traverse(
                &game,
                &states[idx],
                Player::Player1,
                1,
                &net,
                &encoder,
                &mut buffer,
                &mut seq_rng,
                &device,
            )
            .unwrap();
        }

        // Run batched traversals
        let mut batch_rng = StdRng::seed_from_u64(77);
        let batch_samples = traverse_batched(
            &game,
            &states,
            Player::Player1,
            1,
            &net,
            &encoder,
            &device,
            &mut batch_rng,
            50,
            16,
        )
        .unwrap();

        // Both should produce roughly similar samples per traversal
        let seq_per_trav = buffer.len() as f64 / 50.0;
        let batch_per_trav = batch_samples.len() as f64 / 50.0;

        assert!(
            (seq_per_trav - batch_per_trav).abs() < 2.0,
            "Sequential ({seq_per_trav:.1}/trav) and batched ({batch_per_trav:.1}/trav) \
             should produce similar sample rates"
        );
    }

    #[test]
    fn batched_traversal_with_batch_size_one() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();
        let mut rng = StdRng::seed_from_u64(42);

        let samples = traverse_batched(
            &game,
            &states,
            Player::Player1,
            1,
            &net,
            &encoder,
            &device,
            &mut rng,
            5,
            1,
        )
        .unwrap();

        assert!(
            !samples.is_empty(),
            "batch_size=1 should still produce samples"
        );
    }

    #[test]
    fn both_players_produce_samples() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();

        for &player in &[Player::Player1, Player::Player2] {
            let mut rng = StdRng::seed_from_u64(42);
            let samples = traverse_batched(
                &game,
                &states,
                player,
                1,
                &net,
                &encoder,
                &device,
                &mut rng,
                20,
                8,
            )
            .unwrap();

            assert!(
                !samples.is_empty(),
                "Player {player:?} should produce samples",
            );
        }
    }

    #[test]
    fn iteration_field_preserved_in_samples() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();
        let mut rng = StdRng::seed_from_u64(42);

        let iteration = 7;
        let samples = traverse_batched(
            &game,
            &states,
            Player::Player1,
            iteration,
            &net,
            &encoder,
            &device,
            &mut rng,
            10,
            4,
        )
        .unwrap();

        for sample in &samples {
            assert_eq!(
                sample.iteration, iteration,
                "All samples should carry the iteration number"
            );
        }
    }

    #[test]
    fn cache_key_equal_for_identical_features() {
        let f1 = InfoSetFeatures {
            cards: [0, 1, 2, 3, 4, -1, -1],
            bets: [0.5f32; BET_FEATURES],
        };
        let f2 = InfoSetFeatures {
            cards: [0, 1, 2, 3, 4, -1, -1],
            bets: [0.5f32; BET_FEATURES],
        };
        assert_eq!(CacheKey::from_features(&f1), CacheKey::from_features(&f2));
    }

    #[test]
    fn cache_key_differs_for_different_bets() {
        let f1 = InfoSetFeatures {
            cards: [0, 1, 2, 3, 4, -1, -1],
            bets: [0.5f32; BET_FEATURES],
        };
        let mut bets2 = [0.5f32; BET_FEATURES];
        bets2[0] = 0.75;
        let f2 = InfoSetFeatures {
            cards: [0, 1, 2, 3, 4, -1, -1],
            bets: bets2,
        };
        assert_ne!(CacheKey::from_features(&f1), CacheKey::from_features(&f2));
    }

    #[test]
    fn cached_traversal_is_deterministic() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();

        let run = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            traverse_batched(
                &game,
                &states,
                Player::Player1,
                1,
                &net,
                &encoder,
                &device,
                &mut rng,
                30,
                8,
            )
            .unwrap()
        };

        let s1 = run(123);
        let s2 = run(123);
        assert_eq!(s1.len(), s2.len(), "same seed should produce same count");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.advantages, b.advantages, "advantages should match");
        }
    }

    #[test]
    fn cache_gets_hits_on_kuhn() {
        let game = KuhnPoker::new();
        let encoder = KuhnEncoder::new();
        let (net, _varmap) = make_kuhn_net(16);
        let device = Device::Cpu;
        let states = game.initial_states();
        let mut rng = StdRng::seed_from_u64(42);

        let mut cache: StrategyCache = HashMap::new();
        let mut stats = CacheStats::new();

        // Run enough traversals that info sets must repeat (only 12 in Kuhn)
        for _ in 0..5 {
            let b = 10;
            run_batch(
                &game,
                &states,
                Player::Player1,
                1,
                &net,
                &encoder,
                &device,
                &mut rng,
                b,
                &mut cache,
                &mut stats,
            )
            .unwrap();
        }

        assert!(
            stats.hits > 0,
            "With 50 Kuhn traversals and only 12 info sets, \
             cache should have hits (got {hits} hits, {misses} misses)",
            hits = stats.hits,
            misses = stats.misses,
        );
    }
}
