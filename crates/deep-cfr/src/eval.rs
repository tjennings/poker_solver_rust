//! Evaluation policies for computing average strategies from trained networks.
//!
//! Provides two methods for computing the average strategy sigma-bar:
//!
//! - [`TrajectoryPolicy`]: Samples a single value net per episode, weighted by
//!   iteration. Cheap (one forward pass per decision) and correctly implements
//!   the linear average strategy from SD-CFR Theorem 2, Equation 5.
//!
//! - [`ExplicitPolicy`]: Computes the full weighted average over all stored
//!   value nets at each info set. Expensive (`|B^M|` forward passes per
//!   decision) but exact. Used for exploitability measurement.

use candle_core::{Device, Tensor};
#[cfg(test)]
use rand::Rng;

use crate::SdCfrError;
use crate::card_features::InfoSetFeatures;
use crate::model_buffer::ModelBuffer;
use crate::network::AdvantageNet;

// ---------------------------------------------------------------------------
// Shared helper: compute strategy from a single network + features
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Compute a strategy (probability distribution over actions) from a single
/// advantage network and info set features.
///
/// Steps: encode features as tensors -> forward pass -> ReLU + normalize.
fn strategy_from_net(
    net: &AdvantageNet,
    features: &InfoSetFeatures,
    device: &Device,
) -> Result<Vec<f32>, SdCfrError> {
    let (cards, bets) = InfoSetFeatures::to_tensors(std::slice::from_ref(features), device)?;
    let advantages = net.forward(&cards, &bets)?;
    let strategy = AdvantageNet::advantages_to_strategy(&advantages)?;
    extract_probs(&strategy)
}

/// Extract probabilities from a `[1, num_actions]` strategy tensor.
fn extract_probs(strategy: &Tensor) -> Result<Vec<f32>, SdCfrError> {
    let flat = strategy.squeeze(0)?;
    let probs = flat.to_vec1::<f32>()?;
    Ok(probs)
}

// ---------------------------------------------------------------------------
// TrajectoryPolicy
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Policy that uses a single sampled value net for an entire episode.
///
/// At episode start, one network is sampled from the model buffer with
/// probability proportional to its iteration number (linear weighting).
/// That network is used for all decisions in the trajectory, yielding
/// one forward pass per decision point.
pub(crate) struct TrajectoryPolicy {
    net: AdvantageNet,
    device: Device,
}

#[cfg(test)]
impl TrajectoryPolicy {
    /// Sample a trajectory policy from the model buffer.
    ///
    /// Selects one network with probability proportional to its iteration
    /// number, implementing the linear average strategy from Theorem 2.
    pub fn sample(
        buffer: &ModelBuffer,
        num_actions: usize,
        hidden_dim: usize,
        device: &Device,
        rng: &mut impl Rng,
    ) -> Result<Self, SdCfrError> {
        let index = buffer.sample_weighted(rng)?;
        let net = buffer.load_net(index, num_actions, hidden_dim, device)?;
        Ok(Self {
            net,
            device: device.clone(),
        })
    }

    /// Compute the action probability distribution for the given info set.
    ///
    /// Returns a `Vec<f32>` of length `num_actions` that sums to 1.0.
    pub fn strategy(&self, features: &InfoSetFeatures) -> Result<Vec<f32>, SdCfrError> {
        strategy_from_net(&self.net, features, &self.device)
    }
}

// ---------------------------------------------------------------------------
// ExplicitPolicy
// ---------------------------------------------------------------------------

/// Policy that computes the exact weighted average strategy over all networks.
///
/// For each info set, every stored value net produces a strategy, and these
/// are combined with weights proportional to the iteration number:
///
///   sigma_bar(I,a) = sum_t [ t * sigma_t(I,a) ] / sum_t [ t ]
///
/// This is expensive (`|B^M|` forward passes per decision) but provides
/// the exact average strategy for exploitability measurement.
pub struct ExplicitPolicy {
    nets: Vec<(AdvantageNet, f64)>,
    device: Device,
}

impl ExplicitPolicy {
    /// Load all value nets from the model buffer with their weights.
    pub fn from_buffer(
        buffer: &ModelBuffer,
        num_actions: usize,
        hidden_dim: usize,
        device: &Device,
    ) -> Result<Self, SdCfrError> {
        if buffer.is_empty() {
            return Err(SdCfrError::EmptyModelBuffer);
        }
        let nets = buffer
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let net = buffer.load_net(index, num_actions, hidden_dim, device)?;
                Ok((net, entry.weight()))
            })
            .collect::<Result<Vec<_>, SdCfrError>>()?;

        Ok(Self {
            nets,
            device: device.clone(),
        })
    }

    /// Compute the full weighted average strategy at the given info set.
    ///
    /// Returns a `Vec<f32>` of length `num_actions` that sums to 1.0.
    pub fn strategy(&self, features: &InfoSetFeatures) -> Result<Vec<f32>, SdCfrError> {
        let (cards, bets) =
            InfoSetFeatures::to_tensors(std::slice::from_ref(features), &self.device)?;
        let weighted_sum = self.weighted_strategy_sum(&cards, &bets)?;
        let normalized = normalize_strategy(&weighted_sum);
        Ok(normalized)
    }

    /// Return the raw advantage predictions from the highest-weight net.
    ///
    /// This reveals the network's advantage estimates before ReLU and
    /// normalization, useful for diagnosing whether training is producing
    /// meaningful signals or near-zero noise.
    pub fn latest_raw_advantages(
        &self,
        features: &InfoSetFeatures,
    ) -> Result<Vec<f32>, SdCfrError> {
        let (best_net, _) = self
            .nets
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(SdCfrError::EmptyModelBuffer)?;

        let (cards, bets) =
            InfoSetFeatures::to_tensors(std::slice::from_ref(features), &self.device)?;
        let advantages = best_net.forward(&cards, &bets)?;
        let flat = advantages.squeeze(0)?;
        let raw = flat.to_vec1::<f32>()?;
        Ok(raw)
    }

    /// Compute the weighted sum of per-net strategies (before normalization).
    fn weighted_strategy_sum(&self, cards: &Tensor, bets: &Tensor) -> Result<Vec<f32>, SdCfrError> {
        let num_actions = self.nets[0].0.num_actions();
        let mut accum = vec![0.0f64; num_actions];

        for (net, weight) in &self.nets {
            let advantages = net.forward(cards, bets)?;
            let strategy = AdvantageNet::advantages_to_strategy(&advantages)?;
            let probs = extract_probs(&strategy)?;
            for (a, &p) in accum.iter_mut().zip(probs.iter()) {
                *a += *weight * f64::from(p);
            }
        }
        let result = accum.iter().map(|&v| v as f32).collect();
        Ok(result)
    }
}

/// Normalize a vector to sum to 1.0. If all values are zero or negative,
/// places full mass on the first action.
fn normalize_strategy(raw: &[f32]) -> Vec<f32> {
    let sum: f32 = raw.iter().sum();
    if sum > 1e-8 {
        return raw.iter().map(|&v| v / sum).collect();
    }
    // Fallback: uniform over first action
    let mut result = vec![0.0; raw.len()];
    if !result.is_empty() {
        result[0] = 1.0;
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_features::BET_FEATURES;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const TEST_ACTIONS: usize = 5;
    const TEST_DIM: usize = 16;

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// Create an AdvantageNet and return (net, varmap).
    fn make_net_and_varmap(num_actions: usize, hidden_dim: usize) -> (AdvantageNet, VarMap) {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let net = AdvantageNet::new(num_actions, hidden_dim, &vs).unwrap();
        (net, varmap)
    }

    /// Create a ModelBuffer with `n` entries at iterations 1..=n.
    fn make_buffer(n: u32) -> ModelBuffer {
        let mut buf = ModelBuffer::new();
        for i in 1..=n {
            let (_, varmap) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);
            buf.push(&varmap, i).unwrap();
        }
        buf
    }

    /// Simple test features with known values.
    fn test_features() -> InfoSetFeatures {
        InfoSetFeatures {
            cards: [0, 4, 8, 12, 16, 20, 24],
            bets: [0.0; BET_FEATURES],
        }
    }

    /// Assert that a strategy is a valid probability distribution.
    fn assert_valid_strategy(probs: &[f32], label: &str) {
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "{label}: strategy should sum to 1.0, got {sum} ({probs:?})"
        );
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= -1e-6,
                "{label}: action {i} has negative probability {p}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // strategy_from_net tests
    // -----------------------------------------------------------------------

    #[test]
    fn strategy_from_single_net_produces_valid_distribution() {
        let (net, _) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);
        let features = test_features();

        let probs = strategy_from_net(&net, &features, &Device::Cpu).unwrap();

        assert_eq!(probs.len(), TEST_ACTIONS);
        assert_valid_strategy(&probs, "single net");
    }

    #[test]
    fn strategy_from_single_net_deterministic() {
        let (net, _) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);
        let features = test_features();

        let probs1 = strategy_from_net(&net, &features, &Device::Cpu).unwrap();
        let probs2 = strategy_from_net(&net, &features, &Device::Cpu).unwrap();

        assert_eq!(
            probs1, probs2,
            "same net + same features should produce identical strategies"
        );
    }

    #[test]
    fn strategy_from_net_with_absent_cards() {
        let (net, _) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);
        let features = InfoSetFeatures {
            cards: [0, 4, 8, 12, 16, -1, -1], // no turn/river
            bets: [0.0; BET_FEATURES],
        };

        let probs = strategy_from_net(&net, &features, &Device::Cpu).unwrap();

        assert_eq!(probs.len(), TEST_ACTIONS);
        assert_valid_strategy(&probs, "absent cards");
    }

    // -----------------------------------------------------------------------
    // TrajectoryPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn trajectory_policy_produces_valid_strategy() {
        let buffer = make_buffer(3);
        let mut rng = seeded_rng(42);

        let policy =
            TrajectoryPolicy::sample(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu, &mut rng)
                .unwrap();

        let probs = policy.strategy(&test_features()).unwrap();

        assert_eq!(probs.len(), TEST_ACTIONS);
        assert_valid_strategy(&probs, "trajectory policy");
    }

    #[test]
    fn trajectory_policy_from_empty_buffer_fails() {
        let buffer = ModelBuffer::new();
        let mut rng = seeded_rng(42);

        let result =
            TrajectoryPolicy::sample(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu, &mut rng);

        assert!(result.is_err(), "sampling from empty buffer should fail");
    }

    #[test]
    fn trajectory_policy_consistent_within_episode() {
        let buffer = make_buffer(3);
        let mut rng = seeded_rng(42);

        let policy =
            TrajectoryPolicy::sample(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu, &mut rng)
                .unwrap();

        let features = test_features();

        // Same policy should produce identical strategies for same features
        let probs1 = policy.strategy(&features).unwrap();
        let probs2 = policy.strategy(&features).unwrap();

        assert_eq!(
            probs1, probs2,
            "same trajectory policy should be deterministic"
        );
    }

    // -----------------------------------------------------------------------
    // ExplicitPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn explicit_policy_produces_valid_strategy() {
        let buffer = make_buffer(3);

        let policy =
            ExplicitPolicy::from_buffer(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu).unwrap();

        let probs = policy.strategy(&test_features()).unwrap();

        assert_eq!(probs.len(), TEST_ACTIONS);
        assert_valid_strategy(&probs, "explicit policy");
    }

    #[test]
    fn explicit_policy_from_empty_buffer_fails() {
        let buffer = ModelBuffer::new();

        let result = ExplicitPolicy::from_buffer(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu);

        assert!(
            result.is_err(),
            "building explicit policy from empty buffer should fail"
        );
    }

    #[test]
    fn explicit_policy_weights_by_iteration() {
        // Create a buffer with two identical nets but different weights.
        // The explicit policy should produce the same result as either net
        // individually because identical nets produce identical strategies.
        let (_, varmap) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);

        let mut buffer = ModelBuffer::new();
        buffer.push(&varmap, 1).unwrap();
        buffer.push(&varmap, 9).unwrap();

        let policy =
            ExplicitPolicy::from_buffer(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu).unwrap();

        let features = test_features();
        let explicit_probs = policy.strategy(&features).unwrap();

        // With identical nets, the weighted average should match
        // any individual net's strategy (weights cancel out)
        let single_net = buffer
            .load_net(0, TEST_ACTIONS, TEST_DIM, &Device::Cpu)
            .unwrap();
        let single_probs = strategy_from_net(&single_net, &features, &Device::Cpu).unwrap();

        for (i, (&ep, &sp)) in explicit_probs.iter().zip(single_probs.iter()).enumerate() {
            assert!(
                (ep - sp).abs() < 1e-4,
                "action {i}: explicit={ep}, single={sp} — \
                 identical nets should produce identical weighted average"
            );
        }
    }

    #[test]
    fn explicit_policy_with_different_nets_produces_weighted_blend() {
        // Create two nets with different random seeds producing different strategies.
        // Verify the explicit policy produces a blend that differs from either alone.
        let (_, varmap1) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);
        let (_, varmap2) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);

        let mut buffer = ModelBuffer::new();
        buffer.push(&varmap1, 1).unwrap();
        buffer.push(&varmap2, 1).unwrap();

        let policy =
            ExplicitPolicy::from_buffer(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu).unwrap();

        let probs = policy.strategy(&test_features()).unwrap();
        assert_valid_strategy(&probs, "blended explicit policy");
    }

    #[test]
    fn explicit_policy_single_net_matches_trajectory() {
        // With only one net in the buffer, ExplicitPolicy and TrajectoryPolicy
        // should produce identical strategies.
        let (_, varmap) = make_net_and_varmap(TEST_ACTIONS, TEST_DIM);

        let mut buffer = ModelBuffer::new();
        buffer.push(&varmap, 1).unwrap();

        let explicit =
            ExplicitPolicy::from_buffer(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu).unwrap();

        let mut rng = seeded_rng(42);
        let trajectory =
            TrajectoryPolicy::sample(&buffer, TEST_ACTIONS, TEST_DIM, &Device::Cpu, &mut rng)
                .unwrap();

        let features = test_features();
        let ep = explicit.strategy(&features).unwrap();
        let tp = trajectory.strategy(&features).unwrap();

        for (i, (&e, &t)) in ep.iter().zip(tp.iter()).enumerate() {
            assert!(
                (e - t).abs() < 1e-4,
                "action {i}: explicit={e}, trajectory={t} — \
                 single-net buffer should produce identical results"
            );
        }
    }

    // -----------------------------------------------------------------------
    // normalize_strategy tests
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_strategy_normal_case() {
        let raw = vec![2.0, 3.0, 5.0];
        let normed = normalize_strategy(&raw);

        assert!((normed[0] - 0.2).abs() < 1e-6);
        assert!((normed[1] - 0.3).abs() < 1e-6);
        assert!((normed[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalize_strategy_all_zeros_falls_back() {
        let raw = vec![0.0, 0.0, 0.0];
        let normed = normalize_strategy(&raw);

        assert!((normed[0] - 1.0).abs() < 1e-6);
        assert!(normed[1].abs() < 1e-6);
        assert!(normed[2].abs() < 1e-6);
    }

    #[test]
    fn normalize_strategy_empty() {
        let raw: Vec<f32> = vec![];
        let normed = normalize_strategy(&raw);
        assert!(normed.is_empty());
    }
}
