//! Two-branch MLP advantage network for Single Deep CFR.
//!
//! The network has a card branch (embedding-based) and a bet branch (dense),
//! whose outputs are concatenated and fed through a combined trunk to produce
//! per-action advantage values.

use candle_core::{DType, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear};

/// Number of card groups for the card branch output.
const NUM_CARD_GROUPS: usize = 4;

/// Bet feature input size: 4 rounds x 6 positions x 2 features.
const BET_INPUT_DIM: usize = 48;

/// Boundaries for grouping the 7 card slots into 4 groups.
/// Group 0: hole cards (indices 0..2)
/// Group 1: flop cards (indices 2..5)
/// Group 2: turn card  (indices 5..6)
/// Group 3: river card (indices 6..7)
const CARD_GROUP_BOUNDS: [(usize, usize); NUM_CARD_GROUPS] = [(0, 2), (2, 5), (5, 6), (6, 7)];

/// Two-branch MLP that predicts per-action advantages.
///
/// Card branch processes card indices through embeddings, bet branch processes
/// betting features through dense layers, and the trunk combines both.
pub struct AdvantageNet {
    // Embeddings (card branch)
    rank_emb: Embedding,
    suit_emb: Embedding,
    card_emb: Embedding,
    // Card branch dense layers
    card_l1: Linear,
    card_l2: Linear,
    card_l3: Linear,
    // Bet branch dense layers
    bet_l1: Linear,
    bet_l2: Linear,
    // Combined trunk
    trunk_l1: Linear,
    trunk_l2: Linear,
    trunk_l3: Linear,
    // Output head
    output: Linear,
    num_actions: usize,
}

impl AdvantageNet {
    /// Create a new network with random weights.
    ///
    /// `num_actions` is the number of output advantage values.
    /// `hidden_dim` controls the width of all hidden layers.
    pub fn new(
        num_actions: usize,
        hidden_dim: usize,
        vs: &VarBuilder,
    ) -> Result<Self, candle_core::Error> {
        let rank_emb = embedding(13, hidden_dim, vs.pp("rank_emb"))?;
        let suit_emb = embedding(4, hidden_dim, vs.pp("suit_emb"))?;
        let card_emb = embedding(52, hidden_dim, vs.pp("card_emb"))?;

        let card_l1 = linear(NUM_CARD_GROUPS * hidden_dim, hidden_dim, vs.pp("card_l1"))?;
        let card_l2 = linear(hidden_dim, hidden_dim, vs.pp("card_l2"))?;
        let card_l3 = linear(hidden_dim, hidden_dim, vs.pp("card_l3"))?;

        let bet_l1 = linear(BET_INPUT_DIM, hidden_dim, vs.pp("bet_l1"))?;
        let bet_l2 = linear(hidden_dim, hidden_dim, vs.pp("bet_l2"))?;

        let trunk_l1 = linear(2 * hidden_dim, hidden_dim, vs.pp("trunk_l1"))?;
        let trunk_l2 = linear(hidden_dim, hidden_dim, vs.pp("trunk_l2"))?;
        let trunk_l3 = linear(hidden_dim, hidden_dim, vs.pp("trunk_l3"))?;

        let output = linear(hidden_dim, num_actions, vs.pp("output"))?;

        Ok(Self {
            rank_emb,
            suit_emb,
            card_emb,
            card_l1,
            card_l2,
            card_l3,
            bet_l1,
            bet_l2,
            trunk_l1,
            trunk_l2,
            trunk_l3,
            output,
            num_actions,
        })
    }

    /// Forward pass producing raw advantage values.
    ///
    /// `card_indices`: `[B, 7]` i64 tensor with values 0..51 or -1 for absent cards.
    /// `bets`: `[B, 48]` f32 tensor of bet features.
    /// Returns: `[B, num_actions]` f32 tensor of advantages.
    pub fn forward(
        &self,
        card_indices: &Tensor,
        bets: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let card_out = self.card_branch(card_indices)?;
        let bet_out = self.bet_branch(bets)?;
        self.trunk(&card_out, &bet_out)
    }

    /// Convert raw advantages to a valid strategy (probability distribution).
    ///
    /// Clamps negative advantages to zero and normalizes. When all advantages
    /// are non-positive, places full mass on the highest-advantage action.
    pub fn advantages_to_strategy(advantages: &Tensor) -> Result<Tensor, candle_core::Error> {
        let relu_adv = advantages.relu()?;
        let sums = relu_adv.sum_keepdim(1)?;

        // Build mask for rows where all advantages <= 0 (relu sum == 0)
        let all_zero = sums.le(0.0f32)?.to_dtype(DType::F32)?;
        let has_positive = (all_zero.neg()? + 1.0)?;

        // Normalized strategy from ReLU advantages
        let normalized = relu_adv.broadcast_div(&sums.clamp(1e-8, f64::MAX)?)?;

        // Deterministic fallback: argmax for rows where all advantages <= 0
        let argmax_strategy = deterministic_argmax(advantages)?;

        // Blend: where all_zero, use argmax; otherwise use normalized
        let result = argmax_strategy
            .broadcast_mul(&all_zero)?
            .add(&normalized.broadcast_mul(&has_positive)?)?;
        Ok(result)
    }

    /// Number of actions this network outputs.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }
}

// ---------------------------------------------------------------------------
// Private implementation: card branch
// ---------------------------------------------------------------------------

impl AdvantageNet {
    /// Embed all 7 card slots, masking absent cards, and sum within groups.
    fn card_branch(&self, card_indices: &Tensor) -> Result<Tensor, candle_core::Error> {
        let embedded = self.embed_cards(card_indices)?;
        let grouped = sum_card_groups(&embedded)?;
        apply_card_layers(&self.card_l1, &self.card_l2, &self.card_l3, &grouped)
    }

    /// Produce per-card embeddings `[B, 7, dim]` with absent cards zeroed out.
    fn embed_cards(&self, card_indices: &Tensor) -> Result<Tensor, candle_core::Error> {
        let valid_mask = build_valid_mask(card_indices)?;
        let safe_indices = card_indices.clamp(0i64, 51i64)?;

        let ranks = compute_ranks(&safe_indices)?;
        let suits = compute_suits(&safe_indices)?;

        let rank_vec = self.rank_emb.forward(&ranks)?;
        let suit_vec = self.suit_emb.forward(&suits)?;
        let card_vec = self.card_emb.forward(&safe_indices)?;

        let summed = rank_vec.add(&suit_vec)?.add(&card_vec)?;
        // valid_mask is [B, 7, 1] -- broadcast multiply to zero absent cards
        summed.broadcast_mul(&valid_mask)
    }
}

/// Build a `[B, 7, 1]` f32 mask: 1.0 for valid cards, 0.0 for absent (-1).
fn build_valid_mask(card_indices: &Tensor) -> Result<Tensor, candle_core::Error> {
    let is_absent = card_indices.eq(-1i64)?.to_dtype(DType::F32)?;
    let valid = (is_absent.neg()? + 1.0)?;
    valid.unsqueeze(2)
}

/// Compute rank indices (card_index / 4) as i64 tensor.
fn compute_ranks(safe_indices: &Tensor) -> Result<Tensor, candle_core::Error> {
    // Integer division: floor(index / 4)
    let float_idx = safe_indices.to_dtype(DType::F32)?;
    let ranks = (float_idx / 4.0)?.floor()?;
    ranks.to_dtype(DType::I64)
}

/// Compute suit indices (card_index % 4) as i64 tensor.
fn compute_suits(safe_indices: &Tensor) -> Result<Tensor, candle_core::Error> {
    let float_idx = safe_indices.to_dtype(DType::F32)?;
    let div = (float_idx.clone() / 4.0)?.floor()?;
    let remainder = (float_idx - (div * 4.0)?)?;
    remainder.to_dtype(DType::I64)
}

/// Sum embedded cards within each of the 4 groups, returning `[B, 4*dim]`.
fn sum_card_groups(embedded: &Tensor) -> Result<Tensor, candle_core::Error> {
    let groups: Vec<Tensor> = CARD_GROUP_BOUNDS
        .iter()
        .map(|&(start, end)| {
            let slice = embedded.narrow(1, start, end - start)?;
            slice.sum(1)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Tensor::cat(&groups, 1)
}

/// Apply the 3-layer card branch MLP with ReLU activations.
fn apply_card_layers(
    l1: &Linear,
    l2: &Linear,
    l3: &Linear,
    x: &Tensor,
) -> Result<Tensor, candle_core::Error> {
    let h = l1.forward(x)?.relu()?;
    let h = l2.forward(&h)?.relu()?;
    l3.forward(&h)?.relu()
}

// ---------------------------------------------------------------------------
// Private implementation: bet branch
// ---------------------------------------------------------------------------

impl AdvantageNet {
    /// Two-layer bet branch with a skip connection on the second layer.
    fn bet_branch(&self, bets: &Tensor) -> Result<Tensor, candle_core::Error> {
        let h = self.bet_l1.forward(bets)?.relu()?;
        // Skip connection: add input of second linear to its output
        let out = self.bet_l2.forward(&h)?;
        out.add(&h)?.relu()
    }
}

// ---------------------------------------------------------------------------
// Private implementation: trunk
// ---------------------------------------------------------------------------

impl AdvantageNet {
    /// Combined trunk: concat card+bet outputs, 3 layers with skip, normalize, output.
    fn trunk(&self, card_out: &Tensor, bet_out: &Tensor) -> Result<Tensor, candle_core::Error> {
        let combined = Tensor::cat(&[card_out, bet_out], 1)?;
        let h = self.trunk_l1.forward(&combined)?.relu()?;
        let h = apply_skip_relu(&self.trunk_l2, &h)?;
        let h = apply_skip_relu(&self.trunk_l3, &h)?;
        let h = layer_normalize(&h)?;
        self.output.forward(&h)
    }
}

/// Apply a linear layer with a skip connection and ReLU activation.
/// Requires input and output dimensions to match (dim -> dim).
fn apply_skip_relu(layer: &Linear, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    layer.forward(x)?.add(x)?.relu()
}

/// Zero-mean, unit-variance normalization: `(x - mean) / (std + 1e-6)`.
/// Applied per-sample across the feature dimension (last dim).
fn layer_normalize(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mean = x.mean_keepdim(1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let std = (var + 1e-6)?.sqrt()?;
    centered.broadcast_div(&std)
}

/// Produce a one-hot tensor for the argmax action in each batch row.
/// Used as fallback when all advantages are non-positive.
fn deterministic_argmax(advantages: &Tensor) -> Result<Tensor, candle_core::Error> {
    let num_actions = advantages.dim(1)?;
    let batch_size = advantages.dim(0)?;
    // argmax_keepdim returns U32
    let argmax = advantages.argmax_keepdim(1)?;

    // Build range as U32 to match argmax dtype
    let indices = Tensor::arange(0u32, num_actions as u32, advantages.device())?.unsqueeze(0)?;
    let indices = indices.broadcast_as((batch_size, num_actions))?;
    let argmax = argmax.broadcast_as((batch_size, num_actions))?;

    indices.eq(&argmax)?.to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    /// Helper: create an AdvantageNet for testing.
    fn make_net(num_actions: usize, hidden_dim: usize) -> AdvantageNet {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        AdvantageNet::new(num_actions, hidden_dim, &vs).unwrap()
    }

    /// Helper: create dummy card indices `[B, 7]` with valid values.
    fn dummy_cards(batch: usize) -> Tensor {
        let data: Vec<i64> = (0..batch)
            .flat_map(|b| {
                // 2 hole + 3 flop + turn + river, all distinct per sample
                let offset = (b as i64 * 7) % 46;
                (offset..offset + 7).collect::<Vec<_>>()
            })
            .collect();
        Tensor::from_vec(data, (batch, 7), &Device::Cpu).unwrap()
    }

    /// Helper: create dummy bet features `[B, 48]`.
    fn dummy_bets(batch: usize) -> Tensor {
        Tensor::zeros((batch, BET_INPUT_DIM), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn output_shape_matches_num_actions() {
        let num_actions = 10;
        let net = make_net(num_actions, 32);
        let cards = dummy_cards(4);
        let bets = dummy_bets(4);

        let out = net.forward(&cards, &bets).unwrap();
        assert_eq!(out.dims(), &[4, num_actions]);
    }

    #[test]
    fn strategy_sums_to_one() {
        let advantages = Tensor::new(
            &[[1.0f32, -2.0, 3.0, 0.5], [0.1, 0.2, 0.3, 0.4]],
            &Device::Cpu,
        )
        .unwrap();
        let strategy = AdvantageNet::advantages_to_strategy(&advantages).unwrap();
        let sums = strategy.sum(1).unwrap().to_vec1::<f32>().unwrap();

        for (i, s) in sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "Row {i} strategy sum = {s}, expected 1.0"
            );
        }

        // Verify no negative probabilities
        let min = strategy
            .flatten_all()
            .unwrap()
            .min(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(min >= 0.0, "Strategy contains negative probability: {min}");
    }

    #[test]
    fn all_negative_advantages_picks_highest() {
        let advantages = Tensor::new(&[[-3.0f32, -1.0, -5.0, -2.0]], &Device::Cpu).unwrap();
        let strategy = AdvantageNet::advantages_to_strategy(&advantages).unwrap();
        let probs = strategy.to_vec2::<f32>().unwrap();

        // Action index 1 has highest advantage (-1.0)
        assert!(
            (probs[0][1] - 1.0).abs() < 1e-5,
            "Expected action 1 to get all mass, got {:?}",
            probs[0]
        );
        // All others should be zero
        for (i, &p) in probs[0].iter().enumerate() {
            if i != 1 {
                assert!(p.abs() < 1e-5, "Expected action {i} to be 0, got {p}");
            }
        }
    }

    #[test]
    fn skip_connections_affect_output() {
        // Verify that skip connections are active by zeroing a skip-connected
        // layer's weights and checking the output is still non-zero (because
        // the skip adds back the input).
        let num_actions = 5;
        let dim = 16;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let net = AdvantageNet::new(num_actions, dim, &vs).unwrap();

        let cards = dummy_cards(2);
        let bets = Tensor::ones((2, BET_INPUT_DIM), DType::F32, &Device::Cpu).unwrap();

        // Normal forward
        let out_normal = net.forward(&cards, &bets).unwrap();

        // Zero out bet_l2 weights -- skip connection should still pass through
        {
            let data = varmap.data().lock().unwrap();
            let w = data.get("bet_l2.weight").unwrap();
            let zeros = Tensor::zeros(w.shape(), DType::F32, &Device::Cpu).unwrap();
            w.set(&zeros).unwrap();
            let b = data.get("bet_l2.bias").unwrap();
            let zeros_b = Tensor::zeros(b.shape(), DType::F32, &Device::Cpu).unwrap();
            b.set(&zeros_b).unwrap();
        }

        let out_zeroed = net.forward(&cards, &bets).unwrap();

        // Outputs should differ because bet_l2 contributed in the original
        let diff_norm = out_normal
            .sub(&out_zeroed)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff_norm > 1e-6,
            "Zeroing skip-connected layer should change output (diff_norm={diff_norm})"
        );

        // But zeroed-weight output should still be non-zero (skip passes through)
        let out_norm = out_zeroed
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            out_norm > 1e-6,
            "Skip connection should keep output non-zero (out_norm={out_norm})"
        );
    }

    #[test]
    fn absent_cards_produce_zero_embeddings() {
        let dim = 16;
        let net = make_net(5, dim);

        // All cards present
        let cards_present = Tensor::new(&[[0i64, 1, 2, 3, 4, 5, 6]], &Device::Cpu).unwrap();
        // Some cards absent (-1)
        let cards_absent = Tensor::new(&[[0i64, 1, -1, -1, -1, -1, -1]], &Device::Cpu).unwrap();

        let emb_present = net.embed_cards(&cards_present).unwrap();
        let emb_absent = net.embed_cards(&cards_absent).unwrap();

        // Absent cards (indices 2..7) should produce zero vectors
        for slot in 2..7 {
            let vec = emb_absent.narrow(1, slot, 1).unwrap().squeeze(1).unwrap();
            let norm = vec
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                norm.abs() < 1e-10,
                "Absent card at slot {slot} should be zero, got norm={norm}"
            );
        }

        // Present cards (indices 0,1) should be identical in both
        for slot in 0..2 {
            let vec_p = emb_present.narrow(1, slot, 1).unwrap();
            let vec_a = emb_absent.narrow(1, slot, 1).unwrap();
            let diff = vec_p
                .sub(&vec_a)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-10,
                "Present card at slot {slot} should match, diff={diff}"
            );
        }
    }

    #[test]
    fn num_actions_accessor() {
        let net = make_net(14, 32);
        assert_eq!(net.num_actions(), 14);
    }

    #[test]
    fn single_sample_batch() {
        let net = make_net(8, 16);
        let cards = dummy_cards(1);
        let bets = dummy_bets(1);
        let out = net.forward(&cards, &bets).unwrap();
        assert_eq!(out.dims(), &[1, 8]);
    }

    #[test]
    fn strategy_with_mixed_positive_negative() {
        // Only positive advantages should contribute to the strategy
        let advantages = Tensor::new(&[[5.0f32, -1.0, 3.0, -2.0, 2.0]], &Device::Cpu).unwrap();
        let strategy = AdvantageNet::advantages_to_strategy(&advantages).unwrap();
        let probs = strategy.to_vec2::<f32>().unwrap();

        // Negative-advantage actions get zero probability
        assert!(probs[0][1].abs() < 1e-6, "Negative advantage should be 0");
        assert!(probs[0][3].abs() < 1e-6, "Negative advantage should be 0");

        // Positive advantages: 5, 3, 2 -> normalized to 5/10, 3/10, 2/10
        assert!(
            (probs[0][0] - 0.5).abs() < 1e-5,
            "Expected 0.5, got {}",
            probs[0][0]
        );
        assert!(
            (probs[0][2] - 0.3).abs() < 1e-5,
            "Expected 0.3, got {}",
            probs[0][2]
        );
        assert!(
            (probs[0][4] - 0.2).abs() < 1e-5,
            "Expected 0.2, got {}",
            probs[0][4]
        );
    }

    #[test]
    fn all_zero_advantages_picks_first() {
        let advantages = Tensor::new(&[[0.0f32, 0.0, 0.0]], &Device::Cpu).unwrap();
        let strategy = AdvantageNet::advantages_to_strategy(&advantages).unwrap();
        let probs = strategy.to_vec2::<f32>().unwrap();
        let sum: f32 = probs[0].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Strategy should sum to 1, got {sum}"
        );
        // argmax of all-zeros picks index 0 in candle
        assert!(
            probs[0][0] > 0.5,
            "Expected first action to get mass, got {:?}",
            probs[0]
        );
    }
}
