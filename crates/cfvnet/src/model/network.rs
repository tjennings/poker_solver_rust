use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PRelu, PReluConfig,
    },
    tensor::{backend::Backend, Tensor},
};

/// Number of hole-card combinations (52 choose 2).
pub const NUM_COMBOS: usize = 1326;
/// One counterfactual value per combo.
pub const OUTPUT_SIZE: usize = NUM_COMBOS;
/// Fixed input feature size: OOP range (1326) + IP range (1326) + board one-hot (52) + SPR (1).
pub const INPUT_SIZE: usize = NUM_COMBOS + NUM_COMBOS + 52 + 1; // = 2705
/// Dual output: OOP CFVs (1326) + IP CFVs (1326).
pub const DUAL_OUTPUT_SIZE: usize = 2 * NUM_COMBOS; // = 2652

/// Compute input feature size for a given number of board cards.
///
/// Now returns the fixed [`INPUT_SIZE`] (2705) regardless of `board_cards`,
/// since board encoding uses a 52-dim one-hot vector. The parameter is
/// retained for API compatibility.
pub fn input_size(_board_cards: usize) -> usize {
    INPUT_SIZE
}

/// A single hidden block: Linear -> BatchNorm -> PReLU.
#[derive(Module, Debug)]
struct HiddenBlock<B: Backend> {
    linear: Linear<B>,
    norm: BatchNorm<B, 1>,
    activation: PRelu<B>,
}

impl<B: Backend> HiddenBlock<B> {
    fn new(device: &B::Device, in_features: usize, out_features: usize) -> Self {
        Self {
            linear: LinearConfig::new(in_features, out_features).init(device),
            norm: BatchNormConfig::new(out_features).init(device),
            activation: PReluConfig::new()
                .with_num_parameters(out_features)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(x);
        // BatchNorm<B, 1> expects [batch, channels, length]; reshape 2D -> 3D -> 2D.
        let [batch, features] = x.dims();
        let x = x.reshape([batch, features, 1]);
        let x = self.norm.forward(x);
        let x = x.reshape([batch, features]);
        self.activation.forward(x)
    }
}

/// Deep Counterfactual Value Network.
///
/// Architecture: `Input(in_size) -> [Linear -> BatchNorm -> PReLU] x N -> Linear(DUAL_OUTPUT_SIZE)`
///
/// The output layer produces `[batch, 2652]` raw values: the first 1326 are OOP CFVs,
/// the last 1326 are IP CFVs. The forward pass applies a zero-sum correction so that
/// `dot(range_oop, cfv_oop) + dot(range_ip, cfv_ip) ≈ 0`.
#[derive(Module, Debug)]
pub struct CfvNet<B: Backend> {
    hidden: Vec<HiddenBlock<B>>,
    output: Linear<B>,
}

impl<B: Backend> CfvNet<B> {
    /// Build a new network with `num_layers` hidden blocks of width `hidden_size`.
    ///
    /// `in_size` is the input feature dimension, typically from [`input_size`].
    pub fn new(device: &B::Device, num_layers: usize, hidden_size: usize, in_size: usize) -> Self {
        assert!(num_layers > 0, "need at least one hidden layer");

        let mut hidden = Vec::with_capacity(num_layers);
        hidden.push(HiddenBlock::new(device, in_size, hidden_size));
        for _ in 1..num_layers {
            hidden.push(HiddenBlock::new(device, hidden_size, hidden_size));
        }

        let output = LinearConfig::new(hidden_size, DUAL_OUTPUT_SIZE).init(device);

        Self { hidden, output }
    }

    /// Forward pass with zero-sum correction.
    ///
    /// Takes input features `x` of shape `[batch, in_size]` and both players' ranges
    /// `range_oop`, `range_ip` of shape `[batch, NUM_COMBOS]`. Returns `[batch, DUAL_OUTPUT_SIZE]`
    /// corrected CFVs where `dot(range_oop, cfv_oop) + dot(range_ip, cfv_ip) ≈ 0`.
    pub fn forward(
        &self,
        mut x: Tensor<B, 2>,
        range_oop: Tensor<B, 2>,
        range_ip: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        for block in &self.hidden {
            x = block.forward(x);
        }
        let raw = self.output.forward(x);

        // Split into OOP and IP halves.
        let oop_raw = raw.clone().narrow(1, 0, NUM_COMBOS);
        let ip_raw = raw.narrow(1, NUM_COMBOS, NUM_COMBOS);

        // Compute game values: gv = sum(cfv * range) for each player.
        let gv_oop: Tensor<B, 1> = (oop_raw.clone() * range_oop.clone()).sum_dim(1).squeeze(1);
        let gv_ip: Tensor<B, 1> = (ip_raw.clone() * range_ip.clone()).sum_dim(1).squeeze(1);

        // Zero-sum error: should be zero in a perfect model.
        let error: Tensor<B, 1> = (gv_oop + gv_ip).div_scalar(2.0);

        // Distribute the error uniformly across each player's range.
        let sum_oop: Tensor<B, 1> = range_oop.clone().sum_dim(1).squeeze::<1>(1).clamp_min(1e-8);
        let sum_ip: Tensor<B, 1> = range_ip.clone().sum_dim(1).squeeze::<1>(1).clamp_min(1e-8);

        let corr_oop: Tensor<B, 2> = (error.clone() / sum_oop).unsqueeze_dim(1);
        let corr_ip: Tensor<B, 2> = (error / sum_ip).unsqueeze_dim(1);

        let oop_corrected = oop_raw - corr_oop;
        let ip_corrected = ip_raw - corr_ip;

        Tensor::cat(vec![oop_corrected, ip_corrected], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn input_size_correct_for_river() {
        assert_eq!(input_size(5), INPUT_SIZE);
        assert_eq!(INPUT_SIZE, 2705);
    }

    #[test]
    fn input_size_correct_for_turn() {
        assert_eq!(input_size(4), INPUT_SIZE);
    }

    #[test]
    fn input_size_correct_for_flop() {
        assert_eq!(input_size(3), INPUT_SIZE);
    }

    #[test]
    fn dual_output_size_constant() {
        assert_eq!(DUAL_OUTPUT_SIZE, 2652);
    }

    #[test]
    fn model_output_shape() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, INPUT_SIZE);
        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let range_oop = Tensor::<TestBackend, 2>::zeros([1, NUM_COMBOS], &device);
        let range_ip = Tensor::<TestBackend, 2>::zeros([1, NUM_COMBOS], &device);
        let output = model.forward(input, range_oop, range_ip);
        assert_eq!(output.dims(), [1, DUAL_OUTPUT_SIZE]);
    }

    #[test]
    fn model_batch_forward() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, INPUT_SIZE);
        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, INPUT_SIZE], &device);
        let range_oop = Tensor::<TestBackend, 2>::zeros([batch_size, NUM_COMBOS], &device);
        let range_ip = Tensor::<TestBackend, 2>::zeros([batch_size, NUM_COMBOS], &device);
        let output = model.forward(input, range_oop, range_ip);
        assert_eq!(output.dims(), [batch_size, DUAL_OUTPUT_SIZE]);
    }

    #[test]
    fn model_output_changes_with_input() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, INPUT_SIZE);
        let input1 = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let input2 = Tensor::<TestBackend, 2>::ones([1, INPUT_SIZE], &device);
        let range_oop = Tensor::<TestBackend, 2>::zeros([1, NUM_COMBOS], &device);
        let range_ip = Tensor::<TestBackend, 2>::zeros([1, NUM_COMBOS], &device);
        let out1 = model.forward(input1, range_oop.clone(), range_ip.clone());
        let out2 = model.forward(input2, range_oop, range_ip);
        let diff: f32 = (out1 - out2).abs().sum().into_scalar();
        assert!(
            diff > 1e-6,
            "outputs should differ for different inputs, diff={diff}"
        );
    }

    #[test]
    fn model_with_turn_input_size() {
        let device = Default::default();
        let in_size = input_size(4);
        assert_eq!(in_size, INPUT_SIZE);
        let model = CfvNet::<TestBackend>::new(&device, 2, 64, in_size);
        let input = Tensor::<TestBackend, 2>::zeros([1, in_size], &device);
        let range_oop = Tensor::<TestBackend, 2>::zeros([1, NUM_COMBOS], &device);
        let range_ip = Tensor::<TestBackend, 2>::zeros([1, NUM_COMBOS], &device);
        let output = model.forward(input, range_oop, range_ip);
        assert_eq!(output.dims(), [1, DUAL_OUTPUT_SIZE]);
    }

    #[test]
    fn zero_sum_holds_after_forward() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 3, 128, INPUT_SIZE);

        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::ones([batch_size, INPUT_SIZE], &device);

        let range_oop = Tensor::<TestBackend, 2>::ones([batch_size, NUM_COMBOS], &device);
        let range_ip = Tensor::<TestBackend, 2>::ones([batch_size, NUM_COMBOS], &device);

        let output = model.forward(input, range_oop.clone(), range_ip.clone());

        let cfv_oop = output.clone().narrow(1, 0, NUM_COMBOS);
        let cfv_ip = output.narrow(1, NUM_COMBOS, NUM_COMBOS);

        let gv_oop = (cfv_oop * range_oop).sum_dim(1).squeeze::<1>(1);
        let gv_ip = (cfv_ip * range_ip).sum_dim(1).squeeze::<1>(1);
        let zero_sum_error: Tensor<TestBackend, 1> = gv_oop + gv_ip;

        let data = zero_sum_error.to_data();
        let values: Vec<f32> = data.to_vec().unwrap();
        for (i, &v) in values.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "zero-sum violated for sample {i}: |error| = {v}"
            );
        }
    }
}
