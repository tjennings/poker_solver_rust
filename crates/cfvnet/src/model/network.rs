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

/// Compute input feature size for a given number of board cards.
///
/// Layout: OOP range (1326) + IP range (1326) + board cards + pot + stack + player indicator.
pub fn input_size(board_cards: usize) -> usize {
    NUM_COMBOS + NUM_COMBOS + board_cards + 1 + 1 + 1
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
/// Architecture: `Input(in_size) -> [Linear -> BatchNorm -> PReLU] x N -> Linear(1326)`
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

        let output = LinearConfig::new(hidden_size, OUTPUT_SIZE).init(device);

        Self { hidden, output }
    }

    /// Forward pass: returns `[batch, OUTPUT_SIZE]` counterfactual values.
    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for block in &self.hidden {
            x = block.forward(x);
        }
        self.output.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    const RIVER_INPUT: usize = 2660; // input_size(5)

    #[test]
    fn input_size_correct_for_river() {
        assert_eq!(input_size(5), 2660);
    }

    #[test]
    fn input_size_correct_for_turn() {
        assert_eq!(input_size(4), 2659);
    }

    #[test]
    fn input_size_correct_for_flop() {
        assert_eq!(input_size(3), 2658);
    }

    #[test]
    fn model_output_shape() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, RIVER_INPUT);
        let input = Tensor::<TestBackend, 2>::zeros([1, RIVER_INPUT], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, OUTPUT_SIZE]);
    }

    #[test]
    fn model_batch_forward() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, RIVER_INPUT);
        let batch_size = 4;
        let input =
            Tensor::<TestBackend, 2>::zeros([batch_size, RIVER_INPUT], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [batch_size, OUTPUT_SIZE]);
    }

    #[test]
    fn model_output_changes_with_input() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500, RIVER_INPUT);
        let input1 = Tensor::<TestBackend, 2>::zeros([1, RIVER_INPUT], &device);
        let input2 = Tensor::<TestBackend, 2>::ones([1, RIVER_INPUT], &device);
        let out1 = model.forward(input1);
        let out2 = model.forward(input2);
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
        let model = CfvNet::<TestBackend>::new(&device, 2, 64, in_size);
        let input = Tensor::<TestBackend, 2>::zeros([1, in_size], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, OUTPUT_SIZE]);
    }
}
