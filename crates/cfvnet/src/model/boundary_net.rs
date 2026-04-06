use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use super::network::{HiddenBlock, INPUT_SIZE, OUTPUT_SIZE};

/// Boundary value network for depth-bounded range solving.
///
/// Same architecture as CfvNet (MLP with BatchNorm + PReLU hidden blocks),
/// but outputs normalized EVs: `chip_ev / (pot + effective_stack)`.
///
/// Input encoding differs: pot and stack are encoded as fractions of
/// `(pot + effective_stack)` rather than divided by 400.
#[derive(Module, Debug)]
pub struct BoundaryNet<B: Backend> {
    hidden: Vec<HiddenBlock<B>>,
    output: Linear<B>,
}

impl<B: Backend> BoundaryNet<B> {
    pub fn new(device: &B::Device, num_layers: usize, hidden_size: usize) -> Self {
        assert!(num_layers > 0, "need at least one hidden layer");
        let mut hidden = Vec::with_capacity(num_layers);
        hidden.push(HiddenBlock::new(device, INPUT_SIZE, hidden_size));
        for _ in 1..num_layers {
            hidden.push(HiddenBlock::new(device, hidden_size, hidden_size));
        }
        let output = LinearConfig::new(hidden_size, OUTPUT_SIZE).init(device);
        Self { hidden, output }
    }

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

    #[test]
    fn boundary_net_output_shape() {
        let device = Default::default();
        let model = BoundaryNet::<TestBackend>::new(&device, 2, 64);
        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, OUTPUT_SIZE]);
    }

    #[test]
    fn boundary_net_batch_forward() {
        let device = Default::default();
        let model = BoundaryNet::<TestBackend>::new(&device, 2, 64);
        let input = Tensor::<TestBackend, 2>::zeros([8, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [8, OUTPUT_SIZE]);
    }
}
