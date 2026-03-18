use burn::{
    module::Module,
    nn::{Linear, LinearConfig, PRelu, PReluConfig},
    tensor::{backend::Backend, Tensor},
};

/// A single hidden block: Linear -> PReLU with optional residual connection.
///
/// Unlike the standard `HiddenBlock` used in [`CfvNet`](super::network::CfvNet),
/// this block omits `BatchNorm` — the Supremus bucketed architecture does not
/// need it, and `BatchNorm` introduces inference-time issues.
///
/// When `residual` is true (i.e., input and output dimensions match), a skip
/// connection is added: `output = PReLU(Linear(x)) + x`. This improves gradient
/// flow in deep networks (7+ layers).
#[derive(Module, Debug)]
pub struct BucketedHiddenBlock<B: Backend> {
    pub linear: Linear<B>,
    pub activation: PRelu<B>,
    /// Whether to add a skip connection (only when in_features == out_features).
    pub residual: bool,
}

impl<B: Backend> BucketedHiddenBlock<B> {
    /// Create a new hidden block with the given input/output dimensions.
    ///
    /// A residual (skip) connection is automatically enabled when
    /// `in_features == out_features`.
    pub fn new(device: &B::Device, in_features: usize, out_features: usize) -> Self {
        Self {
            linear: LinearConfig::new(in_features, out_features).init(device),
            activation: PReluConfig::new()
                .with_num_parameters(out_features)
                .init(device),
            residual: in_features == out_features,
        }
    }

    /// Forward pass through linear layer then PReLU activation, with optional
    /// residual connection.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.activation.forward(self.linear.forward(x.clone()));
        if self.residual {
            out + x
        } else {
            out
        }
    }
}

/// Supremus-exact Bucketed CFV Network.
///
/// Predicts counterfactual values in **bucket space** rather than per-combo
/// space, enabling efficient training on abstracted game data.
///
/// # Architecture
///
/// ```text
/// Input(2*num_buckets + 1) -> [Linear -> PReLU] x N -> Linear(2*num_buckets)
/// ```
///
/// with hard (differentiable) zero-sum enforcement on the output.
///
/// # Input layout (`2 * num_buckets + 1`)
///
/// | Slice                        | Meaning                           |
/// |------------------------------|-----------------------------------|
/// | `[0 .. num_buckets]`         | OOP range distribution over buckets |
/// | `[num_buckets .. 2*num_buckets]` | IP range distribution over buckets  |
/// | `[2*num_buckets]`            | pot / initial_stack               |
///
/// # Output layout (`2 * num_buckets`)
///
/// | Slice                        | Meaning                            |
/// |------------------------------|------------------------------------|
/// | `[0 .. num_buckets]`         | OOP CFVs per bucket (pot-relative) |
/// | `[num_buckets .. 2*num_buckets]` | IP CFVs per bucket (pot-relative)  |
///
/// # Zero-Sum Enforcement
///
/// After the inner network produces raw outputs, a differentiable correction
/// ensures the game values sum to zero:
///
/// 1. `gv_oop = sum(oop_range * oop_cfv)`
/// 2. `gv_ip  = sum(ip_range  * ip_cfv)`
/// 3. `correction = (gv_oop + gv_ip) / 2`
/// 4. `cfv_oop -= correction`, `cfv_ip -= correction`
///
/// All operations are pure tensor ops so gradients flow through the correction.
#[derive(Module, Debug)]
pub struct BucketedCfvNet<B: Backend> {
    /// Hidden layers: Linear -> PReLU blocks.
    pub hidden: Vec<BucketedHiddenBlock<B>>,
    /// Final output projection to 2*num_buckets.
    pub output: Linear<B>,
    /// Number of buckets (not a trainable parameter).
    pub num_buckets: usize,
}

impl<B: Backend> BucketedCfvNet<B> {
    /// Build a new bucketed CFV network.
    ///
    /// # Arguments
    ///
    /// * `device` — Backend device to allocate tensors on.
    /// * `num_layers` — Number of hidden blocks (must be >= 1).
    /// * `hidden_size` — Width of each hidden layer.
    /// * `num_buckets` — Number of hand buckets; determines input/output size.
    pub fn new(
        device: &B::Device,
        num_layers: usize,
        hidden_size: usize,
        num_buckets: usize,
    ) -> Self {
        assert!(num_layers > 0, "need at least one hidden layer");
        assert!(num_buckets > 0, "need at least one bucket");

        let input_size = 2 * num_buckets + 1;
        let output_size = 2 * num_buckets;

        let mut hidden = Vec::with_capacity(num_layers);
        hidden.push(BucketedHiddenBlock::new(device, input_size, hidden_size));
        for _ in 1..num_layers {
            hidden.push(BucketedHiddenBlock::new(device, hidden_size, hidden_size));
        }

        let output = LinearConfig::new(hidden_size, output_size).init(device);

        Self {
            hidden,
            output,
            num_buckets,
        }
    }

    /// Forward pass **with** zero-sum correction.
    ///
    /// Returns `[batch, 2*num_buckets]` counterfactual values where the
    /// range-weighted game values of OOP and IP sum to zero.
    ///
    /// # Input Scaling
    ///
    /// Reach values are ~1/num_buckets per bucket (range sums to ~1.0 over
    /// num_buckets entries), while pot/stack is ~0.5. To fix the ~250:1 scale
    /// mismatch, the reach portion is scaled by `num_buckets` before feeding
    /// to the network, so both feature groups are ~O(1).
    ///
    /// The zero-sum correction uses the **original** unscaled reach values,
    /// since game value = `sum(reach * cfv)` must use the true reach probabilities.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let nb = self.num_buckets;

        // Scale reach inputs for the network (reach values are ~1/num_buckets, scale to ~1.0).
        let reach = input.clone().slice([None, Some((0, (2 * nb) as i64))]);
        let pot = input.clone().slice([None, Some(((2 * nb) as i64, (2 * nb + 1) as i64))]);
        let scaled_input = Tensor::cat(vec![reach.mul_scalar(nb as f64), pot], 1);

        // Run through network with scaled input.
        let raw = self.forward_raw(scaled_input);

        // Extract per-player raw CFVs: [batch, num_buckets] each.
        let oop_cfv = raw.clone().slice([None, Some((0, nb as i64))]);
        let ip_cfv = raw.slice([None, Some((nb as i64, (2 * nb) as i64))]);

        // Zero-sum correction using ORIGINAL (unscaled) reach for correct game values.
        let oop_range = input.clone().slice([None, Some((0, nb as i64))]);
        let ip_range = input.slice([None, Some((nb as i64, (2 * nb) as i64))]);

        // Range-weighted game values: [batch]
        let gv_oop: Tensor<B, 1> = (oop_range * oop_cfv.clone()).sum_dim(1).squeeze(1);
        let gv_ip: Tensor<B, 1> = (ip_range * ip_cfv.clone()).sum_dim(1).squeeze(1);

        // Correction to enforce zero-sum: [batch] -> [batch, 1] for broadcasting.
        let correction = (gv_oop + gv_ip).div_scalar(2.0).unsqueeze_dim(1);

        // Apply correction to both halves.
        let oop_corrected = oop_cfv - correction.clone();
        let ip_corrected = ip_cfv - correction;

        // Concatenate back: [batch, 2*num_buckets].
        Tensor::cat(vec![oop_corrected, ip_corrected], 1)
    }

    /// Forward pass **without** zero-sum correction (raw network output).
    ///
    /// Useful for debugging or inspecting the uncorrected predictions.
    pub fn forward_raw(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
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

    const NUM_BUCKETS: usize = 50;
    const NUM_LAYERS: usize = 7;
    const HIDDEN_SIZE: usize = 500;

    #[test]
    fn output_shape() {
        let device = Default::default();
        let model =
            BucketedCfvNet::<TestBackend>::new(&device, NUM_LAYERS, HIDDEN_SIZE, NUM_BUCKETS);
        let input_size = 2 * NUM_BUCKETS + 1;
        let input = Tensor::<TestBackend, 2>::zeros([1, input_size], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 2 * NUM_BUCKETS]);
    }

    #[test]
    fn zero_sum_property() {
        let device = Default::default();
        let model =
            BucketedCfvNet::<TestBackend>::new(&device, NUM_LAYERS, HIDDEN_SIZE, NUM_BUCKETS);
        let input_size = 2 * NUM_BUCKETS + 1;

        // Build a random-ish input with valid range distributions.
        let batch = 8;
        let input = Tensor::<TestBackend, 2>::ones([batch, input_size], &device)
            .div_scalar(NUM_BUCKETS as f64);
        let output = model.forward(input.clone());

        // Extract ranges and CFVs.
        let oop_range = input.clone().slice([None, Some((0, NUM_BUCKETS as i64))]);
        let ip_range = input.slice([
            None,
            Some((NUM_BUCKETS as i64, (2 * NUM_BUCKETS) as i64)),
        ]);
        let oop_cfv = output.clone().slice([None, Some((0, NUM_BUCKETS as i64))]);
        let ip_cfv = output.slice([
            None,
            Some((NUM_BUCKETS as i64, (2 * NUM_BUCKETS) as i64)),
        ]);

        let gv_oop: Tensor<TestBackend, 1> = (oop_range * oop_cfv).sum_dim(1).squeeze(1);
        let gv_ip: Tensor<TestBackend, 1> = (ip_range * ip_cfv).sum_dim(1).squeeze(1);
        let game_value_sum = (gv_oop + gv_ip).abs();

        // Each element of the batch should be ~0.
        let max_violation: f32 = game_value_sum.max().into_scalar();
        assert!(
            max_violation < 1e-5,
            "zero-sum violated: max |gv_oop + gv_ip| = {max_violation}"
        );
    }

    #[test]
    fn batch_forward() {
        let device = Default::default();
        let model =
            BucketedCfvNet::<TestBackend>::new(&device, NUM_LAYERS, HIDDEN_SIZE, NUM_BUCKETS);
        let input_size = 2 * NUM_BUCKETS + 1;

        for batch_size in [1, 4, 16, 64] {
            let input = Tensor::<TestBackend, 2>::zeros([batch_size, input_size], &device);
            let output = model.forward(input);
            assert_eq!(
                output.dims(),
                [batch_size, 2 * NUM_BUCKETS],
                "failed for batch_size={batch_size}"
            );
        }
    }

    #[test]
    fn raw_vs_corrected() {
        let device = Default::default();
        let model =
            BucketedCfvNet::<TestBackend>::new(&device, NUM_LAYERS, HIDDEN_SIZE, NUM_BUCKETS);
        let input_size = 2 * NUM_BUCKETS + 1;

        // Use non-trivial input so outputs are non-zero.
        let input = Tensor::<TestBackend, 2>::ones([4, input_size], &device);
        let raw = model.forward_raw(input.clone());
        let corrected = model.forward(input);

        let diff: f32 = (raw - corrected).abs().sum().into_scalar();
        assert!(
            diff > 1e-6,
            "raw and corrected outputs should differ, diff={diff}"
        );
    }

    #[test]
    fn gradient_flows_through_zero_sum() {
        use burn::backend::{Autodiff, NdArray};

        type AdBackend = Autodiff<NdArray>;

        let device = Default::default();
        let num_buckets = 10; // Smaller for speed.
        let model = BucketedCfvNet::<AdBackend>::new(&device, 3, 64, num_buckets);
        let input_size = 2 * num_buckets + 1;

        let input = Tensor::<AdBackend, 2>::ones([2, input_size], &device)
            .div_scalar(num_buckets as f64);
        let output = model.forward(input);

        // Simple MSE-like loss: sum of squares of output.
        let loss = output.powf_scalar(2.0).sum();

        // Backward pass — this panics if any op breaks the autodiff graph.
        let grads = loss.backward();

        // Verify gradients exist for the output layer's weight.
        let output_weight_grad = model.output.weight.grad(&grads);
        assert!(
            output_weight_grad.is_some(),
            "output layer weight gradient should exist"
        );

        // Verify the gradient tensor is non-zero.
        let grad_tensor = output_weight_grad.unwrap();
        let grad_sum: f32 = grad_tensor.abs().sum().into_scalar();
        assert!(
            grad_sum > 1e-10,
            "output layer gradient should be non-zero, got {grad_sum}"
        );
    }
}
