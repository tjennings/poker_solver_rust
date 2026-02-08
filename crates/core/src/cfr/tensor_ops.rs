//! Tensor-based CFR operations.
//!
//! This module provides vectorized implementations of CFR operations
//! for efficient GPU execution.

// Allow doc_markdown for tensor shape notation like [num_info_sets, max_actions]
// Allow needless_pass_by_value since tensors need to be cloned internally anyway
#![allow(clippy::doc_markdown, clippy::needless_pass_by_value)]

use burn::prelude::*;

/// Performs regret matching on a tensor of regrets.
///
/// For each row (info set), converts regrets to a probability distribution:
/// - Positive regrets are normalized to sum to 1
/// - If all regrets are non-positive, returns uniform distribution
///
/// # Arguments
/// * `regrets` - Tensor of shape [num_info_sets, max_actions]
/// * `action_mask` - Boolean mask of valid actions [num_info_sets, max_actions]
///
/// # Returns
/// Strategy tensor of shape [num_info_sets, max_actions]
pub fn regret_match<B: Backend>(
    regrets: Tensor<B, 2>,
    action_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 2> {
    let _device = regrets.device();
    let [num_info_sets, max_actions] = regrets.dims();

    // Clamp negative regrets to 0 (CFR+ style)
    let positive_regrets = regrets.clone().clamp_min(0.0);

    // Mask out invalid actions
    let mask_float = action_mask.clone().float();
    let masked_regrets = positive_regrets.mul(mask_float.clone());

    // Sum positive regrets per info set
    let regret_sums = masked_regrets.clone().sum_dim(1); // [num_info_sets, 1]

    // Count valid actions per info set for uniform fallback
    let num_valid_actions = mask_float.clone().sum_dim(1); // [num_info_sets, 1]

    // Compute normalized strategy where sum > 0
    // Avoid division by zero by adding small epsilon
    let epsilon = 1e-10;
    let regret_sums_safe = regret_sums.clone().clamp_min(epsilon);
    let normalized = masked_regrets.div(regret_sums_safe);

    // Compute uniform strategy for fallback
    let uniform_prob = mask_float.clone().div(num_valid_actions.clamp_min(1.0));

    // Select between normalized and uniform based on whether regret sum > 0
    // mask_where(mask, source) returns source where mask=true, self where mask=false
    // We want uniform where regret_sum <= epsilon (use_uniform=true)
    let use_uniform = regret_sums.lower_equal_elem(epsilon);
    let use_uniform_expanded = use_uniform.expand([num_info_sets, max_actions]);

    // Where regret_sum <= 0, use uniform; else use normalized
    let strategy = normalized.mask_where(use_uniform_expanded, uniform_prob);

    // Ensure invalid actions have 0 probability
    strategy.mul(mask_float)
}

/// Applies CFR+ regret update: new_regret = max(old_regret + delta, 0)
///
/// # Arguments
/// * `regrets` - Current regret tensor [num_info_sets, max_actions]
/// * `delta` - Regret delta to add [num_info_sets, max_actions]
///
/// # Returns
/// Updated regret tensor with CFR+ flooring (non-negative)
pub fn update_regrets_cfr_plus<B: Backend>(
    regrets: Tensor<B, 2>,
    delta: Tensor<B, 2>,
) -> Tensor<B, 2> {
    regrets.add(delta).clamp_min(0.0)
}

/// Accumulates strategy weighted by reach probability.
///
/// # Arguments
/// * `strategy_sum` - Accumulated strategy [num_info_sets, max_actions]
/// * `strategy` - Current strategy [num_info_sets, max_actions]
/// * `reach` - Reach probabilities [num_info_sets, 1] or [num_info_sets]
///
/// # Returns
/// Updated strategy sum
pub fn accumulate_strategy<B: Backend>(
    strategy_sum: Tensor<B, 2>,
    strategy: Tensor<B, 2>,
    reach: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let [num_info_sets, max_actions] = strategy.dims();
    let reach_expanded = reach
        .reshape([num_info_sets, 1])
        .expand([num_info_sets, max_actions]);
    strategy_sum.add(strategy.mul(reach_expanded))
}

/// Computes average strategy from accumulated strategy sums.
///
/// # Arguments
/// * `strategy_sum` - Accumulated strategy [num_info_sets, max_actions]
/// * `action_mask` - Boolean mask of valid actions [num_info_sets, max_actions]
///
/// # Returns
/// Normalized average strategy
pub fn compute_average_strategy<B: Backend>(
    strategy_sum: Tensor<B, 2>,
    action_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 2> {
    let [num_info_sets, max_actions] = strategy_sum.dims();
    let mask_float = action_mask.clone().float();

    // Sum across actions
    let total = strategy_sum.clone().sum_dim(1); // [num_info_sets, 1]

    // Normalize
    let epsilon = 1e-10;
    let total_safe = total.clone().clamp_min(epsilon);
    let normalized = strategy_sum.div(total_safe);

    // For info sets with no accumulated strategy, use uniform
    let num_valid = mask_float.clone().sum_dim(1);
    let uniform = mask_float.clone().div(num_valid.clamp_min(1.0));

    // mask_where(mask, source) returns source where mask=true, self where mask=false
    // We want uniform where total <= epsilon (use_uniform=true)
    let use_uniform = total.lower_equal_elem(epsilon);
    let use_uniform_expanded = use_uniform.expand([num_info_sets, max_actions]);

    let strategy = normalized.mask_where(use_uniform_expanded, uniform);
    strategy.mul(mask_float)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use test_macros::timed_test;

    type TestBackend = NdArray;

    fn create_mask(
        valid: &[&[bool]],
        device: <TestBackend as Backend>::Device,
    ) -> Tensor<TestBackend, 2, Bool> {
        let rows = valid.len();
        let cols = valid[0].len();
        let flat: Vec<i32> = valid
            .iter()
            .flat_map(|row| row.iter().map(|&b| i32::from(b)))
            .collect();
        Tensor::<TestBackend, 1, Int>::from_ints(flat.as_slice(), &device)
            .reshape([rows, cols])
            .equal_elem(1)
    }

    #[timed_test]
    fn regret_match_positive_regrets() {

        let device = NdArrayDevice::default();

        // Two info sets, 3 actions each
        let regrets =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [0.5, 0.5, 0.0]], &device);
        let mask = create_mask(&[&[true, true, true], &[true, true, false]], device);

        let strategy = regret_match(regrets, mask);
        let data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        eprintln!("Strategy data: {data:?}");

        // First row: [1, 2, 3] / 6 = [1/6, 2/6, 3/6]
        assert!(
            (data[0] - 1.0 / 6.0).abs() < 1e-5,
            "data[0] = {}, expected {}",
            data[0],
            1.0 / 6.0
        );
        assert!((data[1] - 2.0 / 6.0).abs() < 1e-5, "data[1] = {}", data[1]);
        assert!((data[2] - 3.0 / 6.0).abs() < 1e-5, "data[2] = {}", data[2]);

        // Second row: [0.5, 0.5, 0] / 1.0 = [0.5, 0.5, 0]
        assert!((data[3] - 0.5).abs() < 1e-5, "data[3] = {}", data[3]);
        assert!((data[4] - 0.5).abs() < 1e-5, "data[4] = {}", data[4]);
        assert!(data[5].abs() < 1e-5, "data[5] = {}", data[5]); // Masked out
    }

    #[timed_test]
    fn regret_match_all_negative_returns_uniform() {

        let device = NdArrayDevice::default();

        let regrets = Tensor::<TestBackend, 2>::from_floats([[-1.0, -2.0, -3.0]], &device);
        let mask = create_mask(&[&[true, true, true]], device);

        let strategy = regret_match(regrets, mask);
        let data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        // All negative -> uniform [1/3, 1/3, 1/3]
        for &p in &data {
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[timed_test]
    fn regret_match_mixed_regrets() {

        let device = NdArrayDevice::default();

        let regrets = Tensor::<TestBackend, 2>::from_floats([[-1.0, 2.0, 4.0]], &device);
        let mask = create_mask(&[&[true, true, true]], device);

        let strategy = regret_match(regrets, mask);
        let data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        // Negative clamped to 0: [0, 2, 4] / 6 = [0, 1/3, 2/3]
        assert!(data[0].abs() < 1e-5);
        assert!((data[1] - 2.0 / 6.0).abs() < 1e-5);
        assert!((data[2] - 4.0 / 6.0).abs() < 1e-5);
    }

    #[timed_test]
    fn update_regrets_cfr_plus_floors_negative() {

        let device = NdArrayDevice::default();

        let regrets = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [0.5, 0.0]], &device);
        let delta = Tensor::<TestBackend, 2>::from_floats([[-2.0, 1.0], [-1.0, 0.5]], &device);

        let updated = update_regrets_cfr_plus(regrets, delta);
        let data: Vec<f32> = updated.to_data().to_vec().unwrap();

        // [1-2, 2+1] clamped = [0, 3]
        assert!(data[0].abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        // [0.5-1, 0+0.5] clamped = [0, 0.5]
        assert!(data[2].abs() < 1e-5);
        assert!((data[3] - 0.5).abs() < 1e-5);
    }

    #[timed_test]
    fn strategy_sums_to_one() {

        let device = NdArrayDevice::default();

        let regrets = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, -2.0, 5.0]],
            &device,
        );
        let mask = create_mask(
            &[
                &[true, true, true],
                &[true, true, false],
                &[true, true, true],
            ],
            device,
        );

        let strategy = regret_match(regrets, mask);

        // Sum each row
        let sums = strategy.clone().sum_dim(1);
        let sum_data: Vec<f32> = sums.to_data().to_vec().unwrap();

        for (i, &s) in sum_data.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "Row {i} sums to {s}, expected 1.0"
            );
        }
    }
}
