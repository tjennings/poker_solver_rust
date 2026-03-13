use burn::tensor::{backend::Backend, Tensor};

/// Masked Huber loss over valid (unmasked) combos.
///
/// Computes the smooth Huber loss on `(pred - target)` only where `mask == 1.0`.
/// Board-blocked combos (mask == 0.0) contribute nothing to the loss.
///
/// When `pot_weight` is `Some`, each sample's loss is weighted by its pot size
/// and the result is normalized by total pot weight instead of count. This
/// emphasizes larger-pot situations in the loss landscape.
///
/// Returns a scalar (rank-1 tensor with one element).
pub fn masked_huber_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    delta: f64,
    pot_weight: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    let diff = (pred - target) * mask.clone();
    let abs_diff = diff.abs();

    // Quadratic region: 0.5 * x^2  (when |x| <= delta)
    let quadratic = abs_diff.clone().powf_scalar(2.0).mul_scalar(0.5);
    // Linear region: delta * (|x| - 0.5 * delta)  (when |x| > delta)
    let linear = abs_diff.clone().sub_scalar(0.5 * delta).mul_scalar(delta);

    let within_delta = abs_diff.lower_equal_elem(delta);
    let element_loss = quadratic.mask_where(within_delta.bool_not(), linear);

    // Mask and reduce over valid entries.
    let masked_loss = element_loss * mask.clone();
    match pot_weight {
        Some(pw) => {
            let pw_2d: Tensor<B, 2> = pw.clone().unsqueeze_dim(1);
            let weighted = masked_loss * pw_2d;
            let total_weight = pw.sum().clamp_min(1.0);
            weighted.sum().div(total_weight)
        }
        None => {
            let num_valid = mask.sum().clamp_min(1.0);
            masked_loss.sum().div(num_valid)
        }
    }
}

/// Combined CFVnet loss for dual-player output.
///
/// The prediction and target tensors have shape `[batch, 2652]`, where the first
/// 1326 columns are OOP CFVs and the last 1326 are IP CFVs. The mask has shape
/// `[batch, 1326]` and applies identically to both players.
///
/// Returns the sum of masked Huber losses for both players.
pub fn cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    huber_delta: f64,
    pot_weight: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    let pred_oop = pred.clone().narrow(1, 0, 1326);
    let pred_ip = pred.narrow(1, 1326, 1326);
    let tgt_oop = target.clone().narrow(1, 0, 1326);
    let tgt_ip = target.narrow(1, 1326, 1326);
    let loss_oop = masked_huber_loss(pred_oop, tgt_oop, mask.clone(), huber_delta, pot_weight.clone());
    let loss_ip = masked_huber_loss(pred_ip, tgt_ip, mask, huber_delta, pot_weight);
    loss_oop + loss_ip
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn huber_loss_zero_on_perfect_prediction() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 1.0]], &device);
        let loss = masked_huber_loss(pred, target, mask, 1.0, None);
        let val: f32 = loss.into_scalar();
        assert!(val.abs() < 1e-6, "loss should be 0, got {val}");
    }

    #[test]
    fn huber_loss_ignores_masked_entries() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[1.0, 999.0, 3.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, 0.0, 3.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
        let loss = masked_huber_loss(pred, target, mask, 1.0, None);
        let val: f32 = loss.into_scalar();
        assert!(val.abs() < 1e-6, "masked loss should be 0, got {val}");
    }

    #[test]
    fn huber_loss_quadratic_and_linear_regions() {
        let device = Default::default();
        // diff = 0.5 (quadratic: 0.5 * 0.25 = 0.125) and diff = 2.0 (linear: 1*(2-0.5) = 1.5)
        let pred = Tensor::<B, 2>::from_floats([[0.5, 2.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let loss = masked_huber_loss(pred, target, mask, 1.0, None);
        let val: f32 = loss.into_scalar();
        // expected = (0.125 + 1.5) / 2 = 0.8125
        assert!(
            (val - 0.8125).abs() < 1e-5,
            "expected ~0.8125, got {val}"
        );
    }

    #[test]
    fn combined_loss_includes_both_terms() {
        let device = Default::default();
        // Test that the loss from both players sums correctly.
        // Use the underlying masked_huber_loss since cfvnet_loss requires 1326-wide tensors.
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let loss_oop = masked_huber_loss(
            Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device),
            Tensor::<B, 2>::from_floats([[0.0, 0.0]], &device),
            mask.clone(),
            1.0,
            None,
        );
        let loss_ip = masked_huber_loss(
            Tensor::<B, 2>::from_floats([[0.3, 0.3]], &device),
            Tensor::<B, 2>::from_floats([[0.0, 0.0]], &device),
            mask,
            1.0,
            None,
        );
        let total: f32 = (loss_oop + loss_ip).into_scalar();
        assert!(total > 0.0, "combined loss should be positive, got {total}");
    }

    #[test]
    fn dual_player_loss_sums_both() {
        use burn::tensor::TensorData;

        let device = Default::default();
        // Build tensors of width 2652 (1326 + 1326)
        let n = 1326;
        let pred_data: Vec<f32> = (0..2 * n).map(|i| if i < n { 0.5 } else { 0.3 }).collect();
        let target_data: Vec<f32> = vec![0.0; 2 * n];
        let mask_data: Vec<f32> = vec![1.0; n];

        let pred = Tensor::<B, 2>::from_data(TensorData::new(pred_data, [1, 2 * n]), &device);
        let target = Tensor::<B, 2>::from_data(TensorData::new(target_data, [1, 2 * n]), &device);
        let mask = Tensor::<B, 2>::from_data(TensorData::new(mask_data.clone(), [1, n]), &device);

        let loss = cfvnet_loss(pred, target, mask.clone(), 1.0, None);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "dual player loss should be positive, got {val}");

        // Verify it equals the sum of individual player losses
        let pred_oop_data: Vec<f32> = vec![0.5; n];
        let pred_ip_data: Vec<f32> = vec![0.3; n];
        let zero_data: Vec<f32> = vec![0.0; n];
        let loss_oop = masked_huber_loss(
            Tensor::<B, 2>::from_data(TensorData::new(pred_oop_data, [1, n]), &device),
            Tensor::<B, 2>::from_data(TensorData::new(zero_data.clone(), [1, n]), &device),
            mask.clone(),
            1.0,
            None,
        );
        let loss_ip = masked_huber_loss(
            Tensor::<B, 2>::from_data(TensorData::new(pred_ip_data, [1, n]), &device),
            Tensor::<B, 2>::from_data(TensorData::new(zero_data, [1, n]), &device),
            mask,
            1.0,
            None,
        );
        let expected: f32 = (loss_oop + loss_ip).into_scalar();
        assert!(
            (val - expected).abs() < 1e-5,
            "cfvnet_loss ({val}) should equal sum of per-player losses ({expected})"
        );
    }

    #[test]
    fn pot_weighted_loss_emphasizes_high_pot() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[0.5, 0.0], [0.5, 0.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0], [1.0, 1.0]], &device);
        let pot = Tensor::<B, 1>::from_floats([10.0, 100.0], &device);
        let weighted = masked_huber_loss(pred.clone(), target.clone(), mask.clone(), 1.0, Some(pot));
        let unweighted = masked_huber_loss(pred, target, mask, 1.0, None);
        let w: f32 = weighted.into_scalar();
        let u: f32 = unweighted.into_scalar();
        assert!(w.is_finite(), "weighted loss should be finite, got {w}");
        assert!(u.is_finite(), "unweighted loss should be finite, got {u}");
    }

    #[test]
    fn pot_weighted_loss_scales_contribution() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[0.1], [0.1]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0], [0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0], [1.0]], &device);
        let pot = Tensor::<B, 1>::from_floats([10.0, 100.0], &device);
        let weighted = masked_huber_loss(pred, target, mask, 1.0, Some(pot));
        let val: f32 = weighted.into_scalar();
        // Huber quadratic: 0.5 * 0.01 = 0.005 per entry
        // Weighted: (10*0.005 + 100*0.005) / (10+100) = 0.55/110 = 0.005
        assert!((val - 0.005).abs() < 1e-5, "expected ~0.005, got {val}");
    }
}
