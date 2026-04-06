use burn::tensor::{backend::Backend, Tensor};

/// Masked Huber loss over valid (unmasked) combos.
///
/// Computes the smooth Huber loss on `(pred - target)` only where `mask == 1.0`.
/// Board-blocked combos (mask == 0.0) contribute nothing to the loss.
/// Returns a scalar (rank-1 tensor with one element).
pub fn masked_huber_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    delta: f64,
) -> Tensor<B, 1> {
    let diff = (pred - target) * mask.clone();
    let abs_diff = diff.abs();

    // Quadratic region: 0.5 * x^2  (when |x| <= delta)
    let quadratic = abs_diff.clone().powf_scalar(2.0).mul_scalar(0.5);
    // Linear region: delta * (|x| - 0.5 * delta)  (when |x| > delta)
    let linear = abs_diff.clone().sub_scalar(0.5 * delta).mul_scalar(delta);

    let within_delta = abs_diff.lower_equal_elem(delta);
    let element_loss = quadratic.mask_where(within_delta.bool_not(), linear);

    // Mask and average over valid entries only.
    let masked_loss = element_loss * mask.clone();
    let num_valid = mask.sum().clamp_min(1.0);
    masked_loss.sum().div(num_valid)
}

/// Auxiliary game-value consistency loss.
///
/// Enforces that the range-weighted sum of predicted CFVs equals the known
/// game value: `L = mean((sum_i(range[i] * cfv[i]) - game_value)^2)`.
pub fn aux_game_value_loss<B: Backend>(
    cfv_pred: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // weighted_sum: [batch, 1326] -> sum over dim 1 -> [batch, 1] -> squeeze -> [batch]
    let weighted_sum: Tensor<B, 1> = (cfv_pred * range).sum_dim(1).squeeze(1);
    let residual = weighted_sum - game_value;
    residual.powf_scalar(2.0).mean()
}

/// Combined CFVnet loss: `L = L_huber + lambda * L_aux`.
pub fn cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
    huber_delta: f64,
    aux_weight: f64,
) -> Tensor<B, 1> {
    let huber = masked_huber_loss(pred.clone(), target, mask, huber_delta);
    let aux = aux_game_value_loss(pred, range, game_value);
    huber + aux.mul_scalar(aux_weight)
}

/// Per-sample weighted Huber loss for importance-weighted training.
///
/// Computes Huber loss per sample (averaging over valid combos within each sample),
/// multiplies by per-sample weight, then takes the weighted mean across the batch.
fn weighted_masked_huber_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    sample_weight: Tensor<B, 1>,
    delta: f64,
) -> Tensor<B, 1> {
    let diff = (pred - target) * mask.clone();
    let abs_diff = diff.abs();

    let quadratic = abs_diff.clone().powf_scalar(2.0).mul_scalar(0.5);
    let linear = abs_diff.clone().sub_scalar(0.5 * delta).mul_scalar(delta);

    let within_delta = abs_diff.lower_equal_elem(delta);
    let element_loss = quadratic.mask_where(within_delta.bool_not(), linear);

    // Per-sample loss: sum masked elements, divide by per-sample valid count.
    let masked_loss = element_loss * mask.clone();
    // [batch, combos] -> sum over combos -> [batch]
    let per_sample_loss: Tensor<B, 1> = masked_loss.sum_dim(1).squeeze(1);
    let per_sample_valid: Tensor<B, 1> = mask.sum_dim(1).squeeze::<1>(1).clamp_min(1.0);
    let per_sample_mean = per_sample_loss / per_sample_valid;

    // Weighted average across batch.
    let weighted = per_sample_mean * sample_weight.clone();
    weighted.sum() / sample_weight.sum().clamp_min(1.0)
}

/// Combined loss with per-sample weighting: `L = weighted_huber + lambda * weighted_aux`.
pub fn weighted_cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
    sample_weight: Tensor<B, 1>,
    huber_delta: f64,
    aux_weight: f64,
) -> Tensor<B, 1> {
    let huber = weighted_masked_huber_loss(pred.clone(), target, mask, sample_weight.clone(), huber_delta);
    // Weighted aux: per-sample squared residual, weighted average.
    let weighted_sum: Tensor<B, 1> = (pred * range).sum_dim(1).squeeze(1);
    let residual = (weighted_sum - game_value).powf_scalar(2.0);
    let aux = (residual * sample_weight.clone()).sum() / sample_weight.sum().clamp_min(1.0);
    huber + aux.mul_scalar(aux_weight)
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
        let loss = masked_huber_loss(pred, target, mask, 1.0);
        let val: f32 = loss.into_scalar();
        assert!(val.abs() < 1e-6, "loss should be 0, got {val}");
    }

    #[test]
    fn huber_loss_ignores_masked_entries() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[1.0, 999.0, 3.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, 0.0, 3.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
        let loss = masked_huber_loss(pred, target, mask, 1.0);
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
        let loss = masked_huber_loss(pred, target, mask, 1.0);
        let val: f32 = loss.into_scalar();
        // expected = (0.125 + 1.5) / 2 = 0.8125
        assert!(
            (val - 0.8125).abs() < 1e-5,
            "expected ~0.8125, got {val}"
        );
    }

    #[test]
    fn aux_loss_zero_when_constraint_met() {
        let device = Default::default();
        let cfv_pred = Tensor::<B, 2>::from_floats([[0.2, 0.4]], &device);
        let range = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        // weighted sum = 0.5*0.2 + 0.5*0.4 = 0.3
        let game_value = Tensor::<B, 1>::from_floats([0.3], &device);
        let loss = aux_game_value_loss(cfv_pred, range, game_value);
        let val: f32 = loss.into_scalar();
        assert!(val.abs() < 1e-5, "aux loss should be ~0, got {val}");
    }

    #[test]
    fn aux_loss_positive_when_constraint_violated() {
        let device = Default::default();
        let cfv_pred = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let range = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        let game_value = Tensor::<B, 1>::from_floats([0.0], &device);
        let loss = aux_game_value_loss(cfv_pred, range, game_value);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.5, "aux loss should be large, got {val}");
    }

    #[test]
    fn combined_loss_includes_both_terms() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let range = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        let game_value = Tensor::<B, 1>::from_floats([0.0], &device);
        let loss = cfvnet_loss(pred, target, mask, range, game_value, 1.0, 1.0);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "combined loss should be positive, got {val}");
    }
}
