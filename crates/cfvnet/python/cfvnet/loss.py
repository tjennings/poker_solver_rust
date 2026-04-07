"""Loss functions for BoundaryNet training."""

import torch


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Per-sample weighted masked Huber loss.

    Args:
        pred: Predictions [batch, combos].
        target: Targets [batch, combos].
        mask: Valid mask [batch, combos], 1.0 for valid.
        sample_weight: Per-sample weight [batch].
        delta: Huber loss threshold.

    Returns:
        Scalar weighted mean Huber loss.
    """
    diff = (pred - target) * mask
    abs_diff = diff.abs()

    quadratic = 0.5 * abs_diff.pow(2)
    linear = delta * (abs_diff - 0.5 * delta)
    element_loss = torch.where(abs_diff <= delta, quadratic, linear)

    masked_loss = element_loss * mask
    per_sample = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    weighted = per_sample * sample_weight
    return weighted.sum() / sample_weight.sum().clamp(min=1.0)


def weighted_aux_loss(
    pred: torch.Tensor,
    player_range: torch.Tensor,
    game_value: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted auxiliary game-value consistency loss.

    Args:
        pred: Predictions [batch, combos].
        player_range: Player's range [batch, combos].
        game_value: Target game value [batch].
        sample_weight: Per-sample weight [batch].

    Returns:
        Scalar weighted mean squared residual.
    """
    weighted_sum = (pred * player_range).sum(dim=1)
    residual = (weighted_sum - game_value).pow(2)
    weighted = residual * sample_weight
    return weighted.sum() / sample_weight.sum().clamp(min=1.0)


def boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    player_range: torch.Tensor,
    game_value: torch.Tensor,
    sample_weight: torch.Tensor,
    delta: float,
    aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined BoundaryNet loss with component breakdown.

    Args:
        pred: Predictions [batch, combos].
        target: Targets [batch, combos].
        mask: Valid mask [batch, combos].
        player_range: Player's range [batch, combos].
        game_value: Target game value [batch].
        sample_weight: Per-sample weight [batch].
        delta: Huber loss threshold.
        aux_weight: Weight for auxiliary loss term.

    Returns:
        Tuple of (combined, huber, aux) loss tensors.
    """
    huber = weighted_huber_loss(pred, target, mask, sample_weight, delta)
    aux = weighted_aux_loss(pred, player_range, game_value, sample_weight)
    combined = huber + aux_weight * aux
    return combined, huber, aux
