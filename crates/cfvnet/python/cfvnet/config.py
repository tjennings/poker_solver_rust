"""Read CfvnetConfig YAML files (same format as Rust)."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """Training configuration matching Rust's TrainingConfig."""

    hidden_layers: int = 7
    hidden_size: int = 500
    batch_size: int = 2048
    epochs: int = 2
    learning_rate: float = 0.001
    lr_min: float = 0.00001
    huber_delta: float = 1.0
    aux_loss_weight: float = 1.0
    validation_split: float = 0.05
    checkpoint_every_n_epochs: int = 1000
    grad_clip_norm: float = 1.0


def load_config(path: Path) -> TrainConfig:
    """Load a CfvnetConfig YAML and extract training parameters.

    Args:
        path: Path to YAML config file.

    Returns:
        TrainConfig with values from file, defaults for missing fields.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    training = raw.get("training", {})
    return TrainConfig(
        hidden_layers=training.get("hidden_layers", 7),
        hidden_size=training.get("hidden_size", 500),
        batch_size=training.get("batch_size", 2048),
        epochs=training.get("epochs", 2),
        learning_rate=training.get("learning_rate", 0.001),
        lr_min=training.get("lr_min", 0.00001),
        huber_delta=training.get("huber_delta", 1.0),
        aux_loss_weight=training.get("aux_loss_weight", 1.0),
        validation_split=training.get("validation_split", 0.05),
        checkpoint_every_n_epochs=training.get("checkpoint_every_n_epochs", 1000),
        grad_clip_norm=training.get("grad_clip_norm", 1.0),
    )
