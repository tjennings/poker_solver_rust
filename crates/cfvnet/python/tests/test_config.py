# tests/test_config.py
import tempfile
from pathlib import Path

from cfvnet.config import load_config


def test_load_config_parses_training_section():
    yaml_content = """\
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  street: "river"
  num_samples: 1000
training:
  hidden_layers: 7
  hidden_size: 768
  batch_size: 8192
  epochs: 100
  learning_rate: 0.002
  lr_min: 0.00002
  huber_delta: 1.0
  aux_loss_weight: 0.1
  validation_split: 0.05
  grad_clip_norm: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.hidden_layers == 7
    assert cfg.hidden_size == 768
    assert cfg.batch_size == 8192
    assert cfg.epochs == 100
    assert abs(cfg.learning_rate - 0.002) < 1e-9
    assert abs(cfg.lr_min - 0.00002) < 1e-9
    assert abs(cfg.huber_delta - 1.0) < 1e-9
    assert abs(cfg.aux_loss_weight - 0.1) < 1e-9
    assert abs(cfg.validation_split - 0.05) < 1e-9
    assert abs(cfg.grad_clip_norm - 1.0) < 1e-9


def test_load_config_uses_defaults_for_missing_fields():
    yaml_content = """\
game:
  initial_stack: 200
  bet_sizes: ["50%"]
datagen:
  num_samples: 100
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.hidden_layers == 7
    assert cfg.hidden_size == 500
    assert cfg.batch_size == 2048
    assert cfg.epochs == 2
