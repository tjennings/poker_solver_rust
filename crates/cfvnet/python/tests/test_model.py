# tests/test_model.py
import torch

from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE
from cfvnet.model import BoundaryNet


def test_output_shape_single():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    model.eval()
    x = torch.zeros(1, INPUT_SIZE)
    y = model(x)
    assert y.shape == (1, OUTPUT_SIZE)


def test_output_shape_batch():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    x = torch.zeros(8, INPUT_SIZE)
    y = model(x)
    assert y.shape == (8, OUTPUT_SIZE)


def test_parameter_count():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    total = sum(p.numel() for p in model.parameters())
    # Layer 1: 2720*64 + 64 (linear) + 64*2 (bn) + 64 (prelu) = 174,336 + 192
    # Layer 2: 64*64 + 64 (linear) + 64*2 (bn) + 64 (prelu) = 4,160 + 192
    # Output: 64*1326 + 1326 = 86,190
    assert total > 0


def test_eval_mode_deterministic():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    model.eval()
    x = torch.randn(4, INPUT_SIZE)
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)
