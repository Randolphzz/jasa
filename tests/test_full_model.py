from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.full_model import BoneAirFusionEnhancementModel
from utils.shape_check import ShapeMismatchError


def _assert_shape_mismatch_raised(fn) -> None:
    try:
        fn()
    except ShapeMismatchError:
        return
    raise AssertionError("Expected ShapeMismatchError to be raised.")


def test_full_model_shapes_and_backward() -> None:
    batch, frames = 2, 12
    y_ac = torch.randn(batch, 2, frames, 161, requires_grad=True)
    y_bc = torch.randn(batch, 2, frames, 161, requires_grad=True)
    model = BoneAirFusionEnhancementModel()

    output = model(y_ac, y_bc)

    assert output.y_af.shape == (batch, 2, frames, 161)
    assert output.backbone_input.shape == (batch, 6, frames, 161)
    assert output.decoder_out.shape == (batch, 2, frames, 161)
    assert output.crm.shape == (batch, 2, frames, 161)
    assert output.enhanced_complex.shape == (batch, 2, frames, 161)

    loss = output.enhanced_complex.square().mean()
    loss.backward()
    assert y_ac.grad is not None
    assert y_bc.grad is not None
    assert y_ac.grad.shape == y_ac.shape
    assert y_bc.grad.shape == y_bc.shape


def test_full_model_invalid_shape_raises() -> None:
    model = BoneAirFusionEnhancementModel()

    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 3, 12, 161), torch.randn(2, 3, 12, 161)))
    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 2, 12, 160), torch.randn(2, 2, 12, 160)))
    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 2, 12, 161), torch.randn(2, 2, 11, 161)))
    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 2, 161), torch.randn(2, 2, 161)))


def _run_all_tests_without_pytest() -> None:
    test_full_model_shapes_and_backward()
    test_full_model_invalid_shape_raises()
    print("All full-model tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
