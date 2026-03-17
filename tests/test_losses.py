from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.losses import LossConfig, MagnitudeLoss, RILoss, SpeechEnhancementLoss
from utils.shape_check import ShapeMismatchError


def _assert_shape_mismatch_raised(fn) -> None:
    try:
        fn()
    except ShapeMismatchError:
        return
    raise AssertionError("Expected ShapeMismatchError to be raised.")


def test_ri_and_magnitude_loss_shapes_and_backward() -> None:
    pred = torch.randn(2, 2, 10, 161, requires_grad=True)
    target = torch.randn(2, 2, 10, 161)

    ri_loss = RILoss()
    mag_loss = MagnitudeLoss()

    ri = ri_loss(pred, target)
    mag = mag_loss(pred, target)

    assert ri.ndim == 0
    assert mag.ndim == 0

    total = ri + mag
    total.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_combined_loss_breakdown_and_backward() -> None:
    pred = torch.randn(2, 2, 8, 161, requires_grad=True)
    target = torch.randn(2, 2, 8, 161)
    loss_fn = SpeechEnhancementLoss(LossConfig(ri_weight=0.5, mag_weight=0.5, base_loss="mse"))

    breakdown = loss_fn(pred, target, return_breakdown=True)

    assert set(breakdown.keys()) == {"total", "ri_loss", "mag_loss"}
    assert breakdown["total"].ndim == 0
    assert breakdown["ri_loss"].ndim == 0
    assert breakdown["mag_loss"].ndim == 0

    breakdown["total"].backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_losses_invalid_shape_raises() -> None:
    ri_loss = RILoss()
    mag_loss = MagnitudeLoss()
    total_loss = SpeechEnhancementLoss()

    _assert_shape_mismatch_raised(lambda: ri_loss(torch.randn(2, 3, 10, 161), torch.randn(2, 3, 10, 161)))
    _assert_shape_mismatch_raised(lambda: mag_loss(torch.randn(2, 2, 10, 161), torch.randn(2, 2, 9, 161)))
    _assert_shape_mismatch_raised(lambda: total_loss(torch.randn(2, 2, 161), torch.randn(2, 2, 161)))


def _run_all_tests_without_pytest() -> None:
    test_ri_and_magnitude_loss_shapes_and_backward()
    test_combined_loss_breakdown_and_backward()
    test_losses_invalid_shape_raises()
    print("All loss tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
