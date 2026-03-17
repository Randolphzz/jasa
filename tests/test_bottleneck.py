from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.bottleneck import DenGCANBottleneck
from utils.shape_check import ShapeMismatchError


def _assert_shape_mismatch_raised(fn) -> None:
    try:
        fn()
    except ShapeMismatchError:
        return
    raise AssertionError("Expected ShapeMismatchError to be raised.")


def test_bottleneck_shape_and_backward() -> None:
    x = torch.randn(2, 64, 15, 3, requires_grad=True)
    bottleneck = DenGCANBottleneck()

    y = bottleneck(x)
    assert y.shape == (2, 64, 15, 3)

    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_bottleneck_invalid_shape_raises() -> None:
    bottleneck = DenGCANBottleneck()

    _assert_shape_mismatch_raised(lambda: bottleneck(torch.randn(2, 63, 15, 3)))
    _assert_shape_mismatch_raised(lambda: bottleneck(torch.randn(2, 64, 15, 4)))
    _assert_shape_mismatch_raised(lambda: bottleneck(torch.randn(2, 64, 3)))


def _run_all_tests_without_pytest() -> None:
    test_bottleneck_shape_and_backward()
    test_bottleneck_invalid_shape_raises()
    print("All bottleneck tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
