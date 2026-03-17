from __future__ import annotations

import os
import sys

import torch
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.sconformer import (
    GroupedSConformerPlaceholder,
    SConformerBlock,
    SConformerConfig,
    SConformerLayer,
)
from utils.shape_check import ShapeMismatchError


def _assert_shape_mismatch_raised(fn) -> None:
    try:
        fn()
    except ShapeMismatchError:
        return
    raise AssertionError("Expected ShapeMismatchError to be raised.")


def test_sconformer_block_shape_and_backward() -> None:
    x = torch.randn(2, 21, 192, requires_grad=True)
    block = SConformerBlock()

    y = block(x)
    assert y.shape == (2, 21, 192)

    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_sconformer_layer_shape_and_backward() -> None:
    x = torch.randn(2, 17, 192, requires_grad=True)
    layer = SConformerLayer()

    y = layer(x)
    assert y.shape == (2, 17, 192)

    loss = y.abs().mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_grouped_sconformer_placeholder_shape_and_backward() -> None:
    x = torch.randn(2, 13, 192, requires_grad=True)
    layer = GroupedSConformerPlaceholder()

    y = layer(x)
    assert y.shape == (2, 13, 192)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_sconformer_invalid_shape_raises() -> None:
    block = SConformerBlock()
    layer = SConformerLayer()
    grouped = GroupedSConformerPlaceholder()

    _assert_shape_mismatch_raised(lambda: block(torch.randn(2, 21, 191)))
    _assert_shape_mismatch_raised(lambda: layer(torch.randn(2, 192)))
    _assert_shape_mismatch_raised(lambda: grouped(torch.randn(2, 12, 193)))


def test_grouped_sconformer_invalid_group_config_raises() -> None:
    with pytest.raises(ValueError):
        GroupedSConformerPlaceholder(SConformerConfig(d_model=190), num_groups=3)


def _run_all_tests_without_pytest() -> None:
    test_sconformer_block_shape_and_backward()
    test_sconformer_layer_shape_and_backward()
    test_grouped_sconformer_placeholder_shape_and_backward()
    test_sconformer_invalid_shape_raises()
    print("All sConformer tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
