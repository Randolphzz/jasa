from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dengcan import DenGCAN
from utils.shape_check import ShapeMismatchError


def _assert_shape_mismatch_raised(fn) -> None:
    try:
        fn()
    except ShapeMismatchError:
        return
    raise AssertionError("Expected ShapeMismatchError to be raised.")


def _capture_shape(name: str, seen: dict[str, tuple[int, ...]]):
    def hook(_module, _inputs, output):
        seen[name] = tuple(output.shape)

    return hook


def test_dengcan_tablei_shapes_and_backward() -> None:
    batch, frames = 2, 11
    x = torch.randn(batch, 6, frames, 161, requires_grad=True)
    model = DenGCAN()

    seen: dict[str, tuple[int, ...]] = {}
    handles = []
    for idx, block in enumerate(model.encoder_blocks, start=1):
        handles.append(block.register_forward_hook(_capture_shape(f"enc{idx}", seen)))
    for idx, gate in enumerate(model.attention_gates, start=1):
        handles.append(gate.register_forward_hook(_capture_shape(f"ag{idx}", seen)))
    handles.append(model.bottleneck.register_forward_hook(_capture_shape("bottleneck", seen)))
    for idx, block in enumerate(model.decoder_blocks, start=1):
        handles.append(block.register_forward_hook(_capture_shape(f"dec{idx}", seen)))

    y = model(x)
    for handle in handles:
        handle.remove()

    expected_shapes = {
        "enc1": (batch, 16, frames, 79),
        "enc2": (batch, 32, frames, 38),
        "enc3": (batch, 48, frames, 18),
        "enc4": (batch, 64, frames, 8),
        "enc5": (batch, 64, frames, 3),
        "ag1": (batch, 64, frames, 3),
        "ag2": (batch, 64, frames, 8),
        "ag3": (batch, 48, frames, 18),
        "ag4": (batch, 32, frames, 38),
        "ag5": (batch, 16, frames, 79),
        "bottleneck": (batch, 64, frames, 3),
        "dec1": (batch, 64, frames, 8),
        "dec2": (batch, 48, frames, 18),
        "dec3": (batch, 32, frames, 38),
        "dec4": (batch, 16, frames, 79),
        "dec5": (batch, 2, frames, 161),
    }

    assert seen == expected_shapes
    assert y.shape == (batch, 2, frames, 161)

    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_dengcan_decoder_first_dense_block_special_case() -> None:
    model = DenGCAN()

    first_decoder_dense = model.decoder_blocks[0].dense
    later_decoder_dense = model.decoder_blocks[1].dense

    for conv_unit in first_decoder_dense.convs:
        assert not any(isinstance(module, nn.BatchNorm2d) for module in conv_unit.net)
        assert not any(isinstance(module, nn.PReLU) for module in conv_unit.net)

    assert any(isinstance(module, nn.BatchNorm2d) for module in later_decoder_dense.convs[0].net)
    assert any(isinstance(module, nn.PReLU) for module in later_decoder_dense.convs[0].net)
    assert model.decoder_blocks[-1].degated.use_post_act is False


def test_dengcan_invalid_shape_raises() -> None:
    model = DenGCAN()

    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 5, 12, 161)))
    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 6, 12, 160)))
    _assert_shape_mismatch_raised(lambda: model(torch.randn(2, 6, 161)))


def _run_all_tests_without_pytest() -> None:
    test_dengcan_tablei_shapes_and_backward()
    test_dengcan_decoder_first_dense_block_special_case()
    test_dengcan_invalid_shape_raises()
    print("All DenGCAN tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
