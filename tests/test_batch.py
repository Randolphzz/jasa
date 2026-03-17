from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.batch import SpeechEnhancementBatch
from utils.shape_check import ShapeMismatchError


def _assert_shape_mismatch_raised(fn) -> None:
    try:
        fn()
    except ShapeMismatchError:
        return
    raise AssertionError("Expected ShapeMismatchError to be raised.")


def test_batch_validation_and_properties() -> None:
    batch = SpeechEnhancementBatch(
        noisy_ac=torch.randn(2, 2, 14, 161),
        noisy_bc=torch.randn(2, 2, 14, 161),
        clean_ac=torch.randn(2, 2, 14, 161),
    )

    assert batch.batch_size == 2
    assert batch.num_frames == 14


def test_batch_to_device_or_dtype_returns_valid_copy() -> None:
    batch = SpeechEnhancementBatch(
        noisy_ac=torch.randn(2, 2, 10, 161),
        noisy_bc=torch.randn(2, 2, 10, 161),
        clean_ac=torch.randn(2, 2, 10, 161),
    )

    moved = batch.to(dtype=torch.float64)

    assert moved.noisy_ac.dtype == torch.float64
    assert moved.noisy_bc.dtype == torch.float64
    assert moved.clean_ac.dtype == torch.float64
    assert moved.noisy_ac.shape == batch.noisy_ac.shape


def test_batch_invalid_shape_raises() -> None:
    _assert_shape_mismatch_raised(
        lambda: SpeechEnhancementBatch(
            noisy_ac=torch.randn(2, 3, 10, 161),
            noisy_bc=torch.randn(2, 3, 10, 161),
            clean_ac=torch.randn(2, 3, 10, 161),
        )
    )
    _assert_shape_mismatch_raised(
        lambda: SpeechEnhancementBatch(
            noisy_ac=torch.randn(2, 2, 10, 161),
            noisy_bc=torch.randn(2, 2, 9, 161),
            clean_ac=torch.randn(2, 2, 10, 161),
        )
    )
    _assert_shape_mismatch_raised(
        lambda: SpeechEnhancementBatch(
            noisy_ac=torch.randn(2, 2, 10, 160),
            noisy_bc=torch.randn(2, 2, 10, 160),
            clean_ac=torch.randn(2, 2, 10, 160),
        )
    )


def _run_all_tests_without_pytest() -> None:
    test_batch_validation_and_properties()
    test_batch_to_device_or_dtype_returns_valid_copy()
    test_batch_invalid_shape_raises()
    print("All batch tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
