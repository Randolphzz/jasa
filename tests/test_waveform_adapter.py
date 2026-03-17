from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.batch import WaveformBatch
from utils.shape_check import ShapeMismatchError
from utils.waveform_adapter import WaveformToSpectralBatchAdapter


def test_waveform_batch_validation_and_from_dict() -> None:
    batch_dict = {
        "mixture": torch.randn(2, 64000),
        "bc": torch.randn(2, 64000),
        "clean": torch.randn(2, 64000),
        "bc_channel": torch.tensor([1, 4], dtype=torch.long),
    }

    batch = WaveformBatch.from_dict(batch_dict)
    assert batch.batch_size == 2
    assert batch.bc_channel is not None
    assert torch.equal(batch.bc_channel, torch.tensor([1, 4], dtype=torch.long))


def test_waveform_to_spectral_adapter_shapes_from_dict_and_batch() -> None:
    adapter = WaveformToSpectralBatchAdapter()
    batch_dict = {
        "mixture": torch.randn(2, 64000),
        "bc": torch.randn(2, 64000),
        "clean": torch.randn(2, 64000),
        "bc_channel": torch.tensor([2, 2], dtype=torch.long),
    }

    spectral_from_dict = adapter(batch_dict)
    waveform_batch = WaveformBatch.from_dict(batch_dict)
    spectral_from_batch = adapter(waveform_batch)

    for spectral_batch in (spectral_from_dict, spectral_from_batch):
        assert spectral_batch.noisy_ac.shape[0] == 2
        assert spectral_batch.noisy_ac.shape[1] == 2
        assert spectral_batch.noisy_bc.shape == spectral_batch.noisy_ac.shape
        assert spectral_batch.clean_ac.shape == spectral_batch.noisy_ac.shape
        assert spectral_batch.noisy_ac.shape[-1] == 161


def test_waveform_batch_invalid_shape_raises() -> None:
    try:
        WaveformBatch(
            mixture=torch.randn(2, 63999),
            bc=torch.randn(2, 63999),
            clean=torch.randn(2, 63999),
        )
    except ShapeMismatchError:
        pass
    else:
        raise AssertionError("Expected invalid waveform length to raise ShapeMismatchError.")


def test_waveform_adapter_invalid_input_raises() -> None:
    adapter = WaveformToSpectralBatchAdapter()

    try:
        adapter(torch.randn(2, 64000))
    except TypeError:
        pass
    else:
        raise AssertionError("Expected invalid adapter input type to raise TypeError.")

    try:
        adapter(
            {
                "mixture": torch.randn(2, 64000),
                "bc": torch.randn(2, 64000),
            }
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Expected missing waveform batch keys to raise KeyError.")


def _run_all_tests_without_pytest() -> None:
    raise SystemExit("Run this module with pytest.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
