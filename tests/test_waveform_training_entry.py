from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.waveform_training_entry import WaveformTrainingEntry, WaveformTrainingEntryConfig


def _write_sample(sample_dir: Path, sample_id: int) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    base = np.linspace(-1.0, 1.0, 64000, dtype=np.float32)
    mix = np.stack(
        [
            base + sample_id,
            base + 0.1 + sample_id,
            base + 0.2 + sample_id,
            base + 0.3 + sample_id,
            base + 0.4 + sample_id,
        ],
        axis=1,
    )
    label = base * 0.25 + sample_id
    sf.write(sample_dir / "mix.wav", mix, 16000, subtype="FLOAT")
    sf.write(sample_dir / "label.wav", label, 16000, subtype="FLOAT")


def test_waveform_training_entry_run_batch_and_optimizer_step(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "train" / "0", sample_id=0)
    _write_sample(root / "train" / "1", sample_id=1)

    entry = WaveformTrainingEntry(
        WaveformTrainingEntryConfig(
            dataset_root=str(root),
            split="train",
            bc_mode="fixed",
            bc_channel=1,
            batch_size=2,
            learning_rate=1e-3,
        )
    )
    loader = entry.build_dataloader()
    waveform_batch = next(iter(loader))

    before = next(entry.training_step.parameters()).detach().clone()
    summary = entry.run_batch(waveform_batch, training=True)
    after = next(entry.training_step.parameters()).detach()

    assert summary.batch_size == 2
    assert summary.did_optimizer_step is True
    assert summary.step_output.model_output.enhanced_complex.shape[0] == 2
    assert summary.step_output.total_loss.ndim == 0
    assert summary.step_output.ri_loss.ndim == 0
    assert summary.step_output.mag_loss.ndim == 0
    assert not torch.equal(before, after)


def test_waveform_training_entry_run_loader_returns_averages(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "valid" / "0", sample_id=0)
    _write_sample(root / "valid" / "1", sample_id=1)
    _write_sample(root / "valid" / "2", sample_id=2)

    entry = WaveformTrainingEntry(
        WaveformTrainingEntryConfig(
            dataset_root=str(root),
            split="valid",
            bc_mode="random",
            batch_size=2,
        )
    )
    loader = entry.build_dataloader(shuffle=False)
    metrics = entry.run_loader(loader, training=False)

    assert set(metrics.keys()) == {"num_batches", "num_examples", "loss", "ri_loss", "mag_loss"}
    assert metrics["num_batches"] == 2.0
    assert metrics["num_examples"] == 3.0
    assert metrics["loss"] >= 0.0
    assert metrics["ri_loss"] >= 0.0
    assert metrics["mag_loss"] >= 0.0


def test_waveform_training_entry_train_one_epoch_returns_typed_summary(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "train" / "0", sample_id=0)
    _write_sample(root / "train" / "1", sample_id=1)
    _write_sample(root / "train" / "2", sample_id=2)

    entry = WaveformTrainingEntry(
        WaveformTrainingEntryConfig(
            dataset_root=str(root),
            split="train",
            bc_mode="fixed",
            bc_channel=2,
            batch_size=2,
            learning_rate=1e-3,
        )
    )

    summary = entry.train_one_epoch(max_batches=1)

    assert summary.split == "train"
    assert summary.training is True
    assert summary.num_batches == 1
    assert summary.num_examples == 2
    assert summary.loss >= 0.0
    assert summary.ri_loss >= 0.0
    assert summary.mag_loss >= 0.0


def test_waveform_training_entry_validate_one_epoch_uses_eval_mode(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "valid" / "0", sample_id=0)
    _write_sample(root / "valid" / "1", sample_id=1)
    _write_sample(root / "valid" / "2", sample_id=2)

    entry = WaveformTrainingEntry(
        WaveformTrainingEntryConfig(
            dataset_root=str(root),
            split="train",
            bc_mode="random",
            batch_size=2,
        )
    )

    before = next(entry.training_step.parameters()).detach().clone()
    summary = entry.validate_one_epoch()
    after = next(entry.training_step.parameters()).detach()

    assert summary.split == "valid"
    assert summary.training is False
    assert summary.num_batches == 2
    assert summary.num_examples == 3
    assert summary.loss >= 0.0
    assert summary.ri_loss >= 0.0
    assert summary.mag_loss >= 0.0
    assert torch.equal(before, after)


def _run_all_tests_without_pytest() -> None:
    raise SystemExit("Run this module with pytest.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
