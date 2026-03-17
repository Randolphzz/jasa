from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import A4BS250hDataset, a4bs250h_collate_fn, build_a4bs250h_dataloader


def _write_sample(sample_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    sample_dir.mkdir(parents=True, exist_ok=True)

    base = np.linspace(-1.0, 1.0, 64000, dtype=np.float32)
    mix = np.stack(
        [
            base + sample_id,
            base + 10 + sample_id,
            base + 20 + sample_id,
            base + 30 + sample_id,
            base + 40 + sample_id,
        ],
        axis=1,
    )
    label = (base * 0.5) + sample_id

    sf.write(sample_dir / "mix.wav", mix, 16000, subtype="FLOAT")
    sf.write(sample_dir / "label.wav", label, 16000, subtype="FLOAT")
    return mix, label


def test_dataset_fixed_bc_mode_reads_expected_fields(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    mix0, label0 = _write_sample(root / "train" / "0", sample_id=0)
    _write_sample(root / "train" / "2", sample_id=2)

    dataset = A4BS250hDataset(str(root), split="train", bc_mode="fixed", bc_channel=2)
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["mixture"].shape == (64000,)
    assert sample["bc"].shape == (64000,)
    assert sample["clean"].shape == (64000,)
    assert sample["mixture"].dtype == torch.float32
    assert sample["bc"].dtype == torch.float32
    assert sample["clean"].dtype == torch.float32
    assert sample["bc_channel"] == 2
    assert torch.allclose(sample["mixture"], torch.from_numpy(label0).float(), atol=1e-4)
    assert torch.allclose(sample["clean"], torch.from_numpy(mix0[:, 0]).float(), atol=1e-4)
    assert torch.allclose(sample["bc"], torch.from_numpy(mix0[:, 2]).float(), atol=1e-4)


def test_dataset_random_bc_mode_uses_selected_channel(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    mix, _label = _write_sample(root / "valid" / "1", sample_id=1)

    dataset = A4BS250hDataset(str(root), split="valid", bc_mode="random")
    with patch("utils.dataset.random.choice", return_value=4):
        sample = dataset[0]

    assert sample["bc_channel"] == 4
    assert torch.allclose(sample["bc"], torch.from_numpy(mix[:, 4]).float(), atol=1e-4)


def test_collate_and_dataloader_stack_fixed_length_samples(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "test" / "0", sample_id=0)
    _write_sample(root / "test" / "1", sample_id=1)

    dataset = A4BS250hDataset(str(root), split="test", bc_mode="fixed", bc_channel=1)
    collated = a4bs250h_collate_fn([dataset[0], dataset[1]])
    assert collated["mixture"].shape == (2, 64000)
    assert collated["bc"].shape == (2, 64000)
    assert collated["clean"].shape == (2, 64000)
    assert collated["bc_channel"].shape == (2,)

    loader = build_a4bs250h_dataloader(
        root=str(root),
        split="test",
        bc_mode="fixed",
        bc_channel=1,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch["mixture"].shape == (2, 64000)
    assert batch["bc"].shape == (2, 64000)
    assert batch["clean"].shape == (2, 64000)
    assert torch.equal(batch["bc_channel"], torch.tensor([1, 1], dtype=torch.long))


def test_dataset_invalid_args_and_bad_shapes_raise(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "train" / "0", sample_id=0)

    try:
        A4BS250hDataset(str(root), split="oops")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid split to raise ValueError.")

    try:
        A4BS250hDataset(str(root), split="train", bc_mode="oops")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid bc_mode to raise ValueError.")

    try:
        A4BS250hDataset(str(root), split="train", bc_channel=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid bc_channel to raise ValueError.")

    bad_root = tmp_path / "bad_root"
    bad_mix = np.zeros((64000, 4), dtype=np.float32)
    bad_label = np.zeros((64000,), dtype=np.float32)
    sample_dir = bad_root / "train" / "0"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sf.write(sample_dir / "mix.wav", bad_mix, 16000, subtype="FLOAT")
    sf.write(sample_dir / "label.wav", bad_label, 16000, subtype="FLOAT")

    dataset = A4BS250hDataset(str(bad_root), split="train")
    try:
        dataset[0]
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid mix shape to raise ValueError.")


def _run_all_tests_without_pytest() -> None:
    raise SystemExit("Run this module with pytest.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
