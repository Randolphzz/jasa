from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_minimal import run_training


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


def test_run_training_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "train" / "0", sample_id=0)
    _write_sample(root / "train" / "1", sample_id=1)
    _write_sample(root / "valid" / "0", sample_id=0)
    _write_sample(root / "valid" / "1", sample_id=1)

    save_dir = tmp_path / "artifacts"
    exit_code = run_training(
        [
            "--dataset-root",
            str(root),
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--train-max-batches",
            "1",
            "--valid-max-batches",
            "1",
            "--save-dir",
            str(save_dir),
            "--save-every-epoch",
        ]
    )

    assert exit_code == 0

    metrics_path = save_dir / "metrics.jsonl"
    best_path = save_dir / "best.pt"
    epoch_path = save_dir / "epoch_1.pt"
    last_path = save_dir / "last.pt"

    assert metrics_path.exists()
    assert best_path.exists()
    assert epoch_path.exists()
    assert last_path.exists()

    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    train_record = json.loads(lines[0])
    valid_record = json.loads(lines[1])
    assert train_record["split"] == "train"
    assert train_record["training"] is True
    assert valid_record["split"] == "valid"
    assert valid_record["training"] is False

    checkpoint = torch.load(best_path, map_location="cpu")
    assert checkpoint["epoch"] == 1
    assert "best_valid_loss" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_run_training_resume_and_scheduler_append_metrics(tmp_path: Path) -> None:
    root = tmp_path / "A4BS_250h"
    _write_sample(root / "train" / "0", sample_id=0)
    _write_sample(root / "train" / "1", sample_id=1)
    _write_sample(root / "valid" / "0", sample_id=0)
    _write_sample(root / "valid" / "1", sample_id=1)

    save_dir = tmp_path / "artifacts"
    exit_code = run_training(
        [
            "--dataset-root",
            str(root),
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--train-max-batches",
            "1",
            "--valid-max-batches",
            "1",
            "--save-dir",
            str(save_dir),
            "--lr-step-size",
            "1",
            "--lr-gamma",
            "0.1",
        ]
    )
    assert exit_code == 0

    resume_path = save_dir / "last.pt"
    exit_code = run_training(
        [
            "--dataset-root",
            str(root),
            "--batch-size",
            "2",
            "--epochs",
            "2",
            "--train-max-batches",
            "1",
            "--valid-max-batches",
            "1",
            "--save-dir",
            str(save_dir),
            "--resume",
            str(resume_path),
            "--lr-step-size",
            "1",
            "--lr-gamma",
            "0.1",
        ]
    )
    assert exit_code == 0

    lines = (save_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    records = [json.loads(line) for line in lines]
    assert records[0]["epoch"] == 1
    assert records[1]["epoch"] == 1
    assert records[2]["epoch"] == 2
    assert records[3]["epoch"] == 2
    assert "lr" in records[0]
    assert "lr" in records[3]
    assert records[2]["lr"] < records[0]["lr"]

    checkpoint = torch.load(resume_path, map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert "scheduler_state_dict" in checkpoint
