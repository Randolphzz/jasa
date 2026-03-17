"""Dataset and DataLoader helpers for the A4BS_250h corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class A4BS250hConfig:
    """Configuration for on-disk A4BS_250h samples."""

    # paper / dataset-specified properties
    sample_rate: int = 16000
    duration_seconds: int = 4
    num_samples: int = 64000
    mix_channels: int = 5
    bc_channels: tuple[int, int, int, int] = (1, 2, 3, 4)


class A4BS250hDataset(Dataset[dict[str, torch.Tensor | int]]):
    """
    Dataset interface for pre-generated A4BS_250h samples.

    This dataset only handles:
        - sample directory discovery
        - `wav` loading
        - BC channel selection
        - tensor conversion

    It does not:
        - regenerate samples
        - modify source files
        - compute STFT
        - apply augmentation

    Per-sample return:
        {
            "mixture": Tensor[T],
            "bc": Tensor[T],
            "clean": Tensor[T],
            "bc_channel": int,
        }
        where `T = 64000`.
    """

    VALID_SPLITS = {"train", "valid", "test"}
    VALID_BC_MODES = {"fixed", "random"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        bc_mode: str = "fixed",
        bc_channel: int = 1,
        cfg: A4BS250hConfig | None = None,
    ):
        self.cfg = cfg or A4BS250hConfig()
        self.root = Path(root)
        self.split = split
        self.bc_mode = bc_mode
        self.bc_channel = bc_channel

        self._validate_init_args()
        self.split_dir = self.root / self.split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.sample_dirs = self._discover_sample_dirs(self.split_dir)
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories found under: {self.split_dir}")

    def __len__(self) -> int:
        """Return the number of numbered sample directories in the split."""
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        """Load one dataset sample and select the requested BC channel."""
        sample_dir = self.sample_dirs[index]
        mix_path = sample_dir / "mix.wav"
        label_path = sample_dir / "label.wav"

        if not mix_path.is_file():
            raise FileNotFoundError(f"mix.wav not found: {mix_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"label.wav not found: {label_path}")

        mix, mix_sr = sf.read(mix_path)
        label, label_sr = sf.read(label_path)

        self._validate_loaded_audio(sample_dir, mix, mix_sr, label, label_sr)

        clean = mix[:, 0]
        k = self.bc_channel if self.bc_mode == "fixed" else random.choice(self.cfg.bc_channels)
        bc = mix[:, k]
        mixture = label

        return {
            "mixture": torch.from_numpy(mixture).to(dtype=torch.float32),
            "bc": torch.from_numpy(bc).to(dtype=torch.float32),
            "clean": torch.from_numpy(clean).to(dtype=torch.float32),
            "bc_channel": int(k),
        }

    def _validate_init_args(self) -> None:
        """Validate constructor arguments."""
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}, got {self.split!r}")
        if self.bc_mode not in self.VALID_BC_MODES:
            raise ValueError(f"bc_mode must be one of {sorted(self.VALID_BC_MODES)}, got {self.bc_mode!r}")
        if self.bc_channel not in self.cfg.bc_channels:
            raise ValueError(
                f"bc_channel must be one of {list(self.cfg.bc_channels)}, got {self.bc_channel!r}"
            )
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

    def _discover_sample_dirs(self, split_dir: Path) -> list[Path]:
        """Discover numbered sample directories under a split."""
        sample_dirs = [path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit()]
        return sorted(sample_dirs, key=lambda path: int(path.name))

    def _validate_loaded_audio(
        self,
        sample_dir: Path,
        mix,
        mix_sr: int,
        label,
        label_sr: int,
    ) -> None:
        """Validate sample-rate and array shapes for one loaded sample."""
        if mix_sr != self.cfg.sample_rate:
            raise ValueError(
                f"Unexpected mix sample rate in {sample_dir}: expected {self.cfg.sample_rate}, got {mix_sr}"
            )
        if label_sr != self.cfg.sample_rate:
            raise ValueError(
                f"Unexpected label sample rate in {sample_dir}: expected {self.cfg.sample_rate}, got {label_sr}"
            )
        if getattr(mix, "shape", None) != (self.cfg.num_samples, self.cfg.mix_channels):
            raise ValueError(
                f"Unexpected mix shape in {sample_dir}: expected "
                f"({self.cfg.num_samples}, {self.cfg.mix_channels}), got {getattr(mix, 'shape', None)}"
            )
        if getattr(label, "shape", None) != (self.cfg.num_samples,):
            raise ValueError(
                f"Unexpected label shape in {sample_dir}: expected ({self.cfg.num_samples},), "
                f"got {getattr(label, 'shape', None)}"
            )


def a4bs250h_collate_fn(
    batch: list[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """
    Collate fixed-length A4BS_250h samples into a batch.

    Output:
        {
            "mixture": Tensor[B, T],
            "bc": Tensor[B, T],
            "clean": Tensor[B, T],
            "bc_channel": Tensor[B],
        }
    """
    if not batch:
        raise ValueError("a4bs250h_collate_fn expects a non-empty batch.")

    return {
        "mixture": torch.stack([sample["mixture"] for sample in batch], dim=0),
        "bc": torch.stack([sample["bc"] for sample in batch], dim=0),
        "clean": torch.stack([sample["clean"] for sample in batch], dim=0),
        "bc_channel": torch.tensor([int(sample["bc_channel"]) for sample in batch], dtype=torch.long),
    }


def build_a4bs250h_dataloader(
    root: str,
    split: str = "train",
    bc_mode: str = "fixed",
    bc_channel: int = 1,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for A4BS_250h.

    engineering assumption:
        `shuffle` defaults to `True` for train and `False` otherwise.
    """
    dataset = A4BS250hDataset(
        root=root,
        split=split,
        bc_mode=bc_mode,
        bc_channel=bc_channel,
    )
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=a4bs250h_collate_fn,
    )
