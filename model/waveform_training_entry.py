"""Minimal training entry that bridges waveform data to the spectral model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from models.training_step import SpeechEnhancementTrainingStep, TrainingStepOutput
from utils.batch import WaveformBatch
from utils.dataset import build_a4bs250h_dataloader
from utils.waveform_adapter import WaveformToSpectralBatchAdapter


@dataclass(frozen=True)
class WaveformTrainingEntryConfig:
    """Configuration for the minimal waveform-domain training entry."""

    dataset_root: str
    split: str = "train"
    bc_mode: str = "fixed"
    bc_channel: int = 1
    batch_size: int = 1
    shuffle: bool | None = None
    num_workers: int = 0
    drop_last: bool = False
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass(frozen=True)
class TrainingIterationSummary:
    """
    Summary returned by one waveform training / evaluation iteration.

    Shape:
        model_output.enhanced_complex: [B, 2, T, 161]
        total_loss / ri_loss / mag_loss: scalar tensors
    """

    step_output: TrainingStepOutput
    batch_size: int
    did_optimizer_step: bool


@dataclass(frozen=True)
class EpochSummary:
    """
    Aggregated metrics for one epoch-level pass over a dataloader.

    Shape:
        loss / ri_loss / mag_loss: python floats averaged over all examples
    """

    split: str
    training: bool
    num_batches: int
    num_examples: int
    loss: float
    ri_loss: float
    mag_loss: float


class WaveformTrainingEntry(nn.Module):
    """
    Minimal runnable training entry for waveform-domain dataset batches.

    This entry wraps exactly the current mainline:
        1. DataLoader emits waveform tensors `[B, 64000]`
        2. `WaveformToSpectralBatchAdapter` converts them to `[B, 2, T, 161]`
        3. `SpeechEnhancementTrainingStep` runs forward + loss
        4. If `training=True`, optimizer step is applied

    engineering assumption:
        The paper does not define an application-level training entry object.
        This class is a thin project runtime utility around the existing model,
        adapter, and loss modules.
    """

    def __init__(
        self,
        cfg: WaveformTrainingEntryConfig,
        adapter: WaveformToSpectralBatchAdapter | None = None,
        training_step: SpeechEnhancementTrainingStep | None = None,
        optimizer: Optimizer | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter or WaveformToSpectralBatchAdapter()
        self.training_step = training_step or SpeechEnhancementTrainingStep()
        self.optimizer = optimizer or Adam(self.training_step.parameters(), lr=self.cfg.learning_rate)

        self.device = torch.device(self.cfg.device)
        self.adapter.to(self.device)
        self.training_step.to(self.device)

    def build_dataloader(self, split: str | None = None, shuffle: bool | None = None) -> DataLoader:
        """Build a waveform DataLoader using the existing dataset interface."""
        return build_a4bs250h_dataloader(
            root=self.cfg.dataset_root,
            split=split or self.cfg.split,
            bc_mode=self.cfg.bc_mode,
            bc_channel=self.cfg.bc_channel,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle if shuffle is None else shuffle,
            num_workers=self.cfg.num_workers,
            drop_last=self.cfg.drop_last,
        )

    def run_batch(
        self,
        waveform_batch: WaveformBatch | dict[str, torch.Tensor],
        training: bool = True,
    ) -> TrainingIterationSummary:
        """Run one collated waveform batch through adapter, model, loss, and optional optimizer step."""
        spectral_batch = self.adapter(waveform_batch)
        spectral_batch = spectral_batch.to(device=self.device)

        self.training_step.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            step_output = self.training_step(spectral_batch)
        if training:
            step_output.total_loss.backward()
            self.optimizer.step()

        return TrainingIterationSummary(
            step_output=step_output,
            batch_size=spectral_batch.batch_size,
            did_optimizer_step=training,
        )

    def run_loader(
        self,
        dataloader: DataLoader,
        max_batches: int | None = None,
        training: bool = True,
    ) -> dict[str, float]:
        """Run a dataloader for a small training / evaluation sweep and return averaged losses."""
        total_loss = 0.0
        total_ri = 0.0
        total_mag = 0.0
        total_examples = 0
        num_batches = 0

        for batch_idx, waveform_batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            summary = self.run_batch(waveform_batch, training=training)
            batch_size = summary.batch_size
            total_examples += batch_size
            num_batches += 1
            total_loss += float(summary.step_output.total_loss.detach()) * batch_size
            total_ri += float(summary.step_output.ri_loss.detach()) * batch_size
            total_mag += float(summary.step_output.mag_loss.detach()) * batch_size

        if total_examples == 0:
            raise ValueError("run_loader received no batches to process.")

        return {
            "num_batches": float(num_batches),
            "num_examples": float(total_examples),
            "loss": total_loss / total_examples,
            "ri_loss": total_ri / total_examples,
            "mag_loss": total_mag / total_examples,
        }

    @staticmethod
    def _epoch_summary_from_metrics(
        metrics: dict[str, float],
        *,
        split: str,
        training: bool,
    ) -> EpochSummary:
        """Convert averaged loader metrics into a typed epoch summary."""
        return EpochSummary(
            split=split,
            training=training,
            num_batches=int(metrics["num_batches"]),
            num_examples=int(metrics["num_examples"]),
            loss=float(metrics["loss"]),
            ri_loss=float(metrics["ri_loss"]),
            mag_loss=float(metrics["mag_loss"]),
        )

    def train_one_epoch(
        self,
        dataloader: DataLoader | None = None,
        *,
        max_batches: int | None = None,
        split: str | None = None,
    ) -> EpochSummary:
        """
        Run one training epoch over waveform batches.

        Args:
            dataloader: Optional external waveform dataloader. If omitted, one is
                built from the current config.
            max_batches: Optional limit for smoke tests or short runs.
            split: Optional dataset split override used only when building the
                dataloader internally.
        """
        resolved_split = split or self.cfg.split
        loader = dataloader or self.build_dataloader(split=resolved_split, shuffle=True)
        metrics = self.run_loader(loader, max_batches=max_batches, training=True)
        return self._epoch_summary_from_metrics(metrics, split=resolved_split, training=True)

    def validate_one_epoch(
        self,
        dataloader: DataLoader | None = None,
        *,
        max_batches: int | None = None,
        split: str = "valid",
    ) -> EpochSummary:
        """
        Run one validation epoch over waveform batches without optimizer steps.

        Args:
            dataloader: Optional external waveform dataloader. If omitted, one is
                built from the requested validation split.
            max_batches: Optional limit for smoke tests or short runs.
            split: Validation split used when building the dataloader internally.
        """
        loader = dataloader or self.build_dataloader(split=split, shuffle=False)
        metrics = self.run_loader(loader, max_batches=max_batches, training=False)
        return self._epoch_summary_from_metrics(metrics, split=split, training=False)
