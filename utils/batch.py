"""Validated batch containers for dual-modal speech enhancement."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from utils.shape_check import assert_freq_bins, assert_rank, assert_same_shape, assert_shape


@dataclass(frozen=True)
class WaveformBatchConfig:
    """Configuration for fixed-length waveform batches."""

    # dataset-specified waveform properties
    waveform_length: int = 64000


@dataclass(frozen=True)
class WaveformBatch:
    """
    Minimal waveform batch container produced by the dataset / DataLoader layer.

    engineering assumption:
        This is a codebase utility that wraps the collated waveform tensors
        before STFT conversion. It is not a paper-specified data structure.

    Shape:
        mixture: [B, 64000]
        bc: [B, 64000]
        clean: [B, 64000]
        bc_channel: [B] optional metadata tensor
    """

    mixture: torch.Tensor
    bc: torch.Tensor
    clean: torch.Tensor
    bc_channel: torch.Tensor | None = None
    cfg: WaveformBatchConfig = field(default_factory=WaveformBatchConfig)

    def __post_init__(self) -> None:
        """Validate waveform tensors immediately on construction."""
        self.validate()

    @classmethod
    def from_dict(
        cls,
        batch_dict: dict[str, torch.Tensor],
        cfg: WaveformBatchConfig | None = None,
    ) -> WaveformBatch:
        """Build a `WaveformBatch` from a collated dataset dictionary."""
        required_keys = {"mixture", "bc", "clean"}
        missing = required_keys - set(batch_dict.keys())
        if missing:
            raise KeyError(f"Waveform batch is missing required keys: {sorted(missing)}")

        bc_channel = batch_dict.get("bc_channel")
        return cls(
            mixture=batch_dict["mixture"],
            bc=batch_dict["bc"],
            clean=batch_dict["clean"],
            bc_channel=bc_channel if isinstance(bc_channel, torch.Tensor) else None,
            cfg=cfg or WaveformBatchConfig(),
        )

    @property
    def batch_size(self) -> int:
        """Return the batch dimension size."""
        return self.mixture.shape[0]

    def validate(self) -> None:
        """Validate waveform tensor shapes and optional BC metadata."""
        assert_shape(self.mixture, [None, self.cfg.waveform_length], "mixture")
        assert_shape(self.bc, [None, self.cfg.waveform_length], "bc")
        assert_shape(self.clean, [None, self.cfg.waveform_length], "clean")
        assert_same_shape([self.mixture, self.bc, self.clean], ["mixture", "bc", "clean"])

        if self.bc_channel is not None:
            assert_rank(self.bc_channel, 1, "bc_channel")
            if self.bc_channel.shape[0] != self.batch_size:
                raise ValueError(
                    f"bc_channel batch size mismatch: expected {self.batch_size}, "
                    f"got {self.bc_channel.shape[0]}."
                )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> WaveformBatch:
        """Return a copy of the waveform batch moved to the requested device / dtype."""
        return replace(
            self,
            mixture=self.mixture.to(device=device, dtype=dtype),
            bc=self.bc.to(device=device, dtype=dtype),
            clean=self.clean.to(device=device, dtype=dtype),
            bc_channel=None if self.bc_channel is None else self.bc_channel.to(device=device),
        )


@dataclass(frozen=True)
class SpeechEnhancementBatchConfig:
    """Configuration for the complex-spectrogram batch interface."""

    # paper-specified tensor interface
    complex_channels: int = 2
    freq_bins: int = 161


@dataclass(frozen=True)
class SpeechEnhancementBatch:
    """
    Minimal batch container for dual-modal enhancement training / evaluation.

    engineering assumption:
        The paper specifies the model I/O tensors, but does not define a data
        loader batch object. This dataclass provides a small validated interface
        around the current project tensors.

    Shape:
        noisy_ac: [B, 2, T, 161]
        noisy_bc: [B, 2, T, 161]
        clean_ac: [B, 2, T, 161]
    """

    noisy_ac: torch.Tensor
    noisy_bc: torch.Tensor
    clean_ac: torch.Tensor
    cfg: SpeechEnhancementBatchConfig = field(default_factory=SpeechEnhancementBatchConfig)

    def __post_init__(self) -> None:
        """Validate the batch tensors immediately on construction."""
        self.validate()

    @property
    def batch_size(self) -> int:
        """Return the batch dimension size."""
        return self.noisy_ac.shape[0]

    @property
    def num_frames(self) -> int:
        """Return the time-frame dimension size."""
        return self.noisy_ac.shape[2]

    def validate(self) -> None:
        """Validate batch tensor shapes against the project interface."""
        assert_shape(self.noisy_ac, [None, self.cfg.complex_channels, None, None], "noisy_ac")
        assert_shape(self.noisy_bc, [None, self.cfg.complex_channels, None, None], "noisy_bc")
        assert_shape(self.clean_ac, [None, self.cfg.complex_channels, None, None], "clean_ac")
        assert_same_shape(
            [self.noisy_ac, self.noisy_bc, self.clean_ac],
            ["noisy_ac", "noisy_bc", "clean_ac"],
        )
        assert_freq_bins(self.noisy_ac, self.cfg.freq_bins, "noisy_ac")
        assert_freq_bins(self.noisy_bc, self.cfg.freq_bins, "noisy_bc")
        assert_freq_bins(self.clean_ac, self.cfg.freq_bins, "clean_ac")

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SpeechEnhancementBatch:
        """Return a copy of the batch moved to the requested device / dtype."""
        return replace(
            self,
            noisy_ac=self.noisy_ac.to(device=device, dtype=dtype),
            noisy_bc=self.noisy_bc.to(device=device, dtype=dtype),
            clean_ac=self.clean_ac.to(device=device, dtype=dtype),
        )
