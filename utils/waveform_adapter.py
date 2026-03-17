"""Adapters that bridge waveform batches to spectral model inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.batch import SpeechEnhancementBatch, WaveformBatch, WaveformBatchConfig
from utils.stft import STFTConfig, STFTProcessor


class WaveformToSpectralBatchAdapter(nn.Module):
    """
    Convert waveform-domain batches into the existing spectral batch interface.

    engineering assumption:
        The paper specifies STFT settings and spectral tensor shapes, but not a
        standalone adapter object. This module is a thin project-level bridge
        from DataLoader waveform batches to `SpeechEnhancementBatch`.

    Input:
        - `WaveformBatch`
        - or a collated dataset dict with keys `mixture`, `bc`, `clean`

    Output:
        `SpeechEnhancementBatch` with:
        - noisy_ac: [B, 2, T, 161]
        - noisy_bc: [B, 2, T, 161]
        - clean_ac: [B, 2, T, 161]
    """

    def __init__(
        self,
        stft_processor: STFTProcessor | None = None,
        waveform_batch_cfg: WaveformBatchConfig | None = None,
    ):
        super().__init__()
        self.stft_processor = stft_processor or STFTProcessor()
        self.waveform_batch_cfg = waveform_batch_cfg or WaveformBatchConfig()

    def forward(self, waveform_batch: WaveformBatch | dict[str, torch.Tensor]) -> SpeechEnhancementBatch:
        """Convert one waveform batch into a spectral batch."""
        batch = self._coerce_waveform_batch(waveform_batch)

        noisy_ac = self.stft_processor.waveform_to_complex(batch.mixture)
        noisy_bc = self.stft_processor.waveform_to_complex(batch.bc)
        clean_ac = self.stft_processor.waveform_to_complex(batch.clean)

        return SpeechEnhancementBatch(
            noisy_ac=noisy_ac,
            noisy_bc=noisy_bc,
            clean_ac=clean_ac,
        )

    def _coerce_waveform_batch(self, waveform_batch: WaveformBatch | dict[str, torch.Tensor]) -> WaveformBatch:
        """Normalize accepted batch inputs into a validated `WaveformBatch`."""
        if isinstance(waveform_batch, WaveformBatch):
            return waveform_batch
        if isinstance(waveform_batch, dict):
            return WaveformBatch.from_dict(waveform_batch, cfg=self.waveform_batch_cfg)
        raise TypeError(
            "WaveformToSpectralBatchAdapter expects a WaveformBatch or collated dict, "
            f"got {type(waveform_batch).__name__}."
        )

    @property
    def stft_cfg(self) -> STFTConfig:
        """Expose the STFT configuration used by this adapter."""
        return self.stft_processor.cfg
