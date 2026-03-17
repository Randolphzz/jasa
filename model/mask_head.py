"""Mask prediction head for complex ratio mask (cRM)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.shape_check import assert_freq_bins, assert_shape


@dataclass(frozen=True)
class MaskHeadConfig:
    # paper-specified
    in_channels: int = 2
    freq_bins: int = 161


class MaskHead(nn.Module):
    """
    Predict real/imag parts of cRM using frequency-wise FC layers.

    paper-specified:
        decoder output [B,2,T,161] is split into real/imag features,
        then two independent FC layers (161->161) produce cRM real and imag.
    """

    def __init__(self, cfg: MaskHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or MaskHeadConfig()
        if self.cfg.in_channels != 2:
            raise ValueError("MaskHead expects in_channels=2.")

        self.real_fc = nn.Linear(self.cfg.freq_bins, self.cfg.freq_bins)
        self.imag_fc = nn.Linear(self.cfg.freq_bins, self.cfg.freq_bins)

    def forward(self, decoder_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_out: [B, 2, T, F]

        Returns:
            crm: [B, 2, T, F]
        """
        assert_shape(decoder_out, [None, self.cfg.in_channels, None, None], "decoder_out")
        assert_freq_bins(decoder_out, self.cfg.freq_bins, "decoder_out", freq_dim=-1)

        # [B, 2, T, F] -> real/imag: [B, T, F]
        real_feat = decoder_out[:, 0, :, :]
        imag_feat = decoder_out[:, 1, :, :]

        crm_real = self.real_fc(real_feat)
        crm_imag = self.imag_fc(imag_feat)

        crm = torch.stack([crm_real, crm_imag], dim=1)
        return crm
