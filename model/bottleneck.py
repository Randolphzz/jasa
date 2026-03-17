"""DenGCAN bottleneck wrapper around grouped sConformer placeholders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.sconformer import GroupedSConformerPlaceholder, SConformerConfig
from utils.shape_check import assert_shape


@dataclass(frozen=True)
class DenGCANBottleneckConfig:
    """Configuration for the DenGCAN bottleneck."""

    # paper-specified interface constraints
    in_channels: int = 64
    bottleneck_freq_bins: int = 3
    d_model: int = 192
    num_grouped_layers: int = 2

    # paper-specified defaults for the sConformer layer
    num_heads: int = 4
    ffn_dim: int = 384
    conv_kernel_size: int = 31
    dropout: float = 0.1

    # engineering assumption for the placeholder grouped implementation
    num_groups: int = 3


class DenGCANBottleneck(nn.Module):
    """
    DenGCAN bottleneck operating on the encoder output `[B, 64, T, 3]`.

    paper-specified:
        The encoder output is reshaped from `[B, 64, T, 3]` to `[B, T, 192]`,
        processed by a 2-layer grouped sConformer bottleneck, and reshaped back.

    engineering assumption:
        The internal grouped sConformer implementation uses
        `GroupedSConformerPlaceholder` until the exact grouped design is fully
        recovered. In the current runnable implementation, `num_groups=3`
        assumes that grouped processing aligns with the 3 bottleneck frequency
        bins after reshaping `[B, 64, T, 3] -> [B, T, 192]`.

    Shape:
        input: [B, 64, T, 3]
        output: [B, 64, T, 3]
    """

    def __init__(self, cfg: DenGCANBottleneckConfig | None = None):
        super().__init__()
        self.cfg = cfg or DenGCANBottleneckConfig()
        flat_dim = self.cfg.in_channels * self.cfg.bottleneck_freq_bins
        if self.cfg.d_model != flat_dim:
            raise ValueError(
                f"d_model={self.cfg.d_model} must equal in_channels * bottleneck_freq_bins = {flat_dim}."
            )

        sconformer_cfg = SConformerConfig(
            d_model=self.cfg.d_model,
            num_heads=self.cfg.num_heads,
            ffn_dim=self.cfg.ffn_dim,
            conv_kernel_size=self.cfg.conv_kernel_size,
            dropout=self.cfg.dropout,
        )
        self.layers = nn.ModuleList(
            [
                GroupedSConformerPlaceholder(
                    cfg=sconformer_cfg,
                    num_groups=self.cfg.num_groups,
                )
                for _ in range(self.cfg.num_grouped_layers)
            ]
        )

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape `[B, 64, T, 3]` to `[B, T, 192]`."""
        batch, channels, frames, freq = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch, frames, channels * freq)

    def _to_feature_map(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        """Reshape `[B, T, 192]` back to `[B, 64, T, 3]`."""
        batch = x.shape[0]
        return x.reshape(batch, frames, self.cfg.in_channels, self.cfg.bottleneck_freq_bins).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2-layer grouped sConformer bottleneck."""
        assert_shape(
            x,
            [None, self.cfg.in_channels, None, self.cfg.bottleneck_freq_bins],
            "bottleneck_input",
        )

        frames = x.shape[2]
        x = self._to_sequence(x)
        for layer in self.layers:
            x = layer(x)
        x = self._to_feature_map(x, frames)

        assert_shape(
            x,
            [None, self.cfg.in_channels, None, self.cfg.bottleneck_freq_bins],
            "bottleneck_output",
        )
        return x
