"""DenGCAN bottleneck wrapper around grouped sConformer blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from models.sconformer import GsConformer
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

    # source-aligned engineering options
    causal: bool = False
    lookahead: int = 0

    # reference-aligned grouped processing
    num_groups: int = 2
    rearrange: bool = True


class DenGCANBottleneck(nn.Module):
    """
    DenGCAN bottleneck operating on the encoder output `[B, 64, T, 3]`.

    paper-specified:
        The encoder output is processed by a grouped sConformer bottleneck
        while preserving `[B, 64, T, 3]` shape.

    reference-aligned implementation:
        This follows src_jasa GsConformer behavior: `[B, 64, T, 3]` is
        flattened to `[B, T, 192]` internally, processed by grouped blocks,
        optionally rearranged between layers, then reshaped back.

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

        self.grouped_sconformer = GsConformer(
            input_dim=self.cfg.d_model,
            num_heads=self.cfg.num_heads,
            ffn_dim=self.cfg.ffn_dim,
            num_layers=self.cfg.num_grouped_layers,
            groups=self.cfg.num_groups,
            rearrange=self.cfg.rearrange,
            depthwise_conv_kernel_size=self.cfg.conv_kernel_size,
            dropout=self.cfg.dropout,
            causal=self.cfg.causal,
            lookahead=self.cfg.lookahead,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2-layer grouped sConformer bottleneck."""
        assert_shape(
            x,
            [None, self.cfg.in_channels, None, self.cfg.bottleneck_freq_bins],
            "bottleneck_input",
        )

        x = self.grouped_sconformer(x)

        assert_shape(
            x,
            [None, self.cfg.in_channels, None, self.cfg.bottleneck_freq_bins],
            "bottleneck_output",
        )
        return x
