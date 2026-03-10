"""iAFF module for AC/BC feature fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.shape_check import assert_same_shape, assert_shape


@dataclass(frozen=True)
class IAFFConfig:
    # paper-specified
    in_channels: int = 2

    # engineering assumption: hidden channels for attention MLP-style mapping
    attn_hidden_channels: int = 8


class ChannelAttentionModule(nn.Module):
    """
    Channel attention with local + global context.

    engineering assumption:
        The paper does not fully specify the internal tensorization of local/global context.
        Here we combine a 1x1 local branch and a GAP-based global branch.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0),
        )
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F]

        Returns:
            attention map a: [B, C, T, F], in [0,1]
        """
        local_logits = self.local_path(x)
        global_logits = self.global_path(x)
        logits = local_logits + global_logits
        return self.sigmoid(logits)


class IAFF(nn.Module):
    """
    Iterative Attentional Feature Fusion (iAFF).

    paper-specified formulas:
        a0 = Fa1(y_AC + y_BC)
        y_AF' = a0 * y_AC + (1 - a0) * y_BC
        a = Fa2(y_AF')
        y_AF = a * y_AC + (1 - a) * y_BC
    """

    def __init__(self, cfg: IAFFConfig | None = None):
        super().__init__()
        self.cfg = cfg or IAFFConfig()
        self.fa1 = ChannelAttentionModule(self.cfg.in_channels, self.cfg.attn_hidden_channels)
        self.fa2 = ChannelAttentionModule(self.cfg.in_channels, self.cfg.attn_hidden_channels)

    def forward(self, y_ac: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_ac: [B, 2, T, 161]
            y_bc: [B, 2, T, 161]

        Returns:
            y_af: [B, 2, T, 161]
        """
        assert_shape(y_ac, [None, self.cfg.in_channels, None, None], "y_ac")
        assert_shape(y_bc, [None, self.cfg.in_channels, None, None], "y_bc")
        assert_same_shape([y_ac, y_bc], ["y_ac", "y_bc"])

        # paper-specified coarse fusion
        a0 = self.fa1(y_ac + y_bc)
        y_af_prime = a0 * y_ac + (1.0 - a0) * y_bc

        # paper-specified refined fusion
        a = self.fa2(y_af_prime)
        y_af = a * y_ac + (1.0 - a) * y_bc
        return y_af
