"""Attention gate for DenGCAN skip connections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.shape_check import assert_rank


@dataclass(frozen=True)
class AttentionGateConfig:
    # engineering assumption: inter-channel reduction ratio for gate projection
    inter_channels: int | None = None


class AttentionGate(nn.Module):
    """
    Additive attention gate over skip features.

    engineering assumption:
        Since paper does not specify AG internals, this implementation follows a
        common additive attention-gate design (Attention U-Net style).
    """

    def __init__(self, x_channels: int, g_channels: int, cfg: AttentionGateConfig | None = None):
        super().__init__()
        self.cfg = cfg or AttentionGateConfig()
        inter_channels = self.cfg.inter_channels or max(1, min(x_channels, g_channels) // 2)

        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_skip: [B, Cx, T, F]
            g: [B, Cg, T, F] (or different F/T that will be aligned)

        Returns:
            gated skip: [B, Cx, T, F]
        """
        assert_rank(x_skip, 4, "x_skip")
        assert_rank(g, 4, "g")

        if x_skip.shape[0] != g.shape[0]:
            raise ValueError(f"Batch mismatch: x_skip={x_skip.shape}, g={g.shape}")

        # engineering assumption: if spatial dims differ, interpolate g to skip size.
        if x_skip.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=x_skip.shape[2:], mode="bilinear", align_corners=False)

        theta = self.theta_x(x_skip)
        phi = self.phi_g(g)
        attn_logits = self.psi(self.relu(theta + phi))
        alpha = self.sigmoid(attn_logits)
        return x_skip * alpha
