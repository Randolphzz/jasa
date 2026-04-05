"""Attention gate for DenGCAN skip connections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.shape_check import assert_rank


@dataclass(frozen=True)
class AttentionGateConfig:
    # if None, use encoder channels as hidden width, matching src_jasa AG usage
    hidden_channels: int | None = None


class AttentionGate(nn.Module):
    """
    Additive attention gate aligned with src_jasa AG behavior.
    """

    def __init__(self, decoder_channels: int, encoder_channels: int, cfg: AttentionGateConfig | None = None):
        super().__init__()
        self.cfg = cfg or AttentionGateConfig()
        hidden_channels = self.cfg.hidden_channels or encoder_channels

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(decoder_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(encoder_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_decoder: [B, C_dec, T, F]
            x_encoder: [B, C_enc, T, F]

        Returns:
            gated encoder feature: [B, C_enc, T, F]
        """
        assert_rank(x_decoder, 4, "x_decoder")
        assert_rank(x_encoder, 4, "x_encoder")

        if x_decoder.shape[0] != x_encoder.shape[0]:
            raise ValueError(f"Batch mismatch: x_decoder={x_decoder.shape}, x_encoder={x_encoder.shape}")
        if x_decoder.shape[2:] != x_encoder.shape[2:]:
            raise ValueError(f"Spatial mismatch: x_decoder={x_decoder.shape}, x_encoder={x_encoder.shape}")

        x_encoder_in = x_encoder
        x_decoder = self.conv_decoder(x_decoder)
        x_encoder = self.conv_encoder(x_encoder)
        attn_logits = self.conv_attn(self.relu(x_decoder + x_encoder))
        alpha = self.sigmoid(attn_logits)
        return alpha * x_encoder_in


class ImprovedAttentionGate(nn.Module):
    """Improved attention gate aligned with src_jasa iAG behavior."""

    def __init__(self, channels: int):
        super().__init__()
        self.local_conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.global_conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor) -> torch.Tensor:
        """Gate encoder features using decoder features and pointwise projection."""
        assert_rank(x_decoder, 4, "x_decoder")
        assert_rank(x_encoder, 4, "x_encoder")

        if x_decoder.shape != x_encoder.shape:
            raise ValueError(f"Shape mismatch: x_decoder={x_decoder.shape}, x_encoder={x_encoder.shape}")

        x_encoder_in = x_encoder
        x = x_decoder + x_encoder
        x_local = self.local_conv(x)
        x_global = self.global_conv(torch.mean(x, dim=-1, keepdim=True))
        x = torch.sigmoid(x_local + x_global) * x_encoder_in
        return self.pwconv(x)


class IdentitySkip(nn.Module):
    """Pass-through skip mapping for cat mode."""

    def forward(self, x_encoder: torch.Tensor) -> torch.Tensor:
        assert_rank(x_encoder, 4, "x_encoder")
        return x_encoder
