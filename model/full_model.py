"""End-to-end dual-modal speech enhancement model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.dengcan import DenGCAN, DenGCANConfig
from models.iaff import IAFF, IAFFConfig
from models.mask_head import MaskHead, MaskHeadConfig
from utils.complex_utils import apply_crm
from utils.shape_check import assert_freq_bins, assert_same_shape, assert_shape


@dataclass(frozen=True)
class FullModelConfig:
    """Configuration for the end-to-end dual-modal enhancement model."""

    # paper-specified interface constraints
    complex_channels: int = 2
    fused_backbone_channels: int = 6
    freq_bins: int = 161


@dataclass(frozen=True)
class FullModelOutput:
    """
    Structured outputs from the full enhancement model.

    Shape:
        y_af: [B, 2, T, 161]
        backbone_input: [B, 6, T, 161]
        decoder_out: [B, 2, T, 161]
        crm: [B, 2, T, 161]
        enhanced_complex: [B, 2, T, 161]
    """

    y_af: torch.Tensor
    backbone_input: torch.Tensor
    decoder_out: torch.Tensor
    crm: torch.Tensor
    enhanced_complex: torch.Tensor


class BoneAirFusionEnhancementModel(nn.Module):
    """
    Full dual-modal complex-domain speech enhancement model.

    paper-specified:
        - inputs are AC and BC complex spectra with shape `[B, 2, T, 161]`
        - iAFF first fuses AC and BC into `y_AF`
        - `[y_AF, y_AC, y_BC]` are concatenated into `[B, 6, T, 161]`
        - DenGCAN predicts a 2-channel decoder feature map
        - the mask head predicts cRM real / imag
        - predicted cRM is complex-multiplied with noisy AC to recover the
          enhanced complex spectrum

    Shape:
        y_ac: [B, 2, T, 161]
        y_bc: [B, 2, T, 161]
        return: `FullModelOutput`
    """

    def __init__(
        self,
        cfg: FullModelConfig | None = None,
        iaff_cfg: IAFFConfig | None = None,
        dengcan_cfg: DenGCANConfig | None = None,
        mask_head_cfg: MaskHeadConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or FullModelConfig()

        self.iaff = IAFF(iaff_cfg or IAFFConfig(in_channels=self.cfg.complex_channels))
        self.backbone = DenGCAN(
            dengcan_cfg
            or DenGCANConfig(
                in_channels=self.cfg.fused_backbone_channels,
                out_channels=self.cfg.complex_channels,
                input_freq_bins=self.cfg.freq_bins,
            )
        )
        self.mask_head = MaskHead(
            mask_head_cfg
            or MaskHeadConfig(
                in_channels=self.cfg.complex_channels,
                freq_bins=self.cfg.freq_bins,
            )
        )

    def forward(self, y_ac: torch.Tensor, y_bc: torch.Tensor) -> FullModelOutput:
        """Run the full AC/BC speech enhancement model."""
        assert_shape(y_ac, [None, self.cfg.complex_channels, None, None], "y_ac")
        assert_shape(y_bc, [None, self.cfg.complex_channels, None, None], "y_bc")
        assert_same_shape([y_ac, y_bc], ["y_ac", "y_bc"])
        assert_freq_bins(y_ac, self.cfg.freq_bins, "y_ac")
        assert_freq_bins(y_bc, self.cfg.freq_bins, "y_bc")

        y_af = self.iaff(y_ac, y_bc)
        backbone_input = torch.cat([y_af, y_ac, y_bc], dim=1)
        assert_shape(
            backbone_input,
            [y_ac.shape[0], self.cfg.fused_backbone_channels, y_ac.shape[2], self.cfg.freq_bins],
            "backbone_input",
        )

        decoder_out = self.backbone(backbone_input)
        crm = self.mask_head(decoder_out)
        enhanced_complex = apply_crm(y_ac, crm)

        return FullModelOutput(
            y_af=y_af,
            backbone_input=backbone_input,
            decoder_out=decoder_out,
            crm=crm,
            enhanced_complex=enhanced_complex,
        )
