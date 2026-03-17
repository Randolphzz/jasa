"""Loss functions for complex-domain speech enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.complex_utils import complex_mag, split_real_imag
from utils.shape_check import assert_same_shape, assert_shape


@dataclass(frozen=True)
class LossConfig:
    """Configuration for the enhancement loss."""

    # paper-specified combination weights
    ri_weight: float = 0.5
    mag_weight: float = 0.5

    # engineering assumption:
    # the paper excerpt fixes the composition of the loss but does not fully
    # specify the pointwise reduction. This implementation uses mean-squared
    # error for RI and magnitude terms.
    base_loss: str = "mse"


class RILoss(nn.Module):
    """
    Real-imaginary loss over 2-channel complex spectra.

    engineering assumption:
        Uses element-wise MSE with mean reduction over real and imaginary parts.

    Shape:
        pred_complex: [B, 2, T, F]
        target_complex: [B, 2, T, F]
        return: scalar tensor
    """

    def __init__(self, base_loss: str = "mse"):
        super().__init__()
        self.base_loss = base_loss
        self.criterion = _build_pointwise_loss(base_loss)

    def forward(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """Compute RI loss between predicted and target complex spectra."""
        _assert_complex_pair(pred_complex, target_complex, "pred_complex", "target_complex")
        pred_real, pred_imag = split_real_imag(pred_complex)
        target_real, target_imag = split_real_imag(target_complex)

        real_loss = self.criterion(pred_real, target_real)
        imag_loss = self.criterion(pred_imag, target_imag)
        return 0.5 * (real_loss + imag_loss)


class MagnitudeLoss(nn.Module):
    """
    Magnitude-domain loss over 2-channel complex spectra.

    engineering assumption:
        Uses element-wise MSE with mean reduction after magnitude conversion.

    Shape:
        pred_complex: [B, 2, T, F]
        target_complex: [B, 2, T, F]
        return: scalar tensor
    """

    def __init__(self, base_loss: str = "mse"):
        super().__init__()
        self.base_loss = base_loss
        self.criterion = _build_pointwise_loss(base_loss)

    def forward(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """Compute magnitude loss between predicted and target complex spectra."""
        _assert_complex_pair(pred_complex, target_complex, "pred_complex", "target_complex")
        pred_mag = complex_mag(pred_complex)
        target_mag = complex_mag(target_complex)
        return self.criterion(pred_mag, target_mag)


class SpeechEnhancementLoss(nn.Module):
    """
    Combined enhancement loss: `0.5 * L_RI + 0.5 * L_Mag`.

    paper-specified:
        `L = 0.5 * L_RI + 0.5 * L_Mag`

    engineering assumption:
        `L_RI` and `L_Mag` each use mean-squared error as their pointwise base
        loss unless configured otherwise.

    Shape:
        pred_complex: [B, 2, T, F]
        target_complex: [B, 2, T, F]
        return: scalar tensor, or a loss-breakdown dict if requested
    """

    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.ri_loss = RILoss(base_loss=self.cfg.base_loss)
        self.mag_loss = MagnitudeLoss(base_loss=self.cfg.base_loss)

    def forward(
        self,
        pred_complex: torch.Tensor,
        target_complex: torch.Tensor,
        return_breakdown: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute total loss, optionally returning RI and magnitude components."""
        ri = self.ri_loss(pred_complex, target_complex)
        mag = self.mag_loss(pred_complex, target_complex)
        total = self.cfg.ri_weight * ri + self.cfg.mag_weight * mag

        if return_breakdown:
            return {
                "total": total,
                "ri_loss": ri,
                "mag_loss": mag,
            }
        return total


def _build_pointwise_loss(base_loss: str) -> nn.Module:
    """Build the pointwise loss used by RI and magnitude terms."""
    if base_loss == "mse":
        return nn.MSELoss()
    if base_loss == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported base_loss={base_loss!r}. Expected 'mse' or 'l1'.")


def _assert_complex_pair(
    pred_complex: torch.Tensor,
    target_complex: torch.Tensor,
    pred_name: str,
    target_name: str,
) -> None:
    """Validate a predicted / target complex-spectrum pair."""
    assert_shape(pred_complex, [None, 2, None, None], pred_name)
    assert_shape(target_complex, [None, 2, None, None], target_name)
    assert_same_shape([pred_complex, target_complex], [pred_name, target_name])
