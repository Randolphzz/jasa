"""Minimal forward + loss wrapper for model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.full_model import BoneAirFusionEnhancementModel, FullModelOutput
from utils.batch import SpeechEnhancementBatch
from utils.losses import LossConfig, SpeechEnhancementLoss


@dataclass(frozen=True)
class TrainingStepOutput:
    """
    Outputs from one forward + loss computation.

    Shape:
        target_complex: [B, 2, T, 161]
        model_output.enhanced_complex: [B, 2, T, 161]
        total_loss / ri_loss / mag_loss: scalar tensors
    """

    model_output: FullModelOutput
    target_complex: torch.Tensor
    total_loss: torch.Tensor
    ri_loss: torch.Tensor
    mag_loss: torch.Tensor


class SpeechEnhancementTrainingStep(nn.Module):
    """
    Minimal training-step wrapper around the full model and enhancement loss.

    engineering assumption:
        The paper defines the model and loss, but not an application-level
        training-step abstraction. This wrapper provides a small reusable
        interface for forward + loss computation without introducing a full
        training script.
    """

    def __init__(
        self,
        model: BoneAirFusionEnhancementModel | None = None,
        loss_module: SpeechEnhancementLoss | None = None,
        loss_cfg: LossConfig | None = None,
    ):
        super().__init__()
        self.model = model or BoneAirFusionEnhancementModel()
        self.loss_module = loss_module or SpeechEnhancementLoss(loss_cfg or LossConfig())

    def forward(self, batch: SpeechEnhancementBatch) -> TrainingStepOutput:
        """Run one forward pass and compute the supervised enhancement loss."""
        if not isinstance(batch, SpeechEnhancementBatch):
            raise TypeError(
                "SpeechEnhancementTrainingStep expects a SpeechEnhancementBatch, "
                f"got {type(batch).__name__}."
            )

        model_output = self.model(batch.noisy_ac, batch.noisy_bc)
        breakdown = self.loss_module(
            model_output.enhanced_complex,
            batch.clean_ac,
            return_breakdown=True,
        )

        return TrainingStepOutput(
            model_output=model_output,
            target_complex=batch.clean_ac,
            total_loss=breakdown["total"],
            ri_loss=breakdown["ri_loss"],
            mag_loss=breakdown["mag_loss"],
        )
