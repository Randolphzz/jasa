from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.training_step import SpeechEnhancementTrainingStep
from utils.batch import SpeechEnhancementBatch


def test_training_step_shapes_and_backward() -> None:
    batch = SpeechEnhancementBatch(
        noisy_ac=torch.randn(2, 2, 9, 161),
        noisy_bc=torch.randn(2, 2, 9, 161),
        clean_ac=torch.randn(2, 2, 9, 161),
    )
    step = SpeechEnhancementTrainingStep()

    output = step(batch)

    assert output.model_output.y_af.shape == (2, 2, 9, 161)
    assert output.model_output.backbone_input.shape == (2, 6, 9, 161)
    assert output.model_output.enhanced_complex.shape == (2, 2, 9, 161)
    assert output.target_complex.shape == (2, 2, 9, 161)
    assert output.total_loss.ndim == 0
    assert output.ri_loss.ndim == 0
    assert output.mag_loss.ndim == 0

    output.total_loss.backward()
    assert any(param.grad is not None for param in step.parameters() if param.requires_grad)


def test_training_step_invalid_batch_type_raises() -> None:
    step = SpeechEnhancementTrainingStep()

    try:
        step(torch.randn(2, 2, 9, 161))
    except TypeError:
        return
    raise AssertionError("Expected TypeError to be raised.")


def _run_all_tests_without_pytest() -> None:
    test_training_step_shapes_and_backward()
    test_training_step_invalid_batch_type_raises()
    print("All training-step tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
