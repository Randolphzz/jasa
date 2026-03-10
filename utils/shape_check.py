"""Shape checking helpers for strict tensor interface validation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


class ShapeMismatchError(ValueError):
    """Raised when a tensor does not satisfy the expected shape contract."""


def _shape_to_str(shape: Sequence[int]) -> str:
    return "[" + ", ".join(str(x) for x in shape) + "]"


def assert_rank(tensor: torch.Tensor, rank: int, name: str) -> None:
    """Assert `tensor.ndim == rank`."""
    if tensor.ndim != rank:
        raise ShapeMismatchError(
            f"{name} rank mismatch: expected {rank}, got {tensor.ndim}. "
            f"Actual shape={_shape_to_str(tuple(tensor.shape))}."
        )


def assert_shape(tensor: torch.Tensor, expected: Sequence[int | None], name: str) -> None:
    """
    Assert tensor shape equals `expected` with optional wildcard dimensions.

    Args:
        tensor: Tensor to validate.
        expected: Shape spec. Use `None` as wildcard for any size.
        name: Tensor name for error messages.
    """
    if tensor.ndim != len(expected):
        raise ShapeMismatchError(
            f"{name} rank mismatch: expected {len(expected)}, got {tensor.ndim}. "
            f"Expected shape={expected}, actual={_shape_to_str(tuple(tensor.shape))}."
        )

    for dim_idx, (actual_dim, expected_dim) in enumerate(zip(tensor.shape, expected)):
        if expected_dim is not None and actual_dim != expected_dim:
            raise ShapeMismatchError(
                f"{name} shape mismatch at dim {dim_idx}: expected {expected_dim}, got {actual_dim}. "
                f"Expected shape={expected}, actual={_shape_to_str(tuple(tensor.shape))}."
            )


def assert_same_shape(tensors: Iterable[torch.Tensor], names: Iterable[str] | None = None) -> None:
    """Assert all tensors in `tensors` have identical shape."""
    tensors = list(tensors)
    if not tensors:
        raise ValueError("assert_same_shape expects at least one tensor.")

    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    names = list(names)
    if len(names) != len(tensors):
        raise ValueError("`names` length must match number of tensors.")

    ref_shape = tuple(tensors[0].shape)
    ref_name = names[0]
    for tensor, name in zip(tensors[1:], names[1:]):
        if tuple(tensor.shape) != ref_shape:
            raise ShapeMismatchError(
                f"Shape mismatch between {ref_name}={_shape_to_str(ref_shape)} and "
                f"{name}={_shape_to_str(tuple(tensor.shape))}."
            )


def assert_freq_bins(tensor: torch.Tensor, freq_bins: int, name: str, freq_dim: int = -1) -> None:
    """Assert specific frequency-dimension width (default: last dim)."""
    actual = tensor.shape[freq_dim]
    if actual != freq_bins:
        raise ShapeMismatchError(
            f"{name} frequency-bin mismatch: expected {freq_bins}, got {actual}. "
            f"Actual shape={_shape_to_str(tuple(tensor.shape))}."
        )
