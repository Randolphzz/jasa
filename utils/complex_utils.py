"""Complex-spectrum utilities for two-channel real/imag tensor representation."""

from __future__ import annotations

import torch

from utils.shape_check import assert_rank, assert_same_shape, assert_shape


# paper-specified: model input/output complex spectra are represented as
# [B, 2, T, F], where channel 0=real and channel 1=imag.
# engineering assumption: helper APIs default `complex_dim=1` and enforce size=2.


def split_real_imag(complex_tensor: torch.Tensor, complex_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 2-channel real/imag tensor.

    Args:
        complex_tensor: [..., 2, ...] complex representation.
        complex_dim: Dimension that stores real/imag channels.

    Returns:
        real, imag tensors with `complex_dim` removed.
    """
    if complex_tensor.shape[complex_dim] != 2:
        raise ValueError(
            f"complex channel size must be 2, got {complex_tensor.shape[complex_dim]} "
            f"for shape {tuple(complex_tensor.shape)}"
        )
    real, imag = torch.unbind(complex_tensor, dim=complex_dim)
    return real, imag


def stack_real_imag(real: torch.Tensor, imag: torch.Tensor, complex_dim: int = 1) -> torch.Tensor:
    """Stack real and imag tensors into a 2-channel complex tensor."""
    assert_same_shape([real, imag], ["real", "imag"])
    return torch.stack([real, imag], dim=complex_dim)


def complex_mul(a: torch.Tensor, b: torch.Tensor, complex_dim: int = 1) -> torch.Tensor:
    """
    Complex multiplication for 2-channel tensors.

    Shape:
        a: [B, 2, T, F] (default complex_dim=1)
        b: [B, 2, T, F]
        return: [B, 2, T, F]
    """
    assert_same_shape([a, b], ["a", "b"])
    ar, ai = split_real_imag(a, complex_dim=complex_dim)
    br, bi = split_real_imag(b, complex_dim=complex_dim)

    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return stack_real_imag(real, imag, complex_dim=complex_dim)


def complex_mag(complex_tensor: torch.Tensor, complex_dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Magnitude of a 2-channel complex tensor.

    Shape:
        complex_tensor: [B, 2, T, F]
        return: [B, T, F]
    """
    real, imag = split_real_imag(complex_tensor, complex_dim=complex_dim)
    return torch.sqrt(real * real + imag * imag + eps)


def apply_crm(noisy_complex: torch.Tensor, crm: torch.Tensor, complex_dim: int = 1) -> torch.Tensor:
    """
    Apply complex ratio mask to noisy complex spectrum.

    Shape:
        noisy_complex: [B, 2, T, F]
        crm: [B, 2, T, F]
        return enhanced_complex: [B, 2, T, F]
    """
    assert_rank(noisy_complex, 4, "noisy_complex")
    assert_shape(noisy_complex, [None, 2, None, None], "noisy_complex")
    assert_shape(crm, [None, 2, None, None], "crm")
    return complex_mul(noisy_complex, crm, complex_dim=complex_dim)


def to_torch_complex(two_channel_complex: torch.Tensor, complex_dim: int = 1) -> torch.Tensor:
    """
    Convert two-channel representation to torch.complex tensor.

    Shape:
        input: [B, 2, T, F]
        return: [B, T, F] complex dtype
    """
    real, imag = split_real_imag(two_channel_complex, complex_dim=complex_dim)
    return torch.complex(real, imag)


def from_torch_complex(complex_tensor: torch.Tensor, complex_dim: int = 1) -> torch.Tensor:
    """
    Convert torch.complex tensor to two-channel representation.

    Shape:
        input: [B, T, F] complex dtype
        return: [B, 2, T, F]
    """
    if not torch.is_complex(complex_tensor):
        raise TypeError("complex_tensor must be a torch complex tensor.")
    return stack_real_imag(complex_tensor.real, complex_tensor.imag, complex_dim=complex_dim)
