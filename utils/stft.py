"""STFT/ISTFT helpers configured for paper settings."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.complex_utils import from_torch_complex, to_torch_complex
from utils.shape_check import assert_rank, assert_shape


@dataclass(frozen=True)
class STFTConfig:
    """Configuration for waveform <-> complex-spectrum conversion."""

    # paper-specified
    sample_rate: int = 16000
    win_ms: float = 20.0
    hop_ms: float = 10.0
    n_fft: int = 320
    window_type: str = "hann"
    onesided: bool = True

    # engineering assumption
    center: bool = True
    normalized: bool = False

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.win_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def freq_bins(self) -> int:
        return self.n_fft // 2 + 1


class STFTProcessor:
    """
    STFT processor using two-channel complex tensors.

    Notes:
        - paper-specified: 16kHz, 20ms window, 10ms hop, Hann window, FFT=320, F=161.
        - engineering assumption: `center=True` and `normalized=False` for `torch.stft`.
    """

    def __init__(self, cfg: STFTConfig | None = None):
        self.cfg = cfg or STFTConfig()
        if self.cfg.window_type.lower() != "hann":
            raise ValueError("Only Hann window is supported in this implementation.")

    def _build_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(self.cfg.win_length, periodic=True, device=device, dtype=dtype)

    def waveform_to_complex(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to two-channel complex spectrum.

        Shape:
            waveform: [B, N]
            return: [B, 2, T, F] where F=161 under paper setting
        """
        assert_rank(waveform, 2, "waveform")

        window = self._build_window(waveform.device, waveform.dtype)
        stft_complex = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            normalized=self.cfg.normalized,
            onesided=self.cfg.onesided,
            return_complex=True,
        )
        # torch.stft: [B, F, T] complex -> [B, T, F] complex
        stft_complex = stft_complex.transpose(1, 2)
        two_channel = from_torch_complex(stft_complex, complex_dim=1)
        assert_shape(two_channel, [waveform.shape[0], 2, None, self.cfg.freq_bins], "stft_two_channel")
        return two_channel

    def complex_to_waveform(self, complex_spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """
        Convert two-channel complex spectrum to waveform via ISTFT.

        Shape:
            complex_spec: [B, 2, T, F]
            return waveform: [B, N]
        """
        assert_shape(complex_spec, [None, 2, None, self.cfg.freq_bins], "complex_spec")

        window = self._build_window(complex_spec.device, complex_spec.dtype)
        spec_complex = to_torch_complex(complex_spec, complex_dim=1)
        # [B, T, F] -> [B, F, T] for torch.istft
        spec_complex = spec_complex.transpose(1, 2)
        waveform = torch.istft(
            spec_complex,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            normalized=self.cfg.normalized,
            onesided=self.cfg.onesided,
            length=length,
        )
        assert_rank(waveform, 2, "waveform")
        return waveform
