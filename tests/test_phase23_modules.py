from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.attention_gate import AttentionGate
from models.dense_block import DenseLayer4Conv, GatedConv2d, GatedConvConfig, GatedDeconv2d
from models.iaff import IAFF
from models.mask_head import MaskHead
from utils.complex_utils import apply_crm, complex_mag
from utils.stft import STFTProcessor


def test_complex_utils_shapes() -> None:
    b, t, f = 2, 20, 161
    noisy = torch.randn(b, 2, t, f)
    crm = torch.randn(b, 2, t, f)
    enhanced = apply_crm(noisy, crm)
    mag = complex_mag(enhanced)

    assert enhanced.shape == (b, 2, t, f)
    assert mag.shape == (b, t, f)


def test_stft_processor_shapes() -> None:
    b, n = 2, 16000
    wav = torch.randn(b, n)
    processor = STFTProcessor()

    spec = processor.waveform_to_complex(wav)
    wav_hat = processor.complex_to_waveform(spec, length=n)

    assert spec.shape[0] == b
    assert spec.shape[1] == 2
    assert spec.shape[-1] == 161
    assert wav_hat.shape == (b, n)


def test_iaff_shape() -> None:
    b, t, f = 2, 24, 161
    ac = torch.randn(b, 2, t, f)
    bc = torch.randn(b, 2, t, f)
    iaff = IAFF()

    y_af = iaff(ac, bc)
    assert y_af.shape == (b, 2, t, f)


def test_dense_and_gated_tablei_frequency_progression() -> None:
    b, t = 2, 16

    x0 = torch.randn(b, 6, t, 161)
    dense = DenseLayer4Conv(in_channels=6)
    x_dense = dense(x0)
    assert x_dense.shape == (b, 38, t, 161)

    down = GatedConv2d(in_channels=38, out_channels=16, cfg=GatedConvConfig(padding=(0, 0)))
    x1 = down(x_dense)
    assert x1.shape == (b, 16, t, 79)

    x = x1
    for out_ch, expect_f in zip([32, 48, 64, 64], [38, 18, 8, 3]):
        x = GatedConv2d(in_channels=x.shape[1], out_channels=out_ch, cfg=GatedConvConfig(padding=(0, 0)))(x)
        assert x.shape == (b, out_ch, t, expect_f)

    # decoder-side transposed conv progression: 3->8->18->38->79->161
    z = torch.randn(b, 128, t, 3)
    z = GatedDeconv2d(
        in_channels=128,
        out_channels=64,
        cfg=GatedConvConfig(padding=(0, 0), output_padding=(0, 0)),
    )(z)
    assert z.shape == (b, 64, t, 8)

    z = GatedDeconv2d(
        in_channels=64,
        out_channels=48,
        cfg=GatedConvConfig(padding=(0, 0), output_padding=(0, 0)),
    )(z)
    assert z.shape == (b, 48, t, 18)

    z = GatedDeconv2d(
        in_channels=48,
        out_channels=32,
        cfg=GatedConvConfig(padding=(0, 0), output_padding=(0, 0)),
    )(z)
    assert z.shape == (b, 32, t, 38)

    z = GatedDeconv2d(
        in_channels=32,
        out_channels=16,
        cfg=GatedConvConfig(padding=(0, 0), output_padding=(0, 1)),
    )(z)
    assert z.shape == (b, 16, t, 79)

    z = GatedDeconv2d(
        in_channels=16,
        out_channels=2,
        cfg=GatedConvConfig(padding=(0, 0), output_padding=(0, 1)),
        use_post_act=False,
    )(z)
    assert z.shape == (b, 2, t, 161)


def test_attention_gate_shape() -> None:
    b, t, f = 2, 20, 38
    x_skip = torch.randn(b, 32, t, f)
    g = torch.randn(b, 64, t, f)
    ag = AttentionGate(x_channels=32, g_channels=64)

    y = ag(x_skip, g)
    assert y.shape == x_skip.shape


def test_mask_head_shape() -> None:
    b, t, f = 2, 30, 161
    decoder_out = torch.randn(b, 2, t, f)
    head = MaskHead()

    crm = head(decoder_out)
    assert crm.shape == (b, 2, t, f)


def _run_all_tests_without_pytest() -> None:
    test_complex_utils_shapes()
    test_stft_processor_shapes()
    test_iaff_shape()
    test_dense_and_gated_tablei_frequency_progression()
    test_attention_gate_shape()
    test_mask_head_shape()
    print("All phase2/phase3 module tests passed.")


if __name__ == "__main__":
    _run_all_tests_without_pytest()
