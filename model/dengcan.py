"""DenGCAN backbone for dual-modal complex-domain speech enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.attention_gate import AttentionGate, AttentionGateConfig
from models.bottleneck import DenGCANBottleneck, DenGCANBottleneckConfig
from models.dense_block import DecoderDenseBlock, EncoderDenseBlock, GatedConvConfig
from utils.shape_check import assert_shape


@dataclass(frozen=True)
class DenGCANConfig:
    """Configuration for the DenGCAN encoder-bottleneck-decoder backbone."""

    # paper-specified interface and Table I channel schedule
    in_channels: int = 6
    out_channels: int = 2
    input_freq_bins: int = 161
    bottleneck_freq_bins: int = 3
    encoder_channels: tuple[int, int, int, int, int] = (16, 32, 48, 64, 64)
    decoder_channels: tuple[int, int, int, int, int] = (64, 48, 32, 16, 2)
    encoder_freq_bins: tuple[int, int, int, int, int] = (79, 38, 18, 8, 3)
    decoder_freq_bins: tuple[int, int, int, int, int] = (8, 18, 38, 79, 161)

    # paper-specified special case
    decoder_first_dense_use_bn_prelu: bool = False

    # engineering assumptions for exact deconvolution recovery and final raw feature output
    decoder_output_paddings: tuple[tuple[int, int], ...] = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 1),
    )
    decoder_use_post_act: tuple[bool, bool, bool, bool, bool] = (True, True, True, True, False)


class DenGCAN(nn.Module):
    """
    DenGCAN backbone with encoder, grouped sConformer bottleneck, and decoder.

    paper-specified:
        - input is `[B, 6, T, 161]`
        - encoder has 5 dense blocks with channel schedule `[16, 32, 48, 64, 64]`
        - bottleneck reshapes `[B, 64, T, 3] -> [B, T, 192]`, applies 2-layer
          grouped sConformer, then reshapes back
        - decoder has 5 dense blocks with channel schedule `[64, 48, 32, 16, 2]`
        - AG is used on skip connections

    engineering assumption:
        - exact AG internals come from the local `AttentionGate` placeholder
        - deconvolution `output_padding` values are chosen to recover the Table I
          frequency widths exactly
        - the final decoder transposed convolution omits BN + PReLU so the output
          remains a raw 2-channel feature map for the cRM head

    Shape:
        input: [B, 6, T, 161]
        output: [B, 2, T, 161]
    """

    def __init__(
        self,
        cfg: DenGCANConfig | None = None,
        bottleneck_cfg: DenGCANBottleneckConfig | None = None,
        attention_gate_cfg: AttentionGateConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or DenGCANConfig()
        self._validate_config()

        encoder_in_channels = (self.cfg.in_channels,) + self.cfg.encoder_channels[:-1]
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderDenseBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    gated_cfg=GatedConvConfig(padding=(0, 0)),
                )
                for in_channels, out_channels in zip(encoder_in_channels, self.cfg.encoder_channels)
            ]
        )

        self.bottleneck = DenGCANBottleneck(
            bottleneck_cfg
            or DenGCANBottleneckConfig(
                in_channels=self.cfg.encoder_channels[-1],
                bottleneck_freq_bins=self.cfg.bottleneck_freq_bins,
            )
        )

        self._skip_channels = (
            self.cfg.encoder_channels[4],
            self.cfg.encoder_channels[3],
            self.cfg.encoder_channels[2],
            self.cfg.encoder_channels[1],
            self.cfg.encoder_channels[0],
        )
        self._decoder_input_channels = (128, 128, 96, 64, 32)
        self._decoder_gate_channels = (64, 64, 48, 32, 16)

        self.attention_gates = nn.ModuleList(
            [
                AttentionGate(
                    x_channels=skip_channels,
                    g_channels=gate_channels,
                    cfg=attention_gate_cfg,
                )
                for skip_channels, gate_channels in zip(self._skip_channels, self._decoder_gate_channels)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderDenseBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    gated_cfg=GatedConvConfig(
                        padding=(0, 0),
                        output_padding=output_padding,
                    ),
                    dense_use_bn_prelu=(
                        self.cfg.decoder_first_dense_use_bn_prelu if idx == 0 else True
                    ),
                    use_post_act=use_post_act,
                )
                for idx, (in_channels, out_channels, output_padding, use_post_act) in enumerate(
                    zip(
                        self._decoder_input_channels,
                        self.cfg.decoder_channels,
                        self.cfg.decoder_output_paddings,
                        self.cfg.decoder_use_post_act,
                    )
                )
            ]
        )

    def _validate_config(self) -> None:
        """Validate DenGCAN schedule lengths and bottleneck consistency."""
        if len(self.cfg.encoder_channels) != 5 or len(self.cfg.decoder_channels) != 5:
            raise ValueError("DenGCAN expects 5 encoder and 5 decoder blocks.")
        if len(self.cfg.encoder_freq_bins) != 5 or len(self.cfg.decoder_freq_bins) != 5:
            raise ValueError("DenGCAN expects 5 encoder and 5 decoder frequency entries.")
        if len(self.cfg.decoder_output_paddings) != 5 or len(self.cfg.decoder_use_post_act) != 5:
            raise ValueError("DenGCAN decoder metadata must define 5 stages.")
        if self.cfg.encoder_channels[-1] * self.cfg.bottleneck_freq_bins != 192:
            raise ValueError("DenGCAN bottleneck reshape expects 64 * 3 = 192 features.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DenGCAN backbone to a fused AC/BC feature map."""
        assert_shape(x, [None, self.cfg.in_channels, None, self.cfg.input_freq_bins], "dengcan_input")
        batch = x.shape[0]
        frames = x.shape[2]

        encoder_outputs: list[torch.Tensor] = []
        cur = x
        for idx, (block, out_channels, freq_bins) in enumerate(
            zip(self.encoder_blocks, self.cfg.encoder_channels, self.cfg.encoder_freq_bins),
            start=1,
        ):
            cur = block(cur)
            assert_shape(cur, [batch, out_channels, frames, freq_bins], f"encoder_block_{idx}_output")
            encoder_outputs.append(cur)

        cur = self.bottleneck(cur)
        assert_shape(
            cur,
            [batch, self.cfg.encoder_channels[-1], frames, self.cfg.bottleneck_freq_bins],
            "bottleneck_output",
        )

        skip_order = list(reversed(encoder_outputs))
        skip_freq_bins = tuple(reversed(self.cfg.encoder_freq_bins))

        for idx, (
            attention_gate,
            decoder_block,
            skip,
            skip_channels,
            gate_channels,
            skip_freq_bins_i,
            out_channels,
            out_freq_bins,
            decoder_in_channels,
        ) in enumerate(
            zip(
                self.attention_gates,
                self.decoder_blocks,
                skip_order,
                self._skip_channels,
                self._decoder_gate_channels,
                skip_freq_bins,
                self.cfg.decoder_channels,
                self.cfg.decoder_freq_bins,
                self._decoder_input_channels,
            ),
            start=1,
        ):
            assert_shape(cur, [batch, gate_channels, frames, skip_freq_bins_i], f"decoder_gate_{idx}_input")
            gated_skip = attention_gate(skip, cur)
            assert_shape(gated_skip, [batch, skip_channels, frames, skip_freq_bins_i], f"ag_{idx}_output")

            cur = torch.cat([cur, gated_skip], dim=1)
            assert_shape(cur, [batch, decoder_in_channels, frames, skip_freq_bins_i], f"decoder_block_{idx}_input")

            cur = decoder_block(cur)
            assert_shape(cur, [batch, out_channels, frames, out_freq_bins], f"decoder_block_{idx}_output")

        return cur
