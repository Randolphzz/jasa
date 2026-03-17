"""Dense block and gated (de)convolution layers used in DenGCAN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.shape_check import assert_rank


@dataclass(frozen=True)
class DenseLayerConfig:
    # paper-specified
    num_convs: int = 4
    growth_channels: int = 8
    kernel_size: tuple[int, int] = (1, 3)
    stride: tuple[int, int] = (1, 1)

    # engineering assumption: use (0,1) padding to preserve F and T in dense conv stack.
    padding: tuple[int, int] = (0, 1)


class _ConvBNPReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        use_bn_prelu: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if use_bn_prelu:
            layers.extend(
                [
                    nn.BatchNorm2d(out_channels),
                    nn.PReLU(num_parameters=out_channels),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseLayer4Conv(nn.Module):
    """
    DenseNet-style layer with 4 conv units.

    paper-specified:
        - each dense layer includes 4 conv layers
        - each conv uses kernel=(1,3), stride=(1,1), out_channels=8
        - each conv followed by BN + PReLU
        - dense concatenation is used

    paper-specified exception:
        - the first decoder dense block omits BN + PReLU inside its dense convs
    """

    def __init__(
        self,
        in_channels: int,
        cfg: DenseLayerConfig | None = None,
        use_bn_prelu: bool = True,
    ):
        super().__init__()
        self.cfg = cfg or DenseLayerConfig()
        self.use_bn_prelu = use_bn_prelu
        if self.cfg.num_convs != 4:
            raise ValueError("This implementation expects num_convs=4 per paper.")

        convs = []
        cur_in = in_channels
        for _ in range(self.cfg.num_convs):
            convs.append(
                _ConvBNPReLU(
                    in_channels=cur_in,
                    out_channels=self.cfg.growth_channels,
                    kernel_size=self.cfg.kernel_size,
                    stride=self.cfg.stride,
                    padding=self.cfg.padding,
                    use_bn_prelu=self.use_bn_prelu,
                )
            )
            cur_in += self.cfg.growth_channels
        self.convs = nn.ModuleList(convs)
        self.out_channels = in_channels + self.cfg.num_convs * self.cfg.growth_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, F]

        Returns:
            y: [B, C_in + 4*growth, T, F]
        """
        assert_rank(x, 4, "dense_input")
        features = [x]
        for conv in self.convs:
            dense_input = torch.cat(features, dim=1)
            out = conv(dense_input)
            features.append(out)
        return torch.cat(features, dim=1)


@dataclass(frozen=True)
class GatedConvConfig:
    # paper-specified
    kernel_size: tuple[int, int] = (1, 4)
    stride: tuple[int, int] = (1, 2)

    # engineering assumption for Table I alignment
    padding: tuple[int, int] = (0, 0)
    output_padding: tuple[int, int] = (0, 0)


class GatedConv2d(nn.Module):
    """
    Gated convolutional downsampling layer.

    paper-specified:
        two branches (main and sigmoid gate), element-wise multiply,
        followed by BN + PReLU (except decoder first block case, not here).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: GatedConvConfig | None = None,
        use_post_act: bool = True,
    ):
        super().__init__()
        self.cfg = cfg or GatedConvConfig()
        self.main_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.cfg.kernel_size,
            stride=self.cfg.stride,
            padding=self.cfg.padding,
        )
        self.gate_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.cfg.kernel_size,
            stride=self.cfg.stride,
            padding=self.cfg.padding,
        )
        self.use_post_act = use_post_act
        self.post = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, F]

        Returns:
            y: [B, C_out, T, F_down]
        """
        assert_rank(x, 4, "gated_conv_input")
        main = self.main_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        out = main * gate
        if self.use_post_act:
            out = self.post(out)
        return out


class GatedDeconv2d(nn.Module):
    """
    Gated transposed convolutional upsampling layer.

    paper-specified:
        decoder uses transposed convolution corresponding to encoder gated conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: GatedConvConfig | None = None,
        use_post_act: bool = True,
    ):
        super().__init__()
        self.cfg = cfg or GatedConvConfig()
        self.main_deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=self.cfg.kernel_size,
            stride=self.cfg.stride,
            padding=self.cfg.padding,
            output_padding=self.cfg.output_padding,
        )
        self.gate_deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=self.cfg.kernel_size,
            stride=self.cfg.stride,
            padding=self.cfg.padding,
            output_padding=self.cfg.output_padding,
        )
        self.use_post_act = use_post_act
        self.post = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, F]

        Returns:
            y: [B, C_out, T, F_up]
        """
        assert_rank(x, 4, "gated_deconv_input")
        main = self.main_deconv(x)
        gate = torch.sigmoid(self.gate_deconv(x))
        out = main * gate
        if self.use_post_act:
            out = self.post(out)
        return out


class EncoderDenseBlock(nn.Module):
    """
    Convenience block: dense layer then gated downsampling.

    Shape:
        input: [B, C_in, T, F]
        output: [B, C_out, T, F_down]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dense_cfg: DenseLayerConfig | None = None,
        gated_cfg: GatedConvConfig | None = None,
    ):
        super().__init__()
        self.dense = DenseLayer4Conv(in_channels, cfg=dense_cfg)
        self.gated = GatedConv2d(self.dense.out_channels, out_channels, cfg=gated_cfg, use_post_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T, F] -> y: [B, C_out, T, F_down]."""
        x = self.dense(x)
        x = self.gated(x)
        return x


class DecoderDenseBlock(nn.Module):
    """
    Convenience block: dense layer then gated transposed-conv upsampling.

    Shape:
        input: [B, C_in, T, F]
        output: [B, C_out, T, F_up]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dense_cfg: DenseLayerConfig | None = None,
        gated_cfg: GatedConvConfig | None = None,
        dense_use_bn_prelu: bool = True,
        use_post_act: bool = True,
    ):
        super().__init__()
        self.dense = DenseLayer4Conv(in_channels, cfg=dense_cfg, use_bn_prelu=dense_use_bn_prelu)
        self.degated = GatedDeconv2d(
            self.dense.out_channels,
            out_channels,
            cfg=gated_cfg,
            use_post_act=use_post_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T, F] -> y: [B, C_out, T, F_up]."""
        x = self.dense(x)
        x = self.degated(x)
        return x
