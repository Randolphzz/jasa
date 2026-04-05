"""sConformer building blocks used by the DenGCAN bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.shape_check import assert_shape


@dataclass(frozen=True)
class SConformerConfig:
    """Configuration shared by sConformer modules."""

    # paper-specified defaults for the bottleneck sequence model
    d_model: int = 192
    num_heads: int = 4
    ffn_dim: int = 384
    conv_kernel_size: int = 31
    dropout: float = 0.1

    # source-aligned engineering options
    causal: bool = False
    lookahead: int = 0


class CumulativeLayerNorm1d(nn.Module):
    """Causal cumulative layer normalization over `[B, T, C]`."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_shape(x, [None, None, self.num_features], "cln_input")

        step_sum = torch.sum(x, dim=2)
        step_squared_sum = torch.sum(x.square(), dim=2)
        cumulative_sum = torch.cumsum(step_sum, dim=1)
        cumulative_squared_sum = torch.cumsum(step_squared_sum, dim=1)

        counts = torch.arange(
            self.num_features,
            self.num_features * (x.shape[1] + 1),
            self.num_features,
            device=x.device,
            dtype=x.dtype,
        )
        cumulative_mean = cumulative_sum / counts
        cumulative_squared_mean = cumulative_squared_sum / counts
        cumulative_var = torch.clamp(cumulative_squared_mean - cumulative_mean.square(), min=0.0)

        normalized = (x - cumulative_mean.unsqueeze(2)) / torch.sqrt(cumulative_var.unsqueeze(2) + self.eps)
        return normalized * self.weight + self.bias


def _build_norm(d_model: int, causal: bool) -> nn.Module:
    return CumulativeLayerNorm1d(d_model) if causal else nn.LayerNorm(d_model)


class _ConformerStyleConvModule(nn.Module):
    """
    Conformer-style depthwise separable convolution module.

    engineering assumption:
        The JASA paper names the sConformer structure but does not publish a
        layer-by-layer tensor definition for the convolution module. This
        implementation uses a reasonable Conformer-style depthwise separable
        convolution with GLU gating, depthwise temporal convolution, BatchNorm,
        SiLU, and a pointwise projection.

    Shape:
        input: [B, T, 192]
        output: [B, T, 192]
    """

    def __init__(self, cfg: SConformerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.layer_norm = _build_norm(cfg.d_model, cfg.causal)
        self.pointwise_in = nn.Conv1d(cfg.d_model, 2 * cfg.d_model, kernel_size=1)
        if cfg.causal:
            self.depthwise_pad = nn.ConstantPad1d((cfg.conv_kernel_size - 1, 0), 0.0)
            depthwise_padding = 0
        else:
            self.depthwise_pad = nn.Identity()
            depthwise_padding = cfg.conv_kernel_size // 2
        self.depthwise = nn.Conv1d(
            cfg.d_model,
            cfg.d_model,
            kernel_size=cfg.conv_kernel_size,
            padding=depthwise_padding,
            groups=cfg.d_model,
        )
        self.batch_norm = nn.BatchNorm1d(cfg.d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution module to a sequence tensor."""
        assert_shape(x, [None, None, self.d_model], "sconformer_conv_input")

        x = self.layer_norm(x).transpose(1, 2)
        x = self.pointwise_in(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_pad(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2).contiguous()


class CausalSelfAttention(nn.Module):
    """Self-attention module with optional causal masking and lookahead."""

    def __init__(self, cfg: SConformerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.dropout = cfg.dropout
        self.is_causal = cfg.causal
        self.lookahead = cfg.lookahead
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_shape(x, [None, None, self.d_model], "causal_self_attention_input")

        batch_size, frames, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        head_dim = self.d_model // self.num_heads

        query = query.view(batch_size, frames, self.num_heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, frames, self.num_heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, frames, self.num_heads, head_dim).transpose(1, 2).contiguous()

        if self.is_causal:
            attn_mask = torch.ones(frames, frames, dtype=torch.bool, device=x.device).tril(diagonal=self.lookahead)
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        attended = attended.transpose(1, 2).reshape(batch_size, frames, self.d_model)
        return self.resid_dropout(self.proj(attended))


class _FeedForwardModule(nn.Module):
    """
    Feed-forward submodule.

    Shape:
        input: [B, T, 192]
        output: [B, T, 192]
    """

    def __init__(self, cfg: SConformerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.net = nn.Sequential(
            _build_norm(cfg.d_model, cfg.causal),
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to a sequence tensor."""
        assert_shape(x, [None, None, self.d_model], "sconformer_ffn_input")
        return self.net(x)


class SConformerBlock(nn.Module):
    """
    sConformer block used inside the DenGCAN bottleneck.

    reference-aligned order:
        Conv module -> MHSA -> FFN -> LayerNorm

    Shape:
        input: [B, T, 192]
        output: [B, T, 192]
    """

    def __init__(self, cfg: SConformerConfig | None = None):
        super().__init__()
        self.cfg = cfg or SConformerConfig()
        if self.cfg.d_model % self.cfg.num_heads != 0:
            raise ValueError(
                f"d_model={self.cfg.d_model} must be divisible by num_heads={self.cfg.num_heads}."
            )

        self.conv_module = _ConformerStyleConvModule(self.cfg)
        self.self_attn_norm = _build_norm(self.cfg.d_model, self.cfg.causal)
        self.self_attention = CausalSelfAttention(self.cfg)
        self.self_attn_dropout = nn.Dropout(self.cfg.dropout)
        self.feed_forward = _FeedForwardModule(self.cfg)
        self.final_norm = _build_norm(self.cfg.d_model, self.cfg.causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one sConformer block."""
        assert_shape(x, [None, None, self.cfg.d_model], "sconformer_block_input")

        residual = x
        x = residual + self.conv_module(x)

        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attention(x)
        x = residual + self.self_attn_dropout(x)

        residual = x
        x = residual + self.feed_forward(x)
        return self.final_norm(x)


class SConformerLayer(nn.Module):
    """Stack one or more sConformer blocks while preserving `[B, T, D]` shape."""

    def __init__(self, cfg: SConformerConfig | None = None, num_blocks: int = 2):
        super().__init__()
        self.cfg = cfg or SConformerConfig()
        self.blocks = nn.ModuleList([SConformerBlock(self.cfg) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_shape(x, [None, None, self.cfg.d_model], "sconformer_layer_input")
        for block in self.blocks:
            x = block(x)
        return x


class GsConformer(nn.Module):
    """Grouped sConformer aligned with src_jasa DenGCAN bottleneck behavior."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int = 2,
        groups: int = 2,
        rearrange: bool = True,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
        lookahead: int = 0,
    ):
        super().__init__()
        if input_dim % groups != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by groups={groups}.")

        self.input_dim = input_dim
        self.groups = groups
        self.num_layers = num_layers
        self.rearrange = rearrange

        input_dim_group = input_dim // groups
        group_cfg = SConformerConfig(
            d_model=input_dim_group,
            num_heads=_infer_group_num_heads(input_dim_group, num_heads),
            ffn_dim=max(input_dim_group * 2, ffn_dim // groups),
            conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            causal=causal,
            lookahead=lookahead,
        )

        self.sconformer_list = nn.ModuleList(
            [
                nn.ModuleList([SConformerBlock(group_cfg) for _ in range(groups)])
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped sConformer on `[B, C, T, D]` and preserve shape."""
        assert_shape(x, [None, None, None, None], "grouped_sconformer_input")

        batch, channels, frames, freq_bins = x.shape
        if channels * freq_bins != self.input_dim:
            raise ValueError(
                f"Expected channels*freq_bins={self.input_dim}, got {channels * freq_bins}."
            )

        out = x.transpose(1, 2).contiguous().view(batch, frames, -1)
        for layer_idx, sconformers in enumerate(self.sconformer_list):
            chunks = torch.chunk(out, self.groups, dim=-1)
            out = torch.cat(
                [sconformer(chunks[i]) for i, sconformer in enumerate(sconformers)],
                dim=-1,
            )

            if self.rearrange and layer_idx < (self.num_layers - 1):
                out = (
                    out.reshape(batch, frames, self.groups, -1)
                    .transpose(-1, -2)
                    .contiguous()
                    .view(batch, frames, -1)
                )

        out = out.view(batch, frames, channels, freq_bins).transpose(1, 2).contiguous()
        return out


def _infer_group_num_heads(group_dim: int, requested_heads: int) -> int:
    """
    Infer a valid number of attention heads for grouped processing.

    engineering assumption:
        The paper does not define per-group head allocation. We choose the
        largest divisor of `group_dim` that does not exceed the full-model head
        count so grouped attention stays close to the requested head budget.
    """
    for num_heads in range(min(group_dim, requested_heads), 0, -1):
        if group_dim % num_heads == 0:
            return num_heads
    raise ValueError(f"Could not infer valid attention heads for group_dim={group_dim}.")
