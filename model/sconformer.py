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
        if cfg.conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve sequence length.")

        self.d_model = cfg.d_model
        self.pointwise_in = nn.Conv1d(cfg.d_model, 2 * cfg.d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            cfg.d_model,
            cfg.d_model,
            kernel_size=cfg.conv_kernel_size,
            padding=cfg.conv_kernel_size // 2,
            groups=cfg.d_model,
        )
        self.batch_norm = nn.BatchNorm1d(cfg.d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution module to a sequence tensor."""
        assert_shape(x, [None, None, self.d_model], "sconformer_conv_input")

        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        x = F.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class _SelfAttentionModule(nn.Module):
    """
    Multi-head self-attention submodule.

    Shape:
        input: [B, T, 192]
        output: [B, T, 192]
    """

    def __init__(self, cfg: SConformerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to a sequence tensor."""
        assert_shape(x, [None, None, self.d_model], "sconformer_mhsa_input")
        y, _ = self.attention(x, x, x, need_weights=False)
        return self.dropout(y)


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

    paper-specified high-level order:
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

        # engineering assumption:
        # Pre-norm residual branches are used for stable optimization, while the
        # overall block order remains Conv -> MHSA -> FFN -> LayerNorm.
        self.conv_norm = nn.LayerNorm(self.cfg.d_model)
        self.attn_norm = nn.LayerNorm(self.cfg.d_model)
        self.ffn_norm = nn.LayerNorm(self.cfg.d_model)
        self.conv_module = _ConformerStyleConvModule(self.cfg)
        self.self_attention = _SelfAttentionModule(self.cfg)
        self.feed_forward = _FeedForwardModule(self.cfg)
        self.final_norm = nn.LayerNorm(self.cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one sConformer block."""
        assert_shape(x, [None, None, self.cfg.d_model], "sconformer_block_input")

        x = x + self.conv_module(self.conv_norm(x))
        x = x + self.self_attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return self.final_norm(x)


class SConformerLayer(nn.Module):
    """
    One sConformer layer composed of two sConformer blocks.

    paper-specified:
        A layer contains two sConformer blocks.

    Shape:
        input: [B, T, 192]
        output: [B, T, 192]
    """
    def __init__(self, cfg: SConformerConfig | None = None):
        super().__init__()
        self.cfg = cfg or SConformerConfig()
        self.block1 = SConformerBlock(self.cfg)
        self.block2 = SConformerBlock(self.cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two sequential sConformer blocks."""
        assert_shape(x, [None, None, self.cfg.d_model], "sconformer_layer_input")
        x = self.block1(x)
        x = self.block2(x)
        return x


class GroupedSConformerPlaceholder(nn.Module):
    """
    Placeholder for the paper's grouped sConformer layer.

    engineering assumption:
        The paper states that the bottleneck uses grouped sConformer layers, but
        the exact grouping rule, tensor partition strategy, and parameter sharing
        details are not fully recovered in the current project.

        This implementation adopts a stronger engineering hypothesis that is
        consistent with the current bottleneck reshape:
            - `[B, 64, T, 3] -> [B, T, 192]`
            - the 192 features are interpreted as `64 channels x 3 bottleneck
              frequency bins`
            - grouped processing is therefore assumed to operate over the 3
              bottleneck frequency groups, each with width 64

        Each group is processed by a smaller sConformer block, the grouped
        outputs are fused back to 192 dimensions, and one full-width
        `SConformerLayer` is then used to model cross-group interactions.
        This is still an engineering placeholder, not a claimed paper-original
        grouped implementation.

    Shape:
        input: [B, T, 192]
        output: [B, T, 192]
    """

    def __init__(self, cfg: SConformerConfig | None = None, num_groups: int = 3):
        super().__init__()
        self.cfg = cfg or SConformerConfig()
        if self.cfg.d_model % num_groups != 0:
            raise ValueError(
                f"d_model={self.cfg.d_model} must be divisible by num_groups={num_groups}."
            )

        self.num_groups = num_groups
        self.group_dim = self.cfg.d_model // num_groups
        self.group_cfg = SConformerConfig(
            d_model=self.group_dim,
            num_heads=_infer_group_num_heads(self.group_dim, self.cfg.num_heads),
            ffn_dim=max(self.group_dim * 2, self.cfg.ffn_dim // num_groups),
            conv_kernel_size=self.cfg.conv_kernel_size,
            dropout=self.cfg.dropout,
        )

        self.group_blocks = nn.ModuleList([SConformerBlock(self.group_cfg) for _ in range(num_groups)])
        self.group_fusion = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.SiLU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.cross_group_layer = SConformerLayer(self.cfg)

    def _to_group_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recover grouped view `[B, T, G, D]` from `[B, T, 192]`.

        engineering assumption:
            Because the bottleneck sequence came from `[B, 64, T, 3]`, we
            interpret the flattened 192 features as `[64, 3]` and regroup them
            into 3 bottleneck-frequency groups of width 64.
        """
        batch, frames, _ = x.shape
        return x.reshape(batch, frames, self.group_dim, self.num_groups).permute(0, 1, 3, 2)

    def _from_group_view(self, x: torch.Tensor) -> torch.Tensor:
        """Merge grouped view `[B, T, G, D]` back to `[B, T, 192]`."""
        batch, frames, _, _ = x.shape
        return x.permute(0, 1, 3, 2).reshape(batch, frames, self.cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped placeholder processing followed by one full-width sConformer layer."""
        assert_shape(x, [None, None, self.cfg.d_model], "grouped_sconformer_input")

        residual = x
        grouped = self._to_group_view(x)
        processed_groups = [
            block(grouped[:, :, group_idx, :]) for group_idx, block in enumerate(self.group_blocks)
        ]
        grouped = torch.stack(processed_groups, dim=2)
        grouped_context = self._from_group_view(grouped)
        x = residual + self.group_fusion(grouped_context)
        return self.cross_group_layer(x)


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
