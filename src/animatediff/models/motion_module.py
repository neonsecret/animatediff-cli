import math
from dataclasses import dataclass
from typing import Optional
from inspect import isfunction

import torch
import torch.nn.functional as F
import xformers.ops as xops
# from diffusers.models.attention import Attention
# from diffusers.utils import maybe_allow_in_graph
from einops import rearrange, repeat
from torch import Tensor, nn


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class AdvancedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None):
        super().__init__()
        self.proj = AdvancedLinear(dim_in, dim_out * 2, dtype=dtype)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            AdvancedLinear(dim, inner_dim, dtype=dtype),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(project_in)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(AdvancedLinear(inner_dim, dim_out, dtype=dtype))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput:
    sample: torch.FloatTensor


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

        output = hidden_states
        return output


# @maybe_allow_in_graph
class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
        )

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


# @maybe_allow_in_graph
class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: int = 768,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        cross_frame_attention_mode=None,
        temporal_position_encoding: bool = False,
        temporal_position_encoding_max_len: int = 24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"))
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout: float = 0.0, max_len: int = 24):
        super().__init__()
        self.dropout: nn.Module = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe: Tensor = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CrossAttentionPytorch(nn.Module):
    def __init__(self, query_dim, cross_attention_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = default(cross_attention_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = AdvancedLinear(query_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_k = AdvancedLinear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_v = AdvancedLinear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype)

        self.to_out = nn.Sequential(AdvancedLinear(inner_dim, query_dim, dtype=dtype), nn.Dropout(dropout))
        self.attention_op = None

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, self.heads, self.dim_head).transpose(1, 2),
            (q, k, v),
        )

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        )

        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, cross_attention_dim=None, heads=8, dim_head=64, dropout=0.0, dtype=None, bias=False):
        super().__init__()
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {cross_attention_dim} and using "
        #       f"{heads} heads.")
        inner_dim = dim_head * heads
        cross_attention_dim = default(cross_attention_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = AdvancedLinear(query_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_k = AdvancedLinear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_v = AdvancedLinear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype)

        self.to_out = nn.Sequential(AdvancedLinear(inner_dim, query_dim, dtype=dtype), nn.Dropout(dropout))
        self.attention_op = None

    def compute_attention(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


# @maybe_allow_in_graph
class VersatileAttention(MemoryEfficientCrossAttention):
    def __init__(
        self,
        attention_mode: str = None,
        cross_frame_attention_mode: Optional[str] = None,
        temporal_position_encoding: bool = False,
        temporal_position_encoding_max_len: int = 24,
        upcast_attention=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if attention_mode.lower() != "temporal":
            raise ValueError(f"Attention mode {attention_mode} is not supported.")

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(kwargs["query_dim"], dropout=0.0, max_len=temporal_position_encoding_max_len)
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states=None, attention_mask=None, video_length=None
    ):
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )
        else:
            raise NotImplementedError

        # attention processor makes this easy so that's nice
        hidden_states = self.compute_attention(hidden_states,
                                               context=encoder_hidden_states,
                                               value=encoder_hidden_states,
                                               mask=attention_mask)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
