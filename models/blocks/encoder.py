# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.
import math

import nncore
import torch
import torch.nn as nn
import x_transformers.x_transformers
from nncore.nn import (
    MODELS,
    build_linear_modules,
    build_model,
    build_norm_layer,
)
from x_transformers import Encoder


@MODELS.register()
class UniModalEncoder(nn.Module):
    def __init__(
        self, dims=None, p=0.5, pos_cfg=None, enc_cfg=None, norm_cfg=None, **kwargs
    ):
        super(UniModalEncoder, self).__init__()

        drop_cfg = dict(type="drop", p=p) if p > 0 else None
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims

        self.dropout = build_norm_layer(drop_cfg)
        self.mapping = build_linear_modules(dims, **kwargs)
        self.pos_enc = build_model(pos_cfg, enc_dims)
        self.encoder = build_model(enc_cfg, enc_dims, bundler="sequential")
        self.norm = build_norm_layer(norm_cfg, enc_dims)

    def forward(self, x, **kwargs):
        if self.dropout is not None:
            x = self.dropout(x)
        if self.mapping is not None:
            x = self.mapping(x)

        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(x)
            x = self.encoder(x, pe=pe, **kwargs)
        if self.norm is not None:
            x = self.norm(x)
        return x


@MODELS.register()
@nncore.bind_getter("dims", "learnable", "p", "max_len")
class SinCosPositionalEncoding(nn.Module):
    """
    SinCosPositional Encoding as referenced in Better plain ViT baselines for ImageNet-1k

    Args:
        dims (int): The input feature dimensions.
        learnable (bool, optional): Whether the positional encoding is
            learnable. Default: ``True``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        max_len (int, optional): The maximum length of the input sequence.
            Default: ``5000``.
    """

    def __init__(self, dims, p=0.1, max_len=5000, temperature=10000):
        super(SinCosPositionalEncoding, self).__init__()

        self._dims = dims
        self._p = p
        self._max_len = max_len
        self._temperature = temperature

        omega = torch.arange(dims // 4) / (dims // 4 - 1)
        omega = 1.0 / (temperature**omega)

        y, x = torch.meshgrid(torch.arange(max_len), torch.arange(dims // 4))
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]

        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(p=p)

    def __repr__(self):
        return "{}(dims={}, p={}, max_len={}, temperature={})".format(
            self.__class__.__name__,
            self._dims,
            self._p,
            self._max_len,
            self._temperature,
        )

    def forward(self, x):
        pe = self.pe[: x.size(1), :].unsqueeze(0)
        pe = pe.repeat(x.size(0), 1, 1)
        pe = self.dropout(pe)
        return pe


@MODELS.register()
class XTransformerEncoderLayer(Encoder):
    def __init__(self, dim, depth=1, **kwargs):
        super(XTransformerEncoderLayer, self).__init__(dim=dim, depth=depth, **kwargs)

    def forward(
        self,
        x,
        context=None,
        pe=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_context_mask=None,
        mems=None,
        return_hiddens=False,
    ):
        if pe is not None:
            x += pe

        if mask is not None:
            mask = mask.bool()

        return super(XTransformerEncoderLayer, self).forward(
            x,
            context=context,
            mask=mask,
            context_mask=context_mask,
            attn_mask=attn_mask,
            self_attn_context_mask=self_attn_context_mask,
            mems=mems,
            return_hiddens=return_hiddens,
        )


@MODELS.register()
class CrossModalEncoder(nn.Module):
    def __init__(
        self,
        dims=None,
        fusion_type="sum",
        pos_cfg=None,
        enc_cfg=None,
        norm_cfg=None,
        **kwargs
    ):
        super(CrossModalEncoder, self).__init__()
        assert fusion_type in ("sum", "mean", "concat")

        map_dims = [2 * dims, dims] if fusion_type == "concat" else None
        self.fusion_type = fusion_type

        self.pos_enc = build_model(pos_cfg, dims)
        self.encoder = build_model(enc_cfg, dims)
        self.mapping = build_linear_modules(map_dims, **kwargs)
        self.norm = build_norm_layer(norm_cfg, dims)

    def forward(self, a, b, **kwargs):
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(a)
            a, b = self.encoder(a, b, pe=pe, **kwargs)
        if self.fusion_type in ("sum", "mean"):
            x = (a + b) / ((self.fusion_type == "mean") + 1)
        else:
            x = torch.cat((a, b), dim=-1)
            x = self.mapping(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
