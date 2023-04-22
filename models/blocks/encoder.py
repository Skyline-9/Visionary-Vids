# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.
import math

import nncore
import torch
import torch.nn as nn
from nncore.nn import (
    MODELS,
    build_linear_modules,
    build_model,
    build_norm_layer,
    Parameter,
)
from vit_pytorch import SimpleViT


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
        print("this is value of x shape BEFORE REREE", x.shape)

        if self.dropout is not None:
            x = self.dropout(x)
        if self.mapping is not None:
            x = self.mapping(x)

        print("this is value of x shape", x.shape)

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
    Positional Encoding introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        learnable (bool, optional): Whether the positional encoding is
            learnable. Default: ``True``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        max_len (int, optional): The maximum length of the input sequence.
            Default: ``5000``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, dims, learnable=True, p=0.1, max_len=5000):
        super(SinCosPositionalEncoding, self).__init__()

        self._dims = dims
        self._learnable = learnable
        self._p = p
        self._max_len = max_len

        if learnable:
            self.pe = Parameter(1, max_len, dims)
        else:
            pos = torch.arange(max_len).unsqueeze(1)
            div1 = torch.exp(torch.arange(0, dims, 2) * -(math.log(10000.0) / dims))
            div2 = torch.exp(torch.arange(1, dims, 2) * -(math.log(10000.0) / dims))
            pe = torch.zeros(1, max_len, dims)
            pos_enc = torch.zeros(max_len, dims)
            pos_enc[:, 0::2] = torch.sin(pos * div1)
            pos_enc[:, 1::2] = torch.cos(pos * div2)
            pe[0, :, :] = pos_enc
            self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(p=p)

    def __repr__(self):
        return "{}(dims={}, learnable={}, p={}, max_len={})".format(
            self.__class__.__name__, self._dims, self._learnable, self._p, self._max_len
        )

    def forward(self, x):
        pe = self.pe[:, : x.size(1)].repeat(x.size(0), 1, 1)
        pe = self.dropout(pe)
        return pe


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
