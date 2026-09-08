"""Minimal CATANet components needed by PWSR.

Extracted from the official CATANet_ implementation (Liu et al., CVPR 2025;
https://github.com/EquationWalker/CATANet, Apache-2.0). Only the classes
required by the PWSR spatial branch are kept:

* ``TAB``   — Token Aggregation Block (global attention)
* ``LRSA``  — Local Receptive-field Self-Attention

.. _CATANet: https://github.com/EquationWalker/CATANet

The full Apache-2.0 license text is included in ``licenses/Apache-2.0.txt``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
import math

from .arch_util import trunc_normal_


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


# ---------------------------------------------------------------------------
# Adaptive Dropout
# ---------------------------------------------------------------------------

class AdaptiveDropout(nn.Module):
    """Adaptive Dropout (Implicit Version).

    Paper: *Adaptive Dropout: Unleashing Dropout across Layers*
    """

    def __init__(self, channels, reduction=16, p=0.5, vector_weight=True):
        super().__init__()
        self.dropout = nn.Dropout2d(p=p)
        hidden = max(channels // reduction, 4)
        self.vector_weight = vector_weight
        out_channel = channels if vector_weight else 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, out_channel)

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            w = x.mean(dim=1)
            w = self.fc1(w)
            w = F.relu(w)
            w = self.fc2(w)
            w = torch.sigmoid(w)
            if self.vector_weight:
                w = w.unsqueeze(1)
            else:
                w = w.view(B, 1, 1)
            if self.training:
                mask = torch.rand(B, 1, C, device=x.device)
                mask = (mask > self.dropout.p).float() / (1 - self.dropout.p)
                x_drop = x * mask
            else:
                x_drop = x
            out = w * x + (1 - w) * x_drop
        else:
            B, C, H, W = x.shape
            w = self.avgpool(x).view(B, C)
            w = self.fc1(w)
            w = F.relu(w)
            w = self.fc2(w)
            w = torch.sigmoid(w)
            if self.vector_weight:
                w = w.view(B, C, 1, 1)
            else:
                w = w.view(B, 1, 1, 1)
            x_drop = self.dropout(x)
            out = w * x + (1 - w) * x_drop
        return out


# ---------------------------------------------------------------------------
# Intra-Group Self-Attention (IASA)
# ---------------------------------------------------------------------------

class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size, dropout=0.5):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size
        self.adp_dropout = (
            AdaptiveDropout(dim, p=dropout, vector_weight=False)
            if dropout > 0 else nn.Identity()
        )

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
        out = self.adp_dropout(out)
        return out


# ---------------------------------------------------------------------------
# Inter-Group Cross-Attention (IRCA)
# ---------------------------------------------------------------------------

class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x, dim=-1),
                                   F.normalize(x_means, dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)
        return k, v, x_global.detach()


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features,
                      kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features),
            nn.GELU(),
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features,
                                   x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features,
                             kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                          h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


# ---------------------------------------------------------------------------
# Patch divide / reverse (for LRSA)
# ---------------------------------------------------------------------------

def patch_divide(x, step, ps):
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


# ============================================================================
# Main blocks exported to PWSR
# ============================================================================

class TAB(nn.Module):
    """Token Aggregation Block.

    Global attention via learned token centres with iterative k-means-style
    assignment and EMA updates (train-time only).
    """

    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128, ema_decay=0.999, dropout=0.5):
        super().__init__()
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.irca_attn = IRCA(dim, qk_dim, heads)
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size, dropout)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat(
                (x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(
                rearrange(paded_x, 'b (cnt n) c->cnt (b n) c',
                          cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1),
                                          F.normalize(x_means, dim=-1))

        k_global, v_global, x_means = self.irca_attn(x, x_means)

        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j',
                                    F.normalize(x, dim=-1),
                                    F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1,
                                    index=idx).unsqueeze(-1)

        y = self.iasa_attn(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x

        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)

        return rearrange(x, 'b (h w) c->b c h w', h=h)


class LRSA(nn.Module):
    """Local Receptive-field Self-Attention.

    Window-based attention that crops the feature map into overlapping
    patches, applies self-attention within each patch, and stitches back.
    """

    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()
        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim)),
        ])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)

        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        return x
