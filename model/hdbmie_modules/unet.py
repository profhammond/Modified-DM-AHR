# =========================
# STABLE DDPM / SR3 / SRDE UNET
# =========================

import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


# -------------------------
# helpers
# -------------------------

def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if isfunction(d) else d)


# -------------------------
# positional encoding
# -------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        half = self.dim // 2
        device = noise_level.device

        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)

        emb = noise_level[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb


# -------------------------
# FiLM conditioning
# -------------------------

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level

        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + use_affine_level)),
            nn.SiLU()
        )

    def forward(self, x, noise_embed):
        b = x.shape[0]
        h = self.noise_func(noise_embed).view(b, -1, 1, 1)

        if self.use_affine_level:
            gamma, beta = h.chunk(2, dim=1)
            return (1 + gamma) * x + beta
        else:
            return x + h


# -------------------------
# basic layers
# -------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()

        num_groups = min(groups, dim)
        while dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# ResNet block
# -------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None,
                 norm_groups=32, dropout=0):
        super().__init__()

        self.mlp = FeatureWiseAffine(
            noise_level_emb_dim,
            dim_out
        ) if noise_level_emb_dim else None

        self.block1 = Block(dim, dim_out, norm_groups, dropout)
        self.block2 = Block(dim_out, dim_out, norm_groups, dropout)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            h = self.mlp(h, time_emb)

        h = self.block2(h)

        return h + self.res_conv(x)


# -------------------------
# Attention
# -------------------------

class SelfAttention(nn.Module):
    def __init__(self, channels, heads=1, norm_groups=32):
        super().__init__()
        self.heads = heads

        self.norm = nn.GroupNorm(min(norm_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)

        q = q.view(b, self.heads, -1, h * w)
        k = k.view(b, self.heads, -1, h * w)
        v = v.view(b, self.heads, -1, h * w)

        attn = torch.einsum("bhci,bhcj->bhij", q, k)
        attn = attn / math.sqrt(c)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(b, c, h, w)

        return self.out(out) + x


# -------------------------
# ResBlock wrapper
# -------------------------

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None,
                 norm_groups=32, dropout=0, with_attn=False):
        super().__init__()

        self.res = ResnetBlock(
            dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups,
            dropout
        )

        self.with_attn = with_attn
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, t):
        x = self.res(x, t)

        if self.with_attn:
            x = self.attn(x)

        return x


# -------------------------
# UNet
# -------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=(),
        res_blocks=2,
        dropout=0,
        image_size=128,
        with_noise_level_emb=True,
        norm_groups=32
    ):
        super().__init__()

        self.attn_res = attn_res
        self.image_size = image_size

        # ---------------- time embedding ----------------
        self.noise_level_mlp = None
        if with_noise_level_emb:
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )

        # ---------------- encoder ----------------
        self.downs = nn.ModuleList()

        ch = inner_channel
        now_res = image_size

        self.downs.append(nn.Conv2d(in_channel, ch, 3, padding=1))

        for i, mult in enumerate(channel_mults):
            out_ch = inner_channel * mult

            for _ in range(res_blocks):
                self.downs.append(
                    ResnetBlocWithAttn(
                        ch,
                        out_ch,
                        inner_channel if with_noise_level_emb else None,
                        norm_groups,
                        dropout,
                        with_attn=False
                    )
                )
                ch = out_ch

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(ch))
                now_res //= 2

        # ---------------- middle ----------------
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(ch, ch, inner_channel, norm_groups, dropout),
            ResnetBlocWithAttn(ch, ch, inner_channel, norm_groups, dropout)
        ])

        # ---------------- decoder ----------------
        self.ups = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = inner_channel * mult

            for _ in range(res_blocks):
                self.ups.append(
                    ResnetBlocWithAttn(
                        ch + ch,   # safe fixed concat assumption
                        out_ch,
                        inner_channel if with_noise_level_emb else None,
                        norm_groups,
                        dropout,
                        with_attn=False
                    )
                )
                ch = out_ch

            if i != 0:
                self.ups.append(Upsample(ch))
                now_res *= 2

        self.final_conv = Block(ch, out_channel, norm_groups)

    # ---------------- forward ----------------

    def forward(self, x, time):
        t = None
        if exists(self.noise_level_mlp):
            t = self.noise_level_mlp(time)
            t = t * 1.5

        feats = []

        # encoder
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        # middle
        for layer in self.mid:
            x = layer(x, t)

        # decoder
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip = feats.pop()

                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:])

                # SAFE channel match
                if skip.shape[1] != x.shape[1]:
                    skip = nn.Conv2d(skip.shape[1], x.shape[1], 1).to(x.device)(skip)

                x = layer(torch.cat([x, skip], dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
