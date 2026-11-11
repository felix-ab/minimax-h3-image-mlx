"""MiniMax-H3-Image VAE in MLX: causal 3D CNN encoder (single frame) and 2D ViT decoder, float32 like the release.

Ported from diffusers' `AutoencoderKLMiniMaxH3Image` (Apache-2.0). Tiling is part of the released numerics (the
official images are blended-tile decodes), so `decode` always tiles exactly as the reference does: 256-px tiles,
>= 64-px overlaps widened in 16-px steps until the tiles cover the image, linear blends against the *unblended*
neighbour tiles, then the overlaps cropped off.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

F32 = mx.float32
RATIO = 16
TILE = 256
MIN_OVERLAP = 64


# ---------------------------------------------------------------------------------------------- helpers

def _linear(x, p, name):
    y = x @ p[f"{name}.weight"].T
    bias = p.get(f"{name}.bias")
    return y + bias if bias is not None else y


def _rms_norm_f32(x, weight, eps):
    """torch nn.RMSNorm on a float32 input: x * rsqrt(mean(x^2) + eps) [* weight]."""
    y = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return y * weight if weight is not None else y


def _layer_norm_f32(x, weight, bias, eps):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * weight + bias


def split_tiles(length: int, tile: int = TILE, min_overlap: int = MIN_OVERLAP):
    """AutoencoderKLMiniMaxH3Image._split_tiles."""
    if tile >= length:
        return [0], [length], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) - length < 0:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    remaining = tile * n - sum(overlaps) - length
    for i in range(remaining // RATIO):
        overlaps[i % (n - 1)] += RATIO
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * n, overlaps


def _blend(a, b, extent, axis):
    """_blend on NHWC-or-NCHW arrays along `axis`; weights in float32 like the reference."""
    extent = min(a.shape[axis], b.shape[axis], extent)
    pos = mx.arange(extent, dtype=F32)
    shape = [1] * a.ndim
    shape[axis] = extent
    wa = (1 - pos / extent).reshape(shape)
    wb = (pos / extent).reshape(shape)
    sa = [slice(None)] * a.ndim
    sa[axis] = slice(a.shape[axis] - extent, None)
    sb = [slice(None)] * b.ndim
    sb[axis] = slice(0, extent)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[axis]:
        return blended
    sr = [slice(None)] * b.ndim
    sr[axis] = slice(extent, None)
    return mx.concatenate([blended, b[tuple(sr)]], axis=axis)


def stitch(tiles, h_overlaps, w_overlaps, h_axis, w_axis):
    rows = []
    for i, row in enumerate(tiles):
        out_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend(tiles[i - 1][j], tile, h_overlaps[i - 1], h_axis)
            if j > 0:
                tile = _blend(row[j - 1], tile, w_overlaps[j - 1], w_axis)
            if i < len(tiles) - 1:
                s = [slice(None)] * tile.ndim
                s[h_axis] = slice(0, tile.shape[h_axis] - h_overlaps[i])
                tile = tile[tuple(s)]
            if j < len(row) - 1:
                s = [slice(None)] * tile.ndim
                s[w_axis] = slice(0, tile.shape[w_axis] - w_overlaps[j])
                tile = tile[tuple(s)]
            out_row.append(tile)
        rows.append(mx.concatenate(out_row, axis=w_axis))
    return mx.concatenate(rows, axis=h_axis)


# ---------------------------------------------------------------------------------------------- decoder

class ViTDecoder:
    """MiniMaxH3ImageViTDecoder2d. Batched over tiles: (B, h, w, 24) latents -> (B, 3, 16h, 16w) pixels."""

    def __init__(self, params: dict, num_layers=24, heads=16, head_dim=64, registers=4, eps=1e-5,
                 rope_theta=100.0, rope_dim_ratio=0.5):
        self.p = params
        self.L, self.heads, self.hd, self.registers, self.eps = num_layers, heads, head_dim, registers, eps
        rope_dim = int(head_dim * rope_dim_ratio)  # 32 rotated channels per head
        # inv_freq = 1 / theta ** arange(0, 1, 2 * num_axes / dim)   (float32, num_axes = 2)
        steps = np.arange(0, 1, 2 * 2 / rope_dim, dtype=np.float32)
        self.inv_freq = mx.array((np.float32(1.0) / np.power(np.float32(rope_theta), steps)).astype(np.float32))

    def rope(self, h, w):
        def grid(size):
            return 2.0 * (mx.arange(0.5, size, dtype=F32) / size) - 1.0
        gh, gw = grid(h), grid(w)
        pos = mx.stack([mx.repeat(gh, w), mx.tile(gw, h)], axis=-1)  # meshgrid ij, flattened
        pos = mx.concatenate([pos, mx.zeros((self.registers + 1, 2), dtype=F32)], axis=0)
        angles = (2.0 * math.pi * pos[:, :, None]) * self.inv_freq[None, None, :]  # (N, 2, 8)
        angles = angles.reshape(pos.shape[0], -1)
        angles = mx.concatenate([angles, angles], axis=-1)  # tile(2) -> (N, 32)
        return mx.cos(angles)[None, :, None, :], mx.sin(angles)[None, :, None, :]

    def attention(self, x, i, cos, sin):
        p, pre = self.p, f"decoder.transformer_blocks.{i}.attn"
        B, N, _ = x.shape
        q = _linear(x, p, f"{pre}.to_q").reshape(B, N, self.heads, self.hd)
        k = _linear(x, p, f"{pre}.to_k").reshape(B, N, self.heads, self.hd)
        v = _linear(x, p, f"{pre}.to_v").reshape(B, N, self.heads, self.hd)
        q = _rms_norm_f32(q, None, self.eps)
        k = _rms_norm_f32(k, None, self.eps)
        r = cos.shape[-1]

        def rot(t):
            tr, tp = t[..., :r], t[..., r:]
            a, b = tr[..., : r // 2], tr[..., r // 2:]
            rotated = mx.concatenate([-b, a], axis=-1)
            return mx.concatenate([tr * cos + rotated * sin, tp], axis=-1)

        q, k = rot(q), rot(k)
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), scale=self.hd ** -0.5)
        o = o.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return _linear(o, p, f"{pre}.to_out.0")

    def block(self, x, i, cos, sin):
        p, pre = self.p, f"decoder.transformer_blocks.{i}"
        h = _rms_norm_f32(x, p[f"{pre}.norm1.weight"], self.eps)
        x = x + self.attention(h, i, cos, sin) * p[f"{pre}.scale1"]
        h = _rms_norm_f32(x, p[f"{pre}.norm2.weight"], self.eps)
        h = _linear(h, p, f"{pre}.ff.net.0.proj")
        value, gate = mx.split(h, 2, axis=-1)
        h = _linear(value * (gate * mx.sigmoid(gate)), p, f"{pre}.ff.net.2")
        return x + h * p[f"{pre}.scale2"]

    def __call__(self, z):  # z: (B, h, w, 24) float32, after post_quant_conv
        p = self.p
        B, h, w, c = z.shape
        x = _linear(z.reshape(B, h * w, c), p, "decoder.proj_in")
        n = x.shape[1]
        reg = mx.broadcast_to(p["decoder.register_tokens"], (B, self.registers, x.shape[-1]))
        x = mx.concatenate([x, reg, mx.zeros((B, 1, x.shape[-1]), dtype=F32)], axis=1)
        cos, sin = self.rope(h, w)
        for i in range(self.L):
            x = self.block(x, i, cos, sin)
            mx.eval(x)
        x = _layer_norm_f32(x, p["decoder.norm_out.weight"], p["decoder.norm_out.bias"], self.eps)
        x = _linear(x, p, "decoder.proj_out")[:, :n]
        x = x.reshape(B, h, w, 3, RATIO, RATIO).transpose(0, 3, 1, 4, 2, 5)
        return x.reshape(B, 3, h * RATIO, w * RATIO)


# ---------------------------------------------------------------------------------------------- encoder

def _reflect_pad(x, top, bottom, left, right):
    """F.pad(mode='reflect') on NHWC for pads of 0 or 1."""
    if top:
        x = mx.concatenate([x[:, 1:2], x], axis=1)
    if bottom:
        x = mx.concatenate([x, x[:, -2:-1]], axis=1)
    if left:
        x = mx.concatenate([x[:, :, 1:2], x], axis=2)
    if right:
        x = mx.concatenate([x, x[:, :, -2:-1]], axis=2)
    return x


class CNNEncoder:
    """MiniMaxH3ImageEncoder3d on a single frame. A causal Conv3d over [0, 0, x] (kernel_t = 3, two zero frames
    prepended) reduces exactly to a Conv2d with the kernel's last temporal slice; temporal strides keep T = 1."""

    def __init__(self, p, block_out=(128, 256, 256, 512, 512, 1024), layers=2, spatial=(2, 2, 2, 2, 1, 1),
                 temporal=(1, 2, 2, 1, 1, 1), groups=32, eps=1e-6):
        self.p, self.block_out, self.layers, self.groups, self.eps = p, block_out, layers, groups, eps
        self.spatial, self.temporal = spatial, temporal
        self._w = {}

    def _kernel(self, name):
        if name not in self._w:
            w = self.p[f"{name}.weight"]  # (O, I, kT, kH, kW)
            self._w[name] = w[:, :, -1].transpose(0, 2, 3, 1)  # (O, kH, kW, I)
        return self._w[name]

    def conv(self, x, name, pad=1, stride=1):
        if pad:
            x = _reflect_pad(x, pad, pad, pad, pad)
        return mx.conv2d(x, self._kernel(name), stride=stride) + self.p[f"{name}.bias"]

    def group_norm(self, x, name):
        n, h, w, c = x.shape
        g = x.reshape(n, h, w, self.groups, c // self.groups)
        mean = mx.mean(g, axis=(1, 2, 4), keepdims=True)
        var = mx.mean(mx.square(g - mean), axis=(1, 2, 4), keepdims=True)
        g = (g - mean) * mx.rsqrt(var + self.eps)
        return g.reshape(n, h, w, c) * self.p[f"{name}.weight"] + self.p[f"{name}.bias"]

    def resnet(self, x, name):
        h = self.group_norm(x, f"{name}.norm1")
        h = self.conv(h * mx.sigmoid(h), f"{name}.conv1")
        h = self.group_norm(h, f"{name}.norm2")
        h = self.conv(h * mx.sigmoid(h), f"{name}.conv2")
        if f"{name}.conv_shortcut.weight" in self.p:
            x = self.conv(x, f"{name}.conv_shortcut", pad=0)
        return x + h

    def __call__(self, x):  # x NHWC float32, ImageNet-normalized
        x = self.conv(x, "encoder.conv_in")
        for i in range(len(self.block_out)):
            for j in range(self.layers):
                x = self.resnet(x, f"encoder.down_blocks.{i}.resnets.{j}")
            if self.spatial[i] * self.temporal[i] > 1:
                if self.spatial[i] == 2:
                    x = _reflect_pad(x, 0, 1, 0, 1)
                x = self.conv(x, f"encoder.down_blocks.{i}.downsamplers.0.conv", pad=0, stride=self.spatial[i])
            mx.eval(x)
        h = self.group_norm(x, "encoder.norm_out")
        x = self.conv(h * mx.sigmoid(h), "encoder.conv_out")
        w = self.p["quant_conv.weight"].reshape(48, 48)
        return x @ w.T + self.p["quant_conv.bias"]  # moments (N, h, w, 48): mean | logvar


class VAE:
    def __init__(self, params: dict, latents_mean, latents_std):
        self.p = {k: v.astype(F32) for k, v in params.items()}
        self.decoder = ViTDecoder(self.p)
        self.encoder = CNNEncoder(self.p)
        self.mean = mx.array(np.asarray(latents_mean, np.float32))
        self.std = mx.array(np.asarray(latents_std, np.float32))

    @classmethod
    def load(cls, path, config):
        return cls(mx.load(str(path)), config["latents_mean"], config["latents_std"])

    def post_quant_conv(self, z):  # 1x1 conv, z (..., 24) channels-last
        w = self.p["post_quant_conv.weight"].reshape(24, 24)
        return z @ w.T + self.p["post_quant_conv.bias"]

    def decode(self, latents_nchw, batch_tiles: int = 32):
        """Normalized latents (1, 24, h, w) float32 -> ImageNet-normalized pixels (1, 3, 16h, 16w) float32.

        Applies `latents * std + mean` first (MiniMaxH3ImageDecodeStep) then the tiled decode."""
        z = latents_nchw.astype(F32) * self.std.reshape(1, -1, 1, 1) + self.mean.reshape(1, -1, 1, 1)
        z = z.transpose(0, 2, 3, 1)  # NHWC
        H, W = z.shape[1] * RATIO, z.shape[2] * RATIO
        ys, ylen, yov = split_tiles(H)
        xs, xlen, xov = split_tiles(W)
        crops = [(i, j, y // RATIO, x // RATIO, hl // RATIO, wl // RATIO)
                 for i, (y, hl) in enumerate(zip(ys, ylen)) for j, (x, wl) in enumerate(zip(xs, xlen))]
        tiles = [[None] * len(xs) for _ in ys]
        # Tiles of equal shape are decoded in batches; each tile's math is independent of its batch-mates.
        by_shape = {}
        for c in crops:
            by_shape.setdefault((c[4], c[5]), []).append(c)
        for (th, tw), group in by_shape.items():
            for s in range(0, len(group), batch_tiles):
                chunk = group[s:s + batch_tiles]
                zt = mx.stack([z[0, cy:cy + th, cx:cx + tw] for (_, _, cy, cx, _, _) in chunk])
                out = self.decoder(self.post_quant_conv(zt))
                mx.eval(out)
                for k, (i, j, *_rest) in enumerate(chunk):
                    tiles[i][j] = out[k:k + 1]
        return stitch(tiles, yov, xov, h_axis=2, w_axis=3)

    def encode_moments(self, pixels_nchw):
        """ImageNet-normalized pixels (1, 3, H, W) -> posterior moments (1, 48, H/16, W/16), tiled like the release."""
        x = pixels_nchw.astype(F32).transpose(0, 2, 3, 1)
        H, W = x.shape[1], x.shape[2]
        ys, ylen, yov = split_tiles(H)
        xs, xlen, xov = split_tiles(W)
        rows = []
        for y, hl in zip(ys, ylen):
            row = []
            for xx, wl in zip(xs, xlen):
                m = self.encoder(x[:, y:y + hl, xx:xx + wl])
                mx.eval(m)
                row.append(m.transpose(0, 3, 1, 2))
            rows.append(row)
        return stitch(rows, [o // RATIO for o in yov], [o // RATIO for o in xov], h_axis=2, w_axis=3)


PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def to_uint8(pixels_nchw) -> np.ndarray:
    """MiniMaxH3ImageDecodeStep: un-normalize, clamp to [0, 1], quantize to 8 bits by truncation. -> (H, W, 3)."""
    mean = mx.array(np.asarray(PIXEL_MEAN, np.float32)).reshape(1, 3, 1, 1)
    std = mx.array(np.asarray(PIXEL_STD, np.float32)).reshape(1, 3, 1, 1)
    image = mx.clip(pixels_nchw.astype(F32) * std + mean, 0, 1)
    image = mx.clip(image * 255, 0, 255)
    return np.asarray(image[0].transpose(1, 2, 0)).astype(np.uint8)  # float -> uint8 truncates like .to(uint8)
