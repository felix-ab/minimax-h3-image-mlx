"""Qwen3-VL vision tower (27 blocks, hidden size 1152) and the edit presentation of the MiniMax-H3 conditioner.

Ported from transformers' Qwen3VLVisionModel / Qwen3VLModel (Apache-2.0), with the same rounding points:
  * patch_embed is a Conv3d whose stride equals its kernel, i.e. one linear map per (3, 2, 16, 16) patch;
  * the learned 48x48 position table is bilinearly resampled (align_corners=True) per image, in float32;
  * 2-D axial RoPE with theta 1e4, applied in float32 and rounded back;
  * LayerNorms (eps 1e-6) compute in float32 and round once (torch's fused kernel);
  * DeepStack: features after blocks 8, 16 and 24 go through their own mergers and are added to the language
    model's hidden states at the image-token rows after decoder layers 0, 1 and 2.
Pixel preprocessing (Qwen2VLImageProcessorFast): rescale to [0, 1], normalize with mean = std = 0.5, repeat the
frame to 2 (temporal patch), and flatten patches in 2x2 spatial-merge-block order.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .ops import linear

BF16, F32 = mx.bfloat16, mx.float32
PATCH, TEMPORAL, MERGE = 16, 2, 2
HIDDEN, HEADS, DEPTH, OUT = 1152, 16, 27, 5120
SIDE = 48  # sqrt(num_position_embeddings)
DEEPSTACK = (8, 16, 24)
EPS = 1e-6
IMAGE_PAD, VISION_START, VISION_END = 151655, 151652, 151653


def smart_resize(height, width, factor=PATCH * MERGE, min_pixels=65536, max_pixels=16777216):
    """Qwen2-VL smart_resize with MiniMax-H3's processor bounds (shortest_edge/longest_edge are pixel counts)."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(image_uint8: np.ndarray):
    """(H, W, 3) uint8 at a size smart_resize leaves unchanged -> (patches (N, 1536) float32, grid (1, h, w))."""
    H, W, _ = image_uint8.shape
    if smart_resize(H, W) != (H, W):
        raise ValueError(f"vision image {W}x{H} must already be a multiple of 32 within bounds")
    x = (image_uint8.astype(np.float32) * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    x = x.transpose(2, 0, 1)                                  # (3, H, W)
    x = np.stack([x, x], axis=0)                              # (T=2, 3, H, W)
    gh, gw = H // PATCH, W // PATCH
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)               # (t, bh, bw, mh, mw, C, T, ph, pw)
    return x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH), (1, gh, gw)


def _layer_norm(x, w, b, eps=EPS):
    xf = x.astype(F32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean(mx.square(xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * mx.rsqrt(var + eps) * w.astype(F32) + b.astype(F32)).astype(x.dtype)


def _gelu_tanh(x):
    xf = x.astype(F32)
    return (0.5 * xf * (1 + mx.tanh(math.sqrt(2 / math.pi) * (xf + 0.044715 * xf ** 3)))).astype(x.dtype)


def _gelu_erf(x):
    xf = x.astype(F32)
    return (0.5 * xf * (1 + mx.erf(xf / math.sqrt(2)))).astype(x.dtype)


def _block_order(h, w):
    """(row, col) of each patch in spatial-merge-block order."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // MERGE, MERGE, w // MERGE, MERGE)
    return rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1), cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)


def _interp_taps(index, size):
    """Bilinear, align_corners=True taps into the 48-long table (transformers' _interpolation_axis_taps_weights)."""
    index = index.astype(np.float32)
    src = (index * np.float32(SIDE - 1) / np.float32(max(size - 1, 1))).astype(np.float32)
    floor = np.floor(src)
    offsets = np.arange(2)
    taps = np.clip(floor.astype(np.int64)[:, None] + offsets, 0, SIDE - 1)
    dist = np.abs(src[:, None] - floor[:, None] - offsets).astype(np.float32)
    return taps, np.clip(1 - dist, 0, None).astype(np.float32)


class VisionTower:
    def __init__(self, params: dict, dtype=BF16):
        self.p = {k[len("model.visual."):]: (v.astype(dtype) if v.dtype in (BF16, F32) else v)
                  for k, v in params.items() if k.startswith("model.visual.")}
        self.dtype = dtype
        w = self.p["patch_embed.proj.weight"]  # (1152, 3, 2, 16, 16)
        self.patch_w = w.reshape(w.shape[0], -1)
        spatial = (HIDDEN // HEADS) // 2  # 36
        self.inv_freq = (np.float32(1.0) / np.power(np.float32(10000.0),
                         np.arange(0, spatial, 2, dtype=np.float32) / np.float32(spatial))).astype(np.float32)

    def _pos_embed(self, h, w):
        rows, cols = _block_order(h, w)
        ht, hw_ = _interp_taps(rows, h)
        wt, ww = _interp_taps(cols, w)
        idx = (ht[:, :, None] * SIDE + wt[:, None, :]).reshape(-1, 4)
        wts = (hw_[:, :, None] * ww[:, None, :]).reshape(-1, 4).astype(np.float32)
        table = self.p["pos_embed.weight"]
        gathered = mx.take(table, mx.array(idx.reshape(-1).astype(np.int32)), axis=0).reshape(-1, 4, HIDDEN)
        return (gathered.astype(F32) * mx.array(wts)[:, :, None]).sum(axis=1)  # bf16 * f32 -> f32, summed in f32

    def _rope(self, h, w):
        rows, cols = _block_order(h, w)
        pos = np.stack([rows, cols], axis=-1).astype(np.float32)
        freqs = (pos[:, :, None] * self.inv_freq[None, None, :]).astype(np.float32)   # (N, 2, 18)
        f = np.concatenate([freqs[:, 0], freqs[:, 1]], axis=-1)
        f = np.concatenate([f, f], axis=-1).astype(np.float64)                         # (N, 72)
        return mx.array(np.cos(f).astype(np.float32)), mx.array(np.sin(f).astype(np.float32))

    def _block(self, x, i, cos, sin):
        p, pre = self.p, f"blocks.{i}"
        n = x.shape[0]
        h = _layer_norm(x, p[f"{pre}.norm1.weight"], p[f"{pre}.norm1.bias"])
        qkv = linear(h, p[f"{pre}.attn.qkv.weight"], p[f"{pre}.attn.qkv.bias"]).reshape(n, 3, HEADS, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        def rope(t):
            tf = t.astype(F32)
            half = tf.shape[-1] // 2
            rotated = mx.concatenate([-tf[..., half:], tf[..., :half]], axis=-1)
            return (tf * cos[:, None] + rotated * sin[:, None]).astype(t.dtype)

        q, k = rope(q), rope(k)
        o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                                                 v.transpose(1, 0, 2)[None], scale=(HIDDEN // HEADS) ** -0.5)
        o = o[0].transpose(1, 0, 2).reshape(n, HIDDEN)
        x = x + linear(o, p[f"{pre}.attn.proj.weight"], p[f"{pre}.attn.proj.bias"])
        h = _layer_norm(x, p[f"{pre}.norm2.weight"], p[f"{pre}.norm2.bias"])
        h = linear(_gelu_tanh(linear(h, p[f"{pre}.mlp.linear_fc1.weight"], p[f"{pre}.mlp.linear_fc1.bias"])),
                   p[f"{pre}.mlp.linear_fc2.weight"], p[f"{pre}.mlp.linear_fc2.bias"])
        return x + h

    def _merger(self, x, pre, postshuffle):
        p = self.p
        if postshuffle:
            x = _layer_norm(x.reshape(-1, HIDDEN * MERGE * MERGE), p[f"{pre}.norm.weight"], p[f"{pre}.norm.bias"])
        else:
            x = _layer_norm(x, p[f"{pre}.norm.weight"], p[f"{pre}.norm.bias"]).reshape(-1, HIDDEN * MERGE * MERGE)
        x = _gelu_erf(linear(x, p[f"{pre}.linear_fc1.weight"], p[f"{pre}.linear_fc1.bias"]))
        return linear(x, p[f"{pre}.linear_fc2.weight"], p[f"{pre}.linear_fc2.bias"])

    def __call__(self, patches: np.ndarray, grid):
        """One image: patches (N, 1536) -> (embeds (N/4, 5120), [deepstack (N/4, 5120)] x 3)."""
        _, h, w = grid
        x = linear(mx.array(patches).astype(self.dtype), self.patch_w, self.p["patch_embed.proj.bias"])
        x = x + self._pos_embed(h, w).astype(x.dtype)
        cos, sin = self._rope(h, w)
        deep = []
        for i in range(DEPTH):
            x = self._block(x, i, cos, sin)
            if i in DEEPSTACK:
                deep.append(self._merger(x, f"deepstack_merger_list.{DEEPSTACK.index(i)}", True))
            mx.eval(x)
        return self._merger(x, "merger", False), deep


def edit_positions(token_ids: list[int], grids) -> np.ndarray:
    """Qwen3VLModel.get_rope_index for one presentation: (3, seq) M-RoPE positions (text runs arange, images
    (t, h, w) grids offset by the running position, which then advances by max(h, w) / 2)."""
    is_image = np.asarray(token_ids) == IMAGE_PAD
    pos, cur, i, n, g = [], 0, 0, len(token_ids), iter(grids)
    while i < n:
        j = i
        while j < n and is_image[j] == is_image[i]:
            j += 1
        if not is_image[i]:
            pos.append(np.tile(np.arange(j - i) + cur, (3, 1)))
            cur += j - i
        else:
            _, gh, gw = next(g)
            lh, lw = gh // MERGE, gw // MERGE
            tt, hh, ww = np.meshgrid(np.arange(1), np.arange(lh) + cur, np.arange(lw) + cur, indexing="ij")
            block = np.stack([tt.reshape(-1) + cur, hh.reshape(-1), ww.reshape(-1)])
            assert block.shape[1] == j - i, "image tokens do not match the grid"
            pos.append(block)
            cur += max(gh, gw) // MERGE
        i = j
    return np.concatenate(pos, axis=1)
