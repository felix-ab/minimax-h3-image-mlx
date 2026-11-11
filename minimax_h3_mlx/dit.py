"""MiniMax-H3-Image diffusion transformer (33.1 B) in MLX.

Ported from diffusers' `MiniMaxH3ImageTransformer2DModel` (Apache-2.0): one packed sequence
`[text | references | image]`, 50 single-stream blocks with per-row AdaLN, a 2-layer text token refiner, 3-axis
RoPE and one PDD output head per sampling step.

Precision follows the release exactly: proj_in, the timestep MLP and the output heads are float32; everything
else is bf16 with torch's rounding points (see ops.py). The per-block AdaLN projections are replaced by
precomputed tables (scripts/make_modulation.py): they depend only on the timestep, and the distilled sampler
only ever uses 7 timesteps, so the tables hold exactly the values the projections would produce.

Weights can be resident or streamed one block at a time from disk (each bf16 block is 0.77 GB; at 4+ MP the
block's compute takes longer than reading it, so streaming costs little and keeps the footprint to a few GB).
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .ops import linear, qlinear, rms_norm_torch, silu

BF16, F32 = mx.bfloat16, mx.float32
HIDDEN, HEADS, HEAD_DIM = 5376, 56, 128
NUM_LAYERS, NUM_REFINER = 50, 2
ROPE_FREQ, ROPE_THETA = 16, 10000.0
EPS = 1e-5
LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


def arrays(tree):
    """All mx.arrays in a nested dict/list (for eval/async_eval of weight trees holding ints too)."""
    if isinstance(tree, mx.array):
        return [tree]
    if isinstance(tree, dict):
        return [a for v in tree.values() for a in arrays(v)]
    if isinstance(tree, (list, tuple)):
        return [a for v in tree for a in arrays(v)]
    return []


def _strip(params: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def _cast(tree, dtype):
    if dtype == BF16:
        return tree
    return {k: (_cast(v, dtype) if isinstance(v, dict) else (v.astype(dtype) if isinstance(v, mx.array)
            and v.dtype in (BF16, F32, mx.float16) else v)) for k, v in tree.items()}


def _layers(params: dict, quant: dict | None = None) -> dict:
    """{'attn.to_q': {'weight': ...[, 'scales', 'biases', 'bits', 'group_size']}, 'norm1': array, ...}."""
    out = {}
    for name in LINEARS:
        layer = {k.split(".")[-1]: v for k, v in params.items() if k.rsplit(".", 1)[0] == name}
        if "scales" in layer:
            q = (quant or {}).get("per_layer", {}).get(name, quant or {})
            layer["bits"], layer["group_size"] = int(q.get("bits", 8)), int(q.get("group_size", 64))
        out[name] = layer
    for name in ("norm1.weight", "norm2.weight", "attn.norm_q.weight", "attn.norm_k.weight"):
        out[name] = params[name]
    return out


# ------------------------------------------------------------------------------------------------- rope

def rope_tables(position_ids: np.ndarray, dtype=BF16):
    """MiniMaxH3ImageRotaryPosEmbed + the bf16 cast in _apply_rotary_emb. -> cos, sin (seq, 96) bf16.

    positions -> float32; freqs = pos * inv_freq in float32 (as torch computes them); cos/sin evaluated in float64
    and rounded (the correctly rounded value of torch's float32 cos/sin), then rounded to bf16 like the reference.
    """
    inv_freq = (np.float32(1.0) / np.power(np.float32(ROPE_THETA),
                (np.arange(0, 2 * ROPE_FREQ, 2, dtype=np.float32) / np.float32(2 * ROPE_FREQ)))).astype(np.float32)
    pos = position_ids.astype(np.float32)
    freqs = (pos[:, :, None] * inv_freq[None, None, :]).astype(np.float32)  # (seq, 3, 16)
    freqs = freqs.reshape(len(pos), -1)                                        # t | h | w
    freqs = np.concatenate([freqs, freqs], axis=-1)                            # (seq, 96)
    f64 = freqs.astype(np.float64)
    cos = mx.array(np.cos(f64).astype(np.float32)).astype(dtype)
    sin = mx.array(np.sin(f64).astype(np.float32)).astype(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    """_apply_rotary_emb on (seq, heads, head_dim) bf16; cos/sin (seq, 96) bf16."""
    r = cos.shape[-1]
    rot, keep = x[..., :r], x[..., r:]
    x1, x2 = rot[..., : r // 2], rot[..., r // 2:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    c, s = cos[:, None, :], sin[:, None, :]
    return mx.concatenate([rot * c + rotated * s, keep], axis=-1)


# ------------------------------------------------------------------------------------------------- blocks

def attention(x, w, cos=None, sin=None):
    n = x.shape[0]
    q = qlinear(x, w["attn.to_q"]).reshape(n, HEADS, HEAD_DIM)
    k = qlinear(x, w["attn.to_k"]).reshape(n, HEADS, HEAD_DIM)
    v = qlinear(x, w["attn.to_v"]).reshape(n, HEADS, HEAD_DIM)
    q = rms_norm_torch(q, w["attn.norm_q.weight"], EPS)
    k = rms_norm_torch(k, w["attn.norm_k.weight"], EPS)
    if cos is not None:
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    # MLX's bf16 SDPA equals float32 attention rounded once (checked bit for bit), like FlashAttention's output.
    o = mx.fast.scaled_dot_product_attention(
        q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None], scale=HEAD_DIM ** -0.5)
    o = o[0].transpose(1, 0, 2).reshape(n, HEADS * HEAD_DIM)
    return qlinear(o, w["attn.to_out.0"])


def feed_forward(x, w):
    h = qlinear(x, w["ff.net.0.proj"])
    value, gate = mx.split(h, 2, axis=-1)
    return qlinear(value * silu(gate), w["ff.net.2"])


def _modulate(x, runs, mods, k_shift, k_scale):
    """norm * (1 + scale) + shift per run of rows; mods[c] is a (6, hidden) bf16 table row set."""
    parts = [x[s:e] * (1.0 + mods[c][k_scale]) + mods[c][k_shift] for s, e, c in runs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


def _gate(y, runs, mods, k_gate):
    parts = [mods[c][k_gate] * y[s:e] for s, e, c in runs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


def transformer_block(h, w, runs, mods, cos, sin):
    """MiniMaxH3ImageTransformerBlock. mods rows: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp."""
    n = _modulate(rms_norm_torch(h, w["norm1.weight"], EPS), runs, mods, 0, 1)
    h = h + _gate(attention(n, w, cos, sin), runs, mods, 2)
    n = _modulate(rms_norm_torch(h, w["norm2.weight"], EPS), runs, mods, 3, 4)
    return h + _gate(feed_forward(n, w), runs, mods, 5)


def refiner_block(h, w):
    h = h + attention(rms_norm_torch(h, w["norm1.weight"], EPS), w)
    return h + feed_forward(rms_norm_torch(h, w["norm2.weight"], EPS), w)


# ------------------------------------------------------------------------------------------------- model

class DiT:
    """Weights layout (directory `root`):
        core.safetensors        proj_in, context_embedder, time_embedder, norm_out, pdd_proj_out.* (full names)
        refiner.safetensors     token_refiner.*
        blocks/NN.safetensors   transformer_blocks.NN.* (no adaln_proj); dense bf16 or MLX-quantized
        modulation/NN.safetensors, modulation/norm_out.safetensors   AdaLN tables over `timesteps`
    """

    def __init__(self, root, stream: bool | None = None, dtype=BF16, quantize: dict | None = None):
        """quantize: {"bits": 8, "group_size": 64} quantizes dense blocks on the fly (evaluation of tiers without
        exporting them; bit-identical to export.py since mx.quantize is deterministic)."""
        self.root = Path(root)
        self.quantize = quantize
        self.dtype = dtype  # bf16 = the release; float32 = exactness tests against a float32 torch reference
        meta_path = self.root / "dit.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        core = mx.load(str(self.root / "core.safetensors"))
        self.proj_in = (core["proj_in.weight"].astype(F32), core["proj_in.bias"].astype(F32))
        self.context_embedder = (core["context_embedder.weight"].astype(dtype), core["context_embedder.bias"].astype(dtype))
        self.norm_out_weight = core["norm_out.norm.weight"].astype(dtype)
        self.heads = [(core[f"pdd_proj_out.{i}.weight"].astype(F32), core[f"pdd_proj_out.{i}.bias"].astype(F32))
                      for i in range(6)]
        mod_out = mx.load(str(self.root / "modulation/norm_out.safetensors"))
        self.timesteps = np.array(mod_out["timesteps"]).astype(np.float32)
        self.norm_out_table = mod_out["table"].astype(dtype)  # (T, 2, hidden): shift, scale
        self.tables = [mx.load(str(self.root / f"modulation/{i:02d}.safetensors"))["table"].astype(dtype)
                       if (self.root / f"modulation/{i:02d}.safetensors").exists() else None
                       for i in range(NUM_LAYERS)]  # all 50 exist in a complete tier
        mx.eval(self.proj_in, self.context_embedder, self.norm_out_weight, self.heads, self.norm_out_table,
                [t for t in self.tables if t is not None])
        self.stream = (not (self.meta.get("quantized", False) or quantize)) if stream is None else stream
        self._resident = None if self.stream else [self._load_block(i) for i in range(NUM_LAYERS)]
        if self._resident is not None:
            mx.eval(arrays(self._resident))

    # weights ---------------------------------------------------------------------------------------
    def _load_block(self, i):
        params = mx.load(str(self.root / f"blocks/{i:02d}.safetensors"))
        layers = _cast(_layers(_strip(params, f"transformer_blocks.{i}."), self.meta), self.dtype)
        if self.quantize:
            for name in LINEARS:
                bits = self.quantize.get("per_layer", {}).get(name, self.quantize["bits"])
                if bits < 16 and "scales" not in layers[name]:
                    w, sc, bi = mx.quantize(layers[name]["weight"], group_size=self.quantize.get("group_size", 64), bits=bits)
                    layers[name] = {"weight": w, "scales": sc, "biases": bi, "bits": bits,
                                    "group_size": self.quantize.get("group_size", 64)}
        return layers

    def block(self, i):
        return self._resident[i] if self._resident is not None else self._load_block(i)

    def t_index(self, t: float) -> int:
        hits = np.nonzero(self.timesteps == np.float32(t))[0]
        if not len(hits):
            raise ValueError(f"timestep {t!r} has no precomputed modulation (have {self.timesteps.tolist()})")
        return int(hits[0])

    # text ------------------------------------------------------------------------------------------
    def embed_text(self, prompt_embeds):
        """context_embedder + token refiner (timestep-independent: run once per image). (n, 5120) -> (n, 5376) bf16."""
        h = linear(prompt_embeds.astype(self.dtype), *self.context_embedder)
        params = mx.load(str(self.root / "refiner.safetensors"))
        for i in range(NUM_REFINER):
            w = _cast(_layers(_strip(params, f"token_refiner.refiner_blocks.{i}.")), self.dtype)
            h = refiner_block(h, w)
            mx.eval(h)
        return rms_norm_torch(h, params["token_refiner.final_norm.weight"].astype(self.dtype), EPS)

    # one denoising forward -------------------------------------------------------------------------
    def __call__(self, image_rows, text_h, runs, rope, step: int, target: slice, on_block=None, n_blocks=NUM_LAYERS):
        """image_rows: (n_img, 96) float32 rows [references | target]; text_h: refined text (n_text, 5376) bf16.
        runs: [(start, end, t_index, tag)] over the packed sequence. Returns the velocity of the target rows (f32)."""
        cos, sin = rope
        image_h = (mx.addmm(self.proj_in[1], image_rows.astype(F32), self.proj_in[0].T)).astype(self.dtype)
        h = mx.concatenate([text_h, image_h], axis=0)
        combos = sorted({(t, tag) for _, _, t, tag in runs})
        key = {c: j for j, c in enumerate(combos)}
        block_runs = [(s, e, key[(t, tag)]) for s, e, t, tag in runs]
        nxt = self.block(0)
        for i in range(n_blocks):
            w = nxt
            table = self.tables[i]
            mods = [table[t, tag] for (t, tag) in combos]  # (6, hidden) each
            h = transformer_block(h, w, block_runs, mods, cos, sin)
            if i + 1 < n_blocks:
                nxt = self.block(i + 1)
                if self.stream:
                    mx.async_eval(arrays(nxt))  # read the next block while this one computes
            mx.eval(h)
            del w
            if on_block:
                on_block(i, h)
        # norm_out over the target rows (per-row timestep modulation), then this step's float32 head
        t_target = runs[-1][2]
        ht = rms_norm_torch(h[target], self.norm_out_weight, EPS)
        shift, scale = self.norm_out_table[t_target, 0], self.norm_out_table[t_target, 1]
        ht = (ht * (1.0 + scale) + shift).astype(F32)
        weight, bias = self.heads[step]
        return mx.addmm(bias, ht, weight.T)
