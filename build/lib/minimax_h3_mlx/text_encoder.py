"""The MiniMax-H3 conditioner: the Qwen3-VL text model (64 layers, hidden 5120), read at hidden_states[50].

Only decoder layers 0..49 are ever evaluated (hidden_states[50] is the raw output of layer 49, before any final
norm), so layers 50..63, the final norm and the LM head are not shipped. Ops follow Hugging Face transformers'
Qwen3-VL text model exactly (Qwen3VLTextRMSNorm rounds before the weight multiply; bf16 rotary tables; SiLU-gated
MLP), which MLX-LM's fused layers do not (see ops.py).

The prompt is used verbatim: no chat template, no special tokens, truncated to 4096 tokens. Text-only prompts
use 1-D positions (all three M-RoPE axes equal), which makes the interleaved M-RoPE an ordinary RoPE.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .dit import arrays
from .ops import qlinear, rms_norm_hf, silu

BF16 = mx.bfloat16
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 5120, 64, 8, 128
LAYER = 50
EPS = 1e-6
THETA = 5_000_000.0
MAX_TOKENS = 4096


def rope_tables(positions: np.ndarray, dtype=BF16):
    """Qwen3VLTextRotaryEmbedding for 1-D positions -> (cos, sin) (n, 128) bf16.

    inv_freq = 1 / theta ** (arange(0, 128, 2) / 128) in float32; freqs = pos * inv_freq in float32;
    cos/sin in float64 then rounded (torch's float32 cos/sin, correctly rounded), then cast to bf16."""
    inv = (np.float32(1.0) / np.power(np.float32(THETA),
           (np.arange(0, HEAD_DIM, 2, dtype=np.int64).astype(np.float32) / np.float32(HEAD_DIM)))).astype(np.float32)
    freqs = (positions.astype(np.float32)[:, None] * inv[None, :]).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float64)
    return (mx.array(np.cos(emb).astype(np.float32)).astype(dtype),
            mx.array(np.sin(emb).astype(np.float32)).astype(dtype))


def mrope_tables(pos3: np.ndarray, dtype=BF16, section=(24, 20, 20)):
    """Interleaved M-RoPE (Qwen3VLTextRotaryEmbedding.recomposition_frequencies) for (3, n) t/h/w positions:
    channel j < 60 with j % 3 == 1 takes the h frequency, j % 3 == 2 the w frequency, all others t."""
    inv = (np.float32(1.0) / np.power(np.float32(THETA),
           (np.arange(0, HEAD_DIM, 2, dtype=np.int64).astype(np.float32) / np.float32(HEAD_DIM)))).astype(np.float32)
    freqs = (pos3.astype(np.float32)[:, :, None] * inv[None, None, :]).astype(np.float32)  # (3, n, 64)
    f = freqs[0].copy()
    f[:, 1:section[1] * 3:3] = freqs[1][:, 1:section[1] * 3:3]
    f[:, 2:section[2] * 3:3] = freqs[2][:, 2:section[2] * 3:3]
    emb = np.concatenate([f, f], axis=-1).astype(np.float64)
    return (mx.array(np.cos(emb).astype(np.float32)).astype(dtype),
            mx.array(np.sin(emb).astype(np.float32)).astype(dtype))


def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _layers(params, prefix, quant=None):
    names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    out = {}
    for name in names:
        layer = {k[len(prefix) + len(name) + 1:]: v for k, v in params.items() if k.startswith(f"{prefix}{name}.")}
        if "scales" in layer:
            layer["bits"], layer["group_size"] = int((quant or {}).get("bits", 8)), int((quant or {}).get("group_size", 64))
        out[name] = layer
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                 "self_attn.k_norm.weight"):
        out[name] = params[prefix + name]
    return out


def decoder_layer(h, w, cos, sin):
    """Qwen3VLTextDecoderLayer, causal, bf16. h: (n, 5120)."""
    n = h.shape[0]
    x = rms_norm_hf(h, w["input_layernorm.weight"], EPS)
    q = rms_norm_hf(qlinear(x, w["self_attn.q_proj"]).reshape(n, HEADS, HEAD_DIM), w["self_attn.q_norm.weight"], EPS)
    k = rms_norm_hf(qlinear(x, w["self_attn.k_proj"]).reshape(n, KV_HEADS, HEAD_DIM), w["self_attn.k_norm.weight"], EPS)
    v = qlinear(x, w["self_attn.v_proj"]).reshape(n, KV_HEADS, HEAD_DIM)
    q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
    c, s = cos[None, None], sin[None, None]
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5, mask="causal")
    o = o[0].transpose(1, 0, 2).reshape(n, HEADS * HEAD_DIM)
    h = h + qlinear(o, w["self_attn.o_proj"])
    x = rms_norm_hf(h, w["post_attention_layernorm.weight"], EPS)
    x = qlinear(silu(qlinear(x, w["mlp.gate_proj"])) * qlinear(x, w["mlp.up_proj"]), w["mlp.down_proj"])
    return h + x


class TextEncoder:
    """Weights layout (`root`): embed.safetensors, layers/NN.safetensors (NN < 50), visual.safetensors (edit)."""

    def __init__(self, root, tokenizer_dir, dtype=BF16, quantize: dict | None = None):
        from tokenizers import Tokenizer

        self.root = Path(root)
        self.dtype = dtype
        self.quantize = quantize
        meta = self.root / "te.json"
        self.quant = __import__("json").loads(meta.read_text()) if meta.exists() else {}
        self.tokenizer = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def tokenize(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a single non-empty string")
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids[:MAX_TOKENS]

    def embed(self, ids: list[int]):
        table = mx.load(str(self.root / "embed.safetensors"))["model.language_model.embed_tokens.weight"]
        rows = mx.take(table, mx.array(np.asarray(ids, np.int32)), axis=0).astype(self.dtype)
        mx.eval(rows)
        del table
        return rows

    def layer(self, i):
        params = mx.load(str(self.root / f"layers/{i:02d}.safetensors"))
        if self.dtype != BF16:
            params = {k: v.astype(self.dtype) for k, v in params.items()}
        layers = _layers(params, f"model.language_model.layers.{i}.", self.quant)
        if self.quantize and self.quantize.get("bits", 16) < 16:
            for name, layer in layers.items():
                if isinstance(layer, dict) and "scales" not in layer:
                    gs, bits = self.quantize.get("group_size", 64), self.quantize["bits"]
                    w, sc, bi = mx.quantize(layer["weight"], group_size=gs, bits=bits)
                    layers[name] = {"weight": w, "scales": sc, "biases": bi, "bits": bits, "group_size": gs}
        return layers

    def __call__(self, ids: list[int], on_layer=None, image_embeds=None, deepstack=None, positions=None, n_layers=LAYER):
        """hidden_states[50]: (n, 5120) bf16.

        Text-only: 1-D positions. Edit: `image_embeds` (rows for every <|image_pad|> token, in order), `deepstack`
        (3 arrays of the same rows, added after decoder layers 0, 1, 2) and `positions` (3, n) M-RoPE ids."""
        h = self.embed(ids)
        image_rows = None
        if image_embeds is not None:
            image_rows = mx.array(np.nonzero(np.asarray(ids) == 151655)[0].astype(np.int32))
            h[image_rows] = image_embeds.astype(h.dtype)  # masked_scatter
        if positions is None:
            cos, sin = rope_tables(np.arange(len(ids)), self.dtype)
        else:
            cos, sin = mrope_tables(positions, self.dtype)
        nxt = self.layer(0)
        for i in range(n_layers):
            w = nxt
            h = decoder_layer(h, w, cos, sin)
            if deepstack is not None and i < len(deepstack):
                h[image_rows] = h[image_rows] + deepstack[i].astype(h.dtype)
            if i + 1 < n_layers:
                nxt = self.layer(i + 1)
                mx.async_eval(arrays(nxt))
            mx.eval(h)
            del w
            if on_layer:
                on_layer(i, h)
        return h
