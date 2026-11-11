"""MiniMax-H3-Image text-to-image on MLX: the released 6-step, guidance-free PDD sampler.

Order of operations (diffusers MiniMaxH3ImageBlocks, text2image):
  canvas    = size scaled up to >= min_generation_area (default 2048x2048) at the same aspect, sides / 32
  prompt    -> tokens (verbatim, <= 4096) -> Qwen3-VL hidden_states[50]
  noise     = torch.randn((1, 24, 1, H/16, W/16), CPU generator(seed)), patchified into rows
  for step i in 0..5:  v = DiT(rows, t_i, head i);  rows += (sigma_i - sigma_{i+1}) * v      (float32)
  image     = tiled ViT-VAE decode, 8-bit by truncation, LANCZOS back to the requested size if the canvas was larger
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from . import layout as L
from .dit import DiT, rope_tables
from .text_encoder import TextEncoder
from .torch_randn import randn
from .vae import VAE, to_uint8


@dataclass
class Result:
    image: Image.Image
    canvas: tuple[int, int]
    timings: dict
    latents: np.ndarray | None = None


def _free():
    gc.collect()
    mx.clear_cache()


class Pipeline:
    def __init__(self, weights, dtype=mx.bfloat16, stream: bool | None = None, dit_quant=None, te_quant=None,
                 cache_limit_gb: float | None = None):
        """cache_limit_gb caps MLX's buffer cache (freed buffers kept for reuse), which otherwise counts toward the
        process footprint. It changes memory, never numerics."""
        self.root = Path(weights)
        if cache_limit_gb is not None:
            mx.set_cache_limit(int(cache_limit_gb * 1e9))
        self.dit_quant, self.te_quant = dit_quant, te_quant
        self.dtype = dtype
        self.stream = stream
        self.config = json.loads((self.root / "vae/config.json").read_text())

    # stages --------------------------------------------------------------------------------------------
    def encode_prompt(self, prompt: str):
        te = TextEncoder(self.root / "te", self.root / "tokenizer", dtype=self.dtype, quantize=self.te_quant)
        ids = te.tokenize(prompt)
        h = te(ids)
        mx.eval(h)
        del te
        _free()
        return ids, h

    def denoise(self, prompt_embeds, n_text: int, canvas: tuple[int, int], seed: int, noise=None, on_step=None):
        height, width = canvas
        lay = L.Layout(n_text=n_text, text_tags=np.full(n_text, L.TEXT_TAG), latent_h=height // L.RATIO,
                       latent_w=width // L.RATIO)
        if noise is None:
            noise = randn((1, L.CHANNELS, 1, lay.latent_h, lay.latent_w), seed)
        rows = mx.array(L.patchify(np.asarray(noise, np.float32).reshape(L.CHANNELS, lay.latent_h, lay.latent_w)))
        dit = DiT(self.root / "dit", stream=self.stream, dtype=self.dtype, quantize=self.dit_quant)
        text_h = dit.embed_text(prompt_embeds)
        mx.eval(text_h)
        rope = rope_tables(lay.position_ids(), self.dtype)
        t, sigmas = L.sampling_grid()
        reference_t = dit.t_index(L.REFERENCE_T)
        target = slice(lay.target_start, lay.length)
        for i in range(len(t)):
            runs = lay.runs(dit.t_index(float(t[i])), reference_t)
            v = dit(rows, text_h, runs, rope, step=i, target=target)
            dsigma = mx.array(np.float32(sigmas[i]) - np.float32(sigmas[i + 1]), dtype=mx.float32)
            rows = rows + dsigma * v
            mx.eval(rows)
            if on_step:
                on_step(i, rows)
        del dit
        _free()
        return rows, lay

    def decode(self, rows, lay):
        vae = VAE.load(self.root / "vae/diffusion_pytorch_model.safetensors", self.config)
        latents = L.unpatchify(rows, lay.latent_h, lay.latent_w)
        pixels = vae.decode(latents)
        mx.eval(pixels)
        del vae
        _free()
        return to_uint8(pixels)

    # editing ---------------------------------------------------------------------------------------------
    @staticmethod
    def _bound_short_edge(width, height, lo, hi, multiple=L.MULTIPLE):
        short = min(width, height)
        if lo <= short <= hi:
            return width, height
        scale = (lo if short < lo else hi) / short
        return (max(multiple, round(width * scale / multiple) * multiple),
                max(multiple, round(height * scale / multiple) * multiple))

    def edit(self, prompt: str, images: list, height: int = 2048, width: int = 2048, seed: int = 0,
             min_generation_area: int = L.DEFAULT_MIN_AREA, on_step=None) -> Result:
        """Reference-image editing: `images` are PIL images, <Picture 1>, <Picture 2>, ... in the prompt."""
        from PIL import ImageOps

        from .torch_randn import Generator
        from .vision import IMAGE_PAD, VISION_END, VISION_START, VisionTower, edit_positions, preprocess

        timings, t0 = {}, time.perf_counter()
        images = [ImageOps.exif_transpose(im).convert("RGB") for im in images]

        def resize(im, lo, hi):
            size = self._bound_short_edge(*im.size, lo, hi)
            return im if im.size == size else im.resize(size, resample=Image.LANCZOS)

        refs = [resize(im, 1024, 2048) for im in images]
        vision_images = [resize(im, 1024, 1024) for im in images]

        # 1. presentation: "Picture i: " <|vision_start|> <|image_pad|> * n <|vision_end|> ... prompt
        te = TextEncoder(self.root / "te", self.root / "tokenizer", dtype=self.dtype, quantize=self.te_quant)
        tower = VisionTower(mx.load(str(self.root / "te/visual.safetensors")), dtype=self.dtype)
        ids, tags, grids, embeds, deep = [], [], [], [], [[], [], []]
        for i, im in enumerate(vision_images):
            patches, grid = preprocess(np.asarray(im))
            e, d = tower(patches, grid)
            label = te.tokenizer.encode(f"Picture {i + 1}: ", add_special_tokens=False).ids[:4096]
            n = grid[1] * grid[2] // 4
            ids += label + [VISION_START] + [IMAGE_PAD] * n + [VISION_END]
            tags += [L.TEXT_TAG] * len(label) + [L.IMAGE_TAG] * (n + 2)
            grids.append(grid)
            embeds.append(e)
            for k in range(3):
                deep[k].append(d[k])
        del tower
        prompt_ids = te.tokenize(prompt)
        ids += prompt_ids
        tags += [L.TEXT_TAG] * len(prompt_ids)
        h = te(ids, image_embeds=mx.concatenate(embeds), deepstack=[mx.concatenate(d) for d in deep],
               positions=edit_positions(ids, grids))
        mx.eval(h)
        del te
        _free()
        timings["text_encoder_s"] = time.perf_counter() - t0

        # 2. reference latents: posterior samples off the request generator, rounded to float16, normalized
        gen = Generator(seed)
        vae = VAE.load(self.root / "vae/diffusion_pytorch_model.safetensors", self.config)
        mean = np.asarray(self.config["latents_mean"], np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(self.config["latents_std"], np.float32).reshape(1, -1, 1, 1)
        from .vae import PIXEL_MEAN, PIXEL_STD
        ref_latents = []
        for im in refs:
            px = np.asarray(im)
            px = px[: px.shape[0] // L.RATIO * L.RATIO, : px.shape[1] // L.RATIO * L.RATIO]
            px = (px.astype(np.float32) / np.float32(255.0) - np.asarray(PIXEL_MEAN, np.float32)) / np.asarray(PIXEL_STD, np.float32)
            moments = np.array(vae.encode_moments(mx.array(px.transpose(2, 0, 1)[None].astype(np.float32))))
            m, logvar = moments[:, :24], np.clip(moments[:, 24:], -30.0, 20.0)
            sample = m + np.exp(np.float32(0.5) * logvar) * gen.randn(m.shape)
            z = sample.astype(np.float16).astype(np.float32)
            z = z[..., : z.shape[-2] // 2 * 2, : z.shape[-1] // 2 * 2]
            ref_latents.append(((z - mean) / std).astype(np.float32))
        del vae
        _free()

        # 3. layout [text | references | target], target noise, then reference noise held at t = 0.999
        ch, cw = L.canvas_size(height, width, min_generation_area)
        lay = L.Layout(n_text=len(ids), text_tags=np.asarray(tags), latent_h=ch // L.RATIO, latent_w=cw // L.RATIO,
                       reference_grids=[tuple(z.shape[-2:]) for z in ref_latents])
        target = gen.randn((1, L.CHANNELS, 1, lay.latent_h, lay.latent_w))
        t_ref = np.float32(L.REFERENCE_T)
        cond = []
        for z in ref_latents:
            noise = gen.randn(z.shape)
            noised = (t_ref * z + (np.float32(1.0) - t_ref) * noise).astype(np.float32)
            cond.append(L.patchify(noised[0]))
        rows = mx.array(np.concatenate(cond + [L.patchify(target.reshape(L.CHANNELS, lay.latent_h, lay.latent_w))]))
        n_ref = lay.n_reference_rows

        # 4. denoise: only the target rows move
        t1 = time.perf_counter()
        dit = DiT(self.root / "dit", stream=self.stream, dtype=self.dtype, quantize=self.dit_quant)
        text_h = dit.embed_text(h)
        mx.eval(text_h)
        rope = rope_tables(lay.position_ids(), self.dtype)
        t, sigmas = L.sampling_grid()
        reference_t = dit.t_index(L.REFERENCE_T)
        tgt = slice(lay.target_start, lay.length)
        for i in range(len(t)):
            v = dit(rows, text_h, lay.runs(dit.t_index(float(t[i])), reference_t), rope, step=i, target=tgt)
            dsigma = mx.array(np.float32(sigmas[i]) - np.float32(sigmas[i + 1]), dtype=mx.float32)
            rows = mx.concatenate([rows[:n_ref], rows[n_ref:] + dsigma * v])
            mx.eval(rows)
            if on_step:
                on_step(i, rows)
        del dit
        _free()
        timings["denoise_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        pixels = self.decode(rows[n_ref:], lay)
        timings["vae_s"] = time.perf_counter() - t2
        image = Image.fromarray(pixels)
        if (height, width) != (ch, cw):
            image = image.resize((width, height), resample=Image.LANCZOS)
        timings["total_s"] = time.perf_counter() - t0
        return Result(image=image, canvas=(ch, cw), timings=timings)

    # all together --------------------------------------------------------------------------------------
    def __call__(self, prompt: str, height: int = 2048, width: int = 2048, seed: int = 0,
                 min_generation_area: int = L.DEFAULT_MIN_AREA, keep_latents: bool = False, on_step=None) -> Result:
        canvas = L.canvas_size(height, width, min_generation_area)
        timings = {}
        t0 = time.perf_counter()
        ids, h = self.encode_prompt(prompt)
        timings["text_encoder_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        rows, lay = self.denoise(h, len(ids), canvas, seed, on_step=on_step)
        timings["denoise_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        pixels = self.decode(rows, lay)
        timings["vae_s"] = time.perf_counter() - t2
        image = Image.fromarray(pixels)
        if (height, width) != canvas:
            image = image.resize((width, height), resample=Image.LANCZOS)  # VaeImageProcessor.resize default
        timings["total_s"] = time.perf_counter() - t0
        return Result(image=image, canvas=canvas, timings=timings,
                      latents=np.array(rows) if keep_latents else None)
