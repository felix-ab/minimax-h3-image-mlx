"""Canvas sizing, the packed `[text | references | image]` layout, the sampling grid and the initial noise.

Everything here is host-side numpy, mirroring diffusers' MiniMax-H3-Image modular blocks (before_denoise.py,
before_encoder.py): the canvas rule, the float64 aspect-normalized rotary grid built with np.linspace, the
float32 round trip sigma -> t = 1 - sigma -> sigma that the released sampler steps on, and the patchify order.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RATIO = 16          # VAE spatial compression
PATCH = 2           # DiT patch (h, w); t patch is 1
MULTIPLE = RATIO * PATCH  # canvas sides are multiples of 32
CHANNELS = 24
ROPE_SPATIAL_SCALE = 32
IMAGE_TAG, TEXT_TAG = 0, 1
REFERENCE_T = 0.999
DEFAULT_MIN_AREA = 4_194_304  # 2048 x 2048
SAMPLE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def canvas_size(height: int, width: int, min_generation_area: int = DEFAULT_MIN_AREA) -> tuple[int, int]:
    """get_image_canvas_size: scale up to min_generation_area at the same aspect, round sides to 32."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if not 1024 <= min_generation_area <= 4096 * 4096:
        raise ValueError(f"min_generation_area must be within [1024, {4096 * 4096}], got {min_generation_area}")
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (max(MULTIPLE, round(height * scale / MULTIPLE) * MULTIPLE),
            max(MULTIPLE, round(width * scale / MULTIPLE) * MULTIPLE))


def sampling_grid():
    """(model timesteps t, sigmas) as the released sampler builds them, both float32.

    t_i = float32(1 - sigma_i) for the 6 steps; the scheduler then steps on sigma_i = float32(1 - t_i) and a final 0.
    """
    t = np.array([1.0 - s for s in SAMPLE_SIGMAS[:-1]], dtype=np.float32)
    sigmas = np.concatenate([(np.float32(1.0) - t).astype(np.float32), np.zeros(1, np.float32)])
    return t, sigmas


def _spatial_grid(dim: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // PATCH, endpoint=False) * ROPE_SPATIAL_SCALE


def frame_grid(latent_h: int, latent_w: int) -> np.ndarray:
    sqrt_area = np.sqrt(latent_h * latent_w)
    gh = _spatial_grid(latent_h, sqrt_area)
    gw = _spatial_grid(latent_w, sqrt_area)
    hh, ww = np.meshgrid(gh, gw, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)  # float64 (rows, 2)


@dataclass
class Layout:
    """One packed request. Rows: text [0, n_text), then each reference, then the generated image."""
    n_text: int
    text_tags: np.ndarray                    # (n_text,) int, 1 text / 0 for rows of a vision block
    reference_grids: list = field(default_factory=list)  # [(latent_h, latent_w)] per reference
    latent_h: int = 0
    latent_w: int = 0

    @property
    def n_reference_rows(self) -> int:
        return sum((h // PATCH) * (w // PATCH) for h, w in self.reference_grids)

    @property
    def n_target_rows(self) -> int:
        return (self.latent_h // PATCH) * (self.latent_w // PATCH)

    @property
    def target_start(self) -> int:
        return self.n_text + self.n_reference_rows

    @property
    def length(self) -> int:
        return self.target_start + self.n_target_rows

    def position_ids(self) -> np.ndarray:
        """build_image_packed_sequence: float64 (seq, 3) rotary coordinates."""
        pos = np.zeros((self.length, 3), dtype=np.float64)
        pos[: self.n_text, 0] = np.arange(self.n_text, dtype=np.float64)
        cursor, rotary_time = self.n_text, float(self.n_text)
        for h, w in self.reference_grids:
            rows = (h // PATCH) * (w // PATCH)
            pos[cursor:cursor + rows, 0] = rotary_time
            pos[cursor:cursor + rows, 1:] = frame_grid(h, w)
            cursor += rows
            rotary_time += 1.0
        pos[cursor:, 0] = rotary_time
        pos[cursor:, 1:] = frame_grid(self.latent_h, self.latent_w)
        return pos

    def tags(self) -> np.ndarray:
        tags = np.full(self.length, IMAGE_TAG, dtype=np.int64)
        tags[: self.n_text] = self.text_tags
        return tags

    def runs(self, step_t_index: int, reference_t_index: int):
        """Contiguous (start, end, timestep_index, tag) runs of the AdaLN table rows for one step.

        Text rows and generated rows sit at the step's timestep; reference rows at t = 0.999."""
        tags = self.tags()
        t_index = np.full(self.length, step_t_index, dtype=np.int64)
        t_index[self.n_text:self.target_start] = reference_t_index
        key = t_index * 3 + tags
        runs, start = [], 0
        for i in range(1, self.length + 1):
            if i == self.length or key[i] != key[start]:
                runs.append((start, i, int(t_index[start]), int(tags[start])))
                start = i
        return runs


def patchify(latents: np.ndarray) -> np.ndarray:
    """(24, H, W) -> rows (H/2 * W/2, 96), feature order (c, ph, pw), row-major over the patch grid."""
    c, h, w = latents.shape
    x = latents.reshape(c, h // PATCH, PATCH, w // PATCH, PATCH).transpose(1, 3, 0, 2, 4)
    return x.reshape(-1, c * PATCH * PATCH)


def unpatchify(rows, latent_h: int, latent_w: int):
    """Inverse of patchify; works on numpy or mlx arrays. -> (1, 24, H, W)."""
    x = rows.reshape(1, latent_h // PATCH, latent_w // PATCH, CHANNELS, PATCH, PATCH)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(1, CHANNELS, latent_h, latent_w)
