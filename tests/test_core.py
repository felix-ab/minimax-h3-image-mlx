"""Torch-free tests (run in CI). Reference values were produced with torch 2.15 / diffusers @875373ac7."""
import hashlib
import sys

import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx import layout as L
from minimax_h3_mlx.ops import rms_norm_hf, rms_norm_torch, silu
from minimax_h3_mlx.torch_randn import Generator, randn
from minimax_h3_mlx.vae import split_tiles


@pytest.mark.skipif(sys.platform != "darwin", reason="bit-exact only against macOS libm")
@pytest.mark.parametrize("shape,seed,digest", [
    ((1, 24, 1, 222, 296), 42, "9d7c76fb8c78e207cef9c9da853fe20197a4fbaa6f8b90555d0250d72b6d0ca0"),
    ((1, 24, 1, 256, 256), 1234, "8b2bb14274e58916d2ef7115353ca6e123ff8a67bcc2c8c9e9d529df165ff89d"),
])
def test_noise_matches_torch_fingerprint(shape, seed, digest):
    assert hashlib.sha256(randn(shape, seed).tobytes()).hexdigest() == digest
    assert hashlib.sha256(Generator(seed).randn(shape).tobytes()).hexdigest() == digest


def test_canvas():
    assert L.canvas_size(3552, 4736) == (3552, 4736)
    assert L.canvas_size(1024, 1024) == (2048, 2048)
    assert L.canvas_size(1024, 1024, 1024) == (1024, 1024)
    assert L.canvas_size(768, 1360) == (1536, 2720)


def test_sampling_grid():
    t, sigmas = L.sampling_grid()
    assert t.dtype == np.float32 and sigmas.dtype == np.float32
    assert t[0] == 0 and sigmas[0] == 1 and sigmas[-1] == 0 and len(t) == 6
    assert (np.float32(1) - t == sigmas[:-1]).all()


def test_tiles_cover_exactly():
    for length in (256, 1024, 3552, 4736, 6272, 2688):
        starts, lengths, overlaps = split_tiles(length)
        assert starts[-1] + lengths[-1] == length
        assert all(o >= 64 and o % 16 == 0 for o in overlaps)
    assert len(split_tiles(3552)[0]) == 19 and len(split_tiles(4736)[0]) == 25


def test_patchify_roundtrip_and_runs():
    x = np.random.default_rng(0).standard_normal((24, 12, 20)).astype(np.float32)
    rows = L.patchify(x)
    assert rows.shape == (60, 96)
    assert (L.unpatchify(rows, 12, 20)[0] == x).all()
    lay = L.Layout(n_text=7, text_tags=np.full(7, L.TEXT_TAG), latent_h=12, latent_w=20)
    assert lay.runs(3, 6) == [(0, 7, 3, L.TEXT_TAG), (7, 67, 3, L.IMAGE_TAG)]
    lay = L.Layout(n_text=4, text_tags=np.array([1, 0, 0, 1]), reference_grids=[(4, 4)], latent_h=4, latent_w=4)
    assert lay.runs(2, 6) == [(0, 1, 2, 1), (1, 3, 2, 0), (3, 4, 2, 1), (4, 8, 6, 0), (8, 12, 2, 0)]


def test_rounding_points():
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((4, 64)).astype(np.float32) * 3).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal(64).astype(np.float32)).astype(mx.bfloat16)
    xf, wf = np.array(x.astype(mx.float32)), np.array(w.astype(mx.float32))
    norm = xf / np.sqrt(np.mean(xf ** 2, -1, keepdims=True) + 1e-5)
    once = mx.array((norm * wf).astype(np.float32)).astype(mx.bfloat16)
    assert (rms_norm_torch(x, w, 1e-5) == once).all()            # torch nn.RMSNorm: one rounding
    twice = mx.array(norm.astype(np.float32)).astype(mx.bfloat16) * w
    assert (rms_norm_hf(x, w, 1e-5) == twice).all()               # HF Qwen RMSNorm: round, then weight
    s = np.array(x.astype(mx.float32))
    assert (silu(x) == mx.array((s / (1 + np.exp(-s))).astype(np.float32)).astype(mx.bfloat16)).sum() >= 250
