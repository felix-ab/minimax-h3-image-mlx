"""torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed)) reproduced bit-exactly in numpy.

MiniMax-H3-Image draws its noise on the CPU generator ("same seed, same initial noise on every machine"), and its
model card warns that last-bit latent differences can grow into visibly different images, so this port is exact,
not merely close. torch's CPU `normal_fill` (aten/src/ATen/native/cpu/DistributionTemplates.h, v2.15, the scalar
`NormalFill16`, which is what torch runs on arm64) does, per block of 16 float32 uniforms u (24-bit, MT19937):
    u1 = 1 - u[j]; u2 = u[j + 8]                       (float32)
    radius = sqrt(-2 * logf(u1))                       (float32)
    theta  = (float)(2.0f * pi<double> * u2)           (product in double, rounded once to float)
    out[j], out[j + 8] = radius * cosf(theta), radius * sinf(theta)
On macOS clang fuses the cosf/sinf pair into Apple's `__sincosf_stret`, whose results differ from separate
cosf/sinf calls (and from vForce) in ~1% of cases; we call the same libm entry points through ctypes. A tail that
is not a multiple of 16 re-draws the last 16 values, as torch does.
Verified bit-identical to torch 2.15 CPU on macOS arm64 for 3.5 M samples (tests/test_noise.py).
"""
import ctypes
import sys

import numpy as np

_TWO_PI = 2.0 * np.pi  # 2.0f * c10::pi<double>, evaluated in double


class _SinCos(ctypes.Structure):
    _fields_ = [("s", ctypes.c_float), ("c", ctypes.c_float)]


if sys.platform == "darwin":
    _libm = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    _libm.logf.restype, _libm.logf.argtypes = ctypes.c_float, [ctypes.c_float]
    _sincosf = _libm.__sincosf_stret
    _sincosf.restype, _sincosf.argtypes = _SinCos, [ctypes.c_float]

    def _log(x):
        return np.array([_libm.logf(v) for v in x.reshape(-1).tolist()], np.float32).reshape(x.shape)

    def _sincos(x):
        pairs = [_sincosf(v) for v in x.reshape(-1).tolist()]
        s = np.array([p.s for p in pairs], np.float32).reshape(x.shape)
        c = np.array([p.c for p in pairs], np.float32).reshape(x.shape)
        return s, c
else:  # not bit-exact off macOS: correctly rounded values, which agree with glibc in the vast majority of cases
    def _log(x):
        return np.log(x.astype(np.float64)).astype(np.float32)

    def _sincos(x):
        x64 = x.astype(np.float64)
        return np.sin(x64).astype(np.float32), np.cos(x64).astype(np.float32)


def _uniform24(rng: np.random.RandomState, n: int) -> np.ndarray:
    raw = rng.randint(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    return (raw & np.uint32((1 << 24) - 1)).astype(np.float32) * np.float32(1.0 / (1 << 24))


def _box_muller16(u: np.ndarray) -> np.ndarray:
    u1 = (np.float32(1.0) - u[..., :8]).astype(np.float32)
    u2 = u[..., 8:]
    radius = np.sqrt((np.float32(-2.0) * _log(u1)).astype(np.float32)).astype(np.float32)
    theta = (_TWO_PI * u2.astype(np.float64)).astype(np.float32)
    sin, cos = _sincos(theta)
    return np.concatenate([radius * cos, radius * sin], axis=-1).astype(np.float32)


def randn(shape, seed: int) -> np.ndarray:
    n = int(np.prod(shape))
    if n < 16:
        raise ValueError("torch uses a different (non Box-Muller-16) path below 16 elements")
    rng = np.random.RandomState(seed)  # MT19937 seeded like at::mt19937(seed)
    u = _uniform24(rng, n)
    out = np.empty(n, np.float32)
    full = (n // 16) * 16
    out[:full] = _box_muller16(u[:full].reshape(-1, 16)).reshape(-1)
    if n % 16:
        tail = _uniform24(rng, 16)  # torch re-draws 16 fresh uniforms for the last 16 slots
        out[n - 16:] = _box_muller16(tail.reshape(1, 16)).reshape(-1)
    return out.reshape(shape)


class Generator:
    """A torch.Generator("cpu").manual_seed(seed) whose successive randn draws share one MT19937 stream, as the
    edit pipeline's posterior samples, target noise and reference noise do (in that order)."""

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def randn(self, shape) -> np.ndarray:
        n = int(np.prod(shape))
        if n < 16:
            raise ValueError("torch uses a different (non Box-Muller-16) path below 16 elements")
        u = _uniform24(self.rng, n)
        out = np.empty(n, np.float32)
        full = (n // 16) * 16
        out[:full] = _box_muller16(u[:full].reshape(-1, 16)).reshape(-1)
        if n % 16:
            out[n - 16:] = _box_muller16(_uniform24(self.rng, 16).reshape(1, 16)).reshape(-1)
        return out.reshape(shape)
