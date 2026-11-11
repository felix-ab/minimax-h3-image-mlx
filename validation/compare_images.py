"""Compare images: PSNR, SSIM, mean |d|, share of identical 8-bit values, plus a high-frequency energy ratio
(grain check: >1 means the first image carries more fine-scale energy than the second).
usage: compare_images.py A.png B.png [--resize-to-b] [--crop x0,y0,x1,y1 (applied to B)] [--json out.json]"""
import argparse, json
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def hf_energy(x):
    g = x.mean(-1)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.mean(lap ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--resize-to-b", action="store_true")
    p.add_argument("--crop")
    p.add_argument("--json")
    a = p.parse_args()
    A = Image.open(a.a).convert("RGB"); B = Image.open(a.b).convert("RGB")
    if a.crop:
        B = B.crop(tuple(int(v) for v in a.crop.split(",")))
    if a.resize_to_b and A.size != B.size:
        A = A.resize(B.size, Image.LANCZOS)
    x = np.asarray(A).astype(np.float64); y = np.asarray(B).astype(np.float64)
    out = {"size": A.size, "psnr_db": float(peak_signal_noise_ratio(y, x, data_range=255)),
           "ssim": float(structural_similarity(y, x, channel_axis=-1, data_range=255)),
           "mean_abs": float(np.abs(x - y).mean()), "identical_frac": float((x == y).mean()),
           "hf_ratio_a_over_b": hf_energy(x) / max(hf_energy(y), 1e-12)}
    print(json.dumps(out))
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
