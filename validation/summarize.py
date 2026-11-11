"""Recompute every parity number from the saved dumps into parity/summary.json (cards and docs read only this file).

rel(a, b) = ||a - b|| / ||b||. "truth" = the torch float32 reference on CPU. Missing inputs are skipped, not guessed.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "parity"
out = {}


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b.astype(np.float64)))


def load(name):
    f = P / name
    return np.load(f) if f.exists() else None


def image_cmp(a, b):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/compare_images.py"), str(a), str(b)], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout else None


# DiT, 50 blocks, small packed layout (40 text rows + 256 image rows), steps 0 and 4
dit = {}
for s in (0, 4):
    f32, cpu, mps = load(f"full_ref_f32_s{s}.npz"), load(f"full_ref_bf16cpu_s{s}.npz"), load(f"full_ref_bf16mps_s{s}.npz")
    m32, m16 = load(f"full_mlx_f32_s{s}.npz"), load(f"full_mlx_bf16_s{s}.npz")
    if f32 is None or m32 is None:
        continue
    dit[f"step{s}"] = {
        "fp32_velocity_rel": rel(m32["v"], f32["v"]), "fp32_block49_rel": rel(m32["block49"], f32["block49"]),
        "bf16_vs_truth": {k: rel(v["v"], f32["v"]) for k, v in (("torch_cpu", cpu), ("torch_mps", mps), ("mlx", m16))},
        "bf16_pairwise": {"cpu_mps": rel(cpu["v"], mps["v"]), "mlx_cpu": rel(m16["v"], cpu["v"]), "mlx_mps": rel(m16["v"], mps["v"])},
    }
out["dit_50_blocks"] = dit

# text encoder, 19 layers (implementation check) and 50 layers (from the end-to-end runs)
f32, m32 = load("te19_ref_f32.npz"), load("te19_mlx_f32.npz")
cpu, mps, m16 = load("te19_ref_bf16cpu.npz"), load("te19_ref_bf16mps.npz"), load("te19_mlx_bf16.npz")
if f32 is not None and m32 is not None:
    out["text_encoder_19_layers"] = {
        "fp32_rel": rel(m32["h50"], f32["h50"]),
        "bf16_vs_truth": {k: rel(v["h50"], f32["h50"]) for k, v in (("torch_cpu", cpu), ("torch_mps", mps), ("mlx", m16))},
    }
e2e = P / "e2e"
if (e2e / "mlx_bf16_prompt_embeds.npy").exists():
    t = np.load(e2e / "ref_f32.npz")["prompt_embeds"]
    out["text_encoder_50_layers_bf16_vs_truth"] = {
        "torch_cpu": rel(np.load(e2e / "ref_bf16cpu.npz")["prompt_embeds"], t),
        "torch_mps": rel(np.load(e2e / "ref_bf16mps.npz")["prompt_embeds"], t),
        "mlx": rel(np.load(e2e / "mlx_bf16_prompt_embeds.npy"), t)}

# end to end, 512x384 canvas, 6 steps
cases = {}
for tag, pre in (("t2i_seed42", ""), ("t2i_seed1", "t2i_s1_"), ("t2i_seed2", "t2i_s2_"), ("portrait_seed7", "portrait_s7_")):
    need = [e2e / f"{pre}{k}.npz" for k in ("ref_f32", "ref_bf16mps", "mlx_bf16")]
    if not all(f.exists() for f in need):
        continue
    f32, mps, mlx = (np.load(f)["steps"] for f in need)
    c = {"latent_final_vs_truth": {"torch_mps": rel(mps[-1], f32[-1]), "mlx": rel(mlx[-1], f32[-1])},
         "latent_final_mlx_vs_mps": rel(mlx[-1], mps[-1]),
         "psnr_db": {"torch_mps_vs_truth": image_cmp(e2e / f"{pre}ref_bf16mps.png", e2e / f"{pre}ref_f32.png")["psnr_db"],
                     "mlx_vs_truth": image_cmp(e2e / f"{pre}mlx_bf16.png", e2e / f"{pre}ref_f32.png")["psnr_db"],
                     "mlx_vs_torch_mps": image_cmp(e2e / f"{pre}mlx_bf16.png", e2e / f"{pre}ref_bf16mps.png")["psnr_db"]}}
    if (e2e / f"{pre}mlx_f32.npz").exists():
        c["fp32_latent_final_rel"] = rel(np.load(e2e / f"{pre}mlx_f32.npz")["steps"][-1], f32[-1])
        c["fp32_psnr_db"] = image_cmp(e2e / f"{pre}mlx_f32.png", e2e / f"{pre}ref_f32.png")["psnr_db"]
    if (e2e / f"{pre}ref_bf16cpu.npz").exists():
        cpu = np.load(e2e / f"{pre}ref_bf16cpu.npz")["steps"]
        c["latent_final_vs_truth"]["torch_cpu"] = rel(cpu[-1], f32[-1])
        c["psnr_db"]["torch_cpu_vs_truth"] = image_cmp(e2e / f"{pre}ref_bf16cpu.png", e2e / f"{pre}ref_f32.png")["psnr_db"]
    cases[tag] = c
out["end_to_end_512x384"] = cases

ed = P / "edit"
if (ed / "ref_f32.png").exists() and (ed / "mlx_f32.png").exists():
    out["edit_512x512_reference1"] = {
        "fp32": image_cmp(ed / "mlx_f32.png", ed / "ref_f32.png"),
        "psnr_db": {k: image_cmp(ed / f"{a}.png", ed / f"{b}.png")["psnr_db"] for k, a, b in (
            ("mlx_bf16_vs_truth", "mlx_bf16", "ref_f32"), ("torch_mps_vs_truth", "ref_bf16mps", "ref_f32"),
            ("mlx_bf16_vs_torch_mps", "mlx_bf16", "ref_bf16mps")) if (ed / f"{a}.png").exists() and (ed / f"{b}.png").exists()}}

official = ROOT / "out/official/mlx_bf16_vs_official.json"
if official.exists():
    out["official_seed42_4736x3552_vs_sglang_panel"] = json.loads(official.read_text())
(P / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
