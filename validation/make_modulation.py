#!/usr/bin/env python3
"""Precompute MiniMax-H3-Image's AdaLN modulation for every timestep the released sampler can ever feed it.

Why this is exact: each block's `adaln_proj` maps the shared timestep embedding (and nothing else) to six
modulation vectors per (timestep, modality). The checkpoint only ever runs at the 6 distilled timesteps
`t = 1 - sigma` plus `t = 0.999` for edit reference rows, so the 13.0 B AdaLN parameters (26 GB bf16, 39% of the
DiT) collapse to a [7, 2, 6, 5376] bf16 table per block. Audio (modality 2) is never addressed by the image model.

Numerics, matching the reference forward:
  temb = time_embedder(time_proj(t))                  float32, computed with diffusers' own modules on CPU
  x    = silu(temb).to(bfloat16)                      as MiniMaxH3ImageAdaLayerNormModulation does
  y    = x @ W.T + b  (bf16 W, b)                     reference: one fp32-accumulated GEMM, rounded once to bf16
We evaluate y in float64 and round once to bf16 (round-to-nearest-even), i.e. the correctly rounded value of the
reference GEMM. A different fp32 accumulation order can only differ where the exact value sits within fp32 error
of a bf16 rounding boundary; we count those cases against torch's own CPU bf16 linear and log them.

Consumes dit/adaln/NN.safetensors as fetch.py writes them, writes dit/modulation/NN.safetensors, then deletes
the temporary adaln file (blocks in KEEP are moved to dit/adaln_keep/ for the parity tests).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

ROOT = Path(sys.argv[1])
UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"
KEEP = {0, 25, 49}
HIDDEN = 5376
NUM_LAYERS = 50


def timesteps():
    sigmas = json.loads((UPSTREAM / "modular_model_index.json").read_text())["sample_sigmas"]
    # before_denoise.MiniMaxH3ImageSetTimestepsStep: float32(1 - sigma) per step; edit reference rows at 0.999.
    steps = torch.tensor([1.0 - s for s in sigmas[:-1]], dtype=torch.float32)
    return torch.cat([steps, torch.tensor([0.999], dtype=torch.float32)])


def bf16_round(x64: np.ndarray) -> torch.Tensor:
    # float64 -> float32 (exact enough: then RNE to bf16). Double rounding f64->f32->bf16 can only differ from a
    # direct f64->bf16 rounding when the f32 value lands exactly on a bf16 tie; handle it by rounding from f64.
    x32 = x64.astype(np.float32)
    bits = x32.view(np.uint32).astype(np.uint64)
    lower = (bits & 0xFFFF0000).astype(np.uint32).view(np.float32).astype(np.float64)
    upper = ((bits & 0xFFFF0000) + 0x10000).astype(np.uint32).view(np.float32).astype(np.float64)
    neg = x64 < 0
    # for negative numbers the "upper" bit pattern is further from zero, same logic holds on magnitudes
    d_low = np.abs(x64 - lower)
    d_up = np.abs(x64 - upper)
    pick_up = (d_up < d_low) | ((d_up == d_low) & (((bits >> 16) & 1) == 1))
    out = np.where(pick_up, (bits & 0xFFFF0000) + 0x10000, bits & 0xFFFF0000).astype(np.uint32)
    del neg
    return torch.from_numpy((out >> 16).astype(np.uint16).view(np.int16)).view(torch.bfloat16)


def wait_for(path: Path):
    while not path.exists():
        time.sleep(5)


def main():
    wait_for(ROOT / "dit/core.json")
    core = load_file(ROOT / "dit/core.safetensors")
    proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    emb = TimestepEmbedding(in_channels=256, time_embed_dim=HIDDEN, out_dim=2688)
    emb.load_state_dict({k[len("time_embedder."):]: v for k, v in core.items() if k.startswith("time_embedder.")})
    ts = timesteps()
    with torch.no_grad():
        temb = emb(proj(ts).to(torch.float32))
        x = torch.nn.functional.silu(temb).to(torch.bfloat16)
    x64 = x.to(torch.float64).numpy()
    out = ROOT / "dit/modulation"
    out.mkdir(parents=True, exist_ok=True)
    save_file({"timesteps": ts, "temb": temb, "silu_temb_bf16": x}, out / "temb.safetensors")

    # norm_out: shift, scale per timestep (no modality axis)
    w = core["norm_out.linear.weight"].to(torch.float64).numpy()
    b = core["norm_out.linear.bias"].to(torch.float64).numpy()
    table = bf16_round(x64 @ w.T + b).reshape(len(ts), 2, HIDDEN)
    save_file({"table": table.contiguous(), "timesteps": ts}, out / "norm_out.safetensors")

    log = []
    for index in range(NUM_LAYERS):
        dst = out / f"{index:02d}.safetensors"
        src = ROOT / f"dit/adaln/{index:02d}.safetensors"
        side = ROOT / f"dit/adaln/{index:02d}.json"
        if dst.exists():
            continue
        wait_for(side)
        tensors = load_file(src)
        w_bf16 = tensors[f"transformer_blocks.{index}.adaln_proj.linear.weight"]
        b_bf16 = tensors[f"transformer_blocks.{index}.adaln_proj.linear.bias"]
        y64 = x64 @ w_bf16.to(torch.float64).numpy().T + b_bf16.to(torch.float64).numpy()
        y = bf16_round(y64)  # (7, 96768) = (7, 3 modalities, 6, 5376) per the reference view(-1, 6*hidden)
        with torch.no_grad():
            y_torch = torch.nn.functional.linear(x, w_bf16, b_bf16)  # torch CPU bf16 GEMM, for the log only
        mismatch = int((y.view(torch.int16) != y_torch.view(torch.int16)).sum())
        table = y.reshape(len(ts), 3, 6, HIDDEN)[:, :2].contiguous()  # drop audio rows (never addressed)
        save_file({"table": table, "timesteps": ts}, dst)
        log.append({"block": index, "bf16_mismatch_vs_torch_cpu": mismatch, "of": int(y.numel())})
        print(json.dumps(log[-1]), flush=True)
        if index in KEEP:
            (ROOT / "dit/adaln_keep").mkdir(exist_ok=True)
            src.rename(ROOT / f"dit/adaln_keep/{index:02d}.safetensors")
            side.rename(ROOT / f"dit/adaln_keep/{index:02d}.json")
        else:
            src.unlink()
            side.rename(side.with_suffix(".consumed.json"))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print("modulation done", flush=True)


if __name__ == "__main__":
    main()
