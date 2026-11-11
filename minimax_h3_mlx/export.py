"""Build a release tier from the local bf16 weights: optionally quantize DiT blocks / text-encoder layers.

    python -m minimax_h3_mlx.export SRC_DIR OUT_DIR --dit-bits 8 --te-bits 8 [--group-size 64]

bits=16 keeps a component bf16. Only the large matmuls are quantized (DiT block q/k/v/o and FFN, text-encoder
attention and MLP projections). Norm weights, the AdaLN tables, the fp32 input/timestep/output layers, the token
refiner, the embedding table, the vision tower and the VAE always stay at their release precision.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

DIT_LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
TE_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def quantize_file(src: Path, dst: Path, linears, bits: int, group_size: int):
    params = mx.load(str(src))
    out = {}
    for name, value in params.items():
        stem = name[: -len(".weight")] if name.endswith(".weight") else None
        if bits < 16 and stem and any(stem.endswith(l) for l in linears):
            w, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
            out[f"{stem}.weight"], out[f"{stem}.scales"], out[f"{stem}.biases"] = w, scales, biases
        else:
            out[name] = value
    mx.eval(out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out, metadata={"format": "mlx"})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--dit-bits", type=int, default=8)
    p.add_argument("--te-bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    a = p.parse_args(argv)
    src, out = a.src, a.out
    out.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        quantize_file(src / f"dit/blocks/{i:02d}.safetensors", out / f"dit/blocks/{i:02d}.safetensors",
                      DIT_LINEARS, a.dit_bits, a.group_size)
        quantize_file(src / f"te/layers/{i:02d}.safetensors", out / f"te/layers/{i:02d}.safetensors",
                      TE_LINEARS, a.te_bits, a.group_size)
        print(f"block/layer {i:02d}", flush=True)
    for rel in ("dit/core.safetensors", "dit/refiner.safetensors", "te/embed.safetensors", "te/visual.safetensors",
                "vae/diffusion_pytorch_model.safetensors", "vae/config.json"):
        if (src / rel).exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, out / rel)
    shutil.copytree(src / "dit/modulation", out / "dit/modulation", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("log.json", "temb.safetensors"))
    shutil.copytree(src / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    quant = lambda bits: ({"quantized": True, "bits": bits, "group_size": a.group_size, "mode": "affine"}
                          if bits < 16 else {"quantized": False})
    (out / "dit/dit.json").write_text(json.dumps(quant(a.dit_bits), indent=1))
    (out / "te/te.json").write_text(json.dumps(quant(a.te_bits), indent=1))


if __name__ == "__main__":
    main()
