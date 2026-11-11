"""Resolve `--weights`: a local tier directory, or a tier name downloaded from the Hugging Face Hub.

Tiers live at VisualInference/MiniMax-H3-Image-MLX-<tier>. The download runs in a child process: huggingface_hub's
Xet transfer buffers would otherwise stay in this process and add GBs to the generation's peak footprint.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TIERS = {"bf16": "VisualInference/MiniMax-H3-Image-MLX-bf16", "8bit": "VisualInference/MiniMax-H3-Image-MLX-8bit"}
REQUIRED = ("config.json", "dit/core.safetensors", "dit/modulation/norm_out.safetensors", "te/embed.safetensors",
            "vae/diffusion_pytorch_model.safetensors", "tokenizer/tokenizer.json")


def is_tier_dir(path: Path) -> bool:
    return all((path / f).exists() for f in REQUIRED)


def resolve(weights: str, revision: str | None = None) -> Path:
    path = Path(weights).expanduser()
    if path.exists():
        if not is_tier_dir(path):
            missing = [f for f in REQUIRED if not (path / f).exists()]
            raise SystemExit(f"{path} is not a MiniMax-H3-Image MLX tier (missing {', '.join(missing)})")
        return path
    repo = TIERS.get(weights, weights if "/" in weights else None)
    if repo is None:
        raise SystemExit(f"--weights must be a directory, one of {sorted(TIERS)}, or a Hub repo id")
    code = ("import sys; from huggingface_hub import snapshot_download; "
            "print(snapshot_download(sys.argv[1], revision=sys.argv[2] or None))")
    out = subprocess.run([sys.executable, "-c", code, repo, revision or ""], check=True, capture_output=True, text=True)
    return Path(out.stdout.strip().splitlines()[-1])
