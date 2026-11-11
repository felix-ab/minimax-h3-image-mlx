"""Torch reference VAE decode (diffusers AutoencoderKLMiniMaxH3Image, CPU float32, tiling on as released)."""
import json, sys, time
import numpy as np, torch
from safetensors.torch import load_file
from diffusers import AutoencoderKLMiniMaxH3Image
W = "weights/bf16/vae/diffusion_pytorch_model.safetensors"
cfg = json.load(open("upstream/vae_diffusers/config.json"))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
vae = AutoencoderKLMiniMaxH3Image(**cfg)
missing, unexpected = vae.load_state_dict(load_file(W), strict=False)
print("missing", missing, "unexpected", unexpected)
vae.eval()
h, w = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0)
lat = rng.standard_normal((1, 24, h, w)).astype(np.float32)
np.save("parity/vae_in.npy", lat)
mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1); std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1)
t = time.time()
with torch.no_grad():
    out = vae.decode(torch.from_numpy(lat) * std + mean, return_dict=False)[0]
print("decode", out.shape, f"{time.time()-t:.1f}s")
np.save("parity/vae_ref_out.npy", out.numpy())
