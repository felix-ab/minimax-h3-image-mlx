import json, sys, time
sys.path.insert(0, "repo")
import numpy as np, mlx.core as mx
from minimax_h3_mlx.vae import VAE
cfg = json.load(open("upstream/vae_diffusers/config.json"))
vae = VAE.load("weights/bf16/vae/diffusion_pytorch_model.safetensors", cfg)
lat = mx.array(np.load("parity/vae_in.npy"))
t = time.time(); out = vae.decode(lat); mx.eval(out); print("mlx decode", out.shape, f"{time.time()-t:.1f}s")
ref = np.load("parity/vae_ref_out.npy"); o = np.array(out)
d = np.abs(o - ref); rng = ref.max() - ref.min()
mse = float(np.mean((o - ref) ** 2)); psnr = 10 * np.log10(rng ** 2 / mse) if mse else float("inf")
print(f"max|d| {d.max():.3e} mean|d| {d.mean():.3e} PSNR(range) {psnr:.1f} dB  ref range {rng:.2f}")
