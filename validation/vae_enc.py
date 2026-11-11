"""VAE encoder parity: torch fp32 (diffusers, tiled) vs MLX fp32 on examples/reference1.png cropped to 640x896."""
import sys, json, numpy as np
from PIL import Image
mode = sys.argv[1]
im = np.asarray(Image.open("upstream/examples/reference1.png").convert("RGB"))[:640, :896].astype(np.float32) / 255
mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
x = ((im - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
if mode == "ref":
    import torch
    from safetensors.torch import load_file
    from diffusers import AutoencoderKLMiniMaxH3Image
    cfg = {k: v for k, v in json.load(open("upstream/vae_diffusers/config.json")).items() if not k.startswith("_")}
    vae = AutoencoderKLMiniMaxH3Image(**cfg); vae.load_state_dict(load_file("weights/bf16/vae/diffusion_pytorch_model.safetensors")); vae.eval()
    with torch.no_grad():
        post = vae.encode(torch.from_numpy(x), return_dict=False)[0]
    np.save("parity/vae_enc_ref.npy", post.parameters.numpy())
else:
    sys.path.insert(0, "repo")
    import mlx.core as mx
    from minimax_h3_mlx.vae import VAE
    vae = VAE.load("weights/bf16/vae/diffusion_pytorch_model.safetensors", json.load(open("upstream/vae_diffusers/config.json")))
    m = np.array(vae.encode_moments(mx.array(x))); ref = np.load("parity/vae_enc_ref.npy")
    print("shape", m.shape, ref.shape, "rel", np.linalg.norm(m - ref) / np.linalg.norm(ref), "max|d|", np.abs(m - ref).max())
