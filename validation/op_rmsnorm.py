import numpy as np, torch, mlx.core as mx
from safetensors.torch import load_file
w = load_file("weights/bf16/dit/blocks/00.safetensors")["transformer_blocks.0.norm1.weight"]
g = torch.Generator().manual_seed(0)
x = (torch.randn(4096, 5376, generator=g) * torch.rand(4096, 1, generator=g) * 60).to(torch.bfloat16)
x64, w64 = x.double(), w.double()
exact = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-5) * w64
ref = exact.float().to(torch.bfloat16)
rms = torch.nn.RMSNorm(5376, eps=1e-5); rms.weight.data = w.clone()
outs = {"torch cpu": rms.to(torch.bfloat16)(x), "torch mps": rms.to("mps", torch.bfloat16)(x.to("mps")).cpu()}
xm, wm = mx.array(x.float().numpy()).astype(mx.bfloat16), mx.array(w.float().numpy()).astype(mx.bfloat16)
xf = xm.astype(mx.float32)
variants = {
    "mlx mean(x^2)": xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + 1e-5) * wm.astype(mx.float32),
    "mlx sum/n": xf * mx.rsqrt(mx.sum(mx.square(xf), axis=-1, keepdims=True) / 5376 + 1e-5) * wm.astype(mx.float32),
    "mlx f64-like (2-pass split)": None,
}
# two-level summation: sum 42 chunks of 128, then sum chunks (pairwise-ish)
sq = mx.square(xf).reshape(4096, 42, 128)
variants["mlx chunked sum"] = xf * mx.rsqrt(mx.sum(mx.sum(sq, axis=-1), axis=-1, keepdims=True) / 5376 + 1e-5) * wm.astype(mx.float32)
variants["mlx fast.rms_norm(no w)*w"] = mx.fast.rms_norm(xf, None, 1e-5) * wm.astype(mx.float32)
for k, v in variants.items():
    if v is not None:
        outs[k] = torch.from_numpy(np.array(v)).to(torch.bfloat16)
for k, v in outs.items():
    bad = (v.view(torch.int16) != ref.view(torch.int16)).float().mean().item() * 100
    print(f"{k:28s} not correctly rounded {bad:.4f}%")
# rstd accuracy directly
r_exact = torch.rsqrt(x64.pow(2).mean(-1) + 1e-5)
r_mlx = torch.from_numpy(np.array(mx.rsqrt(mx.mean(mx.square(xf), axis=-1) + 1e-5))).double()
print("rstd rel err mlx mean:", float(((r_mlx - r_exact) / r_exact).abs().mean()))
