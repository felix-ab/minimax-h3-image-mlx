import numpy as np, torch, mlx.core as mx
from safetensors.torch import load_file
sd = load_file("weights/bf16/dit/blocks/00.safetensors")
g = torch.Generator().manual_seed(0)
for M in (296, 518, 4400):
    for name in ("attn.to_q", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"):
        w = sd[f"transformer_blocks.0.{name}.weight"]
        x = (torch.randn(M, w.shape[1], generator=g) * 2).to(torch.bfloat16)
        exact = x.double() @ w.double().T
        ref = exact.float().to(torch.bfloat16)
        mps = (x.to("mps") @ w.to("mps").T).cpu()
        m = mx.array(x.float().numpy()).astype(mx.bfloat16) @ mx.array(w.float().numpy()).astype(mx.bfloat16).T
        m = torch.from_numpy(np.array(m.astype(mx.float32))).to(torch.bfloat16)
        nc = lambda v: (v.view(torch.int16) != ref.view(torch.int16)).float().mean().item() * 100
        print(f"M={M:5d} {name:14s} K={w.shape[1]:5d} N={w.shape[0]:5d}  not-correctly-rounded: mps {nc(mps):.3f}%  mlx {nc(m):.3f}%  mlx!=mps {(m.view(torch.int16)!=mps.view(torch.int16)).float().mean().item()*100:.3f}%")
