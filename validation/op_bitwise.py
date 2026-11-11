"""Feed identical bf16 inputs to each DiT op in torch (MPS and CPU) and MLX; report bitwise agreement."""
import sys
sys.path.insert(0, "repo")
import numpy as np, torch, mlx.core as mx
from safetensors.torch import load_file
from minimax_h3_mlx.ops import rms_norm_torch, silu
from minimax_h3_mlx.dit import apply_rope, rope_tables
from diffusers.models.transformers.transformer_minimax_h3_image import _apply_rotary_emb, MiniMaxH3ImageRotaryPosEmbed
from minimax_h3_mlx import layout as L
sd = load_file("weights/bf16/dit/blocks/00.safetensors")
g = torch.Generator().manual_seed(0)
N = 448
x = (torch.randn(N, 5376, generator=g) * 20).to(torch.bfloat16)
w = sd["transformer_blocks.0.norm1.weight"]
def to_mx(t): return mx.array(t.float().numpy()).astype(mx.bfloat16)
def from_mx(a): return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)
def report(name, outs):
    base = outs["mlx"]
    for k, v in outs.items():
        if k != "mlx":
            print(f"{name:22s} mlx vs {k:9s}: differing {((v.view(torch.int16) != base.view(torch.int16)).float().mean().item()*100):7.3f}%")
# 1 RMSNorm
rms = torch.nn.RMSNorm(5376, eps=1e-5); rms.weight.data = w.clone()
outs = {"mlx": from_mx(rms_norm_torch(to_mx(x), to_mx(w), 1e-5))}
for dev in ("cpu", "mps"):
    outs[f"torch {dev}"] = rms.to(dev, torch.bfloat16)(x.to(dev)).cpu()
report("rmsnorm", outs)
# 2 modulation
s = (torch.randn(5376, generator=g) * 0.3).to(torch.bfloat16); sh = (torch.randn(5376, generator=g) * 0.3).to(torch.bfloat16)
outs = {"mlx": from_mx(to_mx(x) * (1.0 + to_mx(s)) + to_mx(sh))}
for dev in ("cpu", "mps"):
    outs[f"torch {dev}"] = (x.to(dev) * (1.0 + s.to(dev)) + sh.to(dev)).cpu()
report("modulate", outs)
# 3 swiglu
h = (torch.randn(N, 2 * 14336, generator=g) * 3).to(torch.bfloat16)
val, gate = h.chunk(2, dim=-1)
outs = {"mlx": from_mx(to_mx(val) * silu(to_mx(gate)))}
for dev in ("cpu", "mps"):
    outs[f"torch {dev}"] = (val.to(dev) * torch.nn.SiLU()(gate.to(dev))).cpu()
report("value*silu(gate)", outs)
# 4 rope on (N, 56, 128)
lay = L.Layout(n_text=64, text_tags=np.full(64, 1), latent_h=32, latent_w=24)
pos = torch.from_numpy(lay.position_ids())
q = (torch.randn(lay.length, 56, 128, generator=g) * 2).to(torch.bfloat16)
cos_m, sin_m = rope_tables(lay.position_ids())
outs = {"mlx": from_mx(apply_rope(to_mx(q), cos_m, sin_m))}
rope = MiniMaxH3ImageRotaryPosEmbed(16, 10000.0)
for dev in ("cpu", "mps"):
    p = pos.to(dev) if dev == "cpu" else pos.float().to(dev)
    c, s_ = rope.to(dev)(p)
    outs[f"torch {dev}"] = _apply_rotary_emb(q[None].to(dev), c, s_)[0].cpu()
report("rope", outs)
c, s_ = rope.to("cpu")(pos)
print("   rope tables: cos bf16 differing", float((c.to(torch.bfloat16).view(torch.int16) != from_mx(cos_m).view(torch.int16)).float().mean())*100, "%")
# 5 sdpa
qq = (torch.randn(1, 56, lay.length, 128, generator=g) * 3).to(torch.bfloat16)
kk = (torch.randn(1, 56, lay.length, 128, generator=g) * 3).to(torch.bfloat16)
vv = (torch.randn(1, 56, lay.length, 128, generator=g)).to(torch.bfloat16)
outs = {"mlx": from_mx(mx.fast.scaled_dot_product_attention(to_mx(qq), to_mx(kk), to_mx(vv), scale=128 ** -0.5))}
for dev in ("cpu", "mps"):
    outs[f"torch {dev}"] = torch.nn.functional.scaled_dot_product_attention(qq.to(dev), kk.to(dev), vv.to(dev)).cpu()
exact = torch.nn.functional.scaled_dot_product_attention(qq.double(), kk.double(), vv.double()).to(torch.bfloat16)
outs["exact"] = exact
report("sdpa", outs)
for k in ("torch cpu", "torch mps", "mlx"):
    print(f"   sdpa {k:9s} vs exact differing {((outs[k].view(torch.int16) != exact.view(torch.int16)).float().mean().item()*100):.3f}%")
