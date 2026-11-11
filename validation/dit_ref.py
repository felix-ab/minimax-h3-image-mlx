"""Torch reference DiT forward on a small packed layout; dumps inputs, per-block hidden states, velocity.
usage: dit_ref.py OUT.npz DTYPE(float32|bfloat16) [DEVICE] [STEP] [N_TEXT] [LATENT]"""
import sys, time
sys.path.insert(0, "repo"); sys.path.insert(0, "ref")
import numpy as np, torch
from reference import StreamedDiT
from minimax_h3_mlx import layout as L
from minimax_h3_mlx.torch_randn import randn
out, dtype = sys.argv[1], getattr(torch, sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
step = int(sys.argv[4]) if len(sys.argv) > 4 else 0
n_text = int(sys.argv[5]) if len(sys.argv) > 5 else 40
lat = int(sys.argv[6]) if len(sys.argv) > 6 else 32
inp = "parity/dit_inputs.npz"
try:
    d = np.load(inp); assert d["prompt"].shape[0] == n_text and d["noise"].shape[-1] == lat
    prompt, noise = d["prompt"], d["noise"]
except Exception:
    rng = np.random.default_rng(0)
    prompt = (rng.standard_normal((n_text, 5120)) * 2.0).astype(np.float32)  # synthetic conditioning
    prompt = torch.from_numpy(prompt).to(torch.bfloat16).float().numpy()      # bf16-representable, like TE output
    noise = randn((1, 24, 1, lat, lat), 0)
    np.savez(inp, prompt=prompt, noise=noise)
lay = L.Layout(n_text=n_text, text_tags=np.full(n_text, L.TEXT_TAG), latent_h=lat, latent_w=lat)
t, sig = L.sampling_grid()
rows = L.patchify(noise.reshape(24, lat, lat))
seq = lay.length
model = StreamedDiT(dtype=dtype, device=device, real_adaln=bool(int(__import__("os").environ.get("REAL_ADALN", "0"))))
ts = torch.tensor([t[step]], dtype=torch.float32)
dumps = {}
def dump(i, h):
    if i in (0, 1, 12, 25, 37, 48, 49):
        dumps[f"block{i:02d}"] = h[0].float().cpu().numpy()
pos = torch.from_numpy(lay.position_ids())
if device == "mps": pos = pos.float()
t0 = time.time()
with torch.no_grad():
    text_embeds = model.text(torch.from_numpy(prompt)[None])
    v = model.forward(torch.from_numpy(rows)[None], None, ts, torch.zeros(seq, dtype=torch.long),
                      torch.from_numpy(lay.tags()), pos, torch.arange(n_text, seq), torch.arange(n_text),
                      step, dump=dump, text_embeds=text_embeds, blocks=range(int(__import__("os").environ.get("NB", 50))))
print(f"ref forward {time.time()-t0:.0f}s", v.shape)
np.savez(out, v=v[0].float().cpu().numpy(), text=text_embeds[0].float().cpu().numpy(), **dumps)
