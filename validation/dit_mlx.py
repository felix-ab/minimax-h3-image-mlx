"""MLX DiT forward on the same layout; compares against one or more reference dumps.
usage: dit_mlx.py DTYPE(float32|bfloat16) OUT.npz REF.npz [REF2.npz ...] [--step N]"""
import sys, time
sys.path.insert(0, "repo")
import numpy as np, mlx.core as mx
from minimax_h3_mlx import layout as L
from minimax_h3_mlx.dit import DiT, rope_tables
args = [a for a in sys.argv[1:] if not a.startswith("--step")]
step = int([a for a in sys.argv[1:] if a.startswith("--step")][0].split("=")[1]) if any(a.startswith("--step") for a in sys.argv[1:]) else 0
dtype = getattr(mx, args[0]); out = args[1]; refs = args[2:]
d = np.load("parity/dit_inputs.npz"); prompt, noise = d["prompt"], d["noise"]
n_text, lat = prompt.shape[0], noise.shape[-1]
lay = L.Layout(n_text=n_text, text_tags=np.full(n_text, L.TEXT_TAG), latent_h=lat, latent_w=lat)
t, sig = L.sampling_grid()
dit = DiT("weights/bf16/dit", stream=True, dtype=dtype)
dumps = {}
def on_block(i, h):
    if i in (0, 1, 12, 25, 37, 48, 49):
        dumps[f"block{i:02d}"] = np.array(h.astype(mx.float32))
t0 = time.time()
text = dit.embed_text(mx.array(prompt))
runs = lay.runs(dit.t_index(float(t[step])), dit.t_index(L.REFERENCE_T))
v = dit(mx.array(L.patchify(noise.reshape(24, lat, lat))), text, runs, rope_tables(lay.position_ids(), dtype),
        step=step, target=slice(lay.target_start, lay.length), on_block=on_block, n_blocks=int(__import__("os").environ.get("NB", 50)))
mx.eval(v)
print(f"mlx forward {time.time()-t0:.1f}s")
mine = dict(v=np.array(v), text=np.array(text.astype(mx.float32)), **dumps)
np.savez(out, **mine)
def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
for r in refs:
    ref = np.load(r)
    print("vs", r)
    for k in ["text"] + sorted(k for k in ref.files if k.startswith("block")) + ["v"]:
        if k in ref.files and k in mine:
            print(f"  {k:8s} rel {rel(mine[k], ref[k]):.3e}  max|d| {np.abs(mine[k]-ref[k]).max():.3e}")
