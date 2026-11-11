"""MLX text encoder vs references. usage: te_mlx.py DTYPE OUT.npz [PROMPT_FILE] REF.npz [REF2 ...]"""
import sys, time
sys.path.insert(0, "repo")
import numpy as np, mlx.core as mx
from minimax_h3_mlx.text_encoder import TextEncoder
dtype, out = getattr(mx, sys.argv[1]), sys.argv[2]
rest = sys.argv[3:]
prompt_file = rest[0] if rest and rest[0].endswith(".txt") else "upstream/examples/revised_prompt.t2i.txt"
refs = [r for r in rest if r.endswith(".npz")]
te = TextEncoder("weights/bf16/te", "weights/bf16/tokenizer", dtype=dtype)
ids = te.tokenize(open(prompt_file).read())
dumps = {}
def on_layer(i, h):
    if i in (0, 1, 9, 24, 39, 48):
        dumps[f"layer{i:02d}"] = np.array(h.astype(mx.float32))
t = time.time(); h = te(ids, on_layer=on_layer, n_layers=int(__import__('os').environ.get('NL', 50))); mx.eval(h); print(f"mlx TE {time.time()-t:.1f}s, {len(ids)} tokens")
mine = dict(ids=np.array(ids), h50=np.array(h.astype(mx.float32)), **dumps)
np.savez(out, **mine)
def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
for r in refs:
    ref = np.load(r); print("vs", r, "ids equal:", bool((ref["ids"] == mine["ids"]).all()))
    for k in sorted(k for k in ref.files if k.startswith("layer")) + ["h50"]:
        print(f"  {k:8s} rel {rel(mine[k], ref[k]):.3e}  max|d| {np.abs(mine[k]-ref[k]).max():.3e}  |ref|max {np.abs(ref[k]).max():.1f}")
