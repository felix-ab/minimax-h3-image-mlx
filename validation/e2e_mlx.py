"""MLX end-to-end text-to-image with per-step latent dumps. usage: e2e_mlx.py OUT_PREFIX DTYPE --prompt F --size WxH --seed N [--min-area A] [--ref REF_PREFIX ...]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "repo")
import numpy as np, mlx.core as mx
from minimax_h3_mlx.pipeline import Pipeline
p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("dtype")
p.add_argument("--prompt", required=True); p.add_argument("--size", default="1024x1024")
p.add_argument("--seed", type=int, default=0); p.add_argument("--min-area", type=int, default=4194304)
p.add_argument("--weights", default="weights/bf16"); p.add_argument("--ref", nargs="*", default=[])
p.add_argument("--resident", action="store_true")
p.add_argument("--cache-limit-gb", type=float, default=None)
p.add_argument("--dit-bits", type=int, default=16); p.add_argument("--te-bits", type=int, default=16)
a = p.parse_args()
w, h = (int(v) for v in a.size.split("x"))
q = lambda b: {"bits": b, "group_size": 64} if b < 16 else None
pipe = Pipeline(a.weights, dtype=getattr(mx, a.dtype), stream=False if a.resident else None,
                dit_quant=q(a.dit_bits), te_quant=q(a.te_bits), cache_limit_gb=a.cache_limit_gb)
steps = []
t0 = time.time()
def on_step(i, rows):
    steps.append(np.array(rows)); print(f"step {i+1}/6 {time.time()-t0:.0f}s", flush=True)
res = pipe(Path(a.prompt).read_text(), height=h, width=w, seed=a.seed, min_generation_area=a.min_area, on_step=on_step)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
res.image.save(out.with_suffix(".png"))
np.savez(out.with_suffix(".npz"), steps=np.stack(steps))
meta = {"args": vars(a), "canvas": res.canvas, "timings": res.timings, "mlx_peak_gb": mx.get_peak_memory() / 1e9}
out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta["timings"]))
for r in a.ref:
    ref = np.load(Path(r).with_suffix(".npz"))["steps"]
    rel = [float(np.linalg.norm(s - t) / np.linalg.norm(t)) for s, t in zip(np.stack(steps), ref)]
    print("vs", r, "latent rel per step:", " ".join(f"{x:.2e}" for x in rel))
