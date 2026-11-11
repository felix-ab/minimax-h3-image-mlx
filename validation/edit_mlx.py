"""MLX edit end to end. usage: edit_mlx.py OUT_PREFIX DTYPE --prompt F --image A.png [...] --size WxH --seed N [--min-area A]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "repo")
import mlx.core as mx
from PIL import Image
from minimax_h3_mlx.pipeline import Pipeline
p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("dtype")
p.add_argument("--prompt", required=True); p.add_argument("--image", action="append", required=True)
p.add_argument("--size", default="2048x2048"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--min-area", type=int, default=4194304); p.add_argument("--weights", default="weights/bf16")
a = p.parse_args()
w, h = (int(v) for v in a.size.split("x"))
pipe = Pipeline(a.weights, dtype=getattr(mx, a.dtype))
t0 = time.time()
res = pipe.edit(Path(a.prompt).read_text(), [Image.open(f) for f in a.image], height=h, width=w, seed=a.seed,
                min_generation_area=a.min_area, on_step=lambda i, r: print(f"step {i+1}/6 {time.time()-t0:.0f}s", flush=True))
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
res.image.save(out.with_suffix(".png"))
out.with_suffix(".json").write_text(json.dumps({"args": vars(a), "canvas": res.canvas, "timings": res.timings,
                                                "mlx_peak_gb": mx.get_peak_memory() / 1e9}, indent=1))
print(json.dumps(res.timings))
