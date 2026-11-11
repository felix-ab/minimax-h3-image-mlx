"""minimax-h3-mlx: MiniMax-H3-Image text-to-image on Apple Silicon.

    minimax-h3-mlx "a detailed description of the whole image ..." --size 4096x4096 --seed 42 -o out.png

The model expects a long, complete English description; short prompts work, but MiniMax pairs the model with a
prompt expander (MiniMax-H3-Image-PE) for them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx


def main(argv=None):
    p = argparse.ArgumentParser(prog="minimax-h3-mlx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", help="prompt text, or @file to read it from a file")
    p.add_argument("--size", default="2048x2048", help="WIDTHxHEIGHT of the returned image (default 2048x2048)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", default="minimax-h3.png")
    p.add_argument("--weights", default="bf16", help="tier name (bf16, 8bit), Hub repo id, or a local tier directory")
    p.add_argument("--min-generation-area", type=int, default=4_194_304,
                   help="smaller sizes are generated on a canvas of this many pixels and resized (1024 = off)")
    p.add_argument("--resident", action="store_true", help="keep all DiT blocks in memory instead of streaming")
    p.add_argument("--fp32", action="store_true", help="run the DiT and text encoder in float32 (tests only)")
    p.add_argument("--cache-limit-gb", type=float, default=None,
                   help="cap MLX's buffer cache to lower the footprint (same pixels)")
    p.add_argument("--save-latents", action="store_true")
    p.add_argument("--image", action="append", help="reference image for editing; repeat for <Picture 2>, ...")
    args = p.parse_args(argv)

    from .pipeline import Pipeline
    from .weights import resolve

    prompt = Path(args.prompt[1:]).read_text() if args.prompt.startswith("@") else args.prompt
    width, height = (int(v) for v in args.size.lower().split("x"))
    pipe = Pipeline(resolve(args.weights), dtype=mx.float32 if args.fp32 else mx.bfloat16,
                    stream=False if args.resident else None, cache_limit_gb=args.cache_limit_gb)

    def on_step(i, rows):
        print(f"  step {i + 1}/6 done at {time.perf_counter() - start:.0f}s", flush=True)

    start = time.perf_counter()
    if args.image:
        from PIL import Image
        result = pipe.edit(prompt, [Image.open(f) for f in args.image], height=height, width=width, seed=args.seed,
                           min_generation_area=args.min_generation_area, on_step=on_step)
    else:
        result = pipe(prompt, height=height, width=width, seed=args.seed,
                      min_generation_area=args.min_generation_area, keep_latents=args.save_latents, on_step=on_step)
    result.image.save(args.out)
    meta = {"prompt": prompt, "size": args.size, "canvas": result.canvas, "seed": args.seed,
            "min_generation_area": args.min_generation_area, "timings": result.timings,
            "peak_mlx_gb": mx.get_peak_memory() / 1e9}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    if args.save_latents:
        import numpy as np
        np.save(Path(args.out).with_suffix(".latents.npy"), result.latents)
    print(f"wrote {args.out} ({result.canvas[1]}x{result.canvas[0]} canvas) in {result.timings['total_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
