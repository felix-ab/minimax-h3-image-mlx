"""Vision tower parity. ref: transformers Qwen3VLProcessor + Qwen3VLVisionModel; mlx: minimax_h3_mlx.vision.
usage: vision_parity.py ref DTYPE | mlx DTYPE"""
import sys, json
import numpy as np
from PIL import Image
mode, dtype = sys.argv[1], sys.argv[2]
im = Image.open("upstream/examples/reference1.png").convert("RGB")
w, h = im.size
short = min(w, h)
if short != 1024:  # bound_short_edge(…, 1024, 1024, 32)
    s = 1024 / short
    im = im.resize((max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)), Image.LANCZOS)
print("vision image", im.size)
if mode == "ref":
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3VLProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    proc = Qwen3VLProcessor.from_pretrained("upstream/processor")
    out = proc.image_processor(images=[im], return_tensors="pt")
    cfg = Qwen3VLConfig(**json.load(open("upstream/text_encoder/config.json"))).vision_config
    cfg._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(cfg)
    sd = {k[len("model.visual."):]: v for k, v in load_file("weights/bf16/te/visual.safetensors").items()}
    print(model.load_state_dict(sd, strict=True))
    dt = getattr(torch, dtype)
    for prm in model.parameters():
        prm.data = prm.data.to(dt)
    model.eval()
    with torch.no_grad():
        o = model(out["pixel_values"].to(dt), grid_thw=out["image_grid_thw"], return_dict=True)
    np.savez(f"parity/vision_ref_{dtype}.npz", pixel_values=out["pixel_values"].float().numpy(), grid=out["image_grid_thw"].numpy(),
             embeds=o.pooler_output.float().numpy(), **{f"deep{i}": d.float().numpy() for i, d in enumerate(o.deepstack_features)})
    print("grid", out["image_grid_thw"].tolist(), "embeds", tuple(o.pooler_output.shape))
else:
    sys.path.insert(0, "repo")
    import mlx.core as mx
    from minimax_h3_mlx.vision import VisionTower, preprocess
    patches, grid = preprocess(np.asarray(im))
    ref = np.load(f"parity/vision_ref_{dtype}.npz")
    print("grid", grid, ref["grid"].tolist(), "pixel_values max|d|", float(np.abs(patches - ref["pixel_values"]).max()))
    tower = VisionTower(mx.load("weights/bf16/te/visual.safetensors"), dtype=getattr(mx, dtype))
    e, d = tower(patches, grid)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print("embeds rel", rel(np.array(e.astype(mx.float32)), ref["embeds"]), *[f"deep{i} rel {rel(np.array(x.astype(mx.float32)), ref[f'deep{i}']):.3e}" for i, x in enumerate(d)])
    if dtype == "bfloat16":
        r32 = np.load("parity/vision_ref_float32.npz")
        print("vs fp32 truth: mlx bf16", rel(np.array(e.astype(mx.float32)), r32["embeds"]), "torch bf16", rel(ref["embeds"], r32["embeds"]))
