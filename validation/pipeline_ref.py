"""End-to-end torch reference for text-to-image, built from diffusers' own pieces.

Layout, canvas, noise, scheduler and decode postprocessing are diffusers' functions; the text encoder and DiT
are the streamed reference modules (check_harness.py: equal to the stock forwards). Saves the image, the
latent rows after every step and the prompt embedding.

usage: pipeline_ref.py OUT_PREFIX DEVICE DTYPE --prompt FILE --size WxH --seed N [--min-area A]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import StreamedDiT, StreamedTE, W  # noqa: E402

from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3Scheduler  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("device")
    p.add_argument("dtype")
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-area", type=int, default=4194304)
    p.add_argument("--te-device", default=None)
    p.add_argument("--te-dtype", default=None)
    p.add_argument("--chunk-attention", type=int, default=0, help="query-chunked SDPA above this many rows (MPS)")
    a = p.parse_args()
    if a.chunk_attention:
        from reference import chunked_attention
        chunked_attention(a.chunk_attention)
    dev, dtype = torch.device(a.device), getattr(torch, a.dtype)
    width, height = (int(v) for v in a.size.split("x"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    timings = {}

    # 1. prompt -> hidden_states[50]
    t0 = time.time()
    prompt = Path(a.prompt).read_text()
    ids = AutoTokenizer.from_pretrained(str(W / "tokenizer"))(prompt, add_special_tokens=False)["input_ids"][:4096]
    te = StreamedTE(dtype=getattr(torch, a.te_dtype or a.dtype), device=a.te_device or a.device)
    prompt_embeds = te(ids).to(dtype)
    del te
    timings["te_s"] = time.time() - t0

    # 2. canvas, layout, noise, sampling grid (diffusers helpers)
    ch, cw = bd.get_image_canvas_size(height, width, a.min_area, 32)
    lh, lw = ch // 16, cw // 16
    tags = torch.full((len(ids),), 1, dtype=torch.long)
    pos, token_tags, image_indices, text_indices, nref = bd.build_image_packed_sequence(tags, [], lh, lw, (1, 2, 2), 0)
    if dev.type == "mps":
        pos = pos.float()
    gen = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((1, 24, 1, lh, lw), generator=gen, dtype=torch.float32)
    latents = bd.patchify_video_latents(noise, (1, 2, 2))
    sched = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    ts = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)
    sched.set_timesteps(sigmas=torch.cat([1.0 - ts, torch.zeros(1)]))
    seq = pos.shape[0]

    # 3. denoise
    t1 = time.time()
    dit = StreamedDiT(dtype=dtype, device=a.device, real_adaln=False)
    text_embeds = dit.text(prompt_embeds)
    steps = []
    for i, t in enumerate(sched.timesteps):
        v = dit.forward(latents[None], None, t[None], torch.zeros(seq, dtype=torch.long), token_tags, pos,
                        image_indices, text_indices, i, text_embeds=text_embeds)[0].cpu()
        latents = sched.step(v.float(), t, latents, return_dict=False)[0]
        steps.append(latents.numpy().copy())
        print(f"step {i + 1}/6 {time.time() - t1:.0f}s", flush=True)
    timings["denoise_s"] = time.time() - t1

    # 4. decode (MiniMaxH3ImageUnpackLatentsStep + MiniMaxH3ImageDecodeStep, VAE in float32)
    t2 = time.time()
    cfg = {k: v for k, v in json.loads((W / "vae/config.json").read_text()).items() if not k.startswith("_")}
    vae = AutoencoderKLMiniMaxH3Image(**cfg)
    vae.load_state_dict(load_file(W / "vae/diffusion_pytorch_model.safetensors"))
    vae.eval()
    rows = latents.reshape(-1, lh // 2, lw // 2, 24, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(-1, 24, lh, lw)
    mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1)
    std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1)
    with torch.no_grad():
        image = vae.decode(rows * std + mean, return_dict=False)[0]
    pm = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    ps = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    image = (image.float() * ps + pm).clamp(0, 1)
    image = (image * 255).clamp(0, 255).to(torch.uint8).float() / 255
    proc = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
    if (height, width) != tuple(image.shape[-2:]):
        pil = proc.postprocess(image, output_type="pil")
        pil = [proc.resize(item, height, width) for item in pil]
        image = proc.numpy_to_pt(proc.pil_to_numpy(pil))
    proc.postprocess(image, output_type="pil")[0].save(out.with_suffix(".png"))
    timings["vae_s"] = time.time() - t2
    np.savez(out.with_suffix(".npz"), steps=np.stack(steps), prompt_embeds=prompt_embeds[0].float().cpu().numpy(),
             ids=np.array(ids))
    out.with_suffix(".json").write_text(json.dumps({"args": vars(a), "canvas": [ch, cw], "timings": timings}, indent=1))
    print("wrote", out.with_suffix(".png"), timings)


if __name__ == "__main__":
    main()
