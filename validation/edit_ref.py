"""End-to-end torch reference for image editing, from diffusers' and transformers' own pieces.

Mirrors MiniMaxH3ImageBlocks (image_conditioned): EditResize -> EditTextEncoder -> EditVaeEncoder (posterior
sample off the generator) -> canvas -> edit layout -> target noise -> reference noise (t = 0.999) -> timesteps
-> denoise -> decode. The text encoder and DiT are the streamed reference modules (check_*_harness.py).

usage: edit_ref.py OUT_PREFIX DEVICE DTYPE --prompt FILE --image A.png [--image B.png] --size WxH --seed N [--min-area A]
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import StreamedDiT, StreamedTE, W, UP, chunked_attention  # noqa: E402

from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3Scheduler  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd  # noqa: E402
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer, Qwen3VLProcessor  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLVisionModel  # noqa: E402

SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("device"); p.add_argument("dtype")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", action="append", required=True)
    p.add_argument("--size", default="2048x2048")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-area", type=int, default=4194304)
    p.add_argument("--chunk-attention", type=int, default=0)
    a = p.parse_args()
    if a.chunk_attention:
        chunked_attention(a.chunk_attention)
    dev, dtype = torch.device(a.device), getattr(torch, a.dtype)
    width, height = (int(v) for v in a.size.split("x"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    proc = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)

    # 1. MiniMaxH3ImageEditResizeStep
    images = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in a.image]

    def resize(im, lo, hi):
        size = bound_short_edge(*im.size, lo, hi, 32)
        return im if im.size == size else proc.resize(im, height=size[1], width=size[0])

    refs = [resize(im, 1024, 2048) for im in images]
    vision_images = [resize(im, 1024, 1024) for im in images]

    # 2. MiniMaxH3ImageEditTextEncoderStep (presentation) + Qwen3VLModel.forward (vision, scatter, M-RoPE, DeepStack)
    tok = AutoTokenizer.from_pretrained(str(W / "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(str(UP / "processor"))
    vision = processor.image_processor(images=vision_images, return_tensors="pt")
    merge = processor.image_processor.merge_size ** 2
    ids, tags = [], []
    for i, grid in enumerate(vision["image_grid_thw"]):
        label = tok(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"][:4096]
        vis = ([tok.convert_tokens_to_ids("<|vision_start|>")] + [tok.convert_tokens_to_ids("<|image_pad|>")]
               * (int(grid.prod()) // merge) + [tok.convert_tokens_to_ids("<|vision_end|>")])
        ids += label + vis
        tags += [1] * len(label) + [0] * len(vis)
    pids = tok(Path(a.prompt).read_text(), add_special_tokens=False)["input_ids"][:4096]
    ids += pids
    tags += [1] * len(pids)
    cfg = Qwen3VLConfig(**json.loads((UP / "text_encoder/config.json").read_text()))
    cfg.vision_config._attn_implementation = "sdpa"
    visual = Qwen3VLVisionModel(cfg.vision_config)
    visual.load_state_dict({k[len("model.visual."):]: v for k, v in load_file(W / "te/visual.safetensors").items()})
    for prm in visual.parameters():
        prm.data = prm.data.to(dtype)
    visual = visual.to(dev).eval()
    with torch.no_grad():
        vo = visual(vision["pixel_values"].to(dev, dtype), grid_thw=vision["image_grid_thw"].to(dev), return_dict=True)
    del visual
    input_ids = torch.tensor([ids])
    table = load_file(W / "te/embed.safetensors")["model.language_model.embed_tokens.weight"]
    embeds = table[input_ids].to(dtype)
    del table
    image_mask = input_ids == cfg.image_token_id
    embeds = embeds.masked_scatter(image_mask[..., None], vo.pooler_output.cpu().to(dtype))
    fake = types.SimpleNamespace(config=cfg, get_vision_position_ids=lambda *x, **k: Qwen3VLModel.get_vision_position_ids(None, *x, **k))
    mm = torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long)
    position_ids, _ = Qwen3VLModel.get_rope_index(fake, input_ids, mm, image_grid_thw=vision["image_grid_thw"],
                                                 attention_mask=torch.ones_like(input_ids))
    te = StreamedTE(dtype=dtype, device=a.device)
    prompt_embeds = te(None, inputs_embeds=embeds, position_ids=position_ids, visual_pos_masks=image_mask.to(dev),
                       deepstack_visual_embeds=[d.to(dev) for d in vo.deepstack_features]).to(dtype)
    del te

    # 3. MiniMaxH3ImageEditVaeEncoderStep (float32 VAE), generator draws: posteriors, target noise, reference noise
    gen = torch.Generator().manual_seed(a.seed)
    vcfg = {k: v for k, v in json.loads((W / "vae/config.json").read_text()).items() if not k.startswith("_")}
    vae = AutoencoderKLMiniMaxH3Image(**vcfg)
    vae.load_state_dict(load_file(W / "vae/diffusion_pytorch_model.safetensors"))
    vae.eval()
    pm = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1); ps = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    lm = torch.tensor(vcfg["latents_mean"]).view(1, -1, 1, 1); ls = torch.tensor(vcfg["latents_std"]).view(1, -1, 1, 1)
    ref_latents = []
    with torch.no_grad():
        for im in refs:
            px = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None]
            px = px[..., : px.shape[-2] // 16 * 16, : px.shape[-1] // 16 * 16]
            px = (px.to(torch.float32).div(255.0) - pm) / ps
            post = vae.encode(px, return_dict=False)[0]
            z = post.sample(generator=gen).to(torch.float16).float()
            z = z[..., : z.shape[-2] // 2 * 2, : z.shape[-1] // 2 * 2]
            ref_latents.append((z - lm) / ls)
    ch, cw = bd.get_image_canvas_size(height, width, a.min_area, 32)
    lh, lw = ch // 16, cw // 16
    pos, token_tags, image_indices, text_indices, nref = bd.build_image_packed_sequence(
        torch.tensor(tags), [tuple(z.shape[-2:]) for z in ref_latents], lh, lw, (1, 2, 2), 0)
    if dev.type == "mps":
        pos = pos.float()
    latents = bd.patchify_video_latents(torch.randn((1, 24, 1, lh, lw), generator=gen, dtype=torch.float32), (1, 2, 2))
    sched = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    cond = []
    for z in ref_latents:
        noise = torch.randn(z.shape, generator=gen, dtype=torch.float32)
        cond.append(bd.patchify_video_latents(sched.scale_noise(z, torch.tensor(0.999), noise)[:, :, None], (1, 2, 2)))
    latents = torch.cat(cond + [latents])
    ts = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)
    sched.set_timesteps(sigmas=torch.cat([1.0 - ts, torch.zeros(1)]))
    seq = pos.shape[0]

    # 4. denoise (reference rows held at 0.999) and decode
    dit = StreamedDiT(dtype=dtype, device=a.device, real_adaln=False)
    text_embeds = dit.text(prompt_embeds)
    t0 = time.time()
    for i, t in enumerate(sched.timesteps):
        rt = torch.full((seq,), float(t), dtype=torch.float32)
        rt[image_indices[:nref]] = 0.999
        uniq, tidx = torch.unique(rt, sorted=True, return_inverse=True)
        v = dit.forward(latents[None], None, uniq, tidx, token_tags, pos, image_indices, text_indices, i,
                        text_embeds=text_embeds)[0].cpu()
        latents[nref:] = sched.step(v[nref:].float(), t, latents[nref:], return_dict=False)[0]
        print(f"step {i + 1}/6 {time.time() - t0:.0f}s", flush=True)
    rows = latents[nref:].reshape(-1, lh // 2, lw // 2, 24, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(-1, 24, lh, lw)
    with torch.no_grad():
        image = vae.decode(rows * ls + lm, return_dict=False)[0]
    image = (image.float() * ps + pm).clamp(0, 1)
    image = (image * 255).clamp(0, 255).to(torch.uint8).float() / 255
    if (height, width) != tuple(image.shape[-2:]):
        pil = [proc.resize(x, height, width) for x in proc.postprocess(image, output_type="pil")]
        image = proc.numpy_to_pt(proc.pil_to_numpy(pil))
    proc.postprocess(image, output_type="pil")[0].save(out.with_suffix(".png"))
    np.savez(out.with_suffix(".npz"), latents=latents.numpy(), prompt_embeds=prompt_embeds[0].float().cpu().numpy(),
             ids=np.array(ids), tags=np.array(tags))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
