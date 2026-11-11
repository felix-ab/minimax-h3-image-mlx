"""The streamed reference equals diffusers' stock MiniMaxH3ImageTransformer2DModel.forward (truncated to the
blocks whose real AdaLN weights we kept) bit for bit, and the table AdaLN equals the real projection."""
import sys
sys.path.insert(0, "ref"); sys.path.insert(0, "repo")
import numpy as np, torch
from safetensors.torch import load_file
from diffusers import MiniMaxH3ImageTransformer2DModel
from reference import StreamedDiT, W, CFG, _sub, TableAdaLN
from minimax_h3_mlx import layout as L
from minimax_h3_mlx.torch_randn import randn
keep = [int(p.stem) for p in sorted((W / "dit/adaln_keep").glob("*.safetensors"))]
print("kept blocks", keep)
torch.manual_seed(0)
n_text, lat = 12, 8
lay = L.Layout(n_text=n_text, text_tags=np.full(n_text, 1), latent_h=lat, latent_w=lat)
seq = lay.length
for dtype in (torch.float32, torch.bfloat16):
    cfg = {k: v for k, v in CFG.items() if not k.startswith("_")}
    cfg["num_layers"] = len(keep)
    stock = MiniMaxH3ImageTransformer2DModel(**cfg)
    sd = load_file(W / "dit/core.safetensors"); sd.update(load_file(W / "dit/refiner.safetensors"))
    for j, i in enumerate(keep):
        blk = _sub(load_file(W / f"dit/blocks/{i:02d}.safetensors"), f"transformer_blocks.{i}.")
        blk.update(_sub(load_file(W / f"dit/adaln_keep/{i:02d}.safetensors"), f"transformer_blocks.{i}."))
        sd.update({f"transformer_blocks.{j}.{k}": v for k, v in blk.items()})
    missing, unexpected = stock.load_state_dict(sd, strict=True), None
    for name, module in stock.named_children():  # from_pretrained(torch_dtype=...) keeps these in float32
        if name not in ("proj_in", "time_embedder", "pdd_proj_out", "rope"):
            module.to(dtype)
    stock.eval()
    ours = StreamedDiT(dtype=dtype, real_adaln=True)
    rows = torch.from_numpy(L.patchify(randn((24, lat, lat), 5)))[None]
    enc = (torch.randn(1, n_text, 5120) * 2).to(dtype)
    t, _ = L.sampling_grid()
    for step in (0, 3, 5):
        args = dict(hidden_states=rows, encoder_hidden_states=enc, timestep=torch.tensor([t[step]]),
                    timestep_indices=torch.zeros(seq, dtype=torch.long), token_tags=torch.from_numpy(lay.tags()),
                    position_ids=torch.from_numpy(lay.position_ids()), image_indices=torch.arange(n_text, seq),
                    text_indices=torch.arange(n_text), pdd_step_index=step)
        with torch.no_grad():
            a = stock(**args, return_dict=False)[0]
            b = ours.forward(**args, blocks=keep)
            ours.real_adaln = False
            c = ours.forward(**args, blocks=keep)
            ours.real_adaln = True
        print(f"{str(dtype):15s} step {step}: stock vs streamed max|d| {float((a-b).abs().max()):.3e} | "
              f"real AdaLN vs table: max|d| {float((b-c).abs().max()):.3e} rel {float((b-c).norm()/b.norm()):.2e}")
