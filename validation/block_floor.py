"""One DiT block (and its stages) on an identical realistic bf16 input: torch fp32 truth vs torch CPU/MPS bf16 vs MLX bf16."""
import sys
sys.path.insert(0, "repo"); sys.path.insert(0, "ref")
import numpy as np, torch, mlx.core as mx
from reference import StreamedDiT
from minimax_h3_mlx import layout as L, dit as D
torch.set_grad_enabled(False)
def to_mx(t): return mx.array(t.detach().float().cpu().numpy()).astype(mx.bfloat16)
def np32(t): return t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t.astype(mx.float32))
d = np.load("parity/e2e/ref_f32.npz")
prompt = torch.from_numpy(d["prompt_embeds"]).to(torch.bfloat16)
n_text = prompt.shape[0]; lh, lw = 24, 32
lay = L.Layout(n_text=n_text, text_tags=np.full(n_text, 1), latent_h=lh, latent_w=lw)
step = int(sys.argv[1]) if len(sys.argv) > 1 else 4
blk_i = int(sys.argv[2]) if len(sys.argv) > 2 else 25
t, _ = L.sampling_grid()
# realistic input: run the fp32 reference up to block blk_i at this step, then round to bf16
rows = torch.from_numpy(d["steps"][step - 1] if step > 0 else L.patchify(__import__("minimax_h3_mlx.torch_randn", fromlist=["randn"]).randn((24, lh, lw), 42)))
outs = {}
for name, dev, dt in [("f32", "cpu", torch.float32), ("cpu", "cpu", torch.bfloat16), ("mps", "mps", torch.bfloat16)]:
    ref = StreamedDiT(dtype=dt, device=dev, real_adaln=False)
    caps = {}
    def dump(i, h):
        if i == blk_i - 1: caps["in"] = h.detach().cpu()
        if i == blk_i: caps["out"] = h.detach().cpu()
    if name == "f32":
        pos = torch.from_numpy(lay.position_ids())
        ref.forward(rows[None], None, torch.tensor([t[step]]), torch.zeros(lay.length, dtype=torch.long), torch.from_numpy(lay.tags()),
                    pos, torch.arange(n_text, lay.length), torch.arange(n_text), step, dump=dump, text_embeds=ref.text(prompt[None]),
                    blocks=range(blk_i + 1))
        h_in = caps["in"].to(torch.bfloat16)  # the shared bf16 input of block blk_i
    blk = ref.block(blk_i); blk.adaln_proj.t_rows = [step]
    temb = ref.time_embedder(ref.time_proj(torch.tensor([t[step]]).to(dev)).float())
    pos = torch.from_numpy(lay.position_ids()); pos = pos.float() if dev == "mps" else pos
    rot = ref.rope(pos.to(dev))
    outs[name] = blk(h_in.to(dev, dt), temb, (torch.from_numpy(lay.tags()) + 3 * 0).to(dev) + 3 * 0, rot).float().cpu()
    del ref
w_dit = D.DiT("weights/bf16/dit", stream=True)
w = w_dit.block(blk_i)
runs = [(s, e, 0 if tag == 1 else 1) for s, e, _, tag in lay.runs(step, 6)]
table = w_dit.tables[blk_i]
mods = [table[step, 1], table[step, 0]]
cos, sin = D.rope_tables(lay.position_ids())
outs["mlx"] = torch.from_numpy(np32(D.transformer_block(to_mx(h_in[0]), w, runs, mods, cos, sin)))[None]
rel = lambda a, b: float((a - b).norm() / b.norm())
print(f"step {step} block {blk_i}: vs fp32 truth  cpu {rel(outs['cpu'], outs['f32']):.3e}  mps {rel(outs['mps'], outs['f32']):.3e}  mlx {rel(outs['mlx'], outs['f32']):.3e}"
      f"  | pairwise cpu-mps {rel(outs['cpu'], outs['mps']):.3e} mlx-mps {rel(outs['mlx'], outs['mps']):.3e} mlx-cpu {rel(outs['mlx'], outs['cpu']):.3e}")
# delta (block's contribution) relative error, which is what bf16 rounding perturbs
dl = {k: v - h_in.float() for k, v in outs.items()}
print(f"   block update (out - in) vs fp32: cpu {rel(dl['cpu'], dl['f32']):.3e} mps {rel(dl['mps'], dl['f32']):.3e} mlx {rel(dl['mlx'], dl['f32']):.3e}")
