"""Torch reference text encoder: hidden_states[50] (+ a few layer dumps). usage: te_ref.py OUT.npz DTYPE [DEVICE] [PROMPT_FILE]"""
import sys, time
sys.path.insert(0, "ref")
import numpy as np, torch
from transformers import AutoTokenizer
from reference import StreamedTE
out, dtype = sys.argv[1], getattr(torch, sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
prompt = open(sys.argv[4] if len(sys.argv) > 4 else "upstream/examples/revised_prompt.t2i.txt").read()
ids = AutoTokenizer.from_pretrained("upstream/tokenizer")(prompt, add_special_tokens=False)["input_ids"][:4096]
te = StreamedTE(dtype=dtype, device=device)
dumps = {}
def dump(i, h):
    if i in (0, 1, 9, 24, 39, 48):
        dumps[f"layer{i:02d}"] = h[0].float().cpu().numpy()
t = time.time()
h = te(ids, layers=int(__import__('os').environ.get('NL', 50)), dump=dump)
print(f"ref TE {time.time()-t:.0f}s, {len(ids)} tokens")
np.savez(out, ids=np.array(ids), h50=h[0].float().cpu().numpy(), **dumps)
