"""Streamed TE reference == stock transformers Qwen3VLTextModel (4 layers), fp32 and bf16, hidden_states[3]."""
import sys
sys.path.insert(0, "ref")
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from reference import StreamedTE, te_config, W, _sub
ids = AutoTokenizer.from_pretrained("upstream/tokenizer")(open("upstream/examples/revised_prompt.t2i.txt").read(), add_special_tokens=False)["input_ids"]
for dtype in (torch.float32, torch.bfloat16):
    cfg = te_config(); cfg.num_hidden_layers = 4
    stock = Qwen3VLTextModel(cfg)
    sd = {"embed_tokens.weight": load_file(W / "te/embed.safetensors")["model.language_model.embed_tokens.weight"]}
    for i in range(4):
        sd.update({f"layers.{i}.{k}": v for k, v in _sub(load_file(W / f"te/layers/{i:02d}.safetensors"), f"model.language_model.layers.{i}.").items()})
    missing, unexpected = stock.load_state_dict(sd, strict=False)
    for prm in stock.parameters():
        prm.data = prm.data.to(dtype)  # parameters only: from_pretrained keeps the rotary inv_freq buffer float32
    stock = stock.eval()
    with torch.no_grad():
        hs = stock(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long), output_hidden_states=True).hidden_states
        ours = StreamedTE(dtype=dtype)(ids, layers=3)
    print(dtype, "missing", missing, "| stock hidden_states[3] vs streamed: max|d|", float((hs[3] - ours).abs().max()))
