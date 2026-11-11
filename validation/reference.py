"""PyTorch reference for MiniMax-H3-Image, streamed one block/layer at a time so it runs in 48 GB.

Every module is diffusers' or transformers' own class with the released weights; only the outer loops are
re-written, line for line from `MiniMaxH3ImageTransformer2DModel.forward` and `Qwen3VLTextModel.forward`, so the
weights can be loaded per block. `check_harness()` proves those loops equal the stock forwards on truncated
models. Blocks whose AdaLN projection was consumed into the modulation table (all but 0, 25, 49) use the table
rows, i.e. the correctly rounded values of the projection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers import transformer_minimax_h3_image as dm

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "weights/bf16"
UP = ROOT / "upstream"
CFG = json.loads((UP / "transformer_diffusers/config.json").read_text())


def chunked_attention(max_rows: int = 2048):
    """Patch diffusers' attention dispatch with query-chunked SDPA (each query row's softmax is computed exactly as
    unchunked; only the batching of rows changes). Needed on MPS above ~4 MP, where one SDPA call over 16k rows
    can allocate the full attention matrix (tens of GB)."""
    original = dm.dispatch_attention_fn

    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None, parallel_config=None):
        if query.shape[1] <= max_rows:
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                            backend=backend, parallel_config=parallel_config)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))  # (B, H, N, D)
        out = torch.cat([torch.nn.functional.scaled_dot_product_attention(q[:, :, s:s + max_rows], k, v)
                         for s in range(0, q.shape[2], max_rows)], dim=2)
        return out.transpose(1, 2)

    dm.dispatch_attention_fn = dispatch
    return original


def _sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class TableAdaLN(nn.Module):
    """Stands in for `adaln_proj` with the precomputed table; `t_rows` must hold the table index of each temb row."""

    def __init__(self, table, hidden=5376):
        super().__init__()
        self.table, self.hidden, self.t_rows = table, hidden, None

    def forward(self, temb):
        rows = []
        for t in self.t_rows:
            for mod in range(dm.MINIMAX_H3_MODALITY_NUM):
                rows.append(self.table[t, mod].reshape(-1) if mod < 2 else torch.zeros(6 * self.hidden, dtype=self.table.dtype))
        out = torch.stack(rows).to(temb.device)
        return out.chunk(6, dim=-1)


class TableNormOut(nn.Module):
    """MiniMaxH3ImageAdaLayerNormOut.forward with shift/scale read from the precomputed table."""

    def __init__(self, norm_out, table, table_t, t_rows_ref):
        super().__init__()
        self.norm, self.table, self.table_t, self.t_rows_ref = norm_out.norm, table, table_t, t_rows_ref

    def forward(self, hidden_states, temb, timestep_indices):
        rows = self.table[self.t_rows_ref[0]].to(hidden_states.device, hidden_states.dtype)  # (n_t, 2, H)
        shift, scale = rows[:, 0], rows[:, 1]
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


class StreamedDiT:
    def __init__(self, dtype=torch.float32, device="cpu", real_adaln=True):
        self.dtype, self.device, self.real_adaln = dtype, torch.device(device), real_adaln
        core = load_file(W / "dit/core.safetensors")
        H = CFG["hidden_size"]
        self.proj_in = nn.Linear(96, H).float()
        self.proj_in.load_state_dict(_sub(core, "proj_in."))
        self.context_embedder = nn.Linear(CFG["text_dim"], H)
        self.context_embedder.load_state_dict(_sub(core, "context_embedder."))
        self.context_embedder.to(dtype)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=H, out_dim=CFG["time_embed_dim"]).float()
        self.time_embedder.load_state_dict(_sub(core, "time_embedder."))
        self.rope = dm.MiniMaxH3ImageRotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0)
        self.refiner = dm.MiniMaxH3ImageTokenRefiner(H, 56, 128, CFG["ffn_dim"], 2, 1e-5, 1e-5, 1e-5)
        self.refiner.load_state_dict(_sub(load_file(W / "dit/refiner.safetensors"), "token_refiner."))
        self.refiner.to(dtype)
        self.norm_out = dm.MiniMaxH3ImageAdaLayerNormOut(H, CFG["time_embed_dim"], 1e-5)
        self.norm_out.load_state_dict(_sub(core, "norm_out."))
        self.norm_out.to(dtype)
        self.heads = []
        for i in range(6):
            head = nn.Linear(H, 96).float()
            head.load_state_dict(_sub(core, f"pdd_proj_out.{i}."))
            self.heads.append(head)
        for m in (self.proj_in, self.context_embedder, self.time_embedder, self.refiner, self.norm_out, self.rope, *self.heads):
            m.to(self.device)
        self.table_t = load_file(W / "dit/modulation/norm_out.safetensors")["timesteps"]

    def block(self, i):
        blk = dm.MiniMaxH3ImageTransformerBlock(5376, 56, 128, CFG["ffn_dim"], CFG["time_embed_dim"], 1e-5, 1e-5)
        sd = _sub(load_file(W / f"dit/blocks/{i:02d}.safetensors"), f"transformer_blocks.{i}.")
        keep = W / f"dit/adaln_keep/{i:02d}.safetensors"
        if keep.exists() and self.real_adaln:
            sd.update(_sub(load_file(keep), f"transformer_blocks.{i}."))
            blk.load_state_dict(sd)
        else:
            blk.load_state_dict(sd, strict=False)
            # cast like the projection's output would be in this dtype (bf16 run: no-op)
            blk.adaln_proj = TableAdaLN(load_file(W / f"dit/modulation/{i:02d}.safetensors")["table"].to(self.dtype))
        return blk.to(self.device, self.dtype)

    @torch.no_grad()
    def text(self, prompt_embeds):
        e = self.context_embedder(prompt_embeds.to(self.device, self.dtype))
        return self.refiner(e)

    @torch.no_grad()
    def forward(self, hidden_states, encoder_hidden_states, timestep, timestep_indices, token_tags, position_ids,
                image_indices, text_indices, pdd_step_index, dump=None, text_embeds=None, blocks=range(50)):
        """MiniMaxH3ImageTransformer2DModel.forward with the block loop streamed. Shapes as in diffusers."""
        dev = self.device
        rotary_emb = self.rope(position_ids.to(dev))
        image_embeds = self.proj_in(hidden_states.to(dev, torch.float32))
        if text_embeds is None:
            text_embeds = self.text(encoder_hidden_states)
        seq = position_ids.shape[0]
        h = text_embeds.new_zeros((text_embeds.shape[0], seq, text_embeds.shape[-1]))
        h = h.index_copy(1, text_indices.to(dev), text_embeds)
        h = h.index_copy(1, image_indices.to(dev), image_embeds.to(text_embeds.dtype))
        temb = self.time_embedder(self.time_proj(timestep.to(dev)).to(torch.float32))
        adaln_indices = (timestep_indices * dm.MINIMAX_H3_MODALITY_NUM + token_tags).to(dev)
        t_rows = [int(np.nonzero(self.table_t.numpy() == np.float32(t))[0][0]) for t in timestep.tolist()]
        for i in blocks:
            blk = self.block(i)
            if isinstance(blk.adaln_proj, TableAdaLN):
                blk.adaln_proj.t_rows = t_rows
            h = blk(h, temb, adaln_indices, rotary_emb)
            if dump is not None:
                dump(i, h)
            del blk
        proj_out = self.heads[pdd_step_index]
        norm_out = self.norm_out
        if not self.real_adaln:
            table = load_file(W / "dit/modulation/norm_out.safetensors")["table"]
            norm_out = TableNormOut(self.norm_out, table, self.table_t, [t_rows])
        h = norm_out(h, temb, timestep_indices.to(dev)).to(torch.float32)
        return proj_out(h).index_select(1, image_indices.to(dev))


# ------------------------------------------------------------------------------------------- text encoder

def te_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    cfg = json.loads((UP / "text_encoder/config.json").read_text())["text_config"]
    config = Qwen3VLTextConfig(**cfg)
    config._attn_implementation = "sdpa"
    return config


class StreamedTE:
    def __init__(self, dtype=torch.float32, device="cpu"):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as mq
        self.mq, self.dtype, self.device = mq, dtype, torch.device(device)
        self.config = te_config()
        self.rotary = mq.Qwen3VLTextRotaryEmbedding(self.config).to(self.device)

    def layer(self, i):
        layer = self.mq.Qwen3VLTextDecoderLayer(self.config, i)
        layer.load_state_dict(_sub(load_file(W / f"te/layers/{i:02d}.safetensors"), f"model.language_model.layers.{i}."))
        return layer.to(self.device, self.dtype).eval()

    @torch.no_grad()
    def __call__(self, ids, layers=50, dump=None, inputs_embeds=None, position_ids=None, visual_pos_masks=None,
                 deepstack_visual_embeds=None):
        """Qwen3VLTextModel.forward (no cache) streamed; returns hidden_states[layers].

        Text-only by default. For edits pass `inputs_embeds` (image rows already scattered), `position_ids`
        (3, 1, seq) from Qwen3VLModel.get_rope_index, `visual_pos_masks` (1, seq) and the 3 DeepStack tensors."""
        from transformers.masking_utils import create_causal_mask
        if inputs_embeds is None:
            table = load_file(W / "te/embed.safetensors")["model.language_model.embed_tokens.weight"]
            h = table[torch.tensor(ids)][None].to(self.device, self.dtype)
            del table
        else:
            h = inputs_embeds.to(self.device, self.dtype)
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=self.device).view(1, 1, -1).expand(4, 1, -1)
        if position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
        else:
            text_position_ids = None
        position_embeddings = self.rotary(h, position_ids.to(self.device))
        # the mask exactly as Qwen3VLTextModel.forward builds it from diffusers' all-ones attention_mask
        mask = create_causal_mask(config=self.config, inputs_embeds=h,
                                  attention_mask=torch.ones(1, h.shape[1], dtype=torch.long, device=self.device),
                                  past_key_values=None, position_ids=text_position_ids)
        for i in range(layers):
            layer = self.layer(i)
            h = layer(h, attention_mask=mask, position_ids=text_position_ids, position_embeddings=position_embeddings)
            if deepstack_visual_embeds is not None and i in range(len(deepstack_visual_embeds)):
                h = h.clone()
                h[visual_pos_masks, :] = h[visual_pos_masks, :] + deepstack_visual_embeds[i].to(h.device, h.dtype)
            if dump is not None:
                dump(i, h)
            del layer
        return h
