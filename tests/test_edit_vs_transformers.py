"""Edit-path host logic vs transformers (needs torch + transformers): M-RoPE positions, interleaved rotary tables."""
import json
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
mq = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig  # noqa: E402

from minimax_h3_mlx.text_encoder import mrope_tables  # noqa: E402
from minimax_h3_mlx.vision import IMAGE_PAD, VISION_END, VISION_START, edit_positions  # noqa: E402

CFG = Path(__file__).resolve().parents[2] / "upstream/text_encoder/config.json"


@pytest.fixture(scope="module")
def cfg():
    if not CFG.exists():
        pytest.skip("upstream config not available")
    return Qwen3VLConfig(**json.loads(CFG.read_text()))


def presentation(grids):
    ids = []
    for i, (_, h, w) in enumerate(grids):
        ids += [3, 4, 5] + [VISION_START] + [IMAGE_PAD] * (h * w // 4) + [VISION_END]
    return ids + list(range(100, 140))


@pytest.mark.parametrize("grids", [[(1, 64, 64)], [(1, 64, 96), (1, 80, 64)], [(1, 32, 128)]])
def test_positions_and_rotary(cfg, grids):
    ids = presentation(grids)
    fake = types.SimpleNamespace(config=cfg, get_vision_position_ids=lambda *a, **k: mq.Qwen3VLModel.get_vision_position_ids(None, *a, **k))
    mm = torch.tensor([[1 if t == IMAGE_PAD else 0 for t in ids]])
    ref, _ = mq.Qwen3VLModel.get_rope_index(fake, torch.tensor([ids]), mm, image_grid_thw=torch.tensor(grids),
                                            attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    ours = edit_positions(ids, grids)
    assert (ref[:, 0].numpy() == ours).all()
    rot = mq.Qwen3VLTextRotaryEmbedding(cfg.text_config)
    cos, sin = rot(torch.zeros(1, dtype=torch.bfloat16), ref)
    c, s = mrope_tables(ours)
    for a, b in ((cos[0], c), (sin[0], s)):
        theirs = a.float().numpy()
        mine = np.array(b.astype(mx.float32))
        assert (theirs != mine).mean() < 1e-4  # torch float32 cos vs correctly rounded, then bf16: rare 1-ulp ties
