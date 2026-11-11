import numpy as np
import pytest

from minimax_h3_mlx import torch_randn

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("shape,seed", [((1, 24, 1, 222, 296), 42), ((1, 24, 1, 256, 256), 1234),
                                        ((1, 24, 1, 33, 47), 3), ((5, 17), 11), ((16,), 0)])
def test_bit_exact(shape, seed):
    ref = torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).numpy()
    mine = torch_randn.randn(shape, seed)
    assert (ref.view(np.uint32) == mine.view(np.uint32)).all()


def test_generator_stream():
    g = torch.Generator().manual_seed(1234)
    mine = torch_randn.Generator(1234)
    for shape in [(1, 24, 62, 46), (1, 24, 1, 128, 128), (1, 24, 31, 23)]:
        ref = torch.randn(shape, generator=g, dtype=torch.float32).numpy()
        assert (ref.view(np.uint32) == mine.randn(shape).view(np.uint32)).all()
