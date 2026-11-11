import os

import mlx.core as mx

if os.environ.get("MINIMAX_H3_MLX_TEST_CPU"):
    mx.set_default_device(mx.cpu)  # CI runners have no usable GPU
