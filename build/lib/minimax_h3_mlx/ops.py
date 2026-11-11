"""Small ops written to round exactly where the PyTorch reference rounds.

MLX's fused helpers are fast but do not always round at the same places as torch:
  * torch `nn.RMSNorm` (diffusers DiT) computes x * rsqrt(mean(x^2) + eps) * w in float32 and rounds once
    (`_fused_rms_norm` on CUDA/CPU; the composite path promotes the bf16 weight to float32 too);
  * the Hugging Face Qwen3-VL RMSNorm rounds the normalized value to bf16 first, then multiplies by the bf16
    weight (two roundings);
  * torch `F.silu` on bf16 computes in float32 and rounds once (MLX's `x * sigmoid(x)` would round twice);
  * a torch bf16 `nn.Linear` with bias is one GEMM with the bias added in the float32 epilogue.
Every op here returns the dtype of its input unless stated.
"""
import mlx.core as mx

F32 = mx.float32


def rms_norm_torch(x, weight, eps):
    """torch.nn.RMSNorm / F.rms_norm: one rounding at the end."""
    xf = x.astype(F32)
    y = xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + eps)
    if weight is not None:
        y = y * weight.astype(F32)
    return y.astype(x.dtype)


def rms_norm_hf(x, weight, eps):
    """Qwen3VLTextRMSNorm: normalize in float32, round to the input dtype, then multiply by the weight."""
    xf = x.astype(F32)
    y = (xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + eps)).astype(x.dtype)
    return weight * y


def silu(x):
    xf = x.astype(F32)
    return (xf * mx.sigmoid(xf)).astype(x.dtype)


def linear(x, weight, bias=None):
    """x @ W.T (+ b) with a single rounding, like a torch Linear."""
    if bias is None:
        return x @ weight.T
    return mx.addmm(bias.astype(x.dtype), x, weight.T)


def qlinear(x, layer):
    """Linear from a dict that is either {'weight'} (dense) or {'weight','scales','biases','bits','group_size'}."""
    if "scales" in layer:
        y = mx.quantized_matmul(x, layer["weight"], layer["scales"], layer["biases"], transpose=True,
                                group_size=layer["group_size"], bits=layer["bits"])
        return y + layer["bias"] if "bias" in layer else y
    return linear(x, layer["weight"], layer.get("bias"))
