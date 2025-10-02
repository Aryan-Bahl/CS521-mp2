import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler
import os

# Create a log directory
logdir = "./gpu/traces"

def _out_size(H, W, kernel_size, stride, padding):
    out_h = (H - kernel_size + 2 * padding) // stride + 1
    out_w = (W - kernel_size + 2 * padding) // stride + 1
    return out_h, out_w

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.
    N_, C_, Hp, Wp = x_pad.shape
    
    patches = []
    for i in range(out_h):
        h0 = i * S
        for j in range(out_w):
            w0 = j * S
            patch = x_pad[:, :, h0:h0+KH, w0:w0+KW]
            patches.append(patch.reshape(N, C*KH*KW))
    
    # currently patches is a list of (N, C*KH*KW) tensors with len out_h * out_w
    # we need to stack them into a single (N, out_h*out_w, C*KH*KW) tensor
    patches = jnp.stack(patches, axis=1) # -> (N, out_h*out_w, C*KH*KW) because we stack on the second dimension
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1, tile_k=64):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h, out_w = _out_size(H, W, KH, stride, padding)
    M = out_h * out_w
    K = C * KH * KW
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    W_flat = weight.reshape(C_out, K)
    
    # TO DO: 3) perform tiled matmul after required reshaping is done.
    out_cols = jnp.zeros((N, M, C_out), dtype=x.dtype)
    for k0 in range(0, K, tile_k):
        k1 = min(k0 + tile_k, K)
        partial_out = jnp.matmul(cols[:, :, k0:k1], W_flat[:, k0:k1].T)
        out_cols = out_cols + partial_out

    # TO DO: 4) Add bias.
    out_cols = out_cols + bias.reshape(1, 1, C_out)

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = jnp.transpose(out_cols, (0, 2, 1)).reshape(N, C_out, out_h, out_w) # same concept as in myconv.py
    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model = model.to("cuda")
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W, device=model.weight.device)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    print("JAX devices:", jax.devices())
    print("Jax backend:", jax.default_backend())

    dev = jax.devices("gpu")[0]
    with jax.default_device(dev):
        x_jax = jnp.asarray(x_torch.detach().cpu().numpy())
        weight_jax = jnp.asarray(params["weight"])
        bias_jax = jnp.asarray(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # warmup
    _ = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)

    os.makedirs(logdir, exist_ok=True)
    jax.profiler.start_trace(logdir, create_perfetto_trace=True)
    with jax.profiler.TraceAnnotation("manual_jax_conv"):
        for _ in range(10):
            out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
            out_jax.block_until_ready()
    jax.profiler.stop_trace()
    print("JAX --- trace exported")

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    conv_ref_cpu = conv_ref.detach().cpu()
    
    out_jax_torch = torch.from_numpy(np.array(out_jax))
    print("JAX --- shape check:", out_jax_torch.shape == conv_ref_cpu.shape)
    print("JAX --- correctness check:", torch.allclose(out_jax_torch, conv_ref_cpu, atol=1e-1)) # change atol to 1e-1 for larger kernel size (>=5) - talk about in report