import os, time, csv, numpy as np
import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from jax import jit
from jax.profiler import StepTraceAnnotation
import jax.profiler
import jax.lax as lax
from myconv import ConvModel

LOGDIR = "./traces"
DEVICE_KIND = "gpu"
OUT_CSV = "results/jax_vs_torch_baseline.csv"

def _out_size(H, W, kernel_size, stride, padding):
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1
    return out_h, out_w

def im2col_manual_jax(x, KH, KW, S, P):
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))
    N, C, Hp, Wp = x_pad.shape
    K = C * KH * KW
    out_h, out_w = _out_size(H, W, KH, S, P)
    M = out_h * out_w

    out = jnp.zeros((N, out_h, out_w, K), dtype=x.dtype)

    def body(idx, acc):
        i = idx // out_w
        j = idx %  out_w
        h0 = i * S
        w0 = j * S
        patch = lax.dynamic_slice(x_pad, (0, 0, h0, w0), (N, C, KH, KW))
        flat  = patch.reshape(N, K)
        acc   = lax.dynamic_update_slice(acc, flat[:, None, None, :], (0, i, j, 0))
        return acc

    out = lax.fori_loop(0, M, body, out)
    out = out.reshape(N, M, K)
    return out, out_h, out_w

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1, tile_k=64):
    N, C, H, W = x.shape
    Cout, Cin, KH, KW = weight.shape
    
    cols, out_h, out_w = im2col_manual_jax(x, KH, KW, stride, padding)  # (N,M,K)
    M = out_h * out_w
    K = Cin * KH * KW

    W_flat = weight.reshape(Cout, K)
    out_cols = jnp.zeros((N, M, Cout), dtype=x.dtype)
    for k0 in range(0, K, tile_k):
        k1 = min(k0 + tile_k, K)
        out_cols = out_cols + jnp.matmul(cols[:, :, k0:k1], W_flat[:, k0:k1].T)

    out_cols = out_cols + bias.reshape(1, 1, Cout)
    out = jnp.transpose(out_cols, (0, 2, 1)).reshape(N, Cout, out_h, out_w)
    return out


def measure_jax(fn_jit, *args, warmup=5, iters=50, annotate=None):
    """Return (compile_ms, kernel_ms) using block_until_ready()."""
    if annotate:
        with StepTraceAnnotation(annotate + "|first"):
            t0 = time.time()
            fn_jit(*args).block_until_ready()
            first_ms = (time.time() - t0) * 1000.0
    else:
        t0 = time.time()
        fn_jit(*args).block_until_ready()
        first_ms = (time.time() - t0) * 1000.0

    for _ in range(warmup):
        fn_jit(*args).block_until_ready()

    t1 = time.time()
    for _ in range(iters):
        fn_jit(*args).block_until_ready()
    kernel_ms = (time.time() - t1) * 1000.0 / iters
    compile_ms = max(0.0, first_ms - kernel_ms)
    return compile_ms, kernel_ms

@torch.inference_mode()
def measure_torch_baseline(x, weight, bias, stride=1, padding=1, warmup=5, iters=50):
    """Return wall time for F.conv2d."""
    for _ in range(warmup):
        _ = F.conv2d(x, weight, bias, stride=stride, padding=padding)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = F.conv2d(x, weight, bias, stride=stride, padding=padding)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)  # ms
    return float(sum(times) / len(times)) # average time

if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())
    dev = jax.devices(DEVICE_KIND)[0]

    # (H, W, Cin, Cout, K, stride, padding_or_None)
    SHAPES = [
        (64,  64, 3,  8, 3, 1, None),
        (128,128, 3,  8, 3, 1, None),
        (128,128, 3,  8, 5, 1, None),
        (128,128, 3, 16, 7, 1, None),
        (256,256, 3, 16, 3, 1, None),
    ]

    results = []
    for (H, W, Cin, Cout, K, stride, pad_opt) in SHAPES:
        if pad_opt is None:
            padding = (K - 1) // 2
        else:
            padding = pad_opt
        N = 1
        torch_model = ConvModel(H, W, in_channels=Cin, out_channels=Cout,
                                kernel_size=K, stride=stride, padding=padding).to("cuda").eval()
        x_torch = torch.randn(N, Cin, H, W, device="cuda")

        ref = F.conv2d(x_torch, torch_model.weight, torch_model.bias, stride=stride, padding=padding).detach().cpu()

        with jax.default_device(dev):
            x_jax = jnp.asarray(x_torch.detach().cpu().numpy()) # (N,C,H,W)
            w_jax = jnp.asarray(torch_model.weight.detach().cpu().numpy()) # (Cout,Cin,K,K)
            b_jax = jnp.asarray(torch_model.bias.detach().cpu().numpy()) # (Cout,)
        
        conv2d_manual_jax_jit = jit(lambda x, w, b: conv2d_manual_jax(
            x, w, b, stride=stride, padding=padding
        ), backend="gpu")


        in_shape = tuple(x_jax.shape)
        f_shape  = tuple(w_jax.shape)

        label_manual = f"trial|variant=jax-manual|in={in_shape}|f={f_shape}"
        cm_jax, km_jax = measure_jax(conv2d_manual_jax_jit, x_jax, w_jax, b_jax,
                                     warmup=5, iters=50,
                                     annotate=label_manual)

        kb_torch = measure_torch_baseline(x_torch, torch_model.weight, torch_model.bias,
                                          stride=stride, padding=padding)

        jax_wall_ms = cm_jax + km_jax   
        torch_wall_ms = kb_torch

        print(f"[{in_shape} K={K}]  JAX-manual: compile≈{cm_jax:.2f} ms, kernel≈{km_jax:.3f} ms, wall≈{jax_wall_ms:.3f} ms")
        print(f"[{in_shape} K={K}]  Torch baseline: wall≈{torch_wall_ms:.3f} ms")

        if LOGDIR:
            os.makedirs(LOGDIR, exist_ok=True)
            jax.profiler.start_trace(LOGDIR, create_perfetto_trace=True)
            with jax.profiler.TraceAnnotation(label_manual + "|trace"):
                conv2d_manual_jax_jit(x_jax, w_jax, b_jax).block_until_ready()
            jax.profiler.stop_trace()

        out_jax = np.array(conv2d_manual_jax_jit(x_jax, w_jax, b_jax))
        out_jax_torch = torch.from_numpy(out_jax)
        atol = 1e-2 if K <= 3 else 1e-1
        ok = torch.allclose(out_jax_torch, ref, atol=atol)
        print(f"shape ok? {out_jax_torch.shape == ref.shape}   allclose@{atol}: {ok}")

        results.append({
            "in_shape": f"(N={N},C={Cin},H={H},W={W})",
            "filter": f"(Cout={Cout},Cin={Cin},K={K})",
            "padding": padding,
            "stride": stride,
            "jax_compile_ms": round(cm_jax, 3),
            "jax_kernel_ms": round(km_jax, 3),
            "jax_wall_ms": round(jax_wall_ms, 3),
            "torch_wall_ms": round(torch_wall_ms, 3),
            "speedup_torch_over_jax": round(jax_wall_ms / torch_wall_ms, 3) if torch_wall_ms > 0 else float("nan"),
        })

    os.makedirs("results", exist_ok=True)
    out_csv = "results/jax_vs_torch.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}")
