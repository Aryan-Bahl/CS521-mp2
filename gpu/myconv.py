import os, csv, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda import nvtx

torch.backends.cudnn.benchmark = True  # helps conv baselines

def _out_size(H, W, kernel_size, stride, padding):
    out_h = (H - kernel_size + 2 * padding) // stride + 1
    out_w = (W - kernel_size + 2 * padding) // stride + 1
    return out_h, out_w

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, tile_k=64):
        super().__init__()
        self.tile_k = tile_k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.H = H
        self.W = W
        self.out_h, self.out_w = _out_size(H, W, kernel_size, stride, padding)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def im2col_manual(self, x):
        N = x.shape[0]
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        x_pad = F.pad(x, (P, P, P, P))
        sN, sC, sH, sW = x_pad.stride()

        patches = x_pad.as_strided(
            size=(N, C, out_h, out_w, KH, KW),
            stride=(sN, sC, sH*S, sW*S, sH, sW),
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(N, out_h*out_w, C*KH*KW)
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        C_in = self.in_channels
        KH = KW = self.kernel_size
        out_h = self.out_h
        out_w = self.out_w
        M = out_h * out_w
        K = C_in * KH * KW

        cols = self.im2col_manual(x)
        W_flat = self.weight.reshape(C_out, K)

        T = self.tile_k
        out_cols = torch.zeros((N, M, C_out), device=x.device, dtype=x.dtype)
        for k0 in range(0, K, T):
            k1 = min(k0 + T, K)
            out_cols += cols[:, :, k0:k1] @ W_flat[:, k0:k1].T

        out_cols += self.bias.view(1, 1, C_out)
        out = out_cols.permute(0, 2, 1).reshape(N, C_out, out_h, out_w)
        return out

    def forward(self, x):
        return self.conv2d_manual(x)

def baseline_conv(x, weight, bias, stride, padding):
    return F.conv2d(x, weight, bias, stride=stride, padding=padding)

@torch.inference_mode()
def measure_kernel_ms(run_fn, iters=50, warmup=10):
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)

    for _ in range(warmup):
        _ = run_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        starter.record()
        _ = run_fn()
        ender.record()
        ender.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    return sum(times) / len(times)

@torch.inference_mode()
def measure_wall_time_ms(run_fn, iters=50, warmup=10):
    """Measure total wall time including all overhead."""
    for _ in range(warmup):
        _ = run_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        _ = run_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)  # ms
    return sum(times) / len(times)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # (H, W, Cin, Cout, K, stride, padding)
    SHAPES = [
        (64,  64,  4,  8, 3, 1, 1),
        (128, 128, 4,  8, 3, 1, 1),
        (128, 128, 4,  8, 7, 1, 3),
        (256, 256, 4, 16, 3, 1, 1),
    ]

    results = []
    for (H, W, Cin, Cout, K, stride, padding) in SHAPES:
        N = 2
        x = torch.randn(N, Cin, H, W, device=device)
        manual = ConvModel(H, W, Cin, Cout, K, stride=stride, padding=padding, tile_k=64).to(device).eval()

        with torch.inference_mode():
            y_manual = manual(x)
            y_base   = baseline_conv(x, manual.weight, manual.bias, stride, padding)
            print(f"[{H}x{W} K={K}] same shape? {y_manual.shape == y_base.shape}")
            print(f"[{H}x{W} K={K}] correct? {torch.allclose(y_manual, y_base, atol=1e-3 if K<=3 else 1e-1)}")

        base_ms = measure_wall_time_ms(lambda: baseline_conv(x, manual.weight, manual.bias, stride, padding))
        man_ms  = measure_wall_time_ms(lambda: manual(x))

        results.append({
            "in_shape": f"(N={N},C={Cin},H={H},W={W})",
            "filter":   f"(Cout={Cout},Cin={Cin},K={K})",
            "padding":  padding,
            "stride":   stride,
            "baseline_ms": round(base_ms, 3),
            "manual_ms":   round(man_ms, 3),
            "speedup_baseline_over_manual": round(man_ms / base_ms, 3) if base_ms > 0 else float("nan"),
        })
        print(f"[{H}x{W} K={K}] baseline {base_ms:.3f} ms, manual {man_ms:.3f} ms")

    os.makedirs("results", exist_ok=True)
    out_csv = "results/myconv_vs_baseline.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}")
