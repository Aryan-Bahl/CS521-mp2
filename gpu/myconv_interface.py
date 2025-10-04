import os, time, csv
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity

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
        times.append((time.time() - t0) * 1000.0)
    return sum(times) / len(times)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    conv_module = load(name="myconv",
                       sources=["./myconv_kernel.cu"],
                       verbose=True)

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
        x = torch.randn(N, Cin, H, W, device=device, dtype=torch.float32)
        w = torch.randn(Cout, Cin, K, K, device=device, dtype=torch.float32)
        
        with torch.inference_mode():
            y_custom = conv_module.conv_cuda(x, w, stride, padding)
            y_ref = F.conv2d(x, w, stride=stride, padding=padding)
            print(f"[{H}x{W} K={K}] same shape? {y_custom.shape == y_ref.shape}")
            print(f"[{H}x{W} K={K}] correct? {torch.allclose(y_custom, y_ref, atol=1e-4)}")

        custom_ms = measure_wall_time_ms(lambda: conv_module.conv_cuda(x, w, stride, padding))
        ref_ms = measure_wall_time_ms(lambda: F.conv2d(x, w, b, stride=stride, padding=padding))

        results.append({
            "in_shape": f"(N={N},C={Cin},H={H},W={W})",
            "filter": f"(Cout={Cout},Cin={Cin},K={K})",
            "padding": padding,
            "stride": stride,
            "custom_wall_ms": round(custom_ms, 3),
            "reference_wall_ms": round(ref_ms, 3),
            "speedup_custom_over_reference": round(ref_ms / custom_ms, 3) if custom_ms > 0 else float("nan"),
        })
        print(f"[{H}x{W} K={K}] custom {custom_ms:.3f} ms, reference {ref_ms:.3f} ms")

    os.makedirs("results", exist_ok=True)
    out_csv = "results/custom_vs_reference.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}")