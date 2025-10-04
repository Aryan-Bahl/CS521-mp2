import os, time, csv
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

@torch.inference_mode()
def measure_kernel_ms(fn, warmup=10, iters=50):
    """Accurate GPU kernel time using CUDA events."""
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        starter.record()
        _ = fn()
        ender.record()
        ender.synchronize()
        times.append(starter.elapsed_time(ender))
    return float(sum(times) / len(times)) # average time

def measure_inductor_compile_and_kernel(compiled_call, warmup=10, iters=50):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = compiled_call()
    torch.cuda.synchronize()
    first_ms = (time.time() - t0) * 1000.0

    kernel_ms = measure_kernel_ms(compiled_call, warmup=warmup, iters=iters)
    compile_ms = max(0.0, first_ms - kernel_ms)
    return compile_ms, kernel_ms

@torch.inference_mode()
def measure_wall_time_ms(fn, warmup=10, iters=50):
    """Measure total wall time including all overhead."""
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)
    return float(sum(times) / len(times)) # average time

if __name__ == "__main__":
    assert torch.cuda.is_available(), "This script requires CUDA for timing with CUDA events."
    torch.manual_seed(0)
    device = torch.device("cuda")

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
        model = ConvModel(H, W, in_channels=Cin, out_channels=Cout,
                          kernel_size=K, stride=stride, padding=padding).to(device).eval()
        x = torch.randn(N, Cin, H, W, device=device)

        try:
            torch._dynamo.reset()
        except Exception:
            pass
        scripted_model = torch.compile(model, backend="inductor")

        inductor_call = lambda: scripted_model(x)
        reference_call = lambda: F.conv2d(x, model.weight, model.bias, stride=stride, padding=padding)

        with torch.inference_mode():
            y_ind = inductor_call()
            y_ref = reference_call()
            atol = 1e-4 if K <= 3 else 1e-1
            print(f"[NCHW={tuple(x.shape)}, Cout={Cout}, K={K}] "
                  f"shape_ok={y_ind.shape == y_ref.shape} allclose@{atol}={torch.allclose(y_ind, y_ref, atol=atol)}")

        comp_ms, ind_kernel_ms = measure_inductor_compile_and_kernel(inductor_call, warmup=10, iters=50)
        ref_wall_ms = measure_wall_time_ms(reference_call, warmup=10, iters=50)
        ind_wall_ms = comp_ms + ind_kernel_ms

        print(f"Inductor   => compile≈{comp_ms:.2f} ms, kernel≈{ind_kernel_ms:.3f} ms, wall≈{ind_wall_ms:.3f} ms")
        print(f"Reference  => wall≈{ref_wall_ms:.3f} ms")

        results.append({
            "in_shape": f"(N={N},C={Cin},H={H},W={W})",
            "filter": f"(Cout={Cout},Cin={Cin},K={K})",
            "padding": padding,
            "stride": stride,
            "inductor_compile_ms": round(comp_ms, 3),
            "inductor_kernel_ms": round(ind_kernel_ms, 3),
            "inductor_wall_ms": round(ind_wall_ms, 3),
            "reference_wall_ms": round(ref_wall_ms, 3),
            "speedup_reference_over_inductor": round(ind_wall_ms / ref_wall_ms, 3) if ref_wall_ms > 0 else float("nan"),
        })

    os.makedirs("results", exist_ok=True)
    out_csv = "results/inductor_vs_reference.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}")
