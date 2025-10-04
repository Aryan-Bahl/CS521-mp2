import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity

# Compile and load CUDA extension
conv_module = load(name="myconv",
                     sources=["./myconv_kernel.cu"],
                     verbose=True)

# Input parameters
N, C_in, H, W = 4, 3, 25, 25
C_out, KH, KW = 4, 6, 6
stride, pad = 1, 1

# Allocate tensors
x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
w = torch.randn(C_out, C_in, KH, KW, device="cuda", dtype=torch.float32)

# Run o4 kernel
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("custom_conv"):
        out_custom = conv_module.conv_cuda(x, w, stride, pad)
    with record_function("reference_conv"):
        out_ref = torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)

# Test shape and correctness
print("CUDA --- shape check:", out_custom.shape == out_ref.shape)
print("CUDA --- correctness check:", torch.allclose(out_custom, out_ref, atol=1e-4))

prof.export_chrome_trace(f"traces/myconv_interface_trace.json")
print("interface profiler trace exported")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))