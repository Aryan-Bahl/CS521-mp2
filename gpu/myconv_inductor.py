import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 2, 3, 19, 19
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    scripted_model = torch.compile(model, backend="inductor")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("inductor_conv"):
            out = scripted_model(x)
        with record_function("reference_conv"):
            conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
    
    prof.export_chrome_trace(f"traces/myconv_inductor_trace.json")
    print("inductor profiler trace exported")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))