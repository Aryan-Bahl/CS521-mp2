import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

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

        # TO DO: Define static shapes here. 

        # Precompute output size
        self.out_h, self.out_w = _out_size(H, W, kernel_size, stride, padding)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        '''
        input is x: (N, C, H, W)
        output is patches: (N, out_h*out_w, C*KH*KW)
        '''
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        _, _, Hp, Wp = x.shape
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.
        sN, sC, sH, sW = x_pad.stride()

        patches = x_pad.as_strided(
            size=(N, C, out_h, out_w, KH, KW),
            stride=(sN, sC, sH*S, sW*S, sH, sW),
        )
        
        patches = patches.permute(0, 2, 3, 1, 4, 5) # (N, out_h, out_w, C, KH, KW) - rearranging dimensions for better grouping
        patches = patches.reshape(N, out_h*out_w, C*KH*KW)
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


        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)          

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        W_flat = self.weight.reshape(C_out, K)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        T = self.tile_k
        out_cols = torch.zeros((N, M, C_out), device=x.device, dtype=x.dtype)
        for k0 in range(0, K, T):
            k1 = min(k0 + T, K)
            out_cols += cols[:, :, k0:k1] @ W_flat[:, k0:k1].T

        # TO DO: 4) Add bias.
        out_cols += self.bias.view(1, 1, C_out) # shape = (N, M, C_out) where M = out_h * out_w

        # TO DO: 5) reshape output (N, M, C_out) into shape (N, C_out, out_h, out_w).
        out = out_cols.permute(0, 2, 1).reshape(N, C_out, out_h, out_w)
        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 22, 22
    x = torch.randn(N, C, H, W)
    out_channels=8
    kernel_size=7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1, tile_k=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("manual_conv"):
            out = model(x)
        with record_function("reference_conv"):
            conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)

    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-1)) # change atol to 1e-1 for larger kernel size - talk about in report

    # export profiler trace
    prof.export_chrome_trace(f"gpu/traces/myconv_trace.json")
    print("pytorch profiler trace exported")
    # print profiler trace
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))