#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    extern __shared__ float shmem[];  // shared memory for partial sums
    float *w_tile = shmem;
    
    // TO DO : Tiled matrix multiplication by using shmem
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (n >= N || oh >= out_h || ow >= out_w) return;

    float acc[TILE_C];
    #pragma unroll
    for (int t = 0; t < TILE_C; t++) {
        acc[t] = 0.0f;
    }

    for (int oc0 = 0; oc0 < C_out; oc0 += TILE_C) {
        int tile = min(TILE_C, C_out - oc0); // tile size

        // reduce over (ic, kh, kw)
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * stride + kh - pad;
                bool in_h = (unsigned)ih < (unsigned)H;

                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * stride + kw - pad;
                    bool in_w = (unsigned)iw < (unsigned)W;

                    float x_val = 0.0f;
                    if (in_h && in_w) {
                        x_val = x[((((n * C_in + ic) * H) + ih) * W) + iw];
                    }

                    int lin = threadIdx.y * blockDim.x + threadIdx.x;
                    for (int t = lin; t < tile; t += blockDim.x * blockDim.y) {
                        int oc = oc0 + t;
                        if (oc < C_out) {
                            w_tile[t] = w[((((oc * C_in + ic) * KH) + kh) * KW) + kw];
                        }
                    }
                    __syncthreads();

                    // fma across the tile
                    #pragma unroll
                    for (int t = 0; t < tile; ++t) {
                        acc[t] = fmaf(x_val, w_tile[t], acc[t]);
                    }
                    __syncthreads();
                }
            }
        }  
        
        #pragma unroll
        for (int t = 0; t < tile; ++t) {
            int oc = oc0 + t;
            out[((((n * C_out + oc) * out_h) + oh) * out_w) + ow] = acc[t];
            acc[t] = 0.0f;
        }
    }   
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + 2*pad - KH) / stride + 1;
    int out_w = (W + 2*pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());
    size_t shmem_bytes = static_cast<size_t>(TILE_C) * sizeof(float);

    dim3 block(8, 8);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N);

    gemm_gpu_o4_kernel<<<grid, block, shmem_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
