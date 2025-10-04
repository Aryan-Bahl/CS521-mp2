import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1
    
    assert in_channels % 128 == 0
    assert out_channels % 128 == 0
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = nl.tile_size.pmax
    
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    for b in nl.affine_range(batch_size):
        for oc_tile in nl.affine_range(n_tiles_c_out):
            oc_start = oc_tile * c_out_pmax
            bias_tile = nl.load(bias[nl.ds(oc_start, c_out_pmax)])

            for oh in nl.affine_range(out_height):
                accum = nl.zeros((c_out_pmax, out_width), dtype=X.dtype, buffer=nl.sbuf)
                for ow in nl.affine_range(out_width):
                    accum[:, ow] = bias_tile

                for ic_tile in nl.sequential_range(n_tiles_c_in):
                    ic_start = ic_tile * c_in_pmax
                    
                    w_slab = nl.load(W[
                        nl.ds(oc_start, c_out_pmax),
                        nl.ds(ic_start, c_in_pmax),
                        nl.ds(0, filter_height),
                        nl.ds(0, filter_width)
                    ])
                    
                    for fh in nl.sequential_range(filter_height):
                        ih = oh + fh
                        for fw in nl.sequential_range(filter_width):
                            w_2d = w_slab[:, :, fh, fw]  # shape = (c_out_pmax, c_in_pmax)
                            
                            x_tile = nl.load(X[b, nl.ds(ic_start, c_in_pmax), ih, nl.ds(fw, out_width)])
                            
                            w_2d_t = nl.transpose(w_2d)  # shape = (c_in_pmax, c_out_pmax)
                            
                            # nc_matmul computes w_2d_t.T @ x_tile which is equivalent to w_2d @ x_tile
                            # this is the reason I transposed the weight matrix above -> probably a better way to do this
                            update = nisa.nc_matmul(w_2d_t, x_tile)
                            accum[...] = nl.add(accum, update)

                nl.store(X_out[b, nl.ds(oc_start, c_out_pmax), oh, nl.ds(0, out_width)], value=accum)

    return X_out