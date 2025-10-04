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

    out_pool_height = out_height
    out_pool_width = out_width
    
    assert in_channels % 128 == 0
    assert out_channels % 128 == 0
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for oc in nl.affine_range(out_channels):
            bias_scalar = nl.load(bias[oc])

            for oh in nl.affine_range(out_height):
                accum = nl.add(bias_scalar, nl.zeros((1, out_width), dtype=X.dtype, buffer=nl.sbuf))

                for fh in nl.sequential_range(filter_height):
                    ih = oh + fh
                    for fw in nl.sequential_range(filter_width):
                        for ic_tile in nl.sequential_range(n_tiles_c_in):
                            ic_start = ic_tile * c_in_pmax
                            for i in nl.sequential_range(c_in_pmax):
                                ic = ic_start + i
                                x_row = nl.load(X[b, ic, nl.ds(ih, 1), nl.ds(fw, out_width)])
                                w_val = nl.load(W[oc, ic, fh, fw])
                                update = nl.multiply(w_val, x_row)
                                accum[...] = nl.add(accum, update)

                nl.store(X_out[b, oc, nl.ds(oh, 1), nl.ds(0, out_width)], value=accum)

    return X_out