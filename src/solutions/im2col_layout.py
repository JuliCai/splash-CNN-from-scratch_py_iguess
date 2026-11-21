# answer for the todo 'im2col_layout' - run this cell to define it
def im2col_layout(C, H, W, K):
    """the patches of a K x K convolution as a layout over a C x H x W image stored row-major (strides H*W, W, 1).

    row p    = an output pixel (oy, ox)        -> mode (OH, OW)
    column j = a spot (c, i, j) in the patch    -> mode (C, K, K)
    pixel (c, oy + i, ox + j) lives at  c*H*W + (oy + i)*W + (ox + j)
                                     = oy*W + ox*1  +  c*H*W + i*W + j*1"""
    OH, OW = H - K + 1, W - K + 1
    return Layout(((OH, OW), (C, K, K)),
                  ((W, 1), (H * W, W, 1)))
