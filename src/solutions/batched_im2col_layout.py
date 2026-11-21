# answer for the todo 'batched_im2col_layout' - run this cell to define it
def batched_im2col_layout(N, C, H, W, K):
    """im2col for a whole batch stored (N, C, H, W) row-major, so strides (C*H*W, H*W, W, 1).

    rows    = (n, oy, ox): every output pixel of every image  -> mode (N, OH, OW)
    columns = (c, i, j):   a spot in the patch, same as before"""
    OH, OW = H - K + 1, W - K + 1
    return Layout(((N, OH, OW), (C, K, K)),
                  ((C * H * W, W, 1), (H * W, W, 1)))
