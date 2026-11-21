# answer for the todo 'pool_layout' - run this cell to define it
def pool_layout(C, H, W, P):
    """the P x P max-pool windows of a C x H x W image, one window per row.

    row    = a window (c, oy, ox)    -> mode (C, H//P, W//P), the window starts at c*H*W + oy*P*W + ox*P
    column = a spot (i, j) inside it  -> mode (P, P), offset i*W + j"""
    return Layout(((C, H // P, W // P), (P, P)),
                  ((H * W, P * W, P), (W, 1)))
