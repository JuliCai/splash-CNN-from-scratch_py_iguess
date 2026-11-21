# answer for the todo 'conv2d_backward' - run this cell to define it
def conv2d_backward(dy, x, weight, C, H, W, K, need_dx=True):
    """y = conv2d_forward(x, weight, bias). given dy (Cout x OH x OW), gives back (dx, dweight, dbias).

    forward was  out = filters @ patches^T  (+ a bias per filter), so by the matmul rules:
      dbias[k]    = the sum of feature map k's gradient
      dweight     = dy @ patches              (Cout x n, every weight saw one pixel per output spot)
      dpatches    = dy^T @ filters            (P x n, how much each patch entry mattered)
      dx          = add every dpatches entry back onto the pixel it was read from (the same layout!)"""
    n = C * K * K
    L = im2col_layout(C, H, W, K)
    P = size(L[0])                                 # output pixels
    Cout = len(weight) // n
    dy_rows = rows(dy, P)                          # Cout rows, one feature map's gradient each
    dbias = [sum(r) for r in dy_rows]
    patch_columns = rows(gather(x, transpose(L)), P)    # n rows: patch entry j at every output spot
    dweight = matmul_t(dy_rows, patch_columns)          # Cout x n
    if not need_dx:                                # the first layer's input is the image, nobody needs its gradient
        return None, dweight, dbias
    dy_columns = rows(gather(dy, Layout((P, Cout), (1, P))), Cout)       # P rows: every filter's dy at spot p
    filter_columns = rows(gather(weight, Layout((n, Cout), (1, n))), Cout)  # n rows: entry j of every filter
    dpatches = matmul_t(dy_columns, filter_columns)  # P x n, same order as offsets(L)
    dx = [0.0] * len(x)
    for o, g in zip(offsets(L), dpatches):         # scatter-add, the backward of a gather
        dx[o] += g
    return dx, dweight, dbias
