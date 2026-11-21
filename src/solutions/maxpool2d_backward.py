# answer for the todo 'maxpool2d_backward' - run this cell to define it
def maxpool2d_backward(dy, x, C, H, W, P=2):
    """y = maxpool(x). each window's gradient goes to the one pixel that won the max, the rest get 0"""
    L = pool_layout(C, H, W, P)
    windows = rows(gather(x, L), P * P)
    offs = offsets(L)
    dx = [0.0] * len(x)
    for k, (g, w) in enumerate(zip(dy, windows)):
        winner = w.index(max(w))           # which of the P*P spots had the max
        dx[offs[k * P * P + winner]] += g  # where that spot lives in x
    return dx
