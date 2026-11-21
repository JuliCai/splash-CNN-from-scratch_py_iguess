# answer for the todo 'maxpool2d_forward' - run this cell to define it
def maxpool2d_forward(x, C, H, W, P=2):
    """P x P max-pooling: read every window thru pool_layout and keep the biggest value of each"""
    windows = gather(x, pool_layout(C, H, W, P))  # window after window, P*P values each
    return [max(w) for w in rows(windows, P * P)]
