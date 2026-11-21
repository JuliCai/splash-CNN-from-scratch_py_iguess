# answer for the todo 'linear_forward' - run this cell to define it
def linear_forward(x, weight, bias):
    """fully connected layer: out = W x + b. weight is a flat n_out * n_in list (row o = output o's weights)"""
    out = matmul_t(rows(weight, len(x)), [x])  # n_out x 1
    return [v + b for v, b in zip(out, bias)]
