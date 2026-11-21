# answer for the todo 'linear_backward' - run this cell to define it
def linear_backward(dy, x, weight):
    """y = W x + b with W stored as n_out rows of n_in. given dy = d(loss)/dy, gives back (dx, dweight, dbias).

    dx = W^T dy           (column i of W dotted with dy)
    dW = dy x^T           (every dy times every x: dW[o][i] = dy[o] * x[i])
    db = dy"""
    n_in, n_out = len(x), len(dy)
    columns = rows(gather(weight, Layout((n_in, n_out), (1, n_in))), n_out)  # the transpose of W, as rows
    dx = matmul_t(columns, [dy])
    dweight = [g * v for g in dy for v in x]
    return dx, dweight, list(dy)
