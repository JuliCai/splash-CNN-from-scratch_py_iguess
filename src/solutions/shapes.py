# answer for the todo 'shapes' - run this cell to define it
def conv_out(size, k):
    """how wide the output is after a k x k conv with no padding and stride 1"""
    return size - k + 1


def pool_out(size, p):
    """how wide the output is after p x p max-pooling (the windows dont overlap)"""
    return size // p


def conv_params(c_in, c_out, k):
    """weights + biases in a conv layer: c_out filters, each one c_in x k x k, plus one bias per filter"""
    return c_out * c_in * k * k + c_out


def linear_params(n_in, n_out):
    """weights + biases in a fully connected layer: a weight for every (input, output) pair, plus a bias per output"""
    return n_in * n_out + n_out
