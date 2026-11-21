# answer for the todo 'relu_backward' - run this cell to define it
def relu_backward(dy, x):
    """y = relu(x). the gradient goes thru where x > 0 and gets blocked (0) where x <= 0"""
    return [g if v > 0.0 else 0.0 for g, v in zip(dy, x)]
