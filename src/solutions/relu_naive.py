# answer for the todo 'relu_naive' - run this cell to define it
def relu_naive(x):
    """max(0, v) for every number v in a (nested) list"""
    if isinstance(x, list):
        return [relu_naive(v) for v in x]
    return x if x > 0 else 0.0
