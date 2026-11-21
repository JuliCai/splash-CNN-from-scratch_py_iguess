# answer for the todo 'relu_forward' - run this cell to define it
def relu_forward(x):
    """max(0, v) for every number in a flat list"""
    return [v if v > 0.0 else 0.0 for v in x]
