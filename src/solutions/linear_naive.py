# answer for the todo 'linear_naive' - run this cell to define it
def linear_naive(x, weights, biases):
    """fully connected layer: out[o] = biases[o] + the sum over i of weights[o][i] * x[i]"""
    out = []
    for o in range(len(weights)):
        total = biases[o]
        for i in range(len(x)):
            total += weights[o][i] * x[i]
        out.append(total)
    return out
