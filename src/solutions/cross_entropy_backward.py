# answer for the todo 'cross_entropy_backward' - run this cell to define it
def cross_entropy_backward(probs, label):
    """d(loss)/d(logits) for softmax + cross-entropy: just probs - one_hot(label)"""
    d = list(probs)
    d[label] -= 1.0
    return d
