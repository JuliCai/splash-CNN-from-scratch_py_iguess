# answer for the todo 'softmax' - run this cell to define it
def softmax(logits):
    """turns scores into probabilities that add up to 1: exp(z) / sum(exp(z))"""
    m = max(logits)  # taking away the max doesnt change the answer, it just stops exp() from overflowing
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]
