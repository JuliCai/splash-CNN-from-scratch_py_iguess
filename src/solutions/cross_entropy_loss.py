# answer for the todo 'cross_entropy_loss' - run this cell to define it
def cross_entropy_loss(logits, label):
    """softmax + cross-entropy: loss = -log(probability of the right class). gives back (loss, probs).

    its written as log(sum(exp(z - m))) + m - z[label] so it never ends up doing log(0)"""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    loss = math.log(total) + m - logits[label]
    return loss, [e / total for e in exps]
