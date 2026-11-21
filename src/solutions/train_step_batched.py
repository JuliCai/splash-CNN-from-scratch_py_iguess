# answer for the todo 'train_step_batched' - run this cell to define it
def train_step_batched(model, opt, images, labels):
    """one sgd step on a whole batch. gives back (average loss, how many guesses were right)"""
    logits = forward_batched(model, stack(images))               # 1. forward the whole batch at once
    loss = softmax_cross_entropy_batched(logits, labels)         # 2. average loss over the batch
    opt.zero_grad()                                              # 3. same three lines as before...
    loss.backward()                                              # 4.   ...but once per batch,
    opt.step()                                                   # 5.   not once per image
    n_cls = logits.shape[1]
    guesses = [max(range(n_cls), key=r.__getitem__) for r in rows(logits.data, n_cls)]
    return loss.data[0], sum(g == lab for g, lab in zip(guesses, labels))
