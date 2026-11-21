# answer for the todo 'train_step' - run this cell to define it
def train_step(model, opt, image, label):
    """one step of learning on one image. gives back (loss, was the guess right?)"""
    logits = model(Tensor(image, (3, 32, 32)))      # 1. forward propagation
    loss = softmax_cross_entropy(logits, label)     # 2. how wrong were we
    opt.zero_grad()                                 # 3. forget the last step's gradients
    loss.backward()                                 # 4. back propagation: d(loss)/d(every weight)
    opt.step()                                      # 5. nudge every weight downhill
    guess = max(range(len(logits.data)), key=logits.data.__getitem__)
    return loss.data[0], guess == label
