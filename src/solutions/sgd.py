# answer for the todo 'sgd' - run this cell to define it
class SGD:
    """stochastic gradient descent, like cudagrad's SGD<T>:   w  <-  w - lr * grad

    the learning rate follows a schedule from lr_max (step 0) to lr_min (step total_steps).
    momentum > 0 keeps a running "velocity" v <- momentum * v + grad and steps along v instead"""

    def __init__(self, params, lr_max=0.005, lr_min=1e-5, total_steps=1, schedule=constant_lr, momentum=0.0):
        self.params = list(params)
        self.lr_max, self.lr_min, self.total_steps = lr_max, lr_min, total_steps
        self.schedule, self.momentum = schedule, momentum
        self.velocity = [None] * len(self.params)
        self.t = 0
        self.lr = schedule(0, total_steps, lr_max, lr_min)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        lr, mu = self.lr, self.momentum
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if mu:
                v = self.velocity[i] or [0.0] * len(p.data)
                v = [mu * vi + g for vi, g in zip(v, p.grad)]
                self.velocity[i] = v
                p.data = [w - lr * vi for w, vi in zip(p.data, v)]
            else:
                p.data = [w - lr * g for w, g in zip(p.data, p.grad)]
        self.t += 1
        self.lr = self.schedule(self.t, self.total_steps, self.lr_max, self.lr_min)
