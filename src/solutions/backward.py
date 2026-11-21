# answer for the todo 'backward' - run this cell to define it
def backward(self):
    """backpropagation: fills in .grad of every tensor this one was made from.

    1. topological order: every tensor comes after the tensors it was made from
    2. d(self)/d(self) = 1
    3. go thru that order backwards and call each tensor's backward_fn (the chain rule, one op at a time)"""
    topo, seen = [], set()

    def build(node):
        if id(node) not in seen:
            seen.add(id(node))
            for child in node.children:
                build(child)
            topo.append(node)
    build(self)
    self.grad = [1.0] * len(self.data)
    for node in reversed(topo):
        if node.backward_fn is not None:
            node.backward_fn()


Tensor.backward = backward
