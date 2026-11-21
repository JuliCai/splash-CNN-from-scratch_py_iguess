"""
addons.py - the two bonus labs as code you can run (the answer key for the bonus notebooks).

  bonus a  batching            push N images thru at once. the batch is just one more mode in every layout
  bonus b  split-k reduction   cut the long sum (the K dimension) into slices, add up every slice at the same
                               time, then add the partial sums together (what cudagrad's conv2d_dW_kernel does
                               on the gpu)

the main lab trains one image at a time on purpose, with neither trick. everything here reuses babytorch.py:
the same Layout, gather, matmul_t, Tensor, SGD, modules and saved weights.
"""

import multiprocessing as mp
import time

from babytorch import (DATA, Conv, DataLoader, Flatten, Layout, Linear, MaxPool, SGD, Tensor, cosine_annealing_lr,
                       cross_entropy_backward, cross_entropy_loss, gather, lenet5, matmul_t, maxpool2d_backward,
                       maxpool2d_forward, offsets, relu, rows, softmax_cross_entropy, transpose)

# =====================================================================================================
# bonus a - batching
# =====================================================================================================

# >>> exercise: batched_im2col_layout
def batched_im2col_layout(N, C, H, W, K):
    """im2col for a whole batch stored (N, C, H, W) row-major, so strides (C*H*W, H*W, W, 1).

    rows    = (n, oy, ox): every output pixel of every image  -> mode (N, OH, OW)
    columns = (c, i, j):   a spot in the patch, same as before"""
    OH, OW = H - K + 1, W - K + 1
    return Layout(((N, OH, OW), (C, K, K)),
                  ((C * H * W, W, 1), (H * W, W, 1)))
# <<< exercise: batched_im2col_layout


# >>> exercise: broadcast
def broadcast(n_rows, n_cols):
    """reads a list of n_cols numbers as an n_rows x n_cols matrix where every row is that same list.
    a stride of 0 means stepping that way doesnt move in memory"""
    return Layout((n_rows, n_cols), (0, 1))
# <<< exercise: broadcast


# >>> provided: batched_ops
def conv2d_batched(x, weight, bias):
    """x: (N, C, H, W) -> (N, Cout, OH, OW), with ONE matrix multiply for the whole batch"""
    N, C, H, W = x.shape
    Cout, _, K, _ = weight.shape
    OH, OW = H - K + 1, W - K + 1
    P, n = OH * OW, C * K * K
    L = batched_im2col_layout(N, C, H, W, K)
    patches = rows(gather(x.data, L), n)                        # N*P rows
    out = matmul_t(rows(weight.data, n), patches)               # Cout x (N*P), stored as "C, N, H, W"
    nchw = gather(out, Layout((N, Cout, P), (P, N * P, 1)))     # read it back in N, C, H, W order
    b = gather(bias.data, Layout((N, Cout, P), (0, 1, 0)))      # bias[k] repeated over n and p (stride 0)
    y = Tensor([v + bb for v, bb in zip(nchw, b)], (N, Cout, OH, OW), (x, weight, bias), "conv2d_batched")

    def backward_fn():
        dy = gather(y.grad, Layout((Cout, N, P), (P, Cout * P, 1)))   # back to Cout x (N*P)
        dy_rows = rows(dy, N * P)
        bias.add_grad([sum(r) for r in dy_rows])
        # the weight gradient: Cout x n answers, each one a sum over K = N*P things -> thats bonus b (split-k)
        weight.add_grad(matmul_t(dy_rows, rows(gather(x.data, transpose(L)), N * P)))
        if x.requires_grad:
            dy_cols = rows(gather(dy, Layout((N * P, Cout), (1, N * P))), Cout)
            w_cols = rows(gather(weight.data, Layout((n, Cout), (1, n))), Cout)
            dx = [0.0] * len(x.data)
            for o, g in zip(offsets(L), matmul_t(dy_cols, w_cols)):
                dx[o] += g
            x.add_grad(dx)
    y.backward_fn = backward_fn
    return y


def maxpool2d_batched(x, p=2):
    """batching max-pool is free: an (N, C, H, W) batch is just N*C channels of H x W"""
    N, C, H, W = x.shape
    y = Tensor(maxpool2d_forward(x.data, N * C, H, W, p), (N, C, H // p, W // p), (x,), "maxpool2d_batched")
    y.backward_fn = lambda: x.add_grad(maxpool2d_backward(y.grad, x.data, N * C, H, W, p))
    return y


def flatten_batched(x):
    """(N, C, H, W) -> (N, C*H*W): the same memory, a new shape"""
    N = x.shape[0]
    y = Tensor(x.data, (N, len(x.data) // N), (x,), "flatten_batched")
    y.backward_fn = lambda: x.add_grad(y.grad)
    return y


def linear_batched(x, weight, bias):
    """x: (N, n_in) -> (N, n_out).   Y = X W^T + b, with the bias repeated down the batch"""
    N, n_in = x.shape
    n_out = bias.shape[0]
    X = rows(x.data, n_in)
    out = matmul_t(X, rows(weight.data, n_in))                  # N x n_out
    y = Tensor([v + b for v, b in zip(out, gather(bias.data, broadcast(N, n_out)))], (N, n_out),
               (x, weight, bias), "linear_batched")

    def backward_fn():
        dY_cols = rows(gather(y.grad, Layout((n_out, N), (1, n_out))), N)     # n_out rows of N
        bias.add_grad([sum(c) for c in dY_cols])                              # summed over the batch
        X_cols = rows(gather(x.data, Layout((n_in, N), (1, n_in))), N)        # n_in rows of N
        weight.add_grad(matmul_t(dY_cols, X_cols))                            # summed over the batch
        if x.requires_grad:
            W_cols = rows(gather(weight.data, Layout((n_in, n_out), (1, n_in))), n_out)
            x.add_grad(matmul_t(rows(y.grad, n_out), W_cols))                 # N x n_in
    y.backward_fn = backward_fn
    return y


def softmax_cross_entropy_batched(logits, labels):
    """the AVERAGE cross-entropy over the batch (so the learning rate doesnt depend on the batch size)"""
    N, n_cls = logits.shape
    results = [cross_entropy_loss(z, lab) for z, lab in zip(rows(logits.data, n_cls), labels)]
    out = Tensor([sum(loss for loss, _ in results) / N], (1,), (logits,), "softmax_cross_entropy_batched")

    def backward_fn():
        g = out.grad[0] / N
        logits.add_grad([v * g for (_, probs), lab in zip(results, labels)
                         for v in cross_entropy_backward(probs, lab)])
    out.backward_fn = backward_fn
    return out


def forward_batched(model, x):
    """runs a babytorch CNN (made of Conv / MaxPool / Flatten / Linear) on a whole batch (N, 3, 32, 32)"""
    for layer in model.layers:
        if isinstance(layer, Conv):
            x = conv2d_batched(x, layer.weight, layer.bias)
            x = relu(x) if layer.use_relu else x
        elif isinstance(layer, MaxPool):
            x = maxpool2d_batched(x, layer.p)
        elif isinstance(layer, Flatten):
            x = flatten_batched(x)
        elif isinstance(layer, Linear):
            x = linear_batched(x, layer.weight, layer.bias)
            x = relu(x) if layer.use_relu else x
        else:
            raise TypeError(f"dont know how to batch a {type(layer).__name__}")
    return x


def stack(images):
    """a list of N images (3,072 floats each) -> one (N, 3, 32, 32) tensor"""
    return Tensor([v for image in images for v in image], (len(images), 3, 32, 32))


def batches(loader, batch_size):
    """groups a DataLoader's (image, label) pairs into lists of batch_size"""
    images, labels = [], []
    for image, label in loader:
        images.append(image)
        labels.append(label)
        if len(images) == batch_size:
            yield images, labels
            images, labels = [], []
    if images:
        yield images, labels
# <<< provided: batched_ops


# >>> exercise: train_step_batched
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
# <<< exercise: train_step_batched


# >>> provided: train_batched
def train_batched(model, loader, epochs=1, batch_size=16, lr_max=0.02, lr_min=1e-5, log_every=50):
    """like babytorch.train, but one sgd step per batch. gives back images per second"""
    steps = epochs * -(-len(loader) // batch_size)
    opt = SGD(model.parameters(), lr_max, lr_min, total_steps=steps, schedule=cosine_annealing_lr)
    start, seen = time.time(), 0
    for epoch in range(1, epochs + 1):
        run_loss, run_right, n = 0.0, 0, 0
        for images, labels in batches(loader, batch_size):
            loss, right = train_step_batched(model, opt, images, labels)
            run_loss += loss * len(images)
            run_right += right
            n += len(images)
            seen += len(images)
            if opt.t % log_every == 0 or n == len(loader):
                print(f"epoch {epoch}/{epochs}  image {n:>6}/{len(loader)}  loss {run_loss / n:.3f}  "
                      f"train acc {run_right / n:6.1%}  lr {opt.lr:.5f}  ({seen / (time.time() - start):.0f} images/s)",
                      flush=True)
    return seen / (time.time() - start)
# <<< provided: train_batched


# =====================================================================================================
# bonus b - split-k reduction
# =====================================================================================================

# >>> exercise: tree_reduce
def tree_reduce(partials):
    """adds up a list of same-length lists in pairs, the way a gpu thread block does it in shared memory:
        [p0, p1, p2, p3]  ->  [p0+p1, p2+p3]  ->  [p0+p1+p2+p3]       (log2(n) rounds)
    every add inside one round is independent, so a gpu does a whole round at the same time"""
    while len(partials) > 1:
        paired = [[a + b for a, b in zip(partials[i], partials[i + 1])] for i in range(0, len(partials) - 1, 2)]
        if len(partials) % 2:
            paired.append(partials[-1])   # an odd one out waits for the next round
        partials = paired
    return partials[0]
# <<< exercise: tree_reduce


# >>> exercise: matmul_t_split_k
def matmul_t_split_k(A, B, splits):
    """C = A @ B^T like matmul_t, but the K dimension (the sum) gets cut into `splits` slices.
    every slice gives a partial C (they could all run at the same time), then the partials get added up"""
    K = len(A[0])
    chunk = -(-K // splits)                       # K / splits rounded up
    partials = []
    for s in range(splits):
        lo, hi = s * chunk, min(K, (s + 1) * chunk)
        partials.append(matmul_t([a[lo:hi] for a in A], [b[lo:hi] for b in B]))
    return tree_reduce(partials)
# <<< exercise: matmul_t_split_k


# >>> provided: parallel_split_k
_WORKER_MODEL = None


def worker_partial_grads(job):
    """runs inside a worker process (one "thread block"): forward + backward for its slice of the batch,
    one image at a time, WITHOUT clearing the gradients in between, so they add up into a partial sum"""
    global _WORKER_MODEL
    params, images, labels = job
    if _WORKER_MODEL is None:
        _WORKER_MODEL = lenet5()
    model = _WORKER_MODEL
    for p, data in zip(model.parameters(), params):
        p.data, p.grad = data, None
    total_loss, correct = 0.0, 0
    for image, label in zip(images, labels):
        logits = model(Tensor(image, (3, 32, 32)))
        loss = softmax_cross_entropy(logits, label)
        loss.backward()
        total_loss += loss.data[0]
        correct += max(range(len(logits.data)), key=logits.data.__getitem__) == label
    return [p.grad for p in model.parameters()], total_loss, correct


class SplitKTrainer:
    """training on many cores, seen as split-k: the weight gradient is a sum over the batch, so every worker
    adds up its own slice of the batch (a partial sum), and the main process adds the partials up and steps.

        workers  ~ thread blocks along blockIdx.y in cudagrad's conv2d_dW_kernel
        reduce   ~ the shared memory tree reduction + atomicAdd into dW"""

    def __init__(self, model, opt, workers=4):
        self.model, self.opt, self.workers = model, opt, workers
        self.pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None

    def gradients(self, images, labels, reduce):
        """fills every weight's .grad with the batch average gradient. gives back (average loss, how many right)"""
        params = [p.data for p in self.model.parameters()]
        chunk = -(-len(images) // self.workers)
        jobs = [(params, images[i:i + chunk], labels[i:i + chunk]) for i in range(0, len(images), chunk)]
        results = self.pool.map(worker_partial_grads, jobs) if self.pool else [worker_partial_grads(j) for j in jobs]
        N = len(images)
        for k, p in enumerate(self.model.parameters()):
            total = reduce([grads[k] for grads, _, _ in results])     # the split-k reduction
            p.grad = [g / N for g in total]
        return sum(r[1] for r in results) / N, sum(r[2] for r in results)

    def step(self, images, labels, reduce=tree_reduce):
        loss, correct = self.gradients(images, labels, reduce)
        self.opt.step()
        return loss, correct

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
# <<< provided: parallel_split_k


def _demo():  # python3 addons.py : a quick speed comparison
    loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=20)
    data = list(loader)[:64]
    images, labels = [d[0] for d in data], [d[1] for d in data]
    for workers in (1, 2, 4, 8):
        model = lenet5()
        opt = SGD(model.parameters(), 0.02, 0.02, total_steps=10)
        with SplitKTrainer(model, opt, workers) as trainer:
            trainer.step(images[:8], labels[:8])   # warm up the pool
            t = time.time()
            trainer.step(images, labels)
            print(f"{workers} worker(s): {(time.time() - t) * 1000:.0f} ms for a batch of 64")


if __name__ == "__main__":
    _demo()
