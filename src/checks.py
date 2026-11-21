"""
checks.py - the automatic checks for the lab notebooks.

every check runs your function on a few inputs and compares the answer with the answer key (babytorch.py).
it prints
    pass   when everything matches
    fail   with the first thing that didnt match (and a hint)
    todo   when the function still raises NotImplementedError (you havent written it yet)
"""

import math
import random
import time
import traceback

import babytorch as ref


class _Fail(Exception):
    pass


def _check(name, hint=""):
    def deco(test):
        def run(*args, **kwargs):
            try:
                msg = test(*args, **kwargs)
                print(f"pass  {name}" + (f" - {msg}" if msg else ""))
                return None   # nothing to display, the printed line is the feedback
            except NotImplementedError:
                print(f"todo  {name}: not written yet (it still raises NotImplementedError). "
                      f"write the todo cell, run it, then run this check again")
            except _Fail as e:
                print(f"fail  {name}: {e}" + (f"\n      hint: {hint}" if hint else ""))
            except Exception as e:  # noqa: BLE001 - show students any crash, short
                frames = traceback.extract_tb(e.__traceback__)
                where = next((f for f in reversed(frames) if "checks.py" not in f.filename), frames[-1])
                print(f"fail  {name}: your code crashed with {type(e).__name__}: {e}\n      at line {where.lineno}: {where.line}"
                      + (f"\n      hint: {hint}" if hint else ""))
            return None
        run.__name__ = test.__name__
        run.__doc__ = test.__doc__
        return run
    return deco


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [v for item in x for v in _flat(item)]
    return [x]


def _same(got, want, what, tol=1e-6):
    """compares numbers / (nested) lists and raises _Fail describing the first difference"""
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)):
            raise _Fail(f"{what} should be a list, but got {type(got).__name__}: {str(got)[:80]}")
        if len(got) != len(want):
            raise _Fail(f"{what} should have {len(want)} things in it, but it has {len(got)}")
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{what}[{i}]", tol)
        return
    if got is None or isinstance(got, (list, tuple, str)):
        raise _Fail(f"{what} should be the number {want:.6g}, but got {str(got)[:80]}")
    if not math.isclose(got, want, rel_tol=tol, abs_tol=tol):
        raise _Fail(f"{what} should be {want:.6g}, but got {got:.6g}")


def _rnd(n, seed):
    r = random.Random(seed)
    return [r.uniform(-1, 1) for _ in range(n)]


# ============================================================================================== lab 0
@_check("shapes", "conv: size - k + 1 · pool: size // p · conv weights: c_out*c_in*k*k + c_out · linear: n_in*n_out + n_out")
def shapes(conv_out, pool_out, conv_params, linear_params):
    _same(conv_out(32, 5), 28, "conv_out(32, 5)")
    _same(conv_out(14, 5), 10, "conv_out(14, 5)")
    _same(pool_out(28, 2), 14, "pool_out(28, 2)")
    _same(pool_out(10, 2), 5, "pool_out(10, 2)")
    _same(conv_params(3, 6, 5), 456, "conv_params(3, 6, 5)")
    _same(conv_params(6, 16, 5), 2416, "conv_params(6, 16, 5)")
    _same(linear_params(400, 120), 48120, "linear_params(400, 120)")
    total = (conv_params(3, 6, 5) + conv_params(6, 16, 5) + linear_params(400, 120) + linear_params(120, 84)
             + linear_params(84, 10))
    _same(total, 62006, "lenet-5's total weight count")
    return "lenet-5 has 62,006 numbers to learn"


# ============================================================================================== lab 1
@_check("conv2d_naive", "out[k][y][x] = biases[k] + the sum over c, i, j of image[c][y+i][x+j] * kernels[k][c][i][j]")
def conv2d_naive(fn):
    for (C, H, W, K, Co, s) in [(1, 4, 4, 3, 1, 1), (2, 5, 6, 3, 2, 2), (3, 8, 7, 5, 2, 3)]:
        img = ref.nest(_rnd(C * H * W, s), (C, H, W))
        ker = ref.nest(_rnd(Co * C * K * K, s + 10), (Co, C, K, K))
        b = _rnd(Co, s + 20)
        _same(fn(img, ker, b), ref.conv2d_naive(img, ker, b), f"conv of a {C}x{H}x{W} image with {Co} {K}x{K} filters: out")
    return "matches the answer key on 3 images"


@_check("relu_naive", "if x is a list give back [relu_naive(v) for v in x], otherwise x if x > 0 else 0.0")
def relu_naive(fn):
    x = [[[-1.0, 2.0], [0.5, -3.0]], [[0.0, -0.1], [7.0, 1.0]]]
    _same(fn(x), ref.relu_naive(x), "relu_naive(nested)")
    _same(fn([-2.0, 3.0]), [0.0, 3.0], "relu_naive([-2.0, 3.0])")


@_check("maxpool2d_naive", "window (y, x) covers rows y*p .. y*p+p-1 and columns x*p .. x*p+p-1")
def maxpool2d_naive(fn):
    x = ref.nest(_rnd(2 * 4 * 6, 3), (2, 4, 6))
    _same(fn(x), ref.maxpool2d_naive(x), "maxpool2d_naive(2x4x6)")
    _same(fn([[[1, 2], [3, 4]]]), [[[4]]], "maxpool2d_naive([[[1, 2], [3, 4]]])")


@_check("flatten_naive", "go channel by channel, row by row: every map, every row, every number")
def flatten_naive(fn):
    x = ref.nest(list(range(24)), (2, 3, 4))
    _same(fn(x), list(range(24)), "flatten_naive(2x3x4)")


@_check("linear_naive", "out[o] = biases[o] + the sum over i of weights[o][i] * x[i]")
def linear_naive(fn):
    x, w, b = _rnd(5, 1), ref.nest(_rnd(15, 2), (3, 5)), _rnd(3, 3)
    _same(fn(x, w, b), ref.linear_naive(x, w, b), "linear_naive(5 -> 3)")


@_check("softmax", "exp(z - max) / sum(exp(z - max)). take away the max first so exp() cant overflow")
def softmax(fn):
    z = _rnd(10, 4)
    _same(fn(z), ref.softmax(z), "softmax(random)")
    p = fn([1000.0, 1001.0, 1002.0])
    _same(p, ref.softmax([1000.0, 1001.0, 1002.0]), "softmax([1000, 1001, 1002])")
    _same(sum(fn(z)), 1.0, "the sum of the probabilities")
    return "the probabilities add up to 1, even for huge scores"


# ============================================================================================== lab 2
@_check("crd2idx", "sum(c * d for c, d in zip(coord, stride))")
def crd2idx(fn):
    _same(fn((1, 5, 7), (1024, 32, 1)), 1191, "crd2idx((1, 5, 7), (1024, 32, 1))")
    _same(fn((2, 3), (1, 4)), 14, "crd2idx((2, 3), (1, 4))")
    _same(fn((3, 9), (0, 1)), 9, "crd2idx((3, 9), (0, 1))")
    return "pixel (c=1, y=5, x=7) of a 3x32x32 image is at offset 1191"


@_check("row_major / col_major", "row-major: walk the shape from the right keeping a running product. column-major: from the left")
def strides(row_major, col_major):
    for shape, want in [((3, 32, 32), (1024, 32, 1)), ((4, 8), (8, 1)), ((2, 3, 4), (12, 4, 1)), ((5,), (1,))]:
        got = row_major(shape)
        if tuple(got) != want:
            raise _Fail(f"row_major({shape}) should be {want}, got {got}")
    for shape, want in [((3, 32, 32), (1, 3, 96)), ((4, 8), (1, 4)), ((2, 3, 4), (1, 2, 6))]:
        got = col_major(shape)
        if tuple(got) != want:
            raise _Fail(f"col_major({shape}) should be {want}, got {got}")


@_check("size / cosize", "size = the product of the shape. cosize = 1 + the sum of (s - 1) * d over every mode")
def size_cosize(size, cosize):
    cases = [ref.Layout((3, 32, 32)), ref.Layout((4, 8), (1, 4)), ref.Layout((2, 3), (0, 1)),
             ref.im2col_layout(1, 4, 4, 2), ref.Layout(((2, 2), 3), ((1, 2), 4)), ref.Layout(5, 3)]
    for L in cases:
        _same(size(L), ref.size(L), f"size({L})")
        _same(cosize(L), ref.cosize(L), f"cosize({L})")
    return "the 2x2 patch layout of a 4x4 image has size 36 but cosize 16"


@_check("transpose", "swap both pairs: Layout((shape[1], shape[0]), (stride[1], stride[0]))")
def transpose(fn):
    for L in [ref.Layout((4, 8), (8, 1)), ref.im2col_layout(3, 8, 8, 3), ref.Layout((2, 5), (0, 1))]:
        T, R = fn(L), ref.transpose(L)
        if (T.shape, T.stride) != (R.shape, R.stride):
            raise _Fail(f"transpose({L}) should be {R}, got {T}")


@_check("gather (simple)", "[storage[o] for o in offsets(layout)]")
def gather_simple(fn):
    storage = [v * 10 for v in range(100)]
    for L in [ref.Layout((2, 3), (10, 1)), ref.Layout((3, 2), (1, 10)), ref.Layout(4, 0), ref.im2col_layout(1, 4, 4, 2)]:
        _same(list(fn(storage, L)), list(ref.gather(storage, L)), f"gather(storage, {L})")


def _layout_like(L):
    return ref.Layout(L.shape, L.stride)


@_check("im2col_layout", "rows (OH, OW) step by (W, 1). columns (C, K, K) step by (H*W, W, 1)")
def im2col_layout(fn):
    for (C, H, W, K) in [(1, 4, 4, 2), (3, 32, 32, 5), (6, 14, 14, 5), (2, 5, 7, 3)]:
        mine, want = _layout_like(fn(C, H, W, K)), ref.im2col_layout(C, H, W, K)
        if ref.size(mine[0]) != ref.size(want[0]) or ref.size(mine[1]) != ref.size(want[1]):
            raise _Fail(f"im2col_layout{(C, H, W, K)} should have {ref.size(want[0])} rows and {ref.size(want[1])} "
                        f"columns, got {mine}")
        if ref.offsets(mine) != ref.offsets(want):
            p, j = next((k // ref.size(want[1]), k % ref.size(want[1])) for k, (a, b) in
                        enumerate(zip(ref.offsets(mine), ref.offsets(want))) if a != b)
            raise _Fail(f"for C,H,W,K = {C, H, W, K}: patch {p}, entry {j} should read offset {want(p, j)}, "
                        f"yours reads {mine(p, j)}  (yours: {mine}, should be {want})")
    return "a 5x5 patch of a 3x32x32 image is just a different way of reading the same memory"


@_check("pool_layout", "rows (C, H//P, W//P) step by (H*W, P*W, P). columns (P, P) step by (W, 1)")
def pool_layout(fn):
    for (C, H, W, P) in [(1, 4, 4, 2), (6, 28, 28, 2), (16, 10, 10, 2), (2, 6, 9, 3)]:
        mine, want = _layout_like(fn(C, H, W, P)), ref.pool_layout(C, H, W, P)
        if ref.offsets(mine) != ref.offsets(want):
            raise _Fail(f"pool_layout{(C, H, W, P)} should be {want}, got {mine}")


# ============================================================================================== lab 3
@_check("matmul_t", "[sumprod(a, b) for a in A for b in B], one flat list with row 0 of C first")
def matmul_t(fn):
    A, B = ref.rows(_rnd(15, 1), 5), ref.rows(_rnd(20, 2), 5)
    _same(list(fn(A, B)), ref.matmul_t(A, B), "matmul_t(3x5 rows, 4x5 rows) (a flat list of 12 numbers)")


def _speed(fn, ref_fn, args, label, factor=8.0):
    t = time.perf_counter(); ref_fn(*args); t_ref = time.perf_counter() - t
    t = time.perf_counter(); fn(*args); t_me = time.perf_counter() - t
    return "" if t_me < factor * max(t_ref, 1e-4) else f" (it works, but its {t_me / t_ref:.0f}x slower than the answer key {label})"


@_check("conv2d_forward", "patches = rows(gather(x, im2col_layout(...)), n), filters = rows(weight, n), "
        "out = matmul_t(filters, patches), then add bias[k] to row k")
def conv2d_forward(fn):
    for (C, H, W, K, Co) in [(3, 32, 32, 5, 6), (6, 14, 14, 5, 16), (2, 5, 6, 3, 3)]:
        x, w, b = _rnd(C * H * W, 1), _rnd(Co * C * K * K, 2), _rnd(Co, 3)
        _same(list(fn(x, w, b, C, H, W, K)), ref.conv2d_forward(x, w, b, C, H, W, K), f"conv2d_forward {C}x{H}x{W}, {Co} filters")
    x, w, b = _rnd(3072, 4), _rnd(450, 5), _rnd(6, 6)
    fn(x, w, b, 3, 32, 32, 5)
    return "same numbers as conv2d_naive" + _speed(fn, ref.conv2d_forward, (x, w, b, 3, 32, 32, 5), "conv")


@_check("maxpool2d_forward", "windows = gather(x, pool_layout(...)), then the max of every P*P numbers")
def maxpool2d_forward(fn):
    for (C, H, W) in [(6, 28, 28), (16, 10, 10), (2, 4, 6)]:
        x = _rnd(C * H * W, 7)
        _same(list(fn(x, C, H, W, 2)), ref.maxpool2d_forward(x, C, H, W, 2), f"maxpool2d_forward {C}x{H}x{W}")


@_check("linear_forward", "out = matmul_t(rows(weight, len(x)), [x]), then add the bias")
def linear_forward(fn):
    for (n_in, n_out) in [(400, 120), (5, 3)]:
        x, w, b = _rnd(n_in, 8), _rnd(n_in * n_out, 9), _rnd(n_out, 10)
        _same(list(fn(x, w, b)), ref.linear_forward(x, w, b), f"linear_forward {n_in} -> {n_out}")


@_check("relu_forward", "[v if v > 0.0 else 0.0 for v in x]")
def relu_forward(fn):
    x = _rnd(50, 11)
    _same(list(fn(x)), ref.relu_forward(x), "relu_forward")


# ============================================================================================== lab 4
@_check("cross_entropy_loss", "m = max(logits), loss = log(sum(exp(z - m))) + m - logits[label], probs = the softmax")
def cross_entropy_loss(fn):
    for z, lab in [(_rnd(10, 12), 3), ([0.0] * 10, 0), ([500.0, -500.0, 0.0], 1)]:
        loss, probs = fn(z, lab)
        rl, rp = ref.cross_entropy_loss(z, lab)
        _same(loss, rl, f"loss for label {lab}")
        _same(list(probs), rp, "probs")
    return f"a blind guess (all scores equal) costs ln(10) = {math.log(10):.3f}"


@_check("cosine_annealing_lr", "lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))")
def cosine_annealing_lr(fn):
    for t, T in [(0, 100), (25, 100), (50, 100), (99, 100), (100, 100)]:
        _same(fn(t, T, 0.01, 0.0001), ref.cosine_annealing_lr(t, T, 0.01, 0.0001), f"cosine_annealing_lr({t}, {T}, 0.01, 0.0001)")


@_check("SGD.step", "for every param with a grad: p.data = [w - self.lr * g for w, g in zip(p.data, p.grad)], "
        "then self.t += 1 and update self.lr")
def sgd(SGD, Param):
    for mom in (0.0, 0.9):
        mine = [Param([1.0, 2.0, 3.0]), Param([-1.0]), Param([5.0])]
        theirs = [ref.Param([1.0, 2.0, 3.0]), ref.Param([-1.0]), ref.Param([5.0])]
        o1 = SGD(mine, 0.1, 0.001, total_steps=4, schedule=ref.cosine_annealing_lr, momentum=mom)
        o2 = ref.SGD(theirs, 0.1, 0.001, total_steps=4, schedule=ref.cosine_annealing_lr, momentum=mom)
        for step in range(3):
            for p, q, g in zip(mine, theirs, ([0.5, -1.0, 2.0], [3.0], None)):
                p.grad = None if g is None else list(g)
                q.grad = None if g is None else list(g)
            o1.step(); o2.step()
            for k, (p, q) in enumerate(zip(mine, theirs)):
                _same(p.data, q.data, f"param {k} after step {step + 1} (momentum={mom})")
            _same(o1.lr, o2.lr, f"lr after step {step + 1}")
    return "plain sgd and momentum both match"


@_check("DataLoader", "image(k): the pixels are the 3,072 bytes after the label. channel ch is pixels[ch*1024:(ch+1)*1024], "
        "every byte v -> (v/255 - MEAN[ch]) / STD[ch]")
def dataloader(DataLoader):
    mine, want = DataLoader(ref.DATA / "test.bin"), ref.DataLoader(ref.DATA / "test.bin")
    _same(len(mine), 150, "len(DataLoader(test.bin))")
    _same(list(mine.image(37)), want.image(37), "image 37")
    got = list(mine)
    _same([lab for _, lab in got], [lab for _, lab in want], "the labels in file order")
    shuf = DataLoader(ref.DATA / "test.bin", shuffle=True, seed=1)
    a, b = [lab for _, lab in shuf], [lab for _, lab in shuf]
    if a == [lab for _, lab in got] or a == b:
        raise _Fail("shuffle=True should go thru the images in a new random order every epoch")
    if sorted(a) != sorted(lab for _, lab in got):
        raise _Fail("shuffling should only change the order, not which images you get")
    small = DataLoader(ref.DATA / "train.bin", per_class=3)
    _same(len(small), 30, "len(DataLoader(train.bin, per_class=3))")
    return "150 test images, normalized, shuffled differently every epoch"


# ============================================================================================== lab 5
@_check("Tensor.backward", "1) build a topological order with build(node)  2) self.grad = [1.0] * len(self.data)  "
        "3) call backward_fn on every node in REVERSED order")
def backward(Tensor):
    def scale(t, k):
        out = Tensor([v * k for v in t.data], t.shape, (t,), "scale")
        out.backward_fn = lambda: t.add_grad([g * k for g in out.grad])
        return out

    def add(a, b):
        out = Tensor([u + v for u, v in zip(a.data, b.data)], a.shape, (a, b), "add")

        def fn():
            a.add_grad(out.grad)
            b.add_grad(out.grad)
        out.backward_fn = fn
        return out

    def mul(a, b):
        out = Tensor([u * v for u, v in zip(a.data, b.data)], a.shape, (a, b), "mul")

        def fn():
            a.add_grad([g * v for g, v in zip(out.grad, b.data)])
            b.add_grad([g * u for g, u in zip(out.grad, a.data)])
        out.backward_fn = fn
        return out

    x = Tensor([2.0, -1.0], (2,), requires_grad=True)
    a = scale(x, 3.0)      # 3x
    b = add(x, a)          # 4x       (x gets used twice, a gets used twice: a diamond)
    y = mul(a, b)          # 12x^2  -> dy/dx = 24x
    y.backward()
    _same(x.grad, [48.0, -24.0], "x.grad for y = (3x) * (x + 3x) = 12x^2 at x = [2, -1]")
    _same(y.grad, [1.0, 1.0], "y.grad (the starting gradient)")
    return "the chain rule works, even when a tensor gets used twice"


@_check("relu_backward", "[g if v > 0.0 else 0.0 for g, v in zip(dy, x)]")
def relu_backward(fn):
    x, dy = _rnd(40, 13), _rnd(40, 14)
    _same(list(fn(dy, x)), ref.relu_backward(dy, x), "relu_backward")


@_check("linear_backward", "dx[i] = the sum over o of dy[o] * W[o][i] · dW[o][i] = dy[o] * x[i] · db = dy")
def linear_backward(fn):
    for (n_in, n_out) in [(4, 3), (400, 120)]:
        x, w, dy = _rnd(n_in, 15), _rnd(n_in * n_out, 16), _rnd(n_out, 17)
        dx, dw, db = fn(dy, x, w)
        rx, rw, rb = ref.linear_backward(dy, x, w)
        _same(list(dx), rx, f"dx ({n_in} -> {n_out})")
        _same(list(dw), rw, f"dweight ({n_in} -> {n_out})")
        _same(list(db), rb, f"dbias ({n_in} -> {n_out})")


@_check("maxpool2d_backward", "for window k: winner = w.index(max(w)), then dx[offs[k*P*P + winner]] += dy[k]")
def maxpool2d_backward(fn):
    for (C, H, W) in [(2, 4, 4), (6, 28, 28)]:
        x = _rnd(C * H * W, 18)
        dy = _rnd(C * H * W // 4, 19)
        _same(list(fn(dy, x, C, H, W, 2)), ref.maxpool2d_backward(dy, x, C, H, W, 2), f"dx ({C}x{H}x{W})")


@_check("conv2d_backward", "dbias = the row sums of dy, dweight = matmul_t(dy rows, patch columns), "
        "dx = scatter-add matmul_t(dy columns, filter columns) thru offsets(L)")
def conv2d_backward(fn):
    for (C, H, W, K, Co) in [(1, 4, 4, 2, 1), (2, 6, 5, 3, 3), (6, 14, 14, 5, 16)]:
        OH, OW = H - K + 1, W - K + 1
        x, w, dy = _rnd(C * H * W, 20), _rnd(Co * C * K * K, 21), _rnd(Co * OH * OW, 22)
        dx, dw, db = fn(dy, x, w, C, H, W, K)
        rx, rw, rb = ref.conv2d_backward(dy, x, w, C, H, W, K)
        _same(list(db), rb, f"dbias ({C}x{H}x{W}, {Co} filters of {K}x{K})")
        _same(list(dw), rw, f"dweight ({C}x{H}x{W}, {Co} filters of {K}x{K})")
        _same(list(dx), rx, f"dx ({C}x{H}x{W}, {Co} filters of {K}x{K})")
        none, dw2, _ = fn(dy, x, w, C, H, W, K, need_dx=False)
        if none is not None:
            raise _Fail("with need_dx=False give back (None, dweight, dbias) and skip the dx work")
    return "dx, dweight and dbias all match"


@_check("cross_entropy_backward", "d = list(probs), then d[label] -= 1.0")
def cross_entropy_backward(fn):
    probs = ref.softmax(_rnd(10, 23))
    _same(list(fn(probs, 4)), ref.cross_entropy_backward(probs, 4), "d(loss)/d(logits)")


@_check("gradient check (whole lenet-5)", "if only some layers are off, run the check for that layer's backward again")
def gradcheck_lenet(lenet5, Tensor, softmax_cross_entropy, DataLoader, n=12, eps=1e-5):
    model = lenet5(seed=7)
    image, label = DataLoader(ref.DATA / "test.bin").image(3), 0

    def loss_now():
        return softmax_cross_entropy(model(Tensor(image, (3, 32, 32))), label).data[0]
    loss = softmax_cross_entropy(model(Tensor(image, (3, 32, 32))), label)
    for p in model.parameters():
        p.grad = None
    loss.backward()
    rng = random.Random(0)
    worst = 0.0
    names = ["conv1.w", "conv1.b", "conv2.w", "conv2.b", "fc1.w", "fc1.b", "fc2.w", "fc2.b", "fc3.w", "fc3.b"]
    for _ in range(n):
        k = rng.randrange(len(model.parameters()))
        p = model.parameters()[k]
        i = rng.randrange(len(p.data))
        old = p.data[i]
        p.data[i] = old + eps; up = loss_now()
        p.data[i] = old - eps; down = loss_now()
        p.data[i] = old
        numeric = (up - down) / (2 * eps)
        err = abs(numeric - p.grad[i]) / max(1e-6, abs(numeric) + abs(p.grad[i]))
        worst = max(worst, err)
        if err > 1e-4:
            raise _Fail(f"{names[k]}[{i}]: backprop says {p.grad[i]:.6g} but nudging the weight says {numeric:.6g}")
    return f"backprop agrees with nudging on {n} random weights (worst relative error {worst:.1e})"


# ============================================================================================== lab 6
@_check("train_step", "logits = model(Tensor(image, (3, 32, 32))), loss = softmax_cross_entropy(logits, label), "
        "opt.zero_grad(), loss.backward(), opt.step()")
def train_step(train_step, lenet5, SGD, DataLoader):
    model = lenet5(seed=11)
    opt = SGD(model.parameters(), 0.01, 0.01, total_steps=10)
    image = DataLoader(ref.DATA / "test.bin").image(60)
    losses = []
    for _ in range(4):
        out = train_step(model, opt, image, 4)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise _Fail(f"train_step should give back (loss, right), got {out!r}")
        losses.append(out[0])
    if not losses[-1] < losses[0]:
        raise _Fail(f"training 4 times on the same image should lower its loss, but it went {losses[0]:.3f} -> {losses[-1]:.3f}")
    return f"4 steps on one image: loss {losses[0]:.3f} -> {losses[-1]:.3f}"


@_check("evaluate", "for every (image, label): guess, _ = model.predict(image), confusion[label][guess] += 1, "
        "accuracy = diagonal / total")
def evaluate(evaluate, lenet5, DataLoader):
    model = lenet5(seed=12)
    rmodel = ref.lenet5(seed=12)
    for p, q in zip(model.parameters(), rmodel.parameters()):
        q.data = list(p.data)
    acc, cm = evaluate(model, DataLoader(ref.DATA / "test.bin"), limit=40)
    racc, rcm = ref.evaluate(rmodel, ref.DataLoader(ref.DATA / "test.bin"), limit=40)
    _same(acc, racc, "accuracy on the first 40 test images")
    _same(cm, rcm, "the confusion matrix")


# ======================================================================================= bonus a, b
@_check("batched_im2col_layout", "add the batch to the rows: rows (N, OH, OW) step by (C*H*W, W, 1)")
def batched_im2col_layout(fn):
    import addons
    for (N, C, H, W, K) in [(2, 1, 4, 4, 2), (4, 3, 32, 32, 5), (3, 6, 14, 14, 5)]:
        mine, want = _layout_like(fn(N, C, H, W, K)), addons.batched_im2col_layout(N, C, H, W, K)
        if ref.offsets(mine) != ref.offsets(want):
            raise _Fail(f"batched_im2col_layout{(N, C, H, W, K)} should be {want}, got {mine}")


@_check("broadcast", "a stride of 0 means stepping that way doesnt move: Layout((n_rows, n_cols), (0, 1))")
def broadcast(fn):
    L = _layout_like(fn(3, 4))
    if ref.gather([10, 20, 30, 40], L) != (10, 20, 30, 40) * 3:
        raise _Fail(f"broadcast(3, 4) should read [10, 20, 30, 40] three times, it read {ref.gather([10, 20, 30, 40], L)}")


@_check("train_step_batched", "loss = softmax_cross_entropy_batched(logits, labels) is already the batch average, "
        "then zero_grad / backward / step once per batch")
def train_step_batched(fn):
    import addons  # noqa: F401
    model = ref.lenet5(seed=21)
    opt = ref.SGD(model.parameters(), 0.01, 0.01, total_steps=10)
    loader = ref.DataLoader(ref.DATA / "test.bin")
    images = [loader.image(k) for k in range(0, 150, 19)]
    labels = [loader.labels[k] for k in range(0, 150, 19)]
    first = fn(model, opt, images, labels)
    for _ in range(3):
        last = fn(model, opt, images, labels)
    if opt.t != 4:
        raise _Fail(f"4 calls should take 4 optimizer steps (one per batch), the optimizer took {opt.t}")
    if not last[0] < first[0]:
        raise _Fail(f"repeating a batch should lower its loss, it went {first[0]:.3f} -> {last[0]:.3f}")
    return f"loss on a repeated batch of {len(images)}: {first[0]:.3f} -> {last[0]:.3f}"


@_check("tree_reduce", "while more than one list is left, add neighbours in pairs ([p0+p1, p2+p3, ...]) and keep an odd one out")
def tree_reduce(fn):
    for n in (1, 2, 3, 4, 7, 8):
        parts = [_rnd(5, 30 + i) for i in range(n)]
        want = [sum(col) for col in zip(*parts)]
        _same(list(fn([list(p) for p in parts])), want, f"tree_reduce of {n} partial sums")


@_check("matmul_t_split_k", "chunk = ceil(K / splits), partial s = matmul_t over columns [s*chunk, (s+1)*chunk), "
        "then tree_reduce the partials")
def matmul_t_split_k(fn):
    for (M, N, K, s) in [(3, 4, 12, 3), (2, 5, 10, 4), (6, 75, 784, 8), (1, 1, 7, 7)]:
        A, B = ref.rows(_rnd(M * K, 40), K), ref.rows(_rnd(N * K, 41), K)
        _same(list(fn(A, B, s)), ref.matmul_t(A, B), f"split-k matmul ({M}x{K} times ({N}x{K})^T, {s} splits)", tol=1e-9)
    return "same answer as one big matmul (up to rounding in the last digit)"
