"""
babytorch.py - lenet-5 from scratch in plain python. no numpy, no pytorch, just python and math.

this is the answer key for the lab notebooks in this folder. its in the same order as the lab and the
names copy cudagrad / infanttorch:

  lab 0  the data and the architecture        shapes, how many weights
  lab 1  layers the slow obvious way          conv / relu / max-pool / fully connected on nested lists
  lab 2  how data is stored                   one flat list + cute style layouts (shape:stride), views, im2col
  lab 3  layers = layouts + one operator      the fast forward pass: gather + matrix multiply
  lab 4  learning                             loss, sgd, cosine annealing learning rate, data loading
  lab 5  forward + back propagation           an autograd engine (cudagrad's Node) and the backward kernels
  lab 6  train lenet-5 on memes               modules, the training loop, checking accuracy

only the standard library gets used. python 3.12+ is best becuase of math.sumprod (a dot product done in c).
older versions fall back to sum(map(mul, a, b)) which gives the same answer, just slower.

train from the terminal:
    python3 babytorch.py --per-class 350 --epochs 4            # 4-6 min, about 83% on the real test memes
    python3 babytorch.py --epochs 4 --save weights/mine.json    # all 35,000 images, about 46 min
    python3 babytorch.py --load weights/lenet_meme10.json       # just check saved weights

lines between "# >>> exercise: NAME" and "# <<< exercise: NAME" are what students write in the notebook,
"provided" blocks show up as given code. tools/build_notebooks.py reads these markers.
"""

import json
import math
import operator
import os
import random
import sys
import time
from pathlib import Path

try:
    from math import sumprod  # python 3.12+: sumprod(a, b) = a[0]*b[0] + a[1]*b[1] + ..., done in c
except ImportError:  # pragma: no cover - older pythons
    def sumprod(a, b):
        return sum(map(operator.mul, a, b))


# =====================================================================================================
# lab 0 - the data and the architecture
# =====================================================================================================

# >>> provided: data_basics
RECORD = 1 + 3 * 32 * 32  # one image in a .bin file = 1 label byte + 3,072 pixel bytes (red plane, green plane, blue plane)


def find_data():
    """finds the folder with the meme dataset (the data/ folder next to src/)"""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in [os.environ.get("MEME10_DATA"), here.parent / "data", Path.cwd().parent / "data",
                      Path.cwd() / "data"]:
        if candidate and (Path(candidate) / "classes.txt").exists():
            return Path(candidate)
    raise FileNotFoundError("cant find the data/ folder. put it next to src/ or set MEME10_DATA=/path/to/data")


DATA = find_data()
CLASSES = (DATA / "classes.txt").read_text().split()


def load_bin(path):
    """reads a .bin file into a list of (label, pixels) pairs. pixels is 3,072 bytes, each one 0..255"""
    raw = Path(path).read_bytes()
    return [(raw[i], raw[i + 1:i + RECORD]) for i in range(0, len(raw), RECORD)]
# <<< provided: data_basics


# >>> exercise: shapes
def conv_out(size, k):
    """how wide the output is after a k x k conv with no padding and stride 1"""
    return size - k + 1


def pool_out(size, p):
    """how wide the output is after p x p max-pooling (the windows dont overlap)"""
    return size // p


def conv_params(c_in, c_out, k):
    """weights + biases in a conv layer: c_out filters, each one c_in x k x k, plus one bias per filter"""
    return c_out * c_in * k * k + c_out


def linear_params(n_in, n_out):
    """weights + biases in a fully connected layer: a weight for every (input, output) pair, plus a bias per output"""
    return n_in * n_out + n_out
# <<< exercise: shapes


# =====================================================================================================
# lab 1 - layers the slow obvious way (nested lists, nested loops)
# =====================================================================================================

# >>> provided: nested_image
def to_nested(pixels, C=3, H=32, W=32):
    """flat CHW pixels (bytes or numbers) -> image[c][y][x], scaled to 0..1"""
    return [[[pixels[c * H * W + y * W + x] / 255 for x in range(W)] for y in range(H)] for c in range(C)]
# <<< provided: nested_image


# >>> exercise: conv2d_naive
def conv2d_naive(image, kernels, biases):
    """image[c][y][x] is C x H x W, kernels[k][c][i][j] is Cout x C x K x K, biases[k] is one number per filter.
    gives back out[k][y][x], one feature map per filter.

    every output pixel is one dot product: lay the filter over the image, multiply what lines up, add it all up"""
    C, H, W = len(image), len(image[0]), len(image[0][0])
    K = len(kernels[0][0])
    OH, OW = H - K + 1, W - K + 1
    out = []
    for k in range(len(kernels)):  # one feature map per filter
        fmap = []
        for y in range(OH):
            row = []
            for x in range(OW):
                total = biases[k]
                for c in range(C):
                    for i in range(K):
                        for j in range(K):
                            total += image[c][y + i][x + j] * kernels[k][c][i][j]
                row.append(total)
            fmap.append(row)
        out.append(fmap)
    return out
# <<< exercise: conv2d_naive


# >>> exercise: relu_naive
def relu_naive(x):
    """max(0, v) for every number v in a (nested) list"""
    if isinstance(x, list):
        return [relu_naive(v) for v in x]
    return x if x > 0 else 0.0
# <<< exercise: relu_naive


# >>> exercise: maxpool2d_naive
def maxpool2d_naive(fmaps, p=2):
    """keeps the biggest value in every p x p window (stride p) of every feature map"""
    out = []
    for fmap in fmaps:
        H, W = len(fmap), len(fmap[0])
        out.append([[max(fmap[y * p + i][x * p + j] for i in range(p) for j in range(p))
                     for x in range(W // p)]
                    for y in range(H // p)])
    return out
# <<< exercise: maxpool2d_naive


# >>> exercise: flatten_naive
def flatten_naive(fmaps):
    """C x H x W nested lists -> one flat list (channel by channel, row by row)"""
    return [v for fmap in fmaps for row in fmap for v in row]
# <<< exercise: flatten_naive


# >>> exercise: linear_naive
def linear_naive(x, weights, biases):
    """fully connected layer: out[o] = biases[o] + the sum over i of weights[o][i] * x[i]"""
    out = []
    for o in range(len(weights)):
        total = biases[o]
        for i in range(len(x)):
            total += weights[o][i] * x[i]
        out.append(total)
    return out
# <<< exercise: linear_naive


# >>> exercise: softmax
def softmax(logits):
    """turns scores into probabilities that add up to 1: exp(z) / sum(exp(z))"""
    m = max(logits)  # taking away the max doesnt change the answer, it just stops exp() from overflowing
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]
# <<< exercise: softmax


# >>> provided: lenet_naive
def he_init(n, fan_in):
    """n random weights from a bell curve with std sqrt(2 / fan_in) ("he" init, same as cudagrad's Init::He)"""
    std = math.sqrt(2.0 / fan_in)
    return [random.gauss(0.0, std) for _ in range(n)]


def nest(flat_list, shape):
    """turns a flat list into nested lists: nest(list(range(6)), (2, 3)) -> [[0, 1, 2], [3, 4, 5]]"""
    if len(shape) == 1:
        return list(flat_list)
    step = len(flat_list) // shape[0]
    return [nest(flat_list[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


LENET_SHAPES = {  # name -> shape of every weight and bias in lenet-5
    "conv1.w": (6, 3, 5, 5), "conv1.b": (6,),
    "conv2.w": (16, 6, 5, 5), "conv2.b": (16,),
    "fc1.w": (120, 400), "fc1.b": (120,),
    "fc2.w": (84, 120), "fc2.b": (84,),
    "fc3.w": (10, 84), "fc3.b": (10,),
}


def init_lenet_params(seed=0):
    """random lenet-5 parameters as flat lists (weights use he init, biases start at 0)"""
    random.seed(seed)
    params = {}
    for name, shape in LENET_SHAPES.items():
        n = math.prod(shape)
        params[name] = he_init(n, n // shape[0]) if name.endswith(".w") else [0.0] * n
    return params


def lenet_forward_naive(image, params):
    """lenet-5's forward pass built out of the lab 1 layers. gives back 10 class probabilities"""
    P = {name: nest(v, LENET_SHAPES[name]) for name, v in params.items()}
    x = maxpool2d_naive(relu_naive(conv2d_naive(image, P["conv1.w"], P["conv1.b"])))   # 6 x 14 x 14
    x = maxpool2d_naive(relu_naive(conv2d_naive(x, P["conv2.w"], P["conv2.b"])))       # 16 x 5 x 5
    x = flatten_naive(x)                                                                # 400
    x = relu_naive(linear_naive(x, P["fc1.w"], P["fc1.b"]))                             # 120
    x = relu_naive(linear_naive(x, P["fc2.w"], P["fc2.b"]))                             # 84
    return softmax(linear_naive(x, P["fc3.w"], P["fc3.b"]))                             # 10
# <<< provided: lenet_naive


# =====================================================================================================
# lab 2 - how data is stored. memory is one long list, a *layout* says where things are in it.
# the Layout here copies nvidia's cute: a layout is a shape and a stride, and it maps coordinates to offsets
#     offset = coord[0] * stride[0] + coord[1] * stride[1] + ...
# two differences from real cute, on purpose (to match python / pytorch / cudagrad):
#   * default strides are row-major (cute's LayoutRight). cute defaults to column-major (LayoutLeft)
#   * one integer index counts thru a layout row-major (last mode fastest). cute counts the first mode fastest
# =====================================================================================================

# >>> provided: tuple_helpers
def flat(t):
    """flattens a nested tuple: ((28, 28), (3, 5, 5)) -> (28, 28, 3, 5, 5)"""
    if isinstance(t, tuple):
        return tuple(x for item in t for x in flat(item))
    return (t,)


def unflat(values, like):
    """puts flat values back into the nesting of `like`: unflat((1, 2, 3), ((0, 0), 0)) -> ((1, 2), 3)"""
    values = iter(values)

    def rebuild(t):
        return tuple(rebuild(x) for x in t) if isinstance(t, tuple) else next(values)
    return rebuild(like)


def idx2crd(i, shape):
    """one integer index -> a coordinate, counting row-major: idx2crd(5, (2, 3)) -> (1, 2)"""
    if not isinstance(shape, tuple):
        return i
    crd = []
    for s in reversed(shape):
        n = math.prod(flat(s))
        crd.append(idx2crd(i % n, s))
        i //= n
    return tuple(reversed(crd))


def fmt(t):
    """cute style text for a (nested) tuple: (3, (5, 5)) -> '(3,(5,5))'"""
    return "(" + ",".join(fmt(x) for x in t) + ")" if isinstance(t, tuple) else str(t)
# <<< provided: tuple_helpers


# >>> exercise: crd2idx
def crd2idx(coord, stride):
    """the heart of a layout: offset = coord[0]*stride[0] + coord[1]*stride[1] + ... (both are flat tuples)"""
    return sum(c * d for c, d in zip(coord, stride))
# <<< exercise: crd2idx


# >>> exercise: strides
def row_major(shape):
    """compact row-major strides (cute calls it LayoutRight). the last mode moves fastest. (3, 32, 32) -> (1024, 32, 1)"""
    strides, step = [], 1
    for s in reversed(shape):
        strides.append(step)
        step *= s
    return tuple(reversed(strides))


def col_major(shape):
    """compact column-major strides (cute calls it LayoutLeft). the first mode moves fastest. (3, 32, 32) -> (1, 3, 96)"""
    strides, step = [], 1
    for s in shape:
        strides.append(step)
        step *= s
    return tuple(strides)
# <<< exercise: strides


# >>> provided: layout_class
class Layout:
    """a cute style layout: a shape and a stride with the same (maybe nested) structure.

    its a function from coordinates to memory offsets:  Layout((3, 32, 32), (1024, 32, 1))(1, 5, 7) == 1191.
    it prints the cute way, shape:stride  ->  (3,32,32):(1024,32,1)
    leave out the stride and you get compact row-major strides"""

    def __init__(self, shape, stride=None):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        if stride is None:
            stride = unflat(row_major(flat(self.shape)), self.shape)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        if ... in flat(self.shape) + flat(self.stride):  # a fill in the blank todo that isnt filled in yet
            raise NotImplementedError("todo: replace every ... in the layout with a number")
        if unflat(flat(self.stride), self.shape) != self.stride:
            raise ValueError(f"shape {fmt(self.shape)} and stride {fmt(self.stride)} need the same structure")

    def __call__(self, *coord):
        """coordinate -> offset. each coordinate can be a tuple (one entry per sub-mode) or one integer"""
        if len(coord) == 1 and len(self.shape) > 1:   # one integer for the whole layout: count row-major
            coord = idx2crd(coord[0], self.shape)
        return crd2idx(_flat_coord(coord, self.shape), flat(self.stride))

    def __getitem__(self, i):
        """mode i as its own layout:  Layout((3, 32, 32))[0] == Layout(3, 1024)"""
        return Layout(self.shape[i], self.stride[i])

    def __len__(self):
        return len(self.shape)  # the rank, how many modes

    def __eq__(self, other):
        return isinstance(other, Layout) and (self.shape, self.stride) == (other.shape, other.stride)

    def __hash__(self):
        return hash((self.shape, self.stride))

    def __repr__(self):
        if len(self.shape) == 1:
            return f"{fmt(self.shape[0])}:{fmt(self.stride[0])}"
        return f"{fmt(self.shape)}:{fmt(self.stride)}"


def _flat_coord(coord, shape):
    """turns a coordinate (where some entries can be one integer for a nested mode) into a flat tuple"""
    if not isinstance(shape, tuple):
        return (coord,)
    if not isinstance(coord, tuple):
        coord = idx2crd(coord, shape)
    if len(coord) != len(shape):
        raise ValueError(f"coordinate {coord} doesnt match shape {fmt(shape)}")
    return tuple(x for c, s in zip(coord, shape) for x in _flat_coord(c, s))


def make_layout(*layouts):
    """glues layouts together into one layout whose modes are their modes (cute's make_layout(a, b))"""
    return Layout(tuple(L.shape if len(L) > 1 else L.shape[0] for L in layouts),
                  tuple(L.stride if len(L) > 1 else L.stride[0] for L in layouts))
# <<< provided: layout_class


# >>> exercise: size_cosize
def size(layout):
    """how many coordinates the layout has, the product of its shape"""
    return math.prod(flat(layout.shape))


def cosize(layout):
    """how much memory it reaches: 1 + the offset of its last coordinate (cute: A(size(A) - 1) + 1)"""
    return 1 + sum((s - 1) * d for s, d in zip(flat(layout.shape), flat(layout.stride)))
# <<< exercise: size_cosize


# >>> provided: print_layout
def print_layout(layout):
    """prints a rank-2 layout as a table of offsets, like cute's print_layout"""
    if len(layout) == 1:
        layout = make_layout(Layout(1, 0), layout)
    if len(layout) != 2:
        raise ValueError("print_layout needs a rank-2 layout (rows, columns)")
    M, N = size(layout[0]), size(layout[1])
    cells = [[layout(i, j) for j in range(N)] for i in range(M)]
    w = max(len(str(v)) for row in cells for v in row)
    w = max(w, len(str(N - 1)))
    line = "    +" + "+".join("-" * (w + 2) for _ in range(N)) + "+"
    print(layout)
    print("     " + " ".join(f" {j:>{w}} " for j in range(N)))
    print(line)
    for i, row in enumerate(cells):
        print(f"{i:>3} |" + "|".join(f" {v:>{w}} " for v in row) + "|")
        print(line)
# <<< provided: print_layout


# >>> exercise: transpose
def transpose(layout):
    """swaps the two modes of a rank-2 layout. no data moves, only the layout changes"""
    (s0, s1), (d0, d1) = layout.shape, layout.stride
    return Layout((s1, s0), (d1, d0))
# <<< exercise: transpose


# >>> provided: offsets
_OFFSETS = {}


def offsets(layout):
    """every offset a layout points at, in row-major coordinate order (kept in a cache, it only depends on the layout).

    built one mode at a time: start from [0]. a mode with size s and stride d turns every offset o
    into o, o + d, o + 2d, ..., o + (s-1)d"""
    key = (layout.shape, layout.stride)
    if key not in _OFFSETS:
        offs = [0]
        for s, d in zip(flat(layout.shape), flat(layout.stride)):
            offs = [o + i * d for o in offs for i in range(s)]
        _OFFSETS[key] = offs
    return _OFFSETS[key]
# <<< provided: offsets


# >>> exercise: gather_simple
def gather(storage, layout):
    """reads memory thru a layout: one value for every coordinate, in row-major order"""
    return [storage[o] for o in offsets(layout)]
# <<< exercise: gather_simple


# >>> provided: gather_fast
_GETTERS = {}


def gather(storage, layout):  # noqa: F811 -- replaces the simple version: same answer, much faster
    """reads memory thru a layout (fast version).

    the offsets only depend on the layout so we work them out once and keep an operator.itemgetter,
    which grabs all of them in one call that runs in c. (cutlass precomputes its offsets too)"""
    key = (layout.shape, layout.stride)
    getter = _GETTERS.get(key)
    if getter is None:
        offs = offsets(layout)
        getter = operator.itemgetter(*offs) if len(offs) > 1 else (lambda s, o=offs[0]: (s[o],))
        _GETTERS[key] = getter
    return getter(storage)
# <<< provided: gather_fast


# >>> exercise: im2col_layout
def im2col_layout(C, H, W, K):
    """the patches of a K x K convolution as a layout over a C x H x W image stored row-major (strides H*W, W, 1).

    row p    = an output pixel (oy, ox)        -> mode (OH, OW)
    column j = a spot (c, i, j) in the patch    -> mode (C, K, K)
    pixel (c, oy + i, ox + j) lives at  c*H*W + (oy + i)*W + (ox + j)
                                     = oy*W + ox*1  +  c*H*W + i*W + j*1"""
    OH, OW = H - K + 1, W - K + 1
    return Layout(((OH, OW), (C, K, K)),
                  ((W, 1), (H * W, W, 1)))
# <<< exercise: im2col_layout


# >>> exercise: pool_layout
def pool_layout(C, H, W, P):
    """the P x P max-pool windows of a C x H x W image, one window per row.

    row    = a window (c, oy, ox)    -> mode (C, H//P, W//P), the window starts at c*H*W + oy*P*W + ox*P
    column = a spot (i, j) inside it  -> mode (P, P), offset i*W + j"""
    return Layout(((C, H // P, W // P), (P, P)),
                  ((H * W, P * W, P), (W, 1)))
# <<< exercise: pool_layout


# >>> provided: zipped_divide
def zipped_divide(layout, tiler):
    """cuts every mode of a (flat) layout into tiles, like cute's zipped_divide for simple integer tilers.

    a mode with size s and stride d, tiled by t, turns into (t, s//t) with strides (d, t*d).
    you get ((tile modes...), (rest modes...)), so "where am i inside a tile" and "which tile" """
    tiles, rests, tile_d, rest_d = [], [], [], []
    for s, d, t in zip(layout.shape, layout.stride, tiler):
        if s % t:
            raise ValueError(f"tile {t} doesnt divide a mode of size {s}")
        tiles.append(t); rests.append(s // t); tile_d.append(d); rest_d.append(t * d)
    return Layout((tuple(tiles), tuple(rests)), (tuple(tile_d), tuple(rest_d)))
# <<< provided: zipped_divide


# =====================================================================================================
# lab 3 - layers = layouts + one operator. these are "kernels": plain functions on flat lists, like
# cudagrad's launch_*_forward kernels. every image is a flat C*H*W list in CHW order.
# =====================================================================================================

# >>> provided: rows
def rows(flat_list, n):
    """cuts a flat list into rows of length n, so you can use it as a (len // n) x n matrix"""
    return [flat_list[i:i + n] for i in range(0, len(flat_list), n)]
# <<< provided: rows


# >>> exercise: matmul_t
def matmul_t(A, B):
    """C = A times B-transposed for two lists of rows: C[i][j] = dot(A[i], B[j]).

    every row has the same length (the K dimension, the thing being added up). cutlass would say both
    sides are "K-major". gives back C as one flat row-major list of len(A) * len(B) numbers"""
    return [sumprod(a, b) for a in A for b in B]
# <<< exercise: matmul_t


# >>> exercise: conv2d_forward
def conv2d_forward(x, weight, bias, C, H, W, K):
    """K x K convolution of a C x H x W image (flat CHW list) with Cout filters (flat Cout*C*K*K list).

    the whole layer is: read the patches thru im2col_layout, then one matrix multiply.
    gives back the Cout x OH x OW feature maps as one flat list (CHW order, ready for the next layer)"""
    n = C * K * K                                           # numbers in one patch = numbers in one filter
    patches = rows(gather(x, im2col_layout(C, H, W, K)), n)  # OH*OW rows, one patch per output pixel
    filters = rows(weight, n)                                # Cout rows, one filter each
    out = matmul_t(filters, patches)                         # Cout x (OH*OW), row k = feature map k
    P = len(patches)
    return [v + b for b, fmap in zip(bias, rows(out, P)) for v in fmap]
# <<< exercise: conv2d_forward


# >>> exercise: maxpool2d_forward
def maxpool2d_forward(x, C, H, W, P=2):
    """P x P max-pooling: read every window thru pool_layout and keep the biggest value of each"""
    windows = gather(x, pool_layout(C, H, W, P))  # window after window, P*P values each
    return [max(w) for w in rows(windows, P * P)]
# <<< exercise: maxpool2d_forward


# >>> exercise: linear_forward
def linear_forward(x, weight, bias):
    """fully connected layer: out = W x + b. weight is a flat n_out * n_in list (row o = output o's weights)"""
    out = matmul_t(rows(weight, len(x)), [x])  # n_out x 1
    return [v + b for v, b in zip(out, bias)]
# <<< exercise: linear_forward


# >>> exercise: relu_forward
def relu_forward(x):
    """max(0, v) for every number in a flat list"""
    return [v if v > 0.0 else 0.0 for v in x]
# <<< exercise: relu_forward


# >>> provided: lenet_fast
def lenet_forward(pixels, params):
    """the same lenet-5 forward pass as lenet_forward_naive, on flat lists. gives back 10 class probabilities"""
    x = [v / 255 for v in pixels]
    x = maxpool2d_forward(relu_forward(conv2d_forward(x, params["conv1.w"], params["conv1.b"], 3, 32, 32, 5)), 6, 28, 28)
    x = maxpool2d_forward(relu_forward(conv2d_forward(x, params["conv2.w"], params["conv2.b"], 6, 14, 14, 5)), 16, 10, 10)
    x = relu_forward(linear_forward(x, params["fc1.w"], params["fc1.b"]))   # 400 -> 120 (flatten is free!)
    x = relu_forward(linear_forward(x, params["fc2.w"], params["fc2.b"]))   # 120 -> 84
    return softmax(linear_forward(x, params["fc3.w"], params["fc3.b"]))     # 84 -> 10
# <<< provided: lenet_fast


# =====================================================================================================
# lab 4 - learning: loss, optimizer, learning rate schedule, data loading
# =====================================================================================================

# >>> exercise: cross_entropy_loss
def cross_entropy_loss(logits, label):
    """softmax + cross-entropy: loss = -log(probability of the right class). gives back (loss, probs).

    its written as log(sum(exp(z - m))) + m - z[label] so it never ends up doing log(0)"""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    loss = math.log(total) + m - logits[label]
    return loss, [e / total for e in exps]
# <<< exercise: cross_entropy_loss


# >>> provided: param
class Param:
    """something to optimize: a flat list of numbers (data) and their gradient (grad, same length)"""

    def __init__(self, data):
        self.data = list(data)
        self.grad = None

    def __repr__(self):
        return f"Param({[round(v, 4) for v in self.data]})"
# <<< provided: param


# >>> provided: schedules
# learning rate schedules, same inputs as cudagrad's schedulers.hpp: (t, T, lr_max, lr_min) -> lr
def constant_lr(t, T, lr_max, lr_min):
    return lr_max


def linear_lr(t, T, lr_max, lr_min):
    return lr_max - (lr_max - lr_min) * min(t / T, 1.0)
# <<< provided: schedules


# >>> exercise: cosine_annealing_lr
def cosine_annealing_lr(t, T, lr_max, lr_min):
    """cosine annealing: start at lr_max, slide down half a cosine wave, land on lr_min at step T"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t / T, 1.0)))
# <<< exercise: cosine_annealing_lr


# >>> exercise: sgd
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
# <<< exercise: sgd


# >>> provided: channel_stats
# average and spread of each colour channel over the training images (pixel / 255).
# channel_stats(DATA / "train.bin") works them out again
MEAN = (0.5863, 0.5382, 0.4765)
STD = (0.2864, 0.2874, 0.3036)


def channel_stats(path):
    """mean and standard deviation of every colour channel over a whole .bin file, in 0..1 units"""
    raw = Path(path).read_bytes()
    n = len(raw) // RECORD
    means, stds = [], []
    for ch in range(3):
        plane = b"".join(raw[i * RECORD + 1 + ch * 1024:i * RECORD + 1 + (ch + 1) * 1024] for i in range(n))
        mean = sum(plane) / len(plane) / 255
        var = sumprod(plane, plane) / len(plane) / 255 ** 2 - mean * mean
        means.append(round(mean, 4))
        stds.append(round(math.sqrt(var), 4))
    return tuple(means), tuple(stds)
# <<< provided: channel_stats


# >>> exercise: dataloader
class DataLoader:
    """hands out (image, label) pairs from a .bin file, one image at a time.

    image = 3,072 floats in CHW order, normalized per channel:  (pixel / 255 - mean) / std
    shuffle=True goes thru the images in a new random order every epoch.
    per_class=N keeps only the first N images of every class (a smaller, balanced training set)"""

    def __init__(self, path, shuffle=False, per_class=None, seed=0):
        self.raw = Path(path).read_bytes()  # the whole file stays as compact bytes, we decode one image at a time
        labels = self.raw[::RECORD]         # every record starts with its label byte
        index = list(range(len(labels)))
        if per_class:
            count, keep = [0] * len(CLASSES), []
            for i in index:
                if count[labels[i]] < per_class:
                    keep.append(i)
                    count[labels[i]] += 1
            index = keep
        self.index = index
        self.labels = [labels[i] for i in index]
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.index)

    def image(self, k):
        """the k-th image as 3,072 normalized floats"""
        start = self.index[k] * RECORD + 1
        pixels = self.raw[start:start + 3072]
        x = []
        for ch in range(3):
            m, s = MEAN[ch], STD[ch]
            x += [(v / 255 - m) / s for v in pixels[ch * 1024:(ch + 1) * 1024]]
        return x

    def __iter__(self):
        order = list(range(len(self)))
        if self.shuffle:
            self.rng.shuffle(order)
        for k in order:
            yield self.image(k), self.labels[k]
# <<< exercise: dataloader


# =====================================================================================================
# lab 5 - forward + back propagation. a Tensor is a node in the computation graph (cudagrad's Node<T>):
# it remembers which tensors it was made from (children) and how to send gradients back to them
# (backward_fn). loss.backward() visits the graph in reverse topological order, thats the chain rule.
# =====================================================================================================

# >>> provided: tensor_class
class Tensor:
    """a node in the computation graph: a flat list of numbers + a shape, plus what backprop needs"""

    def __init__(self, data, shape, children=(), op="leaf", requires_grad=False):
        self.data = data                  # the numbers, flat, row-major
        self.shape = tuple(shape)
        self.grad = None                  # d(loss)/d(data), same length as data, backward() fills it in
        self.children = children          # the tensors this one was made from
        self.op = op                      # name of the operation that made it (for printing)
        self.requires_grad = requires_grad or any(c.requires_grad for c in children)
        self.backward_fn = None           # sends self.grad back to the children

    def add_grad(self, g):
        """adds g to this tensor's gradient (a tensor that got used twice collects gradient from both uses)"""
        self.grad = g if self.grad is None else [a + b for a, b in zip(self.grad, g)]

    @property
    def layout(self):
        return Layout(self.shape)

    def __repr__(self):
        head = ", ".join(f"{v:.4f}" for v in self.data[:4]) + (", ..." if len(self.data) > 4 else "")
        return f"Tensor(shape={self.shape}, op={self.op}, data=[{head}])"


def leaf(data, shape):
    """a tensor we want gradients for (a weight), like cudagrad's leaf()"""
    return Tensor(list(data), shape, requires_grad=True)
# <<< provided: tensor_class


# >>> exercise: backward
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
# <<< exercise: backward


# >>> exercise: relu_backward
def relu_backward(dy, x):
    """y = relu(x). the gradient goes thru where x > 0 and gets blocked (0) where x <= 0"""
    return [g if v > 0.0 else 0.0 for g, v in zip(dy, x)]
# <<< exercise: relu_backward


# >>> exercise: linear_backward
def linear_backward(dy, x, weight):
    """y = W x + b with W stored as n_out rows of n_in. given dy = d(loss)/dy, gives back (dx, dweight, dbias).

    dx = W^T dy           (column i of W dotted with dy)
    dW = dy x^T           (every dy times every x: dW[o][i] = dy[o] * x[i])
    db = dy"""
    n_in, n_out = len(x), len(dy)
    columns = rows(gather(weight, Layout((n_in, n_out), (1, n_in))), n_out)  # the transpose of W, as rows
    dx = matmul_t(columns, [dy])
    dweight = [g * v for g in dy for v in x]
    return dx, dweight, list(dy)
# <<< exercise: linear_backward


# >>> exercise: maxpool2d_backward
def maxpool2d_backward(dy, x, C, H, W, P=2):
    """y = maxpool(x). each window's gradient goes to the one pixel that won the max, the rest get 0"""
    L = pool_layout(C, H, W, P)
    windows = rows(gather(x, L), P * P)
    offs = offsets(L)
    dx = [0.0] * len(x)
    for k, (g, w) in enumerate(zip(dy, windows)):
        winner = w.index(max(w))           # which of the P*P spots had the max
        dx[offs[k * P * P + winner]] += g  # where that spot lives in x
    return dx
# <<< exercise: maxpool2d_backward


# >>> exercise: conv2d_backward
def conv2d_backward(dy, x, weight, C, H, W, K, need_dx=True):
    """y = conv2d_forward(x, weight, bias). given dy (Cout x OH x OW), gives back (dx, dweight, dbias).

    forward was  out = filters @ patches^T  (+ a bias per filter), so by the matmul rules:
      dbias[k]    = the sum of feature map k's gradient
      dweight     = dy @ patches              (Cout x n, every weight saw one pixel per output spot)
      dpatches    = dy^T @ filters            (P x n, how much each patch entry mattered)
      dx          = add every dpatches entry back onto the pixel it was read from (the same layout!)"""
    n = C * K * K
    L = im2col_layout(C, H, W, K)
    P = size(L[0])                                 # output pixels
    Cout = len(weight) // n
    dy_rows = rows(dy, P)                          # Cout rows, one feature map's gradient each
    dbias = [sum(r) for r in dy_rows]
    patch_columns = rows(gather(x, transpose(L)), P)    # n rows: patch entry j at every output spot
    dweight = matmul_t(dy_rows, patch_columns)          # Cout x n
    if not need_dx:                                # the first layer's input is the image, nobody needs its gradient
        return None, dweight, dbias
    dy_columns = rows(gather(dy, Layout((P, Cout), (1, P))), Cout)       # P rows: every filter's dy at spot p
    filter_columns = rows(gather(weight, Layout((n, Cout), (1, n))), Cout)  # n rows: entry j of every filter
    dpatches = matmul_t(dy_columns, filter_columns)  # P x n, same order as offsets(L)
    dx = [0.0] * len(x)
    for o, g in zip(offsets(L), dpatches):         # scatter-add, the backward of a gather
        dx[o] += g
    return dx, dweight, dbias
# <<< exercise: conv2d_backward


# >>> exercise: cross_entropy_backward
def cross_entropy_backward(probs, label):
    """d(loss)/d(logits) for softmax + cross-entropy: just probs - one_hot(label)"""
    d = list(probs)
    d[label] -= 1.0
    return d
# <<< exercise: cross_entropy_backward


# >>> provided: ops
# the autograd operations. each one runs a forward kernel, wraps the answer in a Tensor that remembers its
# children, and sticks on a backward_fn that runs the matching backward kernel (cudagrad's Node.cuh pattern)
def relu(x):
    out = Tensor(relu_forward(x.data), x.shape, (x,), "relu")

    def backward_fn():
        x.add_grad(relu_backward(out.grad, x.data))
    out.backward_fn = backward_fn
    return out


def linear(x, weight, bias):
    """cudagrad: bias_add(matmul(x, W), b)"""
    out = Tensor(linear_forward(x.data, weight.data, bias.data), bias.shape, (x, weight, bias), "linear")

    def backward_fn():
        dx, dw, db = linear_backward(out.grad, x.data, weight.data)
        if x.requires_grad:
            x.add_grad(dx)
        weight.add_grad(dw)
        bias.add_grad(db)
    out.backward_fn = backward_fn
    return out


def conv2d(x, weight, bias):
    """cudagrad: conv_bias(conv(x, filters), bias)"""
    C, H, W = x.shape
    Cout, _, K, _ = weight.shape
    out = Tensor(conv2d_forward(x.data, weight.data, bias.data, C, H, W, K),
                 (Cout, H - K + 1, W - K + 1), (x, weight, bias), "conv2d")

    def backward_fn():
        dx, dw, db = conv2d_backward(out.grad, x.data, weight.data, C, H, W, K, need_dx=x.requires_grad)
        if x.requires_grad:
            x.add_grad(dx)
        weight.add_grad(dw)
        bias.add_grad(db)
    out.backward_fn = backward_fn
    return out


def maxpool2d(x, p=2):
    C, H, W = x.shape
    out = Tensor(maxpool2d_forward(x.data, C, H, W, p), (C, H // p, W // p), (x,), "maxpool2d")

    def backward_fn():
        x.add_grad(maxpool2d_backward(out.grad, x.data, C, H, W, p))
    out.backward_fn = backward_fn
    return out


def flatten(x):
    """a view: the same numbers with a new shape. nothing gets copied and the gradient goes straight thru"""
    out = Tensor(x.data, (len(x.data),), (x,), "flatten")

    def backward_fn():
        x.add_grad(out.grad)
    out.backward_fn = backward_fn
    return out


def softmax_cross_entropy(logits, label):
    """the loss as a tensor with one number in it (cudagrad's softmax_cross_entropy)"""
    loss, probs = cross_entropy_loss(logits.data, label)
    out = Tensor([loss], (1,), (logits,), "softmax_cross_entropy")
    out.probs = probs

    def backward_fn():
        logits.add_grad([g * out.grad[0] for g in cross_entropy_backward(probs, label)])
    out.backward_fn = backward_fn
    return out
# <<< provided: ops


# =====================================================================================================
# lab 6 - modules, lenet-5, training on memes
# =====================================================================================================

# >>> provided: modules
class Module:
    """a layer: something you call on a tensor that might own weights (cudagrad's Module<T>)"""

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return []


class Conv(Module):
    """cudagrad's Conv: K x K filters with he init, zero bias, then relu"""

    def __init__(self, c_in, c_out, k, relu=True):
        self.weight = leaf(he_init(c_out * c_in * k * k, c_in * k * k), (c_out, c_in, k, k))
        self.bias = leaf([0.0] * c_out, (c_out,))
        self.use_relu = relu

    def forward(self, x):
        y = conv2d(x, self.weight, self.bias)
        return relu(y) if self.use_relu else y

    def parameters(self):
        return [self.weight, self.bias]


class MaxPool(Module):
    def __init__(self, p=2):
        self.p = p

    def forward(self, x):
        return maxpool2d(x, self.p)


class Flatten(Module):
    def forward(self, x):
        return flatten(x)


class Linear(Module):
    """cudagrad's Linear: fully connected, he init, zero bias, relu if you want it"""

    def __init__(self, n_in, n_out, relu=True):
        self.weight = leaf(he_init(n_out * n_in, n_in), (n_out, n_in))
        self.bias = leaf([0.0] * n_out, (n_out,))
        self.use_relu = relu

    def forward(self, x):
        y = linear(x, self.weight, self.bias)
        return relu(y) if self.use_relu else y

    def parameters(self):
        return [self.weight, self.bias]


class CNN(Module):
    """a stack of layers, like cudagrad's CNN: net.add(layer).add(layer)..."""

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def predict(self, image):
        """(guessed label, probabilities) for one normalized image (3,072 floats)"""
        probs = softmax(self(Tensor(image, (3, 32, 32))).data)
        return max(range(len(probs)), key=probs.__getitem__), probs

    def save(self, path, **info):
        params = [{"shape": p.shape, "data": [float(f"{v:.7g}") for v in p.data]} for p in self.parameters()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"model": "lenet5", "classes": CLASSES, "mean": MEAN, "std": STD,
                                          "info": info, "params": params}))

    def load(self, path):
        saved = json.loads(Path(path).read_text())
        for p, s in zip(self.parameters(), saved["params"]):
            if tuple(s["shape"]) != p.shape:
                raise ValueError(f"shape mismatch: the file has {s['shape']}, the model has {p.shape}")
            p.data = list(s["data"])
        return saved.get("info", {})


def lenet5(num_classes=10, seed=0):
    """lenet-5 for 3 x 32 x 32 images (lecun et al. 1998, with relu and max-pooling)"""
    random.seed(seed)
    return (CNN()
            .add(Conv(3, 6, 5)).add(MaxPool(2))      # 3x32x32 -> 6x28x28 -> 6x14x14
            .add(Conv(6, 16, 5)).add(MaxPool(2))     #         -> 16x10x10 -> 16x5x5
            .add(Flatten())                          #         -> 400
            .add(Linear(400, 120))                   #         -> 120
            .add(Linear(120, 84))                    #         -> 84
            .add(Linear(84, num_classes, relu=False)))  #      -> 10 scores ("logits")
# <<< provided: modules


# >>> exercise: train_step
def train_step(model, opt, image, label):
    """one step of learning on one image. gives back (loss, was the guess right?)"""
    logits = model(Tensor(image, (3, 32, 32)))      # 1. forward propagation
    loss = softmax_cross_entropy(logits, label)     # 2. how wrong were we
    opt.zero_grad()                                 # 3. forget the last step's gradients
    loss.backward()                                 # 4. back propagation: d(loss)/d(every weight)
    opt.step()                                      # 5. nudge every weight downhill
    guess = max(range(len(logits.data)), key=logits.data.__getitem__)
    return loss.data[0], guess == label
# <<< exercise: train_step


# >>> exercise: evaluate
def evaluate(model, loader, limit=None):
    """accuracy on a dataset, plus a confusion matrix: confusion[true label][guessed label] = count"""
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    for k, (image, label) in enumerate(loader):
        if limit is not None and k >= limit:
            break
        guess, _ = model.predict(image)
        confusion[label][guess] += 1
    total = sum(map(sum, confusion))
    correct = sum(confusion[i][i] for i in range(len(CLASSES)))
    return correct / max(total, 1), confusion
# <<< exercise: evaluate


# >>> provided: train
def train(model, train_loader, epochs=1, lr_max=0.005, lr_min=1e-5, schedule=None, val_loader=None,
          val_limit=500, log_every=500, history=None):
    """trains with plain sgd and a cosine annealing learning rate, one image at a time (no batching).
    prints progress and gives back a list of {step, lr, loss, acc} records (also added onto `history`)"""
    opt = SGD(model.parameters(), lr_max, lr_min, total_steps=epochs * len(train_loader),
              schedule=schedule or cosine_annealing_lr)
    history = [] if history is None else history
    start = time.time()
    for epoch in range(1, epochs + 1):
        run_loss, run_right, seen = 0.0, 0, 0
        for image, label in train_loader:
            lr = opt.lr
            loss, right = train_step(model, opt, image, label)
            run_loss += loss
            run_right += right
            seen += 1
            if seen % log_every == 0 or seen == len(train_loader):
                rate = opt.t / (time.time() - start)
                history.append({"step": opt.t, "lr": lr, "loss": run_loss / seen, "acc": run_right / seen})
                print(f"epoch {epoch}/{epochs}  image {seen:>6}/{len(train_loader)}  loss {run_loss / seen:.3f}  "
                      f"train acc {run_right / seen:6.1%}  lr {lr:.5f}  ({rate:.0f} images/s)", flush=True)
        if val_loader is not None:
            acc, _ = evaluate(model, val_loader, limit=val_limit)
            print(f"  -> epoch {epoch}: validation accuracy {acc:.1%} (first {val_limit} val images)", flush=True)
    print(f"done: {opt.t} steps in {time.time() - start:.0f} s")
    return history
# <<< provided: train


# =====================================================================================================
# command line: python3 babytorch.py [--per-class N] [--epochs E] [--lr LR] [--save PATH] [--load PATH]
# =====================================================================================================
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="train / evaluate lenet-5 on meme-10 in plain python.")
    ap.add_argument("--per-class", type=int, default=0, help="training images per class (0 = all 3,500)")
    ap.add_argument("--epochs", type=int, default=None, help="default 4 (0 when --load is given: just evaluate)")
    ap.add_argument("--lr", type=float, default=0.005, help="lr_max of the cosine schedule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="write the trained weights to this .json file")
    ap.add_argument("--load", help="start from (or just evaluate) these weights")
    ap.add_argument("--val-limit", type=int, default=500, help="val images to check after each epoch")
    args = ap.parse_args(argv)
    if args.epochs is None:
        args.epochs = 0 if args.load else 4

    model = lenet5(seed=args.seed)
    if args.load:
        print("loaded", args.load, model.load(args.load))
    test = DataLoader(DATA / "test.bin")
    if args.epochs > 0:
        loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=args.per_class or None, seed=args.seed)
        val = DataLoader(DATA / "val.bin")
        print(f"training on {len(loader)} images for {args.epochs} epochs (lr {args.lr} -> 1e-5, cosine)")
        t0 = time.time()
        train(model, loader, args.epochs, lr_max=args.lr, val_loader=val, val_limit=args.val_limit)
        minutes = (time.time() - t0) / 60
        val_acc, _ = evaluate(model, val)
        test_acc, _ = evaluate(model, test)
        print(f"validation accuracy (all {len(val)}): {val_acc:.1%}   REAL test accuracy: {test_acc:.1%}")
        if args.save:
            model.save(args.save, train_images=len(loader), epochs=args.epochs, lr_max=args.lr, seed=args.seed,
                       minutes=round(minutes, 1), val_acc=round(val_acc, 4), test_acc=round(test_acc, 4),
                       python=sys.version.split()[0])
            print("saved", args.save)
    else:
        test_acc, _ = evaluate(model, test)
        print(f"REAL test accuracy: {test_acc:.1%}")


if __name__ == "__main__":
    main()
