"""
the two bonus notebooks (addon_A_batching.ipynb and addon_B_split_k.ipynb): all the text, the hints inside
the todo cells, and the doc comments on top of the given code. the code itself comes from addons.py.
"""

# =====================================================================================================
# hints inside the todo cells
# =====================================================================================================
BONUS_STUBS = {
    "batched_im2col_layout": {"override": '''
def batched_im2col_layout(N, C, H, W, K):
    """im2col for a whole batch stored (N, C, H, W) row-major, so strides (C*H*W, H*W, W, 1).

    rows    = (n, oy, ox): every output pixel of every image  -> mode (N, OH, OW)
    columns = (c, i, j):   a spot in the patch, same as before"""
    OH, OW = H - K + 1, W - K + 1
    # how to think about it:
    #   its im2col_layout from lab 2 with one more letter. pixel (n, c, oy + i, ox + j) of the batch is at
    #       n*C*H*W + c*H*W + (oy + i)*W + (ox + j)
    #   multiply it out:   n*C*H*W + oy*W + ox*1   +   c*H*W + i*W + j*1
    #   the rows are (n, oy, ox) and the columns are (c, i, j). the number in front of each letter is its
    #   stride. the columns didnt change at all from lab 2
    # stuff youll need:
    #   Layout(shape, stride) with nested tuples. the shape is already filled in, replace every ...
    return Layout(((N, OH, OW), (C, K, K)),
                  ((..., ..., ...), (..., ..., ...)))
'''},

    "broadcast": {"override": '''
def broadcast(n_rows, n_cols):
    """reads a list of n_cols numbers as an n_rows x n_cols matrix where every row is that same list.
    a stride of 0 means stepping that way doesnt move in memory"""
    # how to think about it:
    #   stepping right (to the next column) should move to the next number in memory, so the column stride is 1.
    #   stepping down (to the next row) should land on the SAME numbers again. so how far should one step
    #   down move in memory?
    # stuff youll need:
    #   Layout((n_rows, n_cols), (row_stride, column_stride)), replace the ...
    return Layout((n_rows, n_cols), (..., 1))
'''},

    "train_step_batched": {"hints": {"*": """\
how to think about it:
  its train_step from lab 6, but for a whole batch at once:
  1. glue the images into one (N, 3, 32, 32) tensor with stack(images) and run forward_batched(model, it).
     you get logits with shape (N, 10), one row of 10 scores per image
  2. loss = softmax_cross_entropy_batched(logits, labels). its already the average over the batch
  3. zero_grad, backward, step. ONE optimizer step for the whole batch, thats the whole point
  4. count the right guesses: cut logits.data into rows of 10, take the argmax of every row, and count
     how many match their label
  give back  (loss.data[0], how many were right)
stuff youll need:
  stack(images), forward_batched(model, x), softmax_cross_entropy_batched(logits, labels)   (given above)
  logits.shape[1]                     how many classes (10)
  rows(logits.data, n_cls)            one row of scores per image
  max(range(n_cls), key=r.__getitem__)   the argmax of one row r
  sum(g == lab for g, lab in zip(guesses, labels))   True counts as 1, so sum() counts the matches"""}},

    "tree_reduce": {"hints": {"*": """\
how to think about it:
  you have a list of partial sums, and every one of them is a list of the same length. adding them one after
  another takes (how many there are - 1) steps. adding them in PAIRS takes way fewer rounds:
      [p0, p1, p2, p3]  ->  [p0+p1, p2+p3]  ->  [p0+p1+p2+p3]
  so, while theres more than one list left:
    1. go thru the lists two at a time (0 and 1, then 2 and 3, ...) and add each pair number by number
    2. if theres an odd one left over at the end, carry it into the next round as it is
    3. the new shorter list of lists replaces the old one
  when only one list is left, thats your answer
stuff youll need:
  while len(partials) > 1:          keeps going untill one is left
  range(0, len(partials) - 1, 2)    0, 2, 4, ... the first one of every pair
  [a + b for a, b in zip(p, q)]     adds two lists number by number
  len(partials) % 2                 1 if theres an odd one out
  partials[-1]                      the last one"""}},

    "matmul_t_split_k": {"hints": {"*": """\
how to think about it:
  every row of A and B has K numbers, and each answer C[i][j] is a sum over all K of them. split that sum up:
  1. K = len(A[0]), and chunk = K / splits rounded UP (so the slices cover everything)
  2. for every slice s, it covers positions lo = s * chunk up to hi = min(K, (s + 1) * chunk).
     cut that piece out of every row of A and every row of B, and run your matmul_t on just the pieces.
     that gives a partial C, the same size as the full C
  3. add all the partial Cs together with tree_reduce
  the answer should match matmul_t(A, B), maybe off in the very last digit
stuff youll need:
  -(-K // splits)                   ceiling division trick: -(-7 // 2) is 4
  min(a, b)                         so the last slice doesnt go past K
  [a[lo:hi] for a in A]             the same slice of every row
  matmul_t(...), tree_reduce(...)   yours"""}},
}


# =====================================================================================================
# doc comments on top of the given code cells
# =====================================================================================================
BONUS_NOTES = {
    "batched_ops": """\
what this does: every layer, but for a whole batch at a time
stuff in here:
  N, C, H, W = x.shape              the batch size is the new first number
  conv2d_batched                    one gather + ONE matmul for every image in the batch. the matmul gives the
                                    maps back as (Cout, N, P), so a gather with Layout((N, Cout, P), (P, N*P, 1))
                                    reads them in (N, Cout, P) order without moving anything by hand
  Layout((N, Cout, P), (0, 1, 0))   stride 0 for n and p: bias[k] gets repeated over every image and pixel
  y.backward_fn = lambda: ...       a backward_fn can be a lambda when its just one line
  maxpool2d_batched                 a batch of N images with C channels is just N*C channels, nothing new
  linear_batched                    Y = X W^T + b.  dW = dY^T X,  dX = dY W,  db = the sum over the batch
  softmax_cross_entropy_batched     the AVERAGE loss, so every image's gradient gets divided by N
  forward_batched                   walks the model's layers and calls the batched version of each one
  isinstance(layer, Conv)           checks what kind of layer it is
  batches(loader, n)                a generator (it uses yield) that groups (image, label) pairs into lists of n""",

    "train_batched": """\
what this does: the training loop from lab 6, but one sgd step per batch instead of one per image
stuff in here:
  -(-len(loader) // batch_size)     batches per epoch, rounded up (the last batch can be smaller)
  run_loss += loss * len(images)    loss is an average, so multiply it back to get the batch's total
  opt.t % log_every == 0            prints a line every log_every steps""",
}


# =====================================================================================================
# bonus a - batching
# =====================================================================================================
def addon_a(nb):
    nb.md(r"""
# bonus a - batching

in lab 6 the network updated all 62,006 weights **after every single image**: 35,000 optimizer steps an epoch, each one based on one noisy example. **batching** shows the network N images, averages their gradients, and takes **one** step.

- the gradient is less noisy (its an average of N), so a bigger learning rate is ok
- N times fewer optimizer steps
- the work comes in bigger chunks: one tall matrix multiply instead of N short ones. thats what gpus need to keep thousands of cores busy (cudagrad trains with batches of 128)

with layouts, adding a batch is almost free: **the batch is just one more mode.**

same deal as the main lab: `# todo:` cells are yours, `# given code` cells are already written, and if youre stuck put `%load solutions/NAME.py` in the cell and run it twice. the formula sheet and python stuff for this notebook are right below.
""")
    nb.md(r"""
## formula sheet

- a batch of $N$ images is one tensor $(N, C, H, W)$ with row-major strides $(CHW,\; HW,\; W,\; 1)$. pixel $(n, c, y, x)$ sits at $nCHW + cHW + yW + x$
- batched im2col: rows $(n, oy, ox)$ with strides $(CHW, W, 1)$, columns $(c, i, j)$ with strides $(HW, W, 1)$
- conv for the whole batch is one matrix multiply $Y = F X^T$ where $X$ has $N \cdot OH \cdot OW$ rows (every patch of every image) and $C K^2$ columns
- the batch loss is the average: $L = \dfrac{1}{N} \sum_n L_n$, so the gradient is the average too: $\dfrac{\partial L}{\partial w} = \dfrac{1}{N} \sum_n \dfrac{\partial L_n}{\partial w}$
- softmax + cross-entropy over a batch: $dz[n][k] = \big(p[n][k] - [k = y_n]\big) / N$
- fully connected over a batch: $Y = X W^T + b$ (the same $b$ added to every row), $\;dX = dY\,W$, $\;dW = dY^T X$, $\;db = \sum_n dY[n]$
- broadcasting: a stride of $0$ repeats memory. $(N, n_\text{out}) : (0, 1)$ reads the same bias row $N$ times, $(N, C_\text{out}, P) : (0, 1, 0)$ repeats bias $k$ over every image and pixel
- a weight gradient is a sum over the batch: for conv1 with batch $N$ thats $K = N \cdot 28 \cdot 28$ products per weight ($100{,}352$ when $N = 128$)
- steps per epoch $= \lceil \text{images} / N \rceil$. with fewer steps you can usually raise the learning rate, roughly in proportion to $N$ (the "linear scaling rule", try a few)
""")
    nb.md(r"""
## python stuff for this notebook
- `isinstance(layer, Conv)` checks what kind of thing `layer` is. `forward_batched` uses it to pick the batched version of every layer
- `-(-a // b)` is `a / b` rounded **up** (floor division of a negative rounds the other way): `-(-7 // 2)` is 4
- `lambda: x.add_grad(...)` is a one line function with no arguments, perfect for a short `backward_fn`
- a function with `yield` (like `batches`) hands out one batch at a time
- `time.perf_counter()` twice and subtract = how long something took
- `f"{v:5.1f}"` is a number 5 characters wide with 1 decimal
""")
    nb.code(r"""
import time
from babytorch import *          # everything from the main lab (the answer key versions)
import viz, checks
""")
    nb.md(r"""
### a.1 im2col with a batch mode
a batch of N images stored one after another is the layout `(N, C, H, W):(C·H·W, H·W, W, 1)`. for the convolution, every output pixel of *every image* turns into a row of the patch matrix: rows `(n, oy, ox)`, columns `(c, i, j)` exactly like before. for lenet's first layer with N = 64 thats a 50,176 x 75 patch matrix, and it still never gets built.
""")
    nb.exercise("batched_im2col_layout")
    nb.code("checks.batched_im2col_layout(batched_im2col_layout)")
    nb.code(r"""
viz.show_layout(batched_im2col_layout(2, 1, 3, 3, 2),
                title="2 images of 1x3x3 with 2x2 patches: rows = (image, oy, ox), columns = (c, i, j)")
""")
    nb.md(r"""
### a.2 broadcasting the bias: stride 0
a fully connected layer on a batch works out $Y = XW^T$ (N x out) and then adds the *same* bias row onto every one of the N rows. instead of copying the bias N times, read it thru a layout whose row stride is **0**.
""")
    nb.exercise("broadcast")
    nb.code("checks.broadcast(broadcast)")
    nb.code(r"""print_layout(broadcast(4, 3))      # 4 rows, all reading offsets 0, 1, 2""")
    nb.md(r"""
### a.3 the batched layers
heres every layer for a batch. look at `conv2d_batched` closely:
- **forward**: one gather + one matmul for the whole batch, then a gather to reorder `(Cout, N, P)` → `(N, Cout, P)`, and the bias repeated with a stride-0 layout `(N, Cout, P):(0, 1, 0)`
- **backward**: the weight gradient is now `matmul_t(dy_rows, ...)` with rows of length **N·P**, a sum over every image and every position. remember that one for bonus b

max-pool needs nothing new (a batch of N images with C channels is just N·C channels) and relu is the same function as before.
""")
    nb.provided("batched_ops")
    nb.md(r"""
### a.4 batching doesnt change the math
check it: the batched forward pass gives exactly the one-at-a-time logits, and the batched gradient is exactly the **average** of the one-at-a-time gradients.
""")
    nb.code(r"""
test_loader = DataLoader(DATA / "test.bin")
images = [test_loader.image(k) for k in range(0, 150, 10)]       # 15 test images, one of every 10
labels = [test_loader.labels[k] for k in range(0, 150, 10)]
model = lenet5(seed=4)

batched = forward_batched(model, stack(images)).data
one_by_one = [v for image in images for v in model(Tensor(image, (3, 32, 32))).data]
print("biggest |batched - one at a time| =", max(abs(a - b) for a, b in zip(batched, one_by_one)))

loss = softmax_cross_entropy_batched(forward_batched(model, stack(images)), labels)
loss.backward()
g_batched = [list(p.grad) for p in model.parameters()]
for p in model.parameters():
    p.grad = None
for image, label in zip(images, labels):
    softmax_cross_entropy(model(Tensor(image, (3, 32, 32))), label).backward()    # gradients pile up
g_avg = [[g / len(images) for g in p.grad] for p in model.parameters()]
print("biggest |batched gradient - average of single gradients| =",
      max(abs(a - b) for x, y in zip(g_batched, g_avg) for a, b in zip(x, y)))
""")
    nb.md(r"""
### a.5 a batched training step
""")
    nb.exercise("train_step_batched")
    nb.code("checks.train_step_batched(train_step_batched)")
    nb.provided("train_batched")
    nb.md(r"""
### a.6 is it faster?
time one training step per image for a few batch sizes, against the main lab's `train_step`.
""")
    nb.code(r"""
loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=20)
data = [pair for _, pair in zip(range(64), loader)]                  # 64 training pairs
ms = {}
for B in (1, 4, 16, 64):
    m = lenet5(seed=1)
    opt = SGD(m.parameters(), 0.01)
    imgs, labs = [d[0] for d in data[:B]], [d[1] for d in data[:B]]
    train_step_batched(m, opt, imgs, labs)                                # warm up (builds the offset tables)
    t = time.perf_counter()
    for _ in range(64 // B):                                              # 64 images worth of steps
        train_step_batched(m, opt, imgs, labs)
    ms[B] = (time.perf_counter() - t) / 64 * 1000
m = lenet5(seed=1)
opt = SGD(m.parameters(), 0.01)
t = time.perf_counter()
for image, label in data:
    train_step(m, opt, image, label)
core_ms = (time.perf_counter() - t) / 64 * 1000
print(f"main lab, one image at a time: {core_ms:.1f} ms per image")
for B, v in ms.items():
    print(f"batch of {B:>2}: {v:5.1f} ms per image   ({core_ms / v:.2f}x)")
""")
    nb.md(r"""
talk about it: why so little? in plain python every multiply-add still costs the same, batched or not. batching only saves the stuff *around* the math: the optimizer step (62,006 weights updated once per batch instead of once per image) and some per-call bookkeeping. on a **gpu** its a whole different story. one image gives the first conv only 784 x 6 outputs to compute, nowhere near enough to keep thousands of cores busy. a batch of 128 gives 128x more independent work for almost free. thats why every gpu library trains in batches.

### a.7 does it learn as well?
same data and the same number of epochs, one image at a time vs batches of 16. bigger batches mean fewer, smoother steps, so they need a bigger learning rate (8x bigger here).
""")
    nb.code(r"""
train_loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=200)
val_loader = DataLoader(DATA / "val.bin")
results = {}
for B, lr in [(1, 0.005), (16, 0.04)]:
    random.seed(0)
    m = lenet5(seed=0)
    t = time.time()
    if B == 1:
        train(m, train_loader, epochs=2, lr_max=lr, log_every=1000)
    else:
        train_batched(m, train_loader, epochs=2, batch_size=B, lr_max=lr, log_every=50)
    secs = time.time() - t
    val_acc, _ = evaluate(m, val_loader, limit=500)
    test_acc, _ = evaluate(m, test_loader)
    results[B] = (secs, val_acc, test_acc)
for B, (secs, val_acc, test_acc) in results.items():
    print(f"batch {B:>2}: {secs:5.0f} s   val {val_acc:.1%}   real test {test_acc:.1%}")
""")
    nb.md(r"""
### a.8 matrix shapes with a batch
every layer is a matrix multiply $C_{M\times N} = A_{M\times K}\,B^T$. look at the **weight gradient** ones: the output is tiny (just the filters) but **K**, how many things get added up for each output, grows with the batch.
""")
    nb.code(r"""
N = 128   # cudagrad's batch size
rows_ = [("conv1 forward", 6, N * 28 * 28, 75), ("conv1 weight gradient", 6, 75, N * 28 * 28),
         ("conv2 forward", 16, N * 10 * 10, 150), ("conv2 weight gradient", 16, 150, N * 10 * 10),
         ("fc1 forward", N, 120, 400), ("fc1 weight gradient", 120, 400, N)]
print(f"{'matmul':<24}{'M':>7}{'N':>8}{'K':>9}   outputs   added up per output")
for name, M_, N_, K_ in rows_:
    print(f"{name:<24}{M_:>7}{N_:>8}{K_:>9}   {M_ * N_:>7,}   {K_:>10,}")
""")
    nb.md(r"""
`conv1 weight gradient`: only **450 outputs**, but each one is a sum of **100,352 products**. on a gpu thats 450 threads doing super long loops while thousands of cores sit around doing nothing. the fix is **split-k**, thats [bonus b](addon_B_split_k.ipynb).
""")


# =====================================================================================================
# bonus b - split-k
# =====================================================================================================
def addon_b(nb):
    nb.md(r"""
# bonus b - split-k reduction

### the problem, from a real profiler
cudagrad trains a cnn on an rtx 4080. nvidias profiler (nsight) said two kernels ate **95% of the gpu's time**:

| kernel | share of gpu time |
|---|---|
| `conv2d_dW_kernel` (conv weight gradient) | 67.4% |
| `conv_bias_grad_kernel` (conv bias gradient) | 27.8% |
| everything else, forward passes included | ~5% |

why? each one has very few outputs (72 or 1,152 weights, 8 or 16 biases) and the kernel used **one thread per output**, so only 1-5 thread blocks ran on a gpu with 76 processors (sms). each thread then added up to **86,528** things in a row. thats about 0.2% of what the gpu can do.

### the fix: split the K dimension
a matrix multiply $C = AB^T$ has three sizes: **M** and **N** (the output is M x N) and **K** (each output is a sum of K products). when M x N is small and K is huge, cut K into S slices:

1. every (output, slice) pair works out a **partial sum**, thats S times more independent work for the gpu
2. then **reduce**: add up the S partial sums for every output

thats **split-k**. in cutlass's profiler its the `--split_k_slices` option (with `--split_k_mode serial` or `parallel`). in cudagrad its `blockIdx.y`:

```cpp
// cudagrad: conv2d_dW_kernel (simplified)
const int p  = blockIdx.x;                  // which weight (one output of the matmul)
const int r0 = blockIdx.y * chunk;          // which slice of K this block adds up
for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x)
    acc += x[...] * dOut[...];              // 256 threads each add a strided part of the slice
partial[threadIdx.x] = acc;                 // then a TREE REDUCTION in shared memory:
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
}
if (threadIdx.x == 0) atomicAdd(&dW[p], partial[0]);   // add this slice's partial sum into the answer
```

same deal as the main lab: `# todo:` cells are yours, and `%load solutions/NAME.py` (run it twice) if youre stuck.
""")
    nb.md(r"""
## formula sheet

- matrix multiply: $C[i][j] = \sum_{k=0}^{K-1} A[i][k]\,B[j][k]$, with $M \times N$ outputs and $K$ products added up per output
- split-k with $S$ slices: $\text{chunk} = \lceil K / S \rceil$. slice $s$ covers $k$ from $s \cdot \text{chunk}$ up to $\min(K, (s+1)\cdot\text{chunk})$
- $C = \sum_{s} C_s$ where $C_s[i][j] = \sum_{k \text{ in slice } s} A[i][k]\,B[j][k]$
- independent work goes from $M N$ jobs of $K$ products each to $M N S$ jobs of about $K / S$ products each
- tree reduction of $S$ partial sums takes $\lceil \log_2 S \rceil$ rounds ($8 \to 4 \to 2 \to 1$ is 3 rounds), and $S - 1$ adds per output in total
- cute view of the K mode: `K:1` divided by the chunk is `(chunk, S):(1, chunk)`, so slice $s$ starts at offset $s \cdot \text{chunk}$
- training on many cores is split-k over the batch: $dW = \dfrac{1}{N}\sum_{\text{workers}} \;\sum_{\text{images of that worker}} dW_\text{image}$
- floating point addition isnt perfectly associative: $(a + b) + c$ can differ from $a + (b + c)$ in the last digit
- amdahls law: if a fraction $p$ of the work can run in parallel, $n$ workers give a speed-up of $\dfrac{1}{(1 - p) + p / n}$. with $p \approx 0.87$ and $n = 8$ thats about $4.2\times$, not $8\times$
""")
    nb.md(r"""
## python stuff for this notebook
- `multiprocessing` runs python in seperate processes, so they really do run at the same time on different cores
  - `mp.get_context("spawn").Pool(n)` starts n worker processes
  - `pool.map(f, jobs)` runs `f(job)` for every job on the workers and gives back the results in order
  - the function has to live in a `.py` file (`addons.py` here) becuase every worker imports it fresh. a function you define in the notebook wont work
  - everything sent to a worker and back gets copied (pickled). thats part of why its not 8x faster
- `with SplitKTrainer(...) as trainer:` runs `__enter__` at the start and `__exit__` at the end, even if something crashes. here that shuts the workers down
- `global _WORKER_MODEL` lets a function change a variable that lives outside it (each worker keeps one model around)
- `inspect.getsource(f)` gives back the code of a function as text
- `os.cpu_count()` is how many cores your computer has
- `-(-K // S)` is K / S rounded up
""")
    nb.code(r"""
import time
from babytorch import *          # the main lab (answer key versions)
from addons import batched_im2col_layout, stack, batches
import viz, checks
""")
    nb.md(r"""
### b.1 tree reduction
adding S partial sums one after another takes S − 1 steps. adding them **in pairs**, all the pairs at the same time, takes only $\log_2 S$ rounds: 8 partial sums → 4 → 2 → 1. thats the loop at the end of cudagrad's kernel.
""")
    nb.exercise("tree_reduce")
    nb.code("checks.tree_reduce(tree_reduce)")
    nb.md(r"""
### b.2 slicing K, the cute way
the K mode of a matrix is a 1-D layout `K:1`. cutting it into tiles of `chunk` is cute's `zipped_divide`. you get `(chunk, S):(1, chunk)`, which means "where am i inside the slice" and "which slice". slice `s` covers offsets `s·chunk` up to `s·chunk + chunk − 1`.
""")
    nb.code(r"""
K, chunk = 784, 98
tiles = zipped_divide(Layout(K), (chunk,))
print(tiles, "->", size(tiles[1]), "slices of", size(tiles[0]))
print("slice 3 starts at offset", tiles((0,), (3,)), "and ends at", tiles((chunk - 1,), (3,)))
""")
    nb.exercise("matmul_t_split_k")
    nb.code("checks.matmul_t_split_k(matmul_t_split_k)")
    nb.md(r"""
### b.3 split-k on a real weight gradient
the weight gradient of lenet's first conv for a batch of 16 real images: 6 x 75 outputs, and each one adds up K = 16 x 784 = 12,544 products.
""")
    nb.code(r"""
loader = DataLoader(DATA / "train.bin", per_class=2)
images = [loader.image(k) for k in range(16)]
x = stack(images).data
L = batched_im2col_layout(16, 3, 32, 32, 5)
P = size(L[0])                                                  # 16 * 784 rows, thats the K of the weight gradient
patch_columns = rows(gather(x, transpose(L)), P)                # 75 rows, 12,544 long
random.seed(0)
dy_rows = [[random.uniform(-1, 1) for _ in range(P)] for _ in range(6)]   # a made up output gradient

t = time.perf_counter(); plain = matmul_t(dy_rows, patch_columns); t_plain = time.perf_counter() - t
for S in (1, 2, 8, 64):
    t = time.perf_counter(); split = matmul_t_split_k(dy_rows, patch_columns, S); dt = time.perf_counter() - t
    print(f"S = {S:>2}: biggest difference from plain {max(abs(a - b) for a, b in zip(split, plain)):.1e}   "
          f"({dt * 1000:.0f} ms vs {t_plain * 1000:.0f} ms)")
""")
    nb.md(r"""
**the same answer, but not *exactly* the same digits.** computer addition isnt perfectly associative:
""")
    nb.code(r"""
print((0.1 + 0.2) + 0.3, "vs", 0.1 + (0.2 + 0.3))
""")
    nb.md(r"""
split-k changes the *order* of the additions, so the last digit or two can change. on a gpu, `atomicAdd` makes the order depend on which thread block finishes first, so two runs that start from *the exact same* weights can still end up with slightly different ones.

on one python core split-k isnt faster, its the same multiplies plus a reduction. the point of it is **parallelism**.

### b.4 split-k across your cpu cores
training on a batch: the weight gradient is a sum over every image in the batch (thats part of K). so give each cpu core (a *worker process*) a slice of the batch. every worker runs forward + backward on its images and gives back its **partial sum** of the gradients. the main process **reduces** the partial sums with your `tree_reduce` and takes one sgd step.

| gpu (cudagrad) | here |
|---|---|
| a thread block along `blockIdx.y` | a worker process |
| a slice of K | a slice of the batch |
| shared memory tree reduction + `atomicAdd` | `tree_reduce` in the main process |

the worker code has to live in a `.py` file (`addons.py`) so the worker processes can import it. heres what it looks like:
""")
    nb.code(r"""
# prints the code of the two pieces that matter. read the comments in it
import inspect, addons
from addons import SplitKTrainer
print(inspect.getsource(addons.worker_partial_grads))
print(inspect.getsource(SplitKTrainer.gradients))
""")
    nb.code(r"""
import os
loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=20)
data = [pair for _, pair in zip(range(64), loader)]                   # one batch of 64
imgs, labs = [d[0] for d in data], [d[1] for d in data]
print("cpu cores on this computer:", os.cpu_count())

timings, reference = {}, None
for workers in (1, 2, 4, 8):
    model = lenet5(seed=2)
    opt = SGD(model.parameters(), 0.0)                                # lr 0, we only want the gradients
    with SplitKTrainer(model, opt, workers) as trainer:
        trainer.gradients(imgs[:workers], labs[:workers], tree_reduce)      # wake the workers up first
        t = time.perf_counter()
        trainer.gradients(imgs, labs, tree_reduce)                          # one batch of 64
        timings[workers] = time.perf_counter() - t
        grads = [list(p.grad) for p in model.parameters()]
    reference = reference or grads
    diff = max(abs(a - b) for x, y in zip(grads, reference) for a, b in zip(x, y))
    print(f"{workers} worker(s): {timings[workers] * 1000:5.0f} ms per batch of 64   "
          f"speed-up {timings[1] / timings[workers]:.1f}x   gradient difference vs 1 worker: {diff:.0e}")
viz.line_chart({"measured": [(w, timings[1] / t) for w, t in timings.items()], "perfect": [(w, w) for w in timings]},
               "split-k across cpu cores", "workers", "speed-up")
""")
    nb.md(r"""
talk about it: why not 8x with 8 workers? some of the work doesnt split: sending the weights to every worker, collecting 62,006 partial gradients from each one, the reduction, the sgd step. that serial part caps the speed-up (**amdahls law**, its on the formula sheet). laptops also mix fast and slow cores. gpus win becuase their "workers" share memory, so nothing gets copied.

### b.5 train with it
batches of 32 split across 4 workers (8 images each). compare the images per second with the main lab's ~45-60.
""")
    nb.code(r"""
train_loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=350)
model = lenet5(seed=0)
EPOCHS, BATCH, WORKERS = 3, 32, 4
opt = SGD(model.parameters(), lr_max=0.08, lr_min=1e-5,
          total_steps=EPOCHS * -(-len(train_loader) // BATCH), schedule=cosine_annealing_lr)
start = time.time()
with SplitKTrainer(model, opt, WORKERS) as trainer:
    for epoch in range(EPOCHS):
        seen = right = 0
        for images, labels in batches(train_loader, BATCH):
            loss, correct = trainer.step(images, labels, reduce=tree_reduce)
            seen += len(images); right += correct
        print(f"epoch {epoch + 1}: train acc {right / seen:.1%}   {seen * (epoch + 1) / (time.time() - start):.0f} images/s")
acc, _ = evaluate(model, DataLoader(DATA / "test.bin"))
print(f"real test accuracy after {EPOCHS} epochs on {len(train_loader)} images: {acc:.1%}")
""")
    nb.md(r"""
### where to go from here
- **cutlass** has split-k matrix multiplies and convolutions built in. its convolutions are "implicit gemm": the im2col matrix never gets built, every tile of it gets formed on the fly while its loaded into shared memory. thats your `im2col_layout`, worked out by gpu threads
- **cute dsl** lets you write those kernels in python: `pip install nvidia-cutlass-dsl` (needs an nvidia gpu, ampere or newer, on linux). layouts look just like ours: `cute.make_layout((3, 32, 32), stride=(1024, 32, 1))`
- **cudagrad** has the split-k kernels and the nsight profiles from the top of this notebook: https://github.com/notYnoj/cudagrad
""")
