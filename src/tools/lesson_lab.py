"""
the main lab notebook (lab.ipynb and lab_solutions.ipynb): every bit of text, the hints inside the todo cells,
and the doc comments on top of the given code cells. the code itself comes from babytorch.py.
"""

# =====================================================================================================
# hints inside the todo cells. each one is "how to think about it" (the logic, step by step)
# + "stuff youll need" (every bit of python it takes)
# =====================================================================================================
LAB_STUBS = {
    "shapes": {"hints": {
        "conv_out": """\
how to think about it:
  a k wide filter has to fit inside a row of `size` pixels. it can start at spot 0, 1, 2, ...
  and the last place it can start is size - k (any further and it hangs off the edge).
  so how many starting spots is that? try it on paper: size 5, k 3 -> it fits at 0, 1, 2
stuff youll need:
  return a - b + 1      just math, give the number back with return""",
        "pool_out": """\
how to think about it:
  the windows dont overlap, they go p at a time. how many whole windows fit in `size`?
  a leftover bit at the end that isnt a full window just gets dropped
stuff youll need:
  a // b                floor division, divides and throws away the remainder. 28 // 2 is 14, 7 // 2 is 3""",
        "conv_params": """\
how to think about it:
  one filter is a little block of c_in x k x k weights. there are c_out filters.
  then every filter gets one bias on top
stuff youll need:
  *                     multiply (you can chain them: a * b * c)""",
        "linear_params": """\
how to think about it:
  every input connects to every output with its own weight, so thats n_in * n_out weights.
  then every output gets one bias
stuff youll need:
  * and +"""}},

    "conv2d_naive": {"hints": {"*": """\
how to think about it:
  1. figure out the sizes first. C is how many channels: len(image). H is how many rows: len(image[0]).
     W is how many columns: len(image[0][0]). K is the filter size: len(kernels[0][0])
  2. the output is smaller than the image: OH = H - K + 1 rows and OW = W - K + 1 columns
  3. make an empty list `out`. then for every filter k:
       make an empty feature map
       for every output row y (0 to OH - 1) make an empty row, and for every output column x:
         start a running total at biases[k]
         loop over every channel c, filter row i and filter column j and add
             image[c][y + i][x + j] * kernels[k][c][i][j]
         stick the total on the end of the row
       stick each row on the feature map, then stick the feature map on `out`
  4. return out. out[k][y][x] should be the total for filter k at spot (y, x)
stuff youll need:
  len(xs)               how many items in a list
  range(n)              0, 1, ..., n-1. so `for y in range(OH):` goes over every output row
  for ... in ...:       loops. you can put a loop inside a loop (this one has 6 of them)
  xs = []               an empty list
  xs.append(v)          adds v to the end of a list
  total += v            same as total = total + v
  image[c][y][x]        reading a nested list one level at a time""",}},

    "relu_naive": {"hints": {"*": """\
how to think about it:
  x might be one number, or a list, or a list of lists (of lists...). you dont know how deep it goes.
  so: if x is a list, do relu_naive on every item in it and give back a list of the answers.
      if x is a plain number, give back x if its bigger than 0, otherwise 0.0
  the function calling itself on the smaller pieces is called recursion. it keeps going down
  untill it hits plain numbers
stuff youll need:
  isinstance(x, list)   true if x is a list
  [f(v) for v in xs]    a list comprehension, makes a new list with f done to every item
  a if cond else b      one line if/else that gives back a value"""}},

    "maxpool2d_naive": {"hints": {"*": """\
how to think about it:
  1. do each feature map on its own: for fmap in fmaps
  2. one map has H = len(fmap) rows and W = len(fmap[0]) columns
  3. the pooled map has H // p rows and W // p columns
  4. output spot (y, x) looks at the window that starts at row y * p and column x * p and goes p down and
     p across. the answer is the biggest number in there: fmap[y * p + i][x * p + j] for i, j in range(p)
  5. collect the pooled maps in a list and return it
stuff youll need:
  max(...)              biggest item. works on a list, or on a generator like max(a * 2 for a in xs)
  //                    floor division
  two fors in one generator:  (fmap[...][...] for i in range(p) for j in range(p))  goes over the whole window
  nested comprehension:       [[... for x in range(W // p)] for y in range(H // p)]  builds a grid row by row"""}},

    "flatten_naive": {"hints": {"*": """\
how to think about it:
  go thru every feature map, then every row in that map, then every number in that row,
  and put all the numbers in one list. thats the same order they sit in memory: channel 0 first, row by row
stuff youll need:
  a comprehension can have more than one for. they run left to right like nested loops, so
  [a for row in grid for a in row] flattens a list of lists. you need one more level than that"""}},

    "linear_naive": {"hints": {"*": """\
how to think about it:
  every output o is a dot product: multiply each input x[i] by its weight weights[o][i], add them all up,
  then add the bias biases[o]. so loop over the outputs, and inside that loop over the inputs.
  weights has one row per output, so len(weights) is how many outputs there are
stuff youll need:
  range(len(xs))        every index of a list
  total += a * b        running total
  out.append(total)     build the answer list one output at a time"""}},

    "softmax": {"hints": {"*": """\
how to think about it:
  1. find the biggest score m. this is only so math.exp doesnt overflow (e^1000 is too big for a float).
     subtracting the same m from every score doesnt change the answer
  2. work out e^(z - m) for every score z
  3. add those all up
  4. divide each one by that total, now they add up to 1
stuff youll need:
  max(xs), sum(xs)      biggest item, total of a list
  math.exp(x)           e to the power x
  [... for z in logits] a comprehension"""}},

    "crd2idx": {"hints": {"*": """\
how to think about it:
  offset = coord[0]*stride[0] + coord[1]*stride[1] + ...
  so pair every coordinate with its stride, multiply each pair, and add up all the products.
  example: coord (1, 5, 7) with stride (1024, 32, 1) is 1024 + 160 + 7 = 1191
stuff youll need:
  zip(a, b)             walks two tuples side by side, gives you (a[0], b[0]), (a[1], b[1]), ...
  sum(...)              adds up a list, or a generator like sum(c * d for c, d in pairs)"""}},

    "strides": {"hints": {
        "row_major": """\
how to think about it:
  row-major means the LAST number in a coordinate moves fastest, so its stride is 1.
  the one before it jumps a whole row each step, so its stride is the size of the last mode.
  the one before that jumps a whole (rows x columns) block, and so on.
  so walk the shape from the right and keep a running product:
      shape (3, 32, 32):  last -> 1,  middle -> 32,  first -> 32 * 32 = 1024   ==>   (1024, 32, 1)
stuff youll need:
  reversed(xs)          goes over a list or tuple backwards
  step = 1, step *= s   a running product
  strides.append(step)  build the strides list (it comes out backwards, so flip it at the end)
  tuple(xs)             turns a list into a tuple. the answer has to be a tuple""",
        "col_major": """\
how to think about it:
  same idea as row_major, but the FIRST number moves fastest, so walk the shape from the left.
      shape (3, 32, 32):  first -> 1,  middle -> 3,  last -> 3 * 32 = 96   ==>   (1, 3, 96)
stuff youll need:
  the same stuff as row_major, minus reversed()"""}},

    "size_cosize": {"hints": {
        "size": """\
how to think about it:
  how many cells is the same as how many coordinates: multiply all the sizes in the shape.
  the shape can be nested like ((28, 28), (3, 5, 5)), so flatten it first
stuff youll need:
  layout.shape          the shape tuple stored on the layout
  flat(t)               (given above) turns a nested tuple into a flat one
  math.prod(xs)         multiplies everything together""",
        "cosize": """\
how to think about it:
  the last coordinate is (s - 1) for every size s. its offset is the sum of (s - 1) * d over every
  (size, stride) pair. memory starts at offset 0, so the memory you need is that offset + 1
stuff youll need:
  zip(flat(layout.shape), flat(layout.stride))   pairs every size with its stride
  sum(...)"""}},

    "transpose": {"hints": {"*": """\
how to think about it:
  a layout with two modes has shape (s0, s1) and stride (d0, d1). swapping rows and columns just means
  swapping both pairs: shape (s1, s0) and stride (d1, d0). no numbers move in memory, you only change
  the way you read it
stuff youll need:
  (s0, s1), (d0, d1) = layout.shape, layout.stride    unpacks two pairs in one line
  Layout(shape, stride)                               makes a new layout"""}},

    "gather_simple": {"hints": {"*": """\
how to think about it:
  offsets(layout) (given above) is a list of every offset the layout points at, in order.
  you want the value sitting at each of those offsets: storage[o] for every o
stuff youll need:
  [storage[o] for o in some_list]   a comprehension over the offsets"""}},

    "im2col_layout": {"override": '''
def im2col_layout(C, H, W, K):
    """the patches of a K x K convolution as a layout over a C x H x W image stored row-major (strides H*W, W, 1).

    row p    = an output pixel (oy, ox)        -> mode (OH, OW)
    column j = a spot (c, i, j) in the patch    -> mode (C, K, K)"""
    OH, OW = H - K + 1, W - K + 1
    # how to think about it:
    #   output pixel (oy, ox) reads the patch pixel (c, oy + i, ox + j).
    #   in memory that pixel is at   c*H*W + (oy + i)*W + (ox + j)
    #   multiply it out:             oy*W + ox*1 + c*H*W + i*W + j*1
    #   the number in front of each letter is its stride. the row mode is (oy, ox), the column mode
    #   is (c, i, j). replace every ... below with the right stride
    # stuff youll need:
    #   Layout(shape, stride) with nested tuples. the shape is already filled in for you,
    #   the stride has to have the exact same nesting as the shape
    return Layout(((OH, OW), (C, K, K)),
                  ((..., ...), (..., ..., ...)))
'''},

    "pool_layout": {"override": '''
def pool_layout(C, H, W, P):
    """the P x P max-pool windows of a C x H x W image, one window per row.

    row    = a window (c, oy, ox)    -> mode (C, H//P, W//P)
    column = a spot (i, j) inside it  -> mode (P, P)"""
    # how to think about it:
    #   window (oy, ox) of channel c starts at pixel (c, oy*P, ox*P). spot (i, j) inside it is
    #   pixel (c, oy*P + i, ox*P + j), which lives at   c*H*W + (oy*P + i)*W + (ox*P + j)
    #   multiply it out:   c*H*W + oy*P*W + ox*P + i*W + j*1
    #   same trick as im2col_layout: the number in front of each letter is its stride
    # stuff youll need:
    #   Layout(shape, stride) with nested tuples, replace every ...
    return Layout(((C, H // P, W // P), (P, P)),
                  ((..., ..., ...), (..., ...)))
'''},

    "matmul_t": {"hints": {"*": """\
how to think about it:
  C[i][j] is the dot product of row i of A with row j of B. you need that for every i and every j.
  the answer is ONE flat list: row 0 of C first (A[0] dotted with every row of B), then row 1, ...
stuff youll need:
  sumprod(a, b)                      the dot product, fast (its math.sumprod, set up in the setup cell)
  [f(a, b) for a in A for b in B]    two fors in one comprehension. the first for is the outer loop"""}},

    "conv2d_forward": {"hints": {"*": """\
how to think about it:
  ok so in lab 1 the conv was 6 nested loops. now its: read every patch, then do one big matmul.
  1. n = C * K * K is how many numbers are in one patch (and in one filter, same thing)
  2. gather x thru im2col_layout(C, H, W, K). that gives every patch one after another,
     so cut it into rows of n: one row per output pixel
  3. cut weight into rows of n too: one row per filter
  4. matmul_t(filters, patches) does every dot product at once. row k of the answer is feature map k
     (thats len(patches) = OH*OW numbers)
  5. add bias[k] to every number in row k and give it all back as one flat list
stuff youll need:
  gather(storage, layout)   reads memory thru a layout (you wrote it, then it got sped up)
  rows(flat_list, n)        cuts a flat list into rows of n (given above)
  matmul_t(A, B)            your matrix multiply
  zip(bias, rows(out, P))   pairs every bias with its feature map
  [v + b for ... for v in fmap]   a comprehension with two fors, to add the bias and flatten at once"""}},

    "maxpool2d_forward": {"hints": {"*": """\
how to think about it:
  gather x thru pool_layout(C, H, W, P). thats every window, P*P numbers each, one window after another.
  cut it into rows of P*P and keep the max of each row
stuff youll need:
  gather, rows, max"""}},

    "linear_forward": {"hints": {"*": """\
how to think about it:
  weight is one flat list of n_out * n_in numbers, where row o is the weights for output o.
  cut it into rows of len(x). then its a matmul with x as a matrix that has just one row: [x].
  that gives n_out dot products. add the bias onto each one
stuff youll need:
  rows(weight, len(x)), matmul_t(rows, [x]), zip(out, bias)"""}},

    "relu_forward": {"hints": {"*": """\
how to think about it:
  same as relu_naive but the input is always one flat list, so no recursion needed.
  keep v if its bigger than 0, otherwise 0.0
stuff youll need:
  [... for v in x] and  a if cond else b"""}},

    "cross_entropy_loss": {"hints": {"*": """\
how to think about it:
  1. start exactly like softmax: m = the biggest score, exps = e^(z - m) for every z, total = sum of exps
  2. the loss is -log(probability of the right class). the safe way to write that is
         loss = log(total) + m - logits[label]
     its the same number as -log(exps[label] / total) but it never tries to do log(0)
  3. give back both: the loss, and the probabilities [e / total for e in exps]
stuff youll need:
  math.log(x)           natural log (log base e)
  return a, b           gives back a tuple. whoever calls it can do  loss, probs = cross_entropy_loss(...)"""}},

    "sgd": {"methods": ["step"], "hints": {"step": """\
how to think about it:
  every param p has p.data (its numbers) and p.grad (which way is uphill for each number, or None).
  1. for every param that has a grad: new number = old number - self.lr * its grad (a step downhill)
     and store the new list back in p.data
  2. bonus, momentum: if self.momentum isnt 0, keep a velocity list for every param in self.velocity[i]
     (all zeros the first time). update it  v = momentum * v + g  and step along it:  w - self.lr * v
  3. at the very end count the step and ask the schedule what the next learning rate is:
         self.t += 1
         self.lr = self.schedule(self.t, self.total_steps, self.lr_max, self.lr_min)
stuff youll need:
  self.params, self.lr, self.momentum, ...   stuff __init__ already stored on the object
  enumerate(self.params)        gives (i, p) pairs, so you know which self.velocity[i] goes with p
  if p.grad is None: continue   skip params that didnt get a gradient
  [w - self.lr * g for w, g in zip(p.data, p.grad)]   one way to build the new list"""}},

    "cosine_annealing_lr": {"hints": {"*": """\
how to think about it:
  t / T is how far thru training you are: 0 at the start, 1 at the end.
  cos(pi * t / T) goes from 1 down to -1, so (1 + cos(...)) / 2 goes from 1 down to 0.
  scale that so it goes from lr_max down to lr_min:  lr_min + (lr_max - lr_min) * that.
  use min(t / T, 1.0) so it stays at lr_min if t ever goes past T
stuff youll need:
  math.cos(x), math.pi, min(a, b)"""}},

    "dataloader": {"methods": ["image", "__iter__"], "hints": {
        "image": """\
how to think about it:
  ok so self.raw is the whole .bin file as bytes. image k is record number self.index[k] in that file.
  every record is RECORD (3,073) bytes long and the first byte is the label, so this image's pixels
  start at  self.index[k] * RECORD + 1  and there are 3,072 of them. grab those.
  then do the 3 colour channels one at a time: channel ch is pixels[ch * 1024:(ch + 1) * 1024].
  turn every byte v of that channel into  (v / 255 - MEAN[ch]) / STD[ch]  and put all 3,072 floats
  in one list (channel 0 first, then 1, then 2)
stuff youll need:
  self.raw[a:b]         slicing bytes gives you more bytes
  pixels[i]             indexing bytes gives you an int 0-255
  MEAN, STD             tuples with one number per channel (given above)
  x = []; x += [...]    += sticks a list onto the end of another list""",
        "__iter__": """\
how to think about it:
  this runs when you write  for image, label in loader:
  1. make a list of positions 0, 1, ..., len(self) - 1
  2. if self.shuffle is true, shuffle that list (a new order every epoch)
  3. go thru it and hand out one (image, label) pair at a time
stuff youll need:
  list(range(len(self)))     the positions
  self.rng.shuffle(xs)       shuffles a list in place (self.rng is a random.Random made in __init__)
  yield a, b                 hands out one pair and pauses here untill the for loop asks for the next one.
                             a function with yield in it is a "generator"
  self.image(k), self.labels[k]"""}},

    "backward": {"hints": {"*": """\
how to think about it:
  the loss was made from tensors, which were made from other tensors, and so on (every tensor keeps a
  list of the ones it was made from in .children). to do the chain rule you have to go backwards thru
  all of them, and a tensor can only pass its gradient on once EVERY tensor that used it has already
  sent theirs back. a topological order fixes that:
  1. make an empty list topo and an empty set seen
  2. write a little function build(node): if you havent seen node yet, mark it as seen, call build on
     each of its children, and THEN append node to topo. (that order means kids always land before parents)
  3. call build(self). now the loss itself is the last thing in topo
  4. d(loss)/d(loss) = 1, so set self.grad = [1.0] * len(self.data)
  5. go thru topo BACKWARDS and call node.backward_fn() for every node that has one (not None)
stuff youll need:
  def build(node): ...  a function inside a function. it can see topo and seen from the outside one
  set(), s.add(v), v in s    a set remembers what youve visited, checking "in" is fast
  id(node)              a number unique to that object, a safe thing to put in the set
  reversed(topo)        goes thru the list from the end
  node.backward_fn()    calls the function stored on the node""",}},

    "relu_backward": {"hints": {"*": """\
how to think about it:
  dy is the gradient coming back from the next layer, one number per output.
  where x was bigger than 0, relu passed x straight thru, so the gradient goes straight thru too.
  where x was 0 or less, relu output 0 no matter what, so the gradient there is 0
stuff youll need:
  zip(dy, x) and  g if v > 0.0 else 0.0"""}},

    "linear_backward": {"hints": {"*": """\
how to think about it:
  the forward was  y[o] = b[o] + sum over i of W[o][i] * x[i]
  - dbias: every bias just got added onto its output, so dbias = dy (give back a copy)
  - dweight[o][i]: W[o][i] got multiplied by x[i], so its gradient is dy[o] * x[i].
    thats every dy times every x, in row-major order (o outside, i inside)
  - dx[i]: x[i] got used by EVERY output o, with weight W[o][i]. so dx[i] = sum over o of dy[o] * W[o][i].
    thats column i of W dotted with dy. to get the columns as rows, read W thru the transposed layout:
        columns = rows(gather(weight, Layout((n_in, n_out), (1, n_in))), n_out)
    and then  dx = matmul_t(columns, [dy])
  give back  dx, dweight, dbias
stuff youll need:
  n_in, n_out = len(x), len(dy)
  [g * v for g in dy for v in x]   every dy times every x
  list(dy)                         a copy of dy
  return a, b, c                   three things at once"""}},

    "maxpool2d_backward": {"hints": {"*": """\
how to think about it:
  only the biggest number in each window made it into the output, so only it gets that window's gradient.
  1. L = pool_layout(C, H, W, P), windows = rows(gather(x, L), P * P) gives every window's numbers
  2. offs = offsets(L) says where each of those numbers lives in x (same order)
  3. start dx as all zeros, one per number in x
  4. for window k with gradient dy[k]: find which spot won (w.index(max(w))). that spot lives in x at
     offs[k * P * P + winner]. add dy[k] there
stuff youll need:
  enumerate(zip(dy, windows))   gives  k, (g, w)
  w.index(v)                    where v is in w (the first one if theres a tie)
  [0.0] * len(x)                a list of zeros
  dx[o] += g"""}},

    "conv2d_backward": {"override": '''
def conv2d_backward(dy, x, weight, C, H, W, K, need_dx=True):
    """y = conv2d_forward(x, weight, bias). dy is the gradient for every output number (Cout x OH x OW).
    gives back (dx, dweight, dbias).

    forward was  out = filters @ patches^T  (+ a bias per filter), so by the matmul rules on the formula sheet:
      dbias[k]   = the sum of feature map k's gradient
      dweight    = dy @ patches              (Cout x n)
      dpatches   = dy^T @ filters            (P x n, how much each patch entry mattered)
      dx         = add every dpatches entry back onto the pixel it was read from (same layout as the forward)"""
    n = C * K * K
    L = im2col_layout(C, H, W, K)
    P = size(L[0])                                 # how many output pixels
    Cout = len(weight) // n
    dy_rows = rows(dy, P)                          # Cout rows, one feature map's gradient each
    # how to think about it:
    #   1. dbias. filter k's bias got added onto every pixel of feature map k, so its gradient is the
    #      sum of that whole map. one sum per row of dy_rows
    #   2. dweight. weight j of filter k got multiplied by patch entry j at every output pixel p, so
    #      dweight[k][j] = sum over p of dy[k][p] * patch[p][j]. thats a matmul of dy_rows with the
    #      patch COLUMNS, and the patch columns are just the im2col layout transposed:
    #          patch_columns = rows(gather(x, transpose(L)), P)      # n rows, P long
    #          dweight = matmul_t(dy_rows, patch_columns)
    #      if need_dx is False youre done (the first layer's input is the image, nobody needs its gradient):
    #          return None, dweight, dbias
    #   3. dpatches, how much every patch entry mattered. this part is given, just uncomment it:
    #          dy_columns = rows(gather(dy, Layout((P, Cout), (1, P))), Cout)           # P rows of Cout
    #          filter_columns = rows(gather(weight, Layout((n, Cout), (1, n))), Cout)   # n rows of Cout
    #          dpatches = matmul_t(dy_columns, filter_columns)                         # P x n
    #   4. dx. the forward READ patch entry t from x[offsets(L)[t]]. so going backwards, ADD dpatches[t]
    #      onto dx at that same offset. start dx as [0.0] * len(x). a pixel that showed up in 25 patches
    #      gets 25 things added onto it. this "scatter-add" is the backward of a gather
    #   give back  dx, dweight, dbias
    # stuff youll need:
    #   sum(row)                     adds up a list
    #   transpose(L)                 your transpose, swaps the patch rows and columns
    #   offsets(L)                   every offset the layout reads, in the same order as gather
    #   zip(offsets(L), dpatches)    pairs every offset with its gradient
    #   dx[o] += g                   adds onto one spot of a list
    raise NotImplementedError("todo: conv2d_backward")
'''},

    "cross_entropy_backward": {"hints": {"*": """\
how to think about it:
  the gradient of softmax + cross-entropy works out to the probabilities minus 1 at the right answer.
  copy probs, subtract 1.0 at position label, give it back.
  copy it first, if you change probs itself youd mess up the list whoever called you is still using
stuff youll need:
  list(xs)              a copy of a list
  d[label] -= 1.0"""}},

    "train_step": {"override": '''
def train_step(model, opt, image, label):
    """one step of learning on one image. gives back (loss, was the guess right?)"""
    # how to think about it, these 5 lines are basicly all of deep learning:
    #   1. forward: wrap the image in a Tensor with shape (3, 32, 32) and run the model on it. you get
    #      logits, 10 scores
    #   2. loss: softmax_cross_entropy(logits, label), a tensor with one number in it
    #   3. clear the old gradients, otherwise they pile up from the last image (the optimizer has a method for it)
    #   4. back propagation: call backward() on the loss. after that every weight has a .grad
    #   5. step: let the optimizer nudge every weight downhill
    #   then give back (loss.data[0], whether the biggest logit is at position label)
    # stuff youll need:
    #   Tensor(image, (3, 32, 32))     the input. no gradient needed for the picture itself
    #   model(x)                       runs every layer in order (thats Module.__call__)
    #   opt.zero_grad(), loss.backward(), opt.step()
    #   max(range(len(xs)), key=xs.__getitem__)   argmax: the position of the biggest item in xs
    raise NotImplementedError("todo: train_step")
'''},

    "evaluate": {"hints": {"*": """\
how to think about it:
  1. make a 10 x 10 grid of zeros called confusion. confusion[true][guess] counts how often the network
     guessed `guess` when the answer was `true`
  2. go thru the loader with enumerate so you know how many youve done. if limit isnt None and youve
     hit it, stop
  3. for every image ask the model: guess, _ = model.predict(image). then add 1 to confusion[label][guess]
  4. the right guesses are on the diagonal (confusion[i][i]). accuracy = diagonal total / everything total
  give back  accuracy, confusion
stuff youll need:
  [[0] * n for _ in range(n)]    a grid where every row is its own list. dont use [[0] * n] * n, that
                                 makes n copies of the SAME row so changing one changes all of them
  break                          stops a loop early
  sum(map(sum, grid))            adds up every number in a grid
  max(total, 1)                  so you never divide by zero"""}},
}


# =====================================================================================================
# doc comments that go on top of the given code cells
# =====================================================================================================
NOTES = {
    "data_basics": """\
what this does: finds the data folder and reads a .bin file into python
stuff in here:
  RECORD = 1 + 3 * 32 * 32     how many bytes one image takes up (1 label byte + 3,072 pixels)
  Path(...)                    a path to a file or folder (from pathlib). paths join with /, like DATA / "test.bin"
  Path.cwd()                   the folder python is running in. in jupyter thats the folder this notebook is in (src/)
  Path(__file__).resolve()     where this code's own file is. a notebook doesnt have one, so it uses Path.cwd()
  "__file__" in globals()      true when the code runs from a .py file instead of a notebook
  here.parent                  one folder up, so here.parent / "data" is ../data
  os.environ.get("MEME10_DATA")   reads an environment varible if you set one, otherwise gives None
  .exists()                    true if the file is really there
  .read_text().split()         reads a whole file as text and splits it on spaces/newlines (one class per line)
  .read_bytes()                reads a whole file as bytes (a list of numbers 0-255)
  raw[i]                       one byte, comes out as an int (thats the label)
  raw[i + 1:i + RECORD]        the 3,072 pixel bytes right after it
  range(0, len(raw), RECORD)   0, 3073, 6146, ... where every record starts
  raise FileNotFoundError(...) stops everything with an error message""",

    "nested_image": """\
what this does: turns the flat list of 3,072 pixels into image[c][y][x] and scales them to 0..1
stuff in here:
  C=3, H=32, W=32              default values, you only pass them if your picture is a different size
  [[[... for x in range(W)] for y in range(H)] for c in range(C)]
                               a comprehension inside a comprehension inside a comprehension. the outside one
                               makes the channels, the middle one the rows, the inside one the numbers in a row
  c * H * W + y * W + x        the pixel formula from lab 0 (with H = W = 32 thats c*1024 + y*32 + x)
  / 255                        bytes go 0-255, this squishes them to 0..1""",

    "lenet_naive": """\
what this does: makes random weights for lenet-5 and runs your lab 1 layers in the right order
stuff in here:
  random.gauss(0.0, std)       a random number from a bell curve centered on 0, `std` is how spread out it is
  math.sqrt(2.0 / fan_in)      "he init": layers with more inputs get smaller starting weights so nothing blows up
  for _ in range(n)            _ is a name for a loop varible you dont actually use
  nest(flat_list, shape)       recursion: cut the list into shape[0] pieces and nest every piece the same way
  LENET_SHAPES = {...}         a dict: name -> shape. "conv1.w" is the first conv's filters, "conv1.b" its biases
  params.items()               every (name, value) pair in a dict
  math.prod(shape)             how many numbers fit in that shape (6 * 3 * 5 * 5 = 450)
  name.endswith(".w")          string check, true for weights
  a if cond else b             weights get he init, biases get zeros
  {name: ... for name, v in params.items()}   a dict comprehension, builds a new dict""",

    "tuple_helpers": """\
what this does: little helpers for nested tuples like ((28, 28), (3, 5, 5)), so the layout math only ever
sees flat ones
stuff in here:
  isinstance(t, tuple)         true if t is a tuple
  tuple(x for item in t for x in flat(item))   recursion + a generator: flatten every item, glue them together
  iter(values), next(values)   an iterator hands out the next value every time you call next() on it
  def rebuild(t): inside unflat   a function inside a function, it can use `values` from the outer one
  i % n, i //= n               remainder and floor division. its how you split a number into "digits"
                               (125 -> 1, 2, 5 when every digit goes up to 10)
  ",".join(parts)              glues strings together with commas in between""",

    "layout_class": """\
what this does: the Layout class. a layout is a shape and a stride, and calling it turns a coordinate into
a memory offset
stuff in here:
  class Layout:                a new type of object. Layout((3, 32, 32)) makes one
  def __init__(self, ...)      runs when you make one. self is the new layout. stride=None means its optional
  self.shape = ...             stores stuff on the object so the other methods can use it later
  ... in flat(...)             the ... (its called Ellipsis) check is for the fill in the blank todos later
  raise ValueError(...)        complains if the shape and stride dont have the same structure
  def __call__(self, *coord)   lets you call the object like a function: layout(1, 5, 7).
                               *coord collects all the arguments into one tuple
  def __getitem__(self, i)     makes layout[0] work, it gives one mode as its own layout
  def __len__(self)            makes len(layout) work: how many modes (the "rank")
  def __eq__ / __hash__        makes == work and lets layouts be dict keys
  def __repr__(self)           what you see when you print it, the cute style shape:stride
  f"{a}:{b}"                   an f-string, puts values into a string
  make_layout(L1, L2)          glues two layouts into one that has two modes""",

    "print_layout": """\
what this does: prints a layout with two modes as a table of offsets (cute has a function with the same name)
stuff in here:
  [[layout(i, j) for j in range(N)] for i in range(M)]   every offset in the table
  f"{v:>{w}}"                  right-aligns v in a space w characters wide
  "-" * (w + 2)                repeats a string
  "+".join(...)                glues pieces together with + in between""",

    "offsets": """\
what this does: lists every offset a layout points at, in row-major order, and remembers the answer
stuff in here:
  _OFFSETS = {}                a dict used as a cache, so the same layout only gets worked out once
  key = (layout.shape, layout.stride)   tuples can be dict keys (lists cant)
  [o + i * d for o in offs for i in range(s)]
                               start with [0]. a mode with size s and stride d turns every offset o into
                               o, o + d, o + 2d, ... do that for every mode and youre done""",

    "gather_fast": """\
what this does: the same job as your gather, way faster. it replaces yours for the rest of the notebook
stuff in here:
  operator.itemgetter(*offs)   makes a function that grabs storage[o] for every o at once, and it runs in c.
                               *offs spreads the list out into seperate arguments
  _GETTERS                     a cache again, one getter per layout
  lambda s, o=offs[0]: (s[o],) a tiny one line function for the one offset case (itemgetter with one index
                               gives back a plain value instead of a tuple)""",

    "zipped_divide": """\
what this does: cuts every mode of a layout into tiles, a simple version of cute's zipped_divide
stuff in here:
  zip(layout.shape, layout.stride, tiler)   walks three tuples together
  s % t                        remainder. if its not 0 the tile doesnt fit into the mode evenly
  the rule: a mode with size s and stride d, tiled by t, turns into (t, s // t) with strides (d, t * d)""",

    "rows": """\
what this does: cuts a flat list into rows of n, so a flat list can be used as a matrix
stuff in here:
  range(0, len(flat_list), n)  0, n, 2n, ... where every row starts
  flat_list[i:i + n]           the n items starting at i (works on tuples too)""",

    "lenet_fast": """\
what this does: the same lenet-5 forward pass as lab 1, built out of your fast kernels on flat lists
stuff in here:
  params["conv1.w"]            a flat list now, not nested
  the 3, 32, 32, 5             channels, height, width, filter size for that conv
  no flatten line              a 16 x 5 x 5 list in CHW order already IS the 400 long list""",

    "param": """\
what this does: the simplest thing an optimizer can work on, a list of numbers plus their gradient
stuff in here:
  list(data)                   copies the numbers into a new list
  self.grad = None             no gradient yet
  __repr__                     rounds the numbers when you print it""",

    "schedules": """\
what this does: two learning rate schedules, with the same inputs as cudagrad's schedulers.hpp: (t, T, lr_max, lr_min)
stuff in here:
  t = how many steps so far, T = how many steps in total
  min(t / T, 1.0)              how far thru training you are, never more than 1""",

    "channel_stats": """\
what this does: works out the average and spread of every colour channel over the whole training file.
thats where the numbers in MEAN and STD come from
stuff in here:
  b"".join(...)                glues lots of bytes objects into one
  sum(plane) / len(plane)      the average byte, then / 255 to get it in 0..1
  sumprod(plane, plane)        the sum of every value squared
  var = average(v^2) - average(v)^2   the variance formula. the std is its square root
  round(x, 4)                  rounds to 4 decimals""",

    "tensor_class": """\
what this does: the Tensor class, one node in the computation graph (cudagrad calls it Node<T>)
stuff in here:
  data, shape                  the numbers (flat) and their shape
  grad                         d(loss)/d(each number), backward() fills it in. starts as None
  children                     the tensors this one was made from, thats what makes it a graph
  backward_fn                  a function that sends this tensor's grad back to its children
  any(c.requires_grad for c in children)   true if at least one child needs a gradient
  add_grad(g)                  adds g on top of the gradient thats already there (a tensor used twice gets both)
  @property                    a decorator (a line starting with @ right above a def changes how it works).
                               this one lets you write t.layout without the () even tho layout is a method
  leaf(data, shape)            a tensor we want gradients for, like a weight""",

    "ops": """\
what this does: wires your forward and backward kernels into the graph. every op does 3 things:
  1. runs the forward kernel and wraps the answer in a new Tensor that remembers its children
  2. defines backward_fn, a function INSIDE the op. even after the op returns, backward_fn can still see
     x, weight, out and so on (thats called a closure), so later it can run the backward kernel
     with the right inputs
  3. sticks backward_fn onto the output tensor
stuff in here:
  C, H, W = x.shape            unpacking the shape
  out.grad                     the gradient that shows up at the output when backward() runs
  x.requires_grad              skips work for stuff that doesnt need gradients (like the input picture)
  flatten                      makes a new Tensor with the SAME data list, so nothing gets copied
  out.probs = probs            you can stick any extra attribute onto an object""",

    "modules": """\
what this does: layers that own their weights (like cudagrad's Conv, MaxPool, Flatten, Linear) and a CNN
that stacks them into lenet-5
stuff in here:
  class Conv(Module)           Conv is a kind of Module, it gets __call__ and parameters() from Module for free
                               (thats called inheritance)
  def __call__(self, x)        so model(x) runs model.forward(x)
  return self                  add() gives back the CNN itself, so you can chain .add(...).add(...)
  [p for layer in self.layers for p in layer.parameters()]   every weight in the whole network in one list
  max(range(n), key=probs.__getitem__)    argmax: the position of the biggest probability
  json.dumps(...), json.loads(...)        turn lists and dicts into text and back, thats how weights get saved
  float(f"{v:.7g}")            rounds a number to 7 digits so the file is smaller
  Path(path).parent.mkdir(parents=True, exist_ok=True)   makes the folder if it isnt there yet
  **info                       collects any extra name=value arguments into a dict""",

    "train": """\
what this does: the training loop. sgd with a cosine learning rate, one image per step, and it prints progress
stuff in here:
  schedule or cosine_annealing_lr   "or" picks the second one when the first one is None
  time.time()                  seconds since 1970. take it twice and subtract to time something
  history.append({...})        a list of dicts, one per printed line, used for the charts later
  f"{x:.3f}", f"{x:6.1%}"      3 decimals, and a percent 6 characters wide with 1 decimal
  print(..., flush=True)       shows the line right away instead of waiting
  evaluate(model, val_loader, limit=val_limit)   checks progress on some val images after every epoch""",
}


# =====================================================================================================
# the notebook
# =====================================================================================================
def intro(nb):
    nb.md(r"""
# build a neural network from scratch: lenet-5 vs memes

every ai you've heard of, from chatgpt to self driving cars, is built out of neural networks. most people only ever touch them thru libraries like pytorch. in this notebook you build one yourself with **no ai libraries**, just python and math. you write every piece of lenet-5, train it on 35,000 memes, and then test it on 150 real memes it has never seen (yes that includes people doing the 6-7 hands).

### how this notebook works
- go top to bottom. run a cell with **shift + enter**.
- cells that start with `# todo:` are yours. the comments inside walk thru the logic step by step and list every bit of python you'll need. delete the `raise NotImplementedError(...)` line and write your code there, then run the check cell right under it.
- a check prints `pass`, `fail` (with what went wrong and a hint) or `todo` if you havent written it yet.
- stuck for more than a few minutes? put `%load solutions/NAME.py` in the cell and run it twice. the first run pastes the answer in and the second one runs it. if your editor doesnt paste it, use `%run -i solutions/NAME.py` instead.
- cells that start with `# given code` are already written. read the comments on top, then run them.
- you can only use the python standard library (`math`, `random`, `operator`, `time`, `pathlib`, ...). `viz` only draws pictures and `checks` only grades your answers, neither of them is part of the network.
- the next two cells are a **formula sheet** and a **python cheat sheet**. you dont have to read them now, come back whenever a todo mentions something you dont know.

| | what you build | time |
|---|---|---|
| lab 0 - the big picture | the dataset, lenet-5's shapes, how many weights it has | 10 min |
| lab 1 - layers by hand | convolution, relu, max-pool, fully connected, softmax with plain loops | 20 min |
| lab 2 - how data is stored | layouts (shape + stride), views, the im2col trick | 25 min |
| lab 3 - layers from tensors | a ~10x faster forward pass: one gather + one matrix multiply per layer | 15 min |
| lab 4 - learning | loss, sgd, cosine annealing, data loading | 20 min |
| lab 5 - back propagation | an autograd engine + the backward pass of every layer | 30 min |
| lab 6 - train it on memes | lenet-5, the training loop, accuracy on real memes | 25 min |
| bonus | [batching](addon_A_batching.ipynb) and [split-k](addon_B_split_k.ipynb) | |
""")
    nb.md(r"""
## formula sheet (every formula in the lab, in one place)

notation: an image is $C \times H \times W$ (channels, rows, columns). a filter is $K \times K$. padding is $P$, stride is $S$. $dy$ always means $\partial L / \partial y$, the gradient coming back from the layer after.

**sizes**
- conv output: $\text{out} = \left\lfloor \dfrac{H + 2P - K}{S} \right\rfloor + 1$. in this lab $P = 0$ and $S = 1$ so its just $H - K + 1$ (32 → 28 with a 5x5 filter)
- max-pool output with a $p \times p$ window and stride $p$: $\lfloor H / p \rfloor$ (28 → 14)
- weights in a conv layer: $C_\text{out} \cdot C_\text{in} \cdot K \cdot K + C_\text{out}$ (the $+C_\text{out}$ is one bias per filter)
- weights in a fully connected layer: $n_\text{in} \cdot n_\text{out} + n_\text{out}$
- multiply-adds in a conv layer: $C_\text{out} \cdot OH \cdot OW \cdot C_\text{in} \cdot K \cdot K$

**forward**
- convolution: $y[k][i][j] = b[k] + \sum_{c}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1} x[c][iS + u - P][jS + v - P] \cdot w[k][c][u][v]$. for us thats $x[c][i+u][j+v]$
- relu: $y = \max(0, x)$
- max-pool: $y[c][i][j] = \max_{u, v < p} \; x[c][ip + u][jp + v]$
- fully connected: $y[o] = b[o] + \sum_i W[o][i] \, x[i]$, in matrix form $y = Wx + b$
- softmax: $p_k = \dfrac{e^{z_k - m}}{\sum_j e^{z_j - m}}$ with $m = \max_j z_j$ (subtracting $m$ doesnt change anything, it just stops overflow)
- cross-entropy: $L = -\log p_y$ where $y$ is the right class. the safe way to compute it: $L = \log \sum_j e^{z_j - m} + m - z_y$

**memory and layouts**
- offset of a coordinate: $\text{offset} = \sum_i \text{coord}_i \cdot \text{stride}_i$
- row-major strides for shape $(C, H, W)$: $(HW,\; W,\; 1)$. column-major: $(1,\; C,\; CH)$
- pixel $(c, y, x)$ of a $C \times H \times W$ image sits at $cHW + yW + x$
- size $= \prod_i s_i$, cosize $= 1 + \sum_i (s_i - 1)\, d_i$
- im2col (the patches of a conv as a matrix): row $(oy, ox)$, column $(c, i, j)$, offset $= oy\,W + ox + cHW + iW + j$, so the layout is $((OH, OW), (C, K, K)) : ((W, 1), (HW, W, 1))$
- max-pool windows: $((C, H/p, W/p), (p, p)) : ((HW,\; pW,\; p), (W,\; 1))$
- conv as a matrix multiply: with $F$ = the filters as rows ($C_\text{out} \times CK^2$) and $X$ = the patches as rows ($OH \cdot OW \times CK^2$): $Y = F X^T$, then add $b[k]$ to row $k$

**backward**
- chain rule: $\dfrac{\partial L}{\partial x} = \dfrac{\partial L}{\partial y} \cdot \dfrac{\partial y}{\partial x}$, and if $x$ gets used twice its gradients add up
- relu: $dx = dy$ where $x > 0$, otherwise $0$
- fully connected: $dx = W^T dy$, $\quad dW[o][i] = dy[o]\, x[i]$, $\quad db = dy$
- max-pool: $dx = dy$ at the spot that won the max, $0$ everywhere else in that window
- softmax + cross-entropy together: $dz_k = p_k - [k = y]$ (the probabilities, minus 1 at the right answer)
- matrix multiply $C = A B^T$: $dA = dC \, B$, $\quad dB = dC^T A$
- convolution:
  - $db[k] = \sum_{i,j} dy[k][i][j]$
  - $dw[k][c][u][v] = \sum_{i,j} dy[k][i][j] \cdot x[c][i+u][j+v]$ (matrix form: $dF = dY \, X$)
  - $dx[c][h][w] = \sum_k \sum_{u,v} dy[k][h-u][w-v] \cdot w[k][c][u][v]$ over the $(h-u, w-v)$ that land inside $dy$. thats a "full" convolution of $dy$ with the filter flipped. matrix form: $dX = dY^T F$, then add every entry of $dX$ back onto the pixel it came from (scatter-add)

**learning**
- sgd: $w \leftarrow w - \text{lr} \cdot \dfrac{\partial L}{\partial w}$
- momentum: $v \leftarrow \mu v + g, \quad w \leftarrow w - \text{lr} \cdot v$
- cosine annealing: $\text{lr}(t) = \text{lr}_\min + \tfrac12 (\text{lr}_\max - \text{lr}_\min)\left(1 + \cos\dfrac{\pi t}{T}\right)$
- linear schedule: $\text{lr}(t) = \text{lr}_\max - (\text{lr}_\max - \text{lr}_\min)\, t / T$
- he init: starting weights $\sim \mathcal{N}\!\left(0,\; \sqrt{2 / \text{fan\_in}}\right)$ where fan_in is how many inputs go into one neuron ($C_\text{in} K^2$ for a conv)
- normalizing a pixel: $\hat v = (v / 255 - \text{mean}) / \text{std}$, per channel
- checking a gradient by nudging: $\dfrac{\partial L}{\partial w} \approx \dfrac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$
- accuracy = right guesses / total guesses
""")
    nb.md(r"""
## python cheat sheet (everything the todos use)

**lists and loops**
- `range(n)` is 0, 1, ..., n-1. `range(a, b, step)` works too, like `range(0, 30, 10)` → 0, 10, 20
- `for x in xs:` goes over the values, `for i in range(len(xs)):` goes over the positions
- `xs.append(v)` adds to the end. `xs[i]` gets item i, `xs[-1]` is the last one, `len(xs)` is how many
- slicing: `xs[a:b]` is the items from a up to (not including) b. `xs[::k]` is every k-th item
- list comprehension: `[v * 2 for v in xs]` builds a new list. you can stack fors, `[v for row in grid for v in row]` flattens a grid (the left for is the outer loop)
- one line if/else: `a if cond else b`
- `zip(a, b)` walks two lists side by side: `for x, w in zip(xs, ws):`
- `enumerate(xs)` gives `(position, value)` pairs
- `sum(xs)`, `max(xs)`, `min(xs)`. `max(range(n), key=f)` gives the i where `f(i)` is biggest, thats an argmax
- `[0.0] * n` is a list of n zeros. careful: `[[0] * n] * n` makes n copies of the SAME row, use `[[0] * n for _ in range(n)]`
- `break` stops a loop early, `continue` skips to the next round

**tuples**
- `(3, 32, 32)` is a tuple, like a list you cant change. `(5,)` is a tuple with one thing in it (the comma matters)
- unpacking: `c, h, w = x.shape` or `(a, b), (c, d) = pair_of_pairs`
- `tuple(xs)` turns a list into a tuple. tuples can be dict keys, lists cant

**functions**
- `def f(a, b=2):` gives b a default. `return x, y` gives back a tuple
- `*args` collects extra arguments into a tuple, `**kwargs` collects name=value ones into a dict. `f(*xs)` spreads a list out into seperate arguments
- a function can call itself (recursion), like `relu_naive` calling `relu_naive` on each sub-list
- a function inside a function can use the outer one's variables, even later (a closure). the autograd is built on this
- `lambda x: x * 2` is a tiny unnamed function
- `isinstance(x, list)` checks if x is a list

**classes**
- `class Foo:` makes a new type. `def __init__(self, ...)` runs when you do `Foo(...)`. `self` is the object being made
- `self.data = data` stores something on the object, you get it back later with `obj.data`
- methods with double underscores are special: `__len__` makes `len(obj)` work, `__iter__` makes `for x in obj` work, `__call__` makes `obj(...)` work, `__getitem__` makes `obj[i]` work, `__repr__` is what prints
- `class Conv(Module):` means Conv gets all of Module's methods for free (inheritance)
- decorators: a line like `@property` right above a `def` changes how that function works. `@property` lets you write `t.layout` without the `()`
- you can stick a function onto a class later: `Tensor.backward = backward`

**generators**
- a function with `yield` in it hands out values one at a time instead of building a whole list. `for x, y in loader:` pulls them out one by one. thats how the data loader avoids decoding 35,000 images at once

**math and friends**
- `math.exp(x)`, `math.log(x)` (natural log), `math.sqrt(x)`, `math.cos(x)`, `math.pi`, `math.prod(xs)` (multiplies everything)
- `sumprod(a, b)` is `a[0]*b[0] + a[1]*b[1] + ...` (a dot product). its `math.sumprod` from python 3.12, done in c so its fast
- `operator.itemgetter(*idx)(xs)` grabs `xs[i]` for every i in idx at once (also fast)
- `random.gauss(0, s)` random number from a bell curve, `random.Random(seed)` your own random generator, `.shuffle(xs)` shuffles a list in place
- `time.perf_counter()` a clock for timing stuff

**files**
- `from pathlib import Path`. `Path("data")` is a path. `Path.cwd()` is the folder python is running in (for jupyter thats this notebook's folder)
- `/` joins paths: `DATA / "test.bin"`. `.parent` is one folder up
- `.exists()`, `.read_text()`, `.read_bytes()`, `.write_text(s)`
- a `bytes` object is a list of numbers 0-255. `raw[5]` gives an int, `raw[1:1025]` gives more bytes
- `json.dumps(obj)` / `json.loads(text)` turn lists and dicts into text and back

**jupyter**
- `%load file.py` pastes a file into the cell, `%run -i file.py` runs it right away
""")
    nb.code(r"""
# setup, plain python only
import json, math, operator, os, random, time
from pathlib import Path
try:
    from math import sumprod      # python 3.12+: sumprod(a, b) = a[0]*b[0] + a[1]*b[1] + ...  (a dot product done in c)
except ImportError:               # older python: same answer, just slower
    def sumprod(a, b):
        return sum(map(operator.mul, a, b))
import viz, checks                # viz draws pictures, checks grades your answers. neither is part of the network
random.seed(0)                    # same random numbers every time you run the notebook
""")
    nb.code(r"""
# catch-up: only if youre behind or want to skip ahead. delete the # on the last line and run this cell.
# it loads the answer to every lab. after that, dont re-run todo cells you havent finished,
# an unfinished cell would replace the working answer with the unfinished one.
# from babytorch import *
""")


def lab0(nb):
    nb.md(r"""
---
## lab 0 - the big picture

a neural network is a function with alot of knobs. **lenet-5** (yann lecun and friends, 1998. its 1989 ancestor read handwritten zip codes for the us postal service) takes the 3 x 32 x 32 = 3,072 numbers of an image and turns them into 10 scores, one per meme. it has 62,006 knobs, called *weights*. **training** means turning the knobs untill the right score wins. we use the modern version of it: relu and max-pooling instead of the old tanh and average pooling.
""")
    nb.code("viz.lenet_diagram()   # draws the network with the shape of every tensor")
    nb.md(r"""
**convolution layers** slide small filters over the picture to find patterns (edges first, then eyes, sunglasses, toilets...). **max-pool layers** shrink the picture and keep the strongest signals. **fully connected layers** mix everything into the final 10 scores. you're going to build all of them.

### 0.1 meet the data
- **train**: 35,000 synthetic memes made from 457 real ones by a computer vision pipeline (crops, captions, emoji, blur...)
- **val**: 3,500 more synthetic memes for checking progress while training
- **test**: **150 real memes from the internet** that were never used for training. this is the real exam

every image is shrunk to 32 x 32 pixels, tiny, so plain python can keep up. the files use the cifar-10 format. each image is one record of 3,073 bytes:

| byte | what |
|---|---|
| 0 | the label, 0-9 |
| 1 ... 1024 | red values (0-255), row by row |
| 1025 ... 2048 | green |
| 2049 ... 3072 | blue |
""")
    nb.provided("data_basics")
    nb.code(r"""
test = load_bin(DATA / "test.bin")          # a list of (label, pixels) pairs, 150 of them
label, pixels = test[0]                      # unpack the first pair
print(len(test), "test images. image 0 has label", label, "=", CLASSES[label])
print("its first 8 red values (top left corner):", list(pixels[:8]))   # list() turns bytes into ints so they print nice
viz.show_images([(px, CLASSES[lab]) for lab, px in test[12::15]], scale=3, title="one real test meme per class (32 x 32)")
""")
    nb.md(r"""
pixel (row `y`, column `x`) of channel `c` sits at **`c*1024 + y*32 + x`** inside `pixels`. remember this one, its basicly the whole idea of lab 2.

try it: whats the *blue* value at row 31, column 0 of image 0? change `c, y, x` and run it again.
""")
    nb.code(r"""
c, y, x = 1, 5, 7                            # channel 1 is green
print(f"channel {c}, row {y}, column {x}:", pixels[c * 1024 + y * 32 + x])
""")
    nb.md(r"""
### 0.2 shapes and knobs
a k x k filter with no padding fits into a row of `size` pixels at `size - k + 1` spots. a p x p max-pool keeps one value per p x p window. write four tiny functions, then use them to work out lenet-5's shapes. (all the formulas are on the formula sheet at the top)
""")
    nb.exercise("shapes")
    nb.code("checks.shapes(conv_out, pool_out, conv_params, linear_params)")
    nb.code(r"""
# walks thru lenet-5 layer by layer using your functions
H = 32
print(f"input             3 x {H} x {H}")
H = conv_out(H, 5); print(f"conv 5x5 (6)      6 x {H} x {H}     knobs: {conv_params(3, 6, 5):>6,}")
H = pool_out(H, 2); print(f"max-pool 2x2      6 x {H} x {H}")
H = conv_out(H, 5); print(f"conv 5x5 (16)    16 x {H} x {H}     knobs: {conv_params(6, 16, 5):>6,}")
H = pool_out(H, 2); print(f"max-pool 2x2     16 x {H} x {H}  -> flatten to {16 * H * H} numbers")
print(f"fully connected  {16 * H * H} -> 120          knobs: {linear_params(16 * H * H, 120):>6,}")
print(f"fully connected  120 -> 84           knobs: {linear_params(120, 84):>6,}")
print(f"fully connected  84 -> 10            knobs: {linear_params(84, 10):>6,}")
""")
    nb.md(r"""
talk about it: which layer owns most of the knobs? which layers do most of the *multiplying*? (its not the same ones)
""")


def lab1(nb):
    nb.md(r"""
---
## lab 1 - layers, the slow obvious way

first we write every layer the most direct way possible. an image is a list of channels, a channel is a list of rows, a row is a list of numbers: `image[c][y][x]`. lots of nested loops. its correct and easy to read and slow, thats fine for now.
""")
    nb.provided("nested_image")
    nb.md(r"""
### 1.1 convolution
a **filter** (or *kernel*) is a small block of weights, here C x K x K. put it on top of the image, multiply every pair of numbers that line up, add them all up and add a bias: thats one output pixel. slide one step right and do it again. each filter makes one **feature map**, so a layer with 6 filters makes 6 maps.

$$\text{out}[k][y][x] = \text{bias}[k] + \sum_{c}\sum_{i}\sum_{j} \text{image}[c][y+i][x+j]\cdot\text{kernel}[k][c][i][j]$$

the output is smaller than the input, $OH = H - K + 1$ and $OW = W - K + 1$, becuase the filter cant hang off the edge. (with padding $P$ and stride $S$ its $\lfloor (H + 2P - K)/S \rfloor + 1$. we always use $P = 0$, $S = 1$)

```
 image rows y..y+2      filter          one output number
  3  1  0  2  5         1  0 -1
  2  8  1  0  1    x    2  0 -2    ->   3*1 + 1*0 + 0*(-1) + 2*2 + 8*0 + 1*(-2) + ...
  0  1  4  2  2         1  0 -1
```
""")
    nb.exercise("conv2d_naive")
    nb.code("checks.conv2d_naive(conv2d_naive)")
    nb.md(r"""
### 1.2 relu, max-pool, flatten
- **relu**: `max(0, v)`. keep the positive evidence, throw away the negative. without it the whole network would collapse into one big linear function (stacking linear layers just gives you another linear layer)
- **max-pool**: keep the biggest number in every p x p window. the picture gets half as wide and half as tall, and the network cares less about *exactly* where something is
- **flatten**: line everything up into one long list for the fully connected layers
""")
    nb.exercise("relu_naive")
    nb.code("checks.relu_naive(relu_naive)")
    nb.exercise("maxpool2d_naive")
    nb.code("checks.maxpool2d_naive(maxpool2d_naive)")
    nb.exercise("flatten_naive")
    nb.code("checks.flatten_naive(flatten_naive)")
    nb.md(r"""
### 1.3 watch a convolution work
real networks *learn* their filters. here are three classic hand made ones on a trollface. bright means the filter found its pattern there.
""")
    nb.code(r"""
label, pixels = test[90]                                # test image 90 is a trollface
img = to_nested(pixels)
vertical   = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]      # left to right changes = vertical edges
horizontal = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]      # top to bottom changes = horizontal edges
blur       = [[1 / 9] * 3 for _ in range(3)]           # the average of a 3x3 neighbourhood
kernels = [[f, f, f] for f in (vertical, horizontal, blur)]   # the same 3x3 pattern on red, green and blue
maps = conv2d_naive(img, kernels, [0.0, 0.0, 0.0])            # 3 filters -> 3 maps of 30 x 30
viz.show_image(pixels, caption=CLASSES[label], scale=4)
viz.show_maps(flatten_naive(maps), 3, 30, 30, labels=["vertical edges", "horizontal edges", "blur"], title="conv2d_naive")
pooled = maxpool2d_naive(relu_naive(maps))
viz.show_maps(flatten_naive(pooled), 3, 15, 15, scale=8, labels=["vertical", "horizontal", "blur"],
              title="after relu + 2x2 max-pool: a quarter of the numbers, same story")
""")
    nb.md(r"""
try it: make up your own 3 x 3 filter (`[[0, 1, 0], [1, -4, 1], [0, 1, 0]]` finds outlines) or pick a different picture, `test[0]` ... `test[149]`.

### 1.4 fully connected layer and softmax
a **fully connected** (linear) layer: every output looks at every input. output `o` is a dot product of the input with row `o` of the weight matrix, plus a bias:

$$\text{out}[o] = b[o] + \sum_i W[o][i]\cdot x[i]$$

**softmax** turns the final 10 scores (*logits*) into probabilities that add up to 1: $p_k = e^{z_k} / \sum_j e^{z_j}$. subtract the biggest score first or `math.exp(1000)` overflows.
""")
    nb.exercise("linear_naive")
    nb.code("checks.linear_naive(linear_naive)")
    nb.exercise("softmax")
    nb.code("checks.softmax(softmax)")
    nb.md(r"""
### 1.5 the whole lenet-5, slowly
stack your layers: conv → relu → pool → conv → relu → pool → flatten → linear → relu → linear → relu → linear → softmax. the weights are random for now (**he init**: random numbers with standard deviation $\sqrt{2/\text{fan\_in}}$, the same `Init::He` cudagrad uses), so expect a useless, almost even guess.
""")
    nb.provided("lenet_naive")
    nb.code(r"""
params = init_lenet_params(seed=0)                     # a dict of flat lists, one per weight/bias
t = time.perf_counter()                                # start the clock
probs = lenet_forward_naive(to_nested(pixels), params)
naive_ms = (time.perf_counter() - t) * 1000            # stop the clock, in milliseconds
viz.show_bars(probs, CLASSES, f"untrained lenet-5 looking at a {CLASSES[label]}")
print(f"one forward pass: {naive_ms:.0f} ms")
print(f"training = forward + backward (~3x a forward) for 35,000 images x 4 epochs = "
      f"{3 * naive_ms * 35000 * 4 / 3.6e6:.1f} hours")
""")
    nb.md(r"""
thats way too slow. real libraries never loop over nested lists. they keep a tensor in **one flat block of memory** plus a recipe for finding stuff in it. thats next.
""")


def lab2(nb):
    nb.md(r"""
---
## lab 2 - how data is stored

computer memory is one long row of numbers. a 3 x 32 x 32 image is 3,072 numbers in a row, and in lab 0 you found pixel (c, y, x) at `c*1024 + y*32 + x`. that recipe is called a **layout**.

nvidia's **cute** library (part of cutlass, the code behind the fastest matrix multiplies on gpus) describes every layout with two tuples:

- a **shape**: how many steps there are in each direction, like `(3, 32, 32)`
- a **stride**: how far thru memory one step goes, like `(1024, 32, 1)`

and it prints them as `shape:stride`, here `(3,32,32):(1024,32,1)`. the memory **offset** of a coordinate is just the coordinate times the stride:

$$\text{offset}(c, y, x) = c\cdot 1024 + y\cdot 32 + x\cdot 1$$

thats really the whole idea. transposes, slices, broadcasting, flattening, even convolution patches are all just a different shape and stride over the same memory. **no data gets copied.**

> real cute vs this lab: in cute (c++) you'd write `make_layout(make_shape(3,32,32), make_stride(1024,32,1))`, and in nvidia's python cute dsl `cute.make_layout((3, 32, 32), stride=(1024, 32, 1))` (that one needs an nvidia gpu). we copy cute's notation and names (`size`, `cosize`, `print_layout`, `zipped_divide`) with two differences: our default strides are **row-major** like python and pytorch (cute defaults to column-major), and a single number counts thru a layout row-major (cute counts the first mode fastest).

### 2.1 coordinates → offsets
shapes can be **nested**: `((28, 28), (3, 5, 5))` is a 2-D layout whose rows are themselves 2-D and whose columns are 3-D. these helpers flatten nested tuples so the math only ever sees flat ones.
""")
    nb.provided("tuple_helpers")
    nb.exercise("crd2idx")
    nb.code("checks.crd2idx(crd2idx)")
    nb.md(r"""
### 2.2 row-major and column-major
when you leave the stride out, a layout is **compact**: no gaps and no overlaps. there are two natural ways to do that. **row-major**: the last coordinate moves fastest, so rows sit together in memory (python, c and pytorch do this, cute calls it `LayoutRight`). **column-major**: the first coordinate moves fastest (fortran, matlab, and cute's default `LayoutLeft`).
""")
    nb.exercise("strides")
    nb.code("checks.strides(row_major, col_major)")
    nb.md(r"""
now the `Layout` class itself. its mostly bookkeeping around your `crd2idx` and `row_major`. the important part: a layout is **a function**. you call it with a coordinate and it gives back an offset.
""")
    nb.provided("layout_class")
    nb.code(r"""
image = Layout((3, 32, 32))          # no stride given -> row-major
print(image)                         # (3,32,32):(1024,32,1)
print(image(1, 5, 7))                # channel 1, row 5, column 7 -> 1*1024 + 5*32 + 7
print(pixels[image(1, 5, 7)])        # a layout + memory = a tensor. this is the actual pixel
""")
    nb.md(r"""
### 2.3 size and cosize
- **size**: how many coordinates a layout has (multiply the shape together)
- **cosize**: how much memory it reaches (1 + the offset of its last coordinate)

for a compact layout the two are the same. soon they wont be.
""")
    nb.exercise("size_cosize")
    nb.code("checks.size_cosize(size, cosize)")
    nb.provided("print_layout")
    nb.code(r"""
print_layout(Layout((4, 8), (8, 1)))   # row-major: walk along a row and the offset goes up by 1
print_layout(Layout((4, 8), (1, 4)))   # column-major: the SAME 32 numbers in a different order
""")
    nb.md(r"""
### 2.4 views: same memory, new layout
most "operations" on tensors dont touch the numbers at all, they only change the layout.

- **transpose**: swap the two modes. `(4,8):(8,1)` becomes `(8,4):(1,8)`
- **flatten**: `(16,5,5):(25,5,1)` and `400:1` reach the exact same offsets in the same order
- **slice**: one channel of the image is `(32,32):(32,1)` starting at offset 1024
- **broadcast**: a stride of **0** means that direction doesnt move. `(4,3):(0,1)` is 4 rows that all read the same 3 numbers
""")
    nb.exercise("transpose")
    nb.code("checks.transpose(transpose)")
    nb.provided("offsets")
    nb.code(r"""
print_layout(transpose(Layout((2, 3))))                              # a 2x3 matrix read column by column
print("flatten is free:", offsets(Layout((16, 5, 5))) == offsets(Layout(400)))
print("green(5, 7) =", pixels[1024 + Layout((32, 32))(5, 7)])        # a slice is a smaller layout + a starting offset
viz.show_layout(Layout((4, 3), (0, 1)), title="broadcast (4,3):(0,1), same colour = same spot in memory")
""")
    nb.md(r"""
### 2.5 gather: read memory thru a layout
to compute with a view you read the numbers it points at, in order. thats a **gather**.
""")
    nb.exercise("gather_simple")
    nb.code("checks.gather_simple(gather)")
    nb.provided("gather_fast")
    nb.md(r"""
### 2.6 the big one: convolution patches as a layout (im2col)
look at your `conv2d_naive` again. for output pixel `(oy, ox)` it reads a **patch**: `image[c][oy+i][ox+j]` for every `c, i, j`. thats 3 x 5 x 5 = 75 numbers for lenet's first layer, at 28 x 28 = 784 output pixels. put every patch in a row and you get a 784 x 75 matrix, which people call the **im2col** matrix.

copying it out would cost 58,800 numbers. instead, write it as a layout over the image's memory. pixel `(c, oy+i, ox+j)` lives at

$$c\cdot HW + (oy+i)\cdot W + (ox+j) \;=\; \underbrace{oy\cdot W + ox\cdot 1}_{\text{which patch}} \;+\; \underbrace{c\cdot HW + i\cdot W + j\cdot 1}_{\text{where in the patch}}$$

so the rows are the mode `(OH, OW)` and the columns are the mode `(C, K, K)`. read the strides right off that formula.
""")
    nb.exercise("im2col_layout")
    nb.code("checks.im2col_layout(im2col_layout)")
    nb.code(r"""
L = im2col_layout(1, 4, 4, 2)        # a tiny 1 x 4 x 4 image with 2 x 2 patches
print(L, "  size", size(L), "  cosize", cosize(L))
viz.show_layout(L, title=f"im2col_layout(1, 4, 4, 2) = {L}: 9 patches x 4 numbers out of only 16 memory spots")
""")
    nb.md(r"""
**size 36 but cosize 16**: the layout reads the same memory spot more than once (same colour = same spot) becuase neighbouring patches overlap. the patch matrix is real enough to compute with but it never got built.

same trick on lenet's first layer:
""")
    nb.code(r"""
L1 = im2col_layout(3, 32, 32, 5)
print(L1)
print(f"a {size(L1[0])} x {size(L1[1])} patch matrix = {size(L1):,} numbers, read out of {cosize(L1):,} numbers of memory")
""")
    nb.md(r"""
### 2.7 max-pool windows as a layout
same game: a layout where every row is a 2 x 2 window and the columns are the 4 spots inside it.
""")
    nb.exercise("pool_layout")
    nb.code("checks.pool_layout(pool_layout)")
    nb.code(r"""viz.show_layout(pool_layout(1, 4, 4, 2), title="pool_layout(1, 4, 4, 2): 4 windows, no overlap")""")
    nb.md(r"""
> in cute you dont write that one by hand, you *divide* the image layout by a tile. `zipped_divide(layout, (1, 2, 2))` splits every mode into (inside the tile, which tile) and groups them. cutlass uses exactly this to hand out tiles of a matrix to gpu threads.
""")
    nb.provided("zipped_divide")
    nb.code(r"""
Z = zipped_divide(Layout((1, 4, 4)), (1, 2, 2))
print(Z)
print("same windows as pool_layout:", offsets(transpose(Z)) == offsets(pool_layout(1, 4, 4, 2)))
""")


def lab3(nb):
    nb.md(r"""
---
## lab 3 - layers = tensors + operators

with layouts, every layer turns into **read thru a layout (gather) + one operator**:

| layer | layout | operator |
|---|---|---|
| convolution | `im2col_layout` | matrix multiply (+ bias) |
| max-pool | `pool_layout` | max of each row |
| fully connected | the weights as rows | matrix multiply (+ bias) |
| flatten | a new shape | nothing at all |

so the one operator that really matters is **matrix multiply**. ours takes two lists of rows and dots every row of A with every row of B: `C[i][j] = A[i] · B[j]`, which is A times B-transposed. cutlass would say both inputs are **K-major**: the numbers being added up (the *K* dimension) sit next to each other in memory. thats why one `sumprod` (a dot product done in c) can do each output.
""")
    nb.provided("rows")
    nb.exercise("matmul_t")
    nb.code("checks.matmul_t(matmul_t)")
    nb.md(r"""
### 3.1 convolution = gather + matmul
lenet's first layer turns into: gather the 784 x 75 patch matrix, then multiply the 6 x 75 filter matrix by it. every image is now a flat list in **CHW** order (all of channel 0, then channel 1, ...), so the output is ready for the next layer.
""")
    nb.exercise("conv2d_forward")
    nb.code("checks.conv2d_forward(conv2d_forward)")
    nb.code(r"""
x = [v / 255 for v in pixels]                          # the trollface as a flat list, 0..1
t = time.perf_counter(); fast = conv2d_forward(x, params["conv1.w"], params["conv1.b"], 3, 32, 32, 5); t_fast = time.perf_counter() - t
t = time.perf_counter(); slow = conv2d_naive(to_nested(pixels), nest(params["conv1.w"], (6, 3, 5, 5)), params["conv1.b"]); t_slow = time.perf_counter() - t
print("biggest difference from conv2d_naive:", max(abs(a - b) for a, b in zip(fast, flatten_naive(slow))))
print(f"conv2d_naive {t_slow * 1000:.1f} ms   vs   layout + matmul {t_fast * 1000:.1f} ms   ({t_slow / t_fast:.0f}x faster)")
""")
    nb.exercise("maxpool2d_forward")
    nb.code("checks.maxpool2d_forward(maxpool2d_forward)")
    nb.exercise("linear_forward")
    nb.code("checks.linear_forward(linear_forward)")
    nb.exercise("relu_forward")
    nb.code("checks.relu_forward(relu_forward)")
    nb.md(r"""
### 3.2 the fast lenet-5
same network, same weights, same answer, built from your new kernels. notice theres no flatten line: a `16 x 5 x 5` list in CHW order already *is* the 400 number list, so flatten costs nothing.
""")
    nb.provided("lenet_fast")
    nb.code(r"""
t = time.perf_counter(); fast_probs = lenet_forward(pixels, params); fast_ms = (time.perf_counter() - t) * 1000
print("same answer as lenet_forward_naive:", max(abs(a - b) for a, b in zip(fast_probs, probs)) < 1e-9)
print(f"naive {naive_ms:.0f} ms  ->  layouts + matmul {fast_ms:.1f} ms per image")
macs = 6 * 28 * 28 * 75 + 16 * 10 * 10 * 150 + 400 * 120 + 120 * 84 + 84 * 10   # multiply-adds per image
print(f"lenet-5 does {macs:,} multiply-adds per image: {macs / fast_ms / 1000:.0f} million per second in plain python")
""")
    nb.md(r"""
talk about it: a laptop gpu does trillions of multiply-adds a second. why is python so far behind? (every number is a python object and every step goes thru the interpreter. the c code inside `sumprod` and `itemgetter` is the reason lab 3 is ~10x faster than lab 1)
""")


def lab4(nb):
    nb.md(r"""
---
## lab 4 - learning: loss, optimizers, learning rate schedules, data loading

the network spits out 10 scores. to learn from that it needs four things:

1. **how wrong was it?** a *loss*: one number, small when the right class gets a high probability
2. **which way should each knob turn?** the *gradient*. lab 5 works that out, here we use toy problems where we already know it
3. **how far should it turn?** the *learning rate* and the *optimizer*
4. **what does it practice on, and in what order?** the *data loader*

### 4.1 cross-entropy loss
$\text{loss} = -\log(p_\text{correct})$. if the network gives the right class 100%, the loss is 0. an even guess over 10 classes costs $\ln 10 \approx 2.30$. being confident and wrong costs alot.
""")
    nb.exercise("cross_entropy_loss")
    nb.code("checks.cross_entropy_loss(cross_entropy_loss)")
    nb.code(r"""
for p in (0.99, 0.5, 0.1, 0.01):
    print(f"probability of the right class {p:>4}:  loss {-math.log(p):.2f}")
loss, _ = cross_entropy_loss([math.log(p) for p in probs], label)    # log(probs) works as scores, softmax undoes it
print(f"our untrained network on the {CLASSES[label]}: loss {loss:.2f} (ln 10 = {math.log(10):.2f})")
""")
    nb.md(r"""
### 4.2 gradient descent
the gradient says which way is *uphill* for the loss, so step the other way: $w \leftarrow w - \text{lr}\cdot\frac{\partial \text{loss}}{\partial w}$. doing it with the gradient of one example (or a few) at a time is **stochastic gradient descent (sgd)**. its the optimizer in cudagrad and wordhunt-ai, and in this lab.

first a toy problem: fit a line $y = w\,x + b$ thru 5 points by making the mean squared error small. its gradients are easy to work out by hand, theyre in `loss_and_grads`.
""")
    nb.provided("param")
    nb.code(r"""
xs = [0.0, 1.0, 2.0, 3.0, 4.0]
ys = [1.1, 2.9, 5.2, 7.1, 8.8]                      # roughly y = 2x + 1

def loss_and_grads(w, b):
    errors = [w * x + b - y for x, y in zip(xs, ys)]
    loss = sum(e * e for e in errors) / len(xs)                 # mean squared error
    dw = sum(2 * e * x for e, x in zip(errors, xs)) / len(xs)   # d loss / d w
    db = sum(2 * e for e in errors) / len(xs)                   # d loss / d b
    return loss, dw, db

print(loss_and_grads(0.0, 0.0))   # starting at w = b = 0: big loss, and both gradients say "make them bigger"
""")
    nb.provided("schedules")
    nb.exercise("sgd")
    nb.code("checks.sgd(SGD, Param)")
    nb.code(r"""
def fit(lr, steps=60, schedule=constant_lr, momentum=0.0):
    w, b = Param([0.0]), Param([0.0])
    opt = SGD([w, b], lr_max=lr, lr_min=lr / 100, total_steps=steps, schedule=schedule, momentum=momentum)
    losses = []
    for _ in range(steps):
        loss, dw, db = loss_and_grads(w.data[0], b.data[0])
        losses.append(loss)
        w.grad, b.grad = [dw], [db]                    # hand the optimizer the gradients
        opt.step()
    return losses, w.data[0], b.data[0]

runs = {f"lr {lr}": fit(lr)[0] for lr in (0.01, 0.05, 0.14, 0.16)}
viz.line_chart(runs, "sgd fitting a line: the learning rate matters", "step", "loss (log scale)", logy=True)
losses, w, b = fit(0.05)
print(f"lr 0.05 found y = {w:.2f}x + {b:.2f}")
""")
    nb.md(r"""
too small is slow. too big overshoots back and forth, and past a certain point it **blows up** (lr 0.16).

try it: find the biggest learning rate that still works. then try `fit(0.01, momentum=0.9)`. momentum keeps a running "velocity", kind of like a ball rolling downhill.

### 4.3 learning rate schedules: cosine annealing
take big steps early (explore) and small steps at the end (settle down). **cosine annealing** slides the learning rate along half a cosine wave from `lr_max` to `lr_min`:

$$\text{lr}(t) = \text{lr}_\min + \tfrac{1}{2}(\text{lr}_\max - \text{lr}_\min)\left(1 + \cos\frac{\pi t}{T}\right)$$

its the `cosine_annealing_LR` in cudagrad's `schedulers.hpp`, and what wordhunt-ai trains with.
""")
    nb.exercise("cosine_annealing_lr")
    nb.code("checks.cosine_annealing_lr(cosine_annealing_lr)")
    nb.code(r"""
T = 100
viz.line_chart({name: [f(t, T, 0.01, 0.001) for t in range(T + 1)]
                for name, f in [("constant", constant_lr), ("linear", linear_lr), ("cosine annealing", cosine_annealing_lr)]},
               "learning rate schedules (cudagrad's schedulers.hpp)", "step t", "learning rate")
viz.line_chart({"constant lr 0.14": fit(0.14)[0], "cosine 0.14 -> 0.0014": fit(0.14, schedule=cosine_annealing_lr)[0]},
               "same start, but cosine annealing calms down at the end", "step", "loss (log scale)", logy=True)
""")
    nb.md(r"""
### 4.4 data loading
35,000 images x 3,072 numbers = 107 million numbers. as python floats thats several gigabytes, way too much. so the loader keeps the file as compact **bytes** (107 mb) and turns *one image at a time* into floats, only when its needed.

it also does two more jobs:
- **normalize** every colour channel to average 0 and spread 1: $(v/255 - \text{mean}) / \text{std}$. networks learn alot better when the inputs are centered and about the same size
- **shuffle** every epoch so the network cant learn the order
""")
    nb.provided("channel_stats")
    nb.code(r"""
t = time.perf_counter()
print("mean, std per channel:", channel_stats(DATA / "train.bin"), f"({time.perf_counter() - t:.1f} s for 107 million pixels)")
""")
    nb.exercise("dataloader")
    nb.code("checks.dataloader(DataLoader)")
    nb.code(r"""
train_loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=350)
print(len(train_loader), "training images (350 per class)")

def denormalize(x):   # back to 0..1 so we can look at it
    return [x[c * 1024 + k] * STD[c] + MEAN[c] for c in range(3) for k in range(1024)]

batch = [pair for _, pair in zip(range(16), train_loader)]       # zip with range(16) stops after 16 pairs
viz.show_images([(denormalize(x), CLASSES[y]) for x, y in batch], scale=3, title="synthetic training memes, shuffled")
""")


def lab5(nb):
    nb.md(r"""
---
## lab 5 - forward and back propagation

to train, we need to know for **every one of the 62,006 knobs**: if i nudge this knob up a tiny bit, does the loss go up or down, and how fast? thats the gradient $\partial\,\text{loss} / \partial w$. there are two ways to get it:

1. **nudge every knob and re-run the network.** thats 62,006 extra forward passes per image. hopeless for training, but great for *checking*
2. **back propagation.** run forward once, then go *backwards* thru the layers using the **chain rule**. all 62,006 gradients cost about two more forward passes. every deep learning library does this

### 5.1 the chain rule on one neuron
""")
    nb.code(r"""
x, w, b, target = 2.0, 0.5, -0.2, 1.5
z = w * x + b                # forward, one step at a time
y = max(0.0, z)
L = (y - target) ** 2

dL_dy = 2 * (y - target)     # backward: the slope of each step...
dy_dz = 1.0 if z > 0 else 0.0
dz_dw = x
dL_dw = dL_dy * dy_dz * dz_dw   # ...all multiplied together, thats the chain rule

eps = 1e-6                    # check it the slow way: nudge w a tiny bit and watch L
L_nudged = (max(0.0, (w + eps) * x + b) - target) ** 2
print(f"chain rule: {dL_dw:.6f}    nudging: {(L_nudged - L) / eps:.6f}")
""")
    nb.md(r"""
### 5.2 the computation graph
a library cant know your formula ahead of time, so it **records** it while it runs forward. every operation makes a new `Tensor` that remembers:
- its **children**: the tensors it was made from
- a **`backward_fn`**: given this tensor's gradient, add the right gradients into its children

thats cudagrad's `Node<T>` (`value`, `children`, `backward_fn`, `op`). then `loss.backward()`:
1. sorts the graph so every tensor comes after the tensors it was made from (a **topological sort**)
2. sets $\partial\,\text{loss}/\partial\,\text{loss} = 1$
3. walks that order **backwards** and calls every `backward_fn`

```cpp
// cudagrad, Node.cuh
void backward() {
    build_topo(this, topo, seen);
    launch_fill<T>(value.grad, T(1), value.size);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn();
}
```

why the sort? a tensor thats used twice (say by `b` *and* by `y`) has to collect gradient from *both* uses before it passes anything on. thats also why `add_grad` adds instead of overwriting.
""")
    nb.provided("tensor_class")
    nb.exercise("backward")
    nb.code("checks.backward(Tensor)")
    nb.md(r"""
### 5.3 the backward kernels
for every layer: given `dy` (the gradient of the loss for the layer's output), work out the gradients for its input and its weights. like cudagrad's `launch_*_backward` kernels, these are plain functions on flat lists.

**relu.** $y = \max(0, x)$ has slope 1 where $x > 0$ and 0 everywhere else, so the gradient either goes straight thru or gets blocked.
""")
    nb.exercise("relu_backward")
    nb.code("checks.relu_backward(relu_backward)")
    nb.md(r"""
**fully connected.** $y = Wx + b$, so $y_o = b_o + \sum_i W_{oi}\,x_i$. every $x_i$ touches every $y_o$ with weight $W_{oi}$, so:

$$\frac{\partial L}{\partial x_i} = \sum_o dy_o\,W_{oi}\;(\text{thats } W^T dy),\qquad \frac{\partial L}{\partial W_{oi}} = dy_o\,x_i,\qquad \frac{\partial L}{\partial b_o} = dy_o$$

$W^T$ is just W's memory read thru the transposed **layout**, nothing gets copied untill we gather it.
""")
    nb.exercise("linear_backward")
    nb.code("checks.linear_backward(linear_backward)")
    nb.md(r"""
**max-pool.** only the winner of each window had any effect on the output, so only the winner gets the gradient. `pool_layout` tells you where the winner lives in memory.
""")
    nb.exercise("maxpool2d_backward")
    nb.code("checks.maxpool2d_backward(maxpool2d_backward)")
    nb.md(r"""
**convolution.** the forward pass was `out = filters @ patches^T` (+ bias), a matrix multiply, so its backward pass follows the same rules as the fully connected layer:

| gradient of | is | shape |
|---|---|---|
| bias | the sum of each feature map's `dy` | Cout |
| filters | `dy @ patches` | Cout x 75 |
| patches | `dy^T @ filters` | 784 x 75 |
| image | every patch gradient **added back** onto the pixel it was read from | 3 x 32 x 32 |

written out with indices:

$$db[k] = \sum_{i,j} dy[k][i][j] \qquad dw[k][c][u][v] = \sum_{i,j} dy[k][i][j]\; x[c][i+u][j+v] \qquad dx[c][h][w] = \sum_k \sum_{u,v} dy[k][h-u][w-v]\; w[k][c][u][v]$$

that last row of the table is the deep idea: **the backward of a gather is a scatter-add thru the same layout.** a pixel that showed up in 25 patches collects gradient from all 25.

(the first layer doesnt need the image's gradient, the image isnt a knob. so `need_dx=False` skips that work)
""")
    nb.exercise("conv2d_backward")
    nb.code("checks.conv2d_backward(conv2d_backward)")
    nb.md(r"""
**softmax + cross-entropy.** the calculus works out to something realy simple: $\partial L / \partial z_k = p_k - [k = \text{label}]$. the network's probabilities, minus 1 at the right answer.
""")
    nb.exercise("cross_entropy_backward")
    nb.code("checks.cross_entropy_backward(cross_entropy_backward)")
    nb.md(r"""
### 5.4 wiring it together
every operation runs its forward kernel, wraps the answer in a `Tensor` that remembers its children, and sticks on a `backward_fn` that calls the matching backward kernel. same pattern as cudagrad's `Node.cuh`.
""")
    nb.provided("ops")
    nb.code(r"""
# a gradient check: backprop vs "nudge and measure" for one conv -> relu -> max-pool on a random 2x6x6 input
random.seed(1)
xin = leaf([random.uniform(-1, 1) for _ in range(2 * 6 * 6)], (2, 6, 6))
wc = leaf([random.uniform(-1, 1) for _ in range(3 * 2 * 3 * 3)], (3, 2, 3, 3))
bc = leaf([0.1, 0.0, -0.1], (3,))

def forward():
    return maxpool2d(relu(conv2d(xin, wc, bc)))   # the "loss" here is just the sum of this output

out = forward()
out.backward()                                    # backward() starts from all ones: d(sum)/d(out) = 1
eps = 1e-6
for name, t in [("input", xin), ("weight", wc), ("bias", bc)]:
    i = random.choice([k for k, g in enumerate(t.grad) if g != 0.0])     # pick an entry that matters
    old = t.data[i]
    t.data[i] = old + eps; up = sum(forward().data)
    t.data[i] = old - eps; down = sum(forward().data)
    t.data[i] = old
    print(f"{name:>6}[{i:>2}]   backprop {t.grad[i]: .6f}    nudging {(up - down) / (2 * eps): .6f}")
""")


def lab6(nb):
    nb.md(r"""
---
## lab 6 - train lenet-5 on memes

### 6.1 modules
layers that own their weights, stacked in a container. these copy cudagrad's `Conv`, `MaxPool`, `Flatten`, `Linear` and `CNN` (`net.add(...).add(...)`).
""")
    nb.provided("modules")
    nb.code(r"""
model = lenet5(seed=0)
print(sum(len(p.data) for p in model.parameters()), "numbers to learn in", len(model.parameters()), "tensors")
logits = model(Tensor(DataLoader(DATA / "test.bin").image(90), (3, 32, 32)))
print(logits)
""")
    nb.md("before training, make sure backprop thru the *whole* network matches nudging, on random knobs from every layer:")
    nb.code("checks.gradcheck_lenet(lenet5, Tensor, softmax_cross_entropy, DataLoader)")
    nb.md(r"""
### 6.2 one step of learning
forward → loss → clear the old gradients → backward → step. those five lines are the core of every neural network ever trained, from lenet to chatgpt.
""")
    nb.exercise("train_step")
    nb.code("checks.train_step(train_step, lenet5, SGD, DataLoader)")
    nb.code(r"""
test_loader = DataLoader(DATA / "test.bin")
image, true = test_loader.image(105), test_loader.labels[105]      # test image 105 is a surprised pikachu
model = lenet5(seed=0)
opt = SGD(model.parameters(), lr_max=0.005)
before = model.predict(image)[1][true]
steps = [train_step(model, opt, image, true) for _ in range(15)]
viz.line_chart({"loss": [s[0] for s in steps]}, "15 steps on ONE image: the network memorizes it", "step", "loss")
print(f"P({CLASSES[true]}) went from {before:.0%} to {model.predict(image)[1][true]:.0%}")
""")
    nb.md(r"""
### 6.3 measuring accuracy
**accuracy** = the fraction of images where the highest score is the right class. a **confusion matrix** shows *which* mistakes it makes: row = the right answer, column = the guess.
""")
    nb.exercise("evaluate")
    nb.code("checks.evaluate(evaluate, lenet5, DataLoader)")
    nb.md(r"""
### 6.4 train it
one image at a time, plain sgd, cosine annealed learning rate. thats the same recipe as wordhunt-ai. plain python does about 45-60 images a second on a fast laptop (an older one maybe a third of that), so pick a size that fits your time:

| `PER_CLASS` | `EPOCHS` | `LR` | images seen | time: fast laptop (measured) → older laptop (guess) | real meme accuracy |
|---|---|---|---|---|---|
| 100 | 5 | 0.01 | 5,000 | ~2 min → ~6 min | ~77% |
| 350 | 4 | 0.005 | 14,000 | ~4-6 min → ~15 min | ~83% |
| `None` (all) | 4 | 0.005 | 140,000 | ~46 min → 2-3 hours | ~89% (see 6.6) |

while it trains: talk about what the network might be looking at, or peek at the bonus notebooks.
""")
    nb.provided("train")
    nb.code(r"""
PER_CLASS = 350      # images per class. 100 = quick, 350 = the class run, None = everything
EPOCHS = 4
LR = 0.005           # lr_max, cosine annealing takes it down to 1e-5

model = lenet5(seed=0)
train_loader = DataLoader(DATA / "train.bin", shuffle=True, per_class=PER_CLASS)
val_loader = DataLoader(DATA / "val.bin")
print(f"{len(train_loader):,} images x {EPOCHS} epochs = {len(train_loader) * EPOCHS:,} steps of sgd")
history = train(model, train_loader, EPOCHS, lr_max=LR, val_loader=val_loader, val_limit=500)
""")
    nb.code(r"""
viz.line_chart({"training loss (running average inside each epoch)": [(h["step"], h["loss"]) for h in history]},
               "learning curve", "sgd step", "loss")
viz.line_chart({"learning rate": [(h["step"], h["lr"]) for h in history]}, "cosine annealing", "sgd step", "lr")
""")
    nb.md(r"""
### 6.5 the real exam: 150 real memes
the network only ever practiced on *synthetic* memes. does it work on real ones from the internet?
""")
    nb.code(r"""
acc, confusion = evaluate(model, test_loader)
print(f"accuracy on 150 real memes it has never seen: {acc:.1%}   (random guessing: 10%)")
viz.show_confusion(confusion, CLASSES)
""")
    nb.code(r"""
import csv
originals = [row["original"] for row in csv.DictReader(open(DATA / "test.csv"))]   # full size file for every test image

def show(*indices, net=None):
    net = net or model
    viz.show_predictions([(DATA / originals[i], test[i][1], net.predict(test_loader.image(i))[1],
                           test_loader.labels[i], CLASSES) for i in indices])

show(12, 16, 32, 46, 61, 77, 91, 106, 121, 136)    # one per class (12 = six_seven: someone doing the hands)
""")
    nb.md(r"""
try it: `show(7)`, `show(20, 21, 22)`... test images 0-14 are six_seven, 15-29 tung tung tung sahur, 30-44 tralalero tralala, 45-59 skibidi toilet, 60-74 chill guy, 75-89 doge, 90-104 trollface, 105-119 surprised pikachu, 120-134 drake, 135-149 distracted boyfriend. which mistakes make sense? (hint: at 32 x 32 the network leans on **colour** alot. a person in a blue shirt can look like the blue shark)

### 6.6 a fully trained model
your teacher trained the *same code* on all 35,000 images for 4 epochs (`weights/lenet_meme10.json` says how long it took).
""")
    nb.code(r"""
pretrained = lenet5()
info = pretrained.load("weights/lenet_meme10.json")
print(info)
acc_pre, confusion_pre = evaluate(pretrained, test_loader)
print(f"pretrained on all 35,000 images: {acc_pre:.1%} on the real memes")
viz.show_confusion(confusion_pre, CLASSES, "pretrained model (rows = the right answer, columns = the guess)")
show(2, 4, 5, 10, 12, net=pretrained)              # some six_seven memes
""")
    nb.code(r"""
model.save("weights/my_lenet.json", note="trained in class")   # load it again any time with lenet5().load(...)
print("saved weights/my_lenet.json")
""")
    nb.md(r"""
---
## thats it, you built a neural network from scratch

everything that just ran is your code (or code you read) plus the python standard library: layouts and gathers, one matrix multiply, a topological sort, the chain rule, sgd with cosine annealing, and a data loader. thats the same skeleton pytorch, cudagrad and infanttorch have. they just run it on gpus.

### talk about it: how would we make it faster?
- **where does the time go?** each training step is ~0.65 million multiply-adds forward and ~1.3 million backward, plus updating all 62,006 weights *after every single image*
- **batching**: average the gradient over N images and update once. one bigger matrix multiply instead of N small ones → [bonus A](addon_A_batching.ipynb)
- **split-k reduction**: a weight's gradient is a giant sum over every image and every position (the *K* dimension). cut K into slices, add up each slice on a different core (or gpu thread block), then add the partial sums together → [bonus B](addon_B_split_k.ipynb)
- **gpus**: thousands of cores. cudagrad trains a similar cnn on 124,800 images in ~6 seconds per epoch on an rtx 4080. cute layouts like the ones you wrote decide which gpu thread reads which number

more reading: [cudagrad](https://github.com/notYnoj/cudagrad) (cuda autograd with split-k kernels) · [infanttorch](https://github.com/TorchLearners/infantTorch) · [wordhunt-ai](https://github.com/notYnoj/WordHunt-AI) (a lenet style cnn in plain c++) · [cute docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)
""")


def core(nb):
    intro(nb)
    lab0(nb)
    lab1(nb)
    lab2(nb)
    lab3(nb)
    lab4(nb)
    lab5(nb)
    lab6(nb)
