# Build a Neural Network From Scratch: the lab

An interactive, lab-based lesson in which students write **LeNet‑5 from scratch in pure Python**: no numpy, no PyTorch,
only the standard library. They then train it on the Meme‑10 dataset in `../data/` and test it on 150 real memes.

The core lab trains **one image at a time** and uses no tricks. **Batching** and **split‑K reduction** are two
separate add‑on labs: the class description's closing "discussion on optimization techniques".

**Web overview for students:** `web/index.html` (interactive demos of the lab's ideas). To host it, run
`python3 -m http.server 9997 --directory src/web` from the class folder and open http://localhost:9997.

## What's in this folder

| file | what it is |
|---|---|
| `lab.ipynb` | **the student notebook**: Labs 0–6 with todo cells (step-by-step hints and python docs in the comments), automatic checks, a formula sheet and a python cheat sheet |
| `lab_solutions.ipynb` | the same notebook with every TODO filled in and all outputs shown: the teacher copy to project |
| `addon_A_batching.ipynb` (+ `_solutions`) | add‑on lab: batching (the batch as a layout mode, stride‑0 broadcasting) |
| `addon_B_split_k.ipynb` (+ `_solutions`) | add‑on lab: split‑K reduction (tree reduction, K‑slicing, multi‑core training) |
| `babytorch.py` | the complete reference implementation in one readable file (also a command‑line trainer) |
| `addons.py` | reference code for the two add‑ons |
| `checks.py` | the friendly autograder the notebooks call after every TODO |
| `viz.py` | pictures in pure Python: a PNG encoder plus HTML/SVG charts (not part of the network) |
| `solutions/*.py` | one file per TODO, for `%load solutions/NAME.py` when a student is stuck |
| `weights/lenet_meme10.json` | LeNet‑5 trained by `babytorch.py` on all 35,000 images, for the demo in Lab 6 |
| `tools/build_notebooks.py`, `tools/lesson_content.py` | regenerate all notebooks and `solutions/` (see "Editing the lesson") |

## Setup (students)

1. **Python 3.12 or newer** (for `math.sumprod`, a C-speed dot product). Older versions work too, just ~2–5× slower.
2. A notebook editor, any of:
   * `python3 -m pip install notebook`, then `jupyter notebook` in this folder;
   * VS Code with the Python + Jupyter extensions (open `lab.ipynb`, pick a Python 3.12+ kernel);
   * Google Colab: upload `src/` and `data/` keeping their side-by-side layout (`train.bin`, `val.bin`, `test.bin`,
     `test.csv`, `classes.txt` and `test_full/` are all `data/` needs), or set the `MEME10_DATA` environment variable
     to the data folder.
3. Open the notebook **from this folder** so `import babytorch, checks, viz` work and `../data` is found.

Nothing gets installed for the lab itself: no numpy, no PyTorch, no matplotlib, no Pillow.

## How students use the notebook

* **`# todo:` cells**: the comments walk through the logic and list the python each one needs. Replace the
  `raise NotImplementedError(...)` line, run the cell, then run the check below it. Checks print `pass`, `fail` with
  the first mismatch and a hint, or `todo` (not written yet).
* **Stuck:** `%load solutions/NAME.py` in the cell, run it twice. The first run pastes the answer, the second defines it.
  If an editor doesn't support `%load`, `%run -i solutions/NAME.py` defines it without pasting.
* **Behind, or skipping a lab:** the ⏩ catch-up cell near the top (`from babytorch import *`) loads every reference
  answer. Afterwards, don't re-run unfinished todo cells, or they'll replace the working answers.
* Labs 3–6 use everything written before. That's the point: the network that trains in Lab 6 is the students' own code.

## How the lab follows the class description

| class description | where |
|---|---|
| "the overarching architecture, then break layers down: convolution, MaxPool, fully connected" | Lab 0 (architecture, shapes, parameter counts) and Lab 1 (each layer with plain loops) |
| "Tensors and seeing how data is stored" | Lab 2: flat storage + CuTe‑style layouts (`shape:stride`), views, `print_layout`, im2col as a layout |
| "how to use tensors and operators to create these layers" | Lab 3: every layer = gather through a layout + one operator (matmul / max) |
| "optimizers, learning rate schedulers (cosine annealing), data loading" | Lab 4: cross‑entropy, SGD (+ momentum), cosine annealing, a streaming `DataLoader` |
| "forward propagation and back propagation" | Lab 5: an autograd engine like cudagrad's `Node` + every backward kernel + gradient checks |
| "run our model and see how well it does at identifying memes (six seven hands)" | Lab 6: train, confusion matrix, predictions on the real test memes, a pretrained model |
| "discussion on optimization techniques including batching and split k reduction" | end of Lab 6 + Add‑on A (batching) + Add‑on B (split‑K) |

## Suggested schedules

Times are for the in-class parts; everything else makes good homework.

| length | plan |
|---|---|
| **~90 min** | Lab 0 (10) → Lab 1 conv + demo only (15) → Lab 2 through im2col (20) → Lab 3 demo cells only (5) → Lab 4 skim + cosine (10) → Lab 5 `backward` + `conv2d_backward` (15) → Lab 6 start `PER_CLASS = 100` training, discuss batching/split‑K while it runs (15). Students use `%load` / catch‑up freely. |
| **~2.5 h** | all of Labs 0–6 with `PER_CLASS = 350` training (start it, then do the discussion during the run) |
| **half day** | Labs 0–6, then Add‑on A and Add‑on B |

Lab 1's naive layers and Lab 2's layouts take the longest the first time; the Lab 5 backward kernels are the
hardest. Every TODO has a hint and a `%load` escape hatch, so nobody gets stuck for good.

## Speed and accuracy (measured)

All numbers below are from the reference code on an **Apple M5 Pro** (Python 3.13). Older laptops are typically
2–3× slower; that factor is an estimate, not a measurement.

| what | measured |
|---|---|
| LeNet‑5 forward pass, Lab 1 nested loops | ~41 ms per image |
| LeNet‑5 forward pass, Lab 3 layouts + matmul | ~4.6 ms per image (≈9× faster) |
| one training step (forward + backward + SGD) | ~19 ms (≈53 images/s); ~24 ms with other work running |

| training run (one image at a time, plain SGD, cosine LR) | wall time (M5 Pro) | real test accuracy |
|---|---|---|
| `PER_CLASS = 100`, 5 epochs, `lr = 0.01` | ~2 min* | 77% |
| `PER_CLASS = 350`, 4 epochs, `lr = 0.005` (the notebook default) | 4–6 min (247 s and 345 s in two Jupyter runs) | 83% |
| all 35,000 images, 4 epochs, `lr = 0.005` (`weights/lenet_meme10.json`) | 46 min (2,768 s) | 88.7% (val 85.0%) |

\* computed from the measured step time. The accuracies are from real runs of the same code.
The 150‑image test set is small (one image = 0.7% of the score), so expect a few points of run‑to‑run noise.

Command‑line training, outside the notebook:

```bash
python3 babytorch.py --per-class 350 --epochs 4                 # the in-class run
python3 babytorch.py --epochs 4 --save weights/mine.json        # everything (46 min here)
python3 babytorch.py --load weights/lenet_meme10.json           # evaluate saved weights
```

On macOS, don't start long runs as `cmd &` from zsh: zsh's `BG_NICE` option drops background jobs to low
priority, and they ran about 7× slower in testing. Use a separate terminal tab instead.

## The add-ons

**Add‑on A, batching.** The batch is one more layout mode (`batched_im2col_layout`); the bias is broadcast with a
stride‑0 layout; the batched loss is the mean. The notebook checks that batched logits equal the per‑image ones and
that the batched gradient equals the average of per‑image gradients (both to ~1e‑16). Honest result: in pure
Python, batching saves little time (measured: 1.2–1.7× per image at batch 16 across runs; laptop timings move around
with load), because every multiply still costs the same. The notebook
uses that to explain why GPUs need batches. It also compares accuracy: same epochs, batch 16 with `lr=0.04` vs one
image at a time with `lr=0.005` (measured on 2,000 images × 2 epochs: 70.7% vs 76.0% on the real test set, because fewer, larger
steps learn less per epoch).

**Add‑on B, split‑K.** It starts from cudagrad's own Nsight profile: `conv2d_dW_kernel` and `conv_bias_grad_kernel`
take 95% of GPU time, with 1–5 blocks on a 76‑SM GPU. Students then write `tree_reduce` (the shared‑memory
reduction) and `matmul_t_split_k`, and slice K with CuTe's `zipped_divide`. Then they split real gradient computation
across CPU cores with `multiprocessing`, where each worker is a "thread block" summing its slice of the batch.
Measured here on a batch of 64: 2.0× with 2 workers, 3.7× with 4, 4.2× with 8 (and slower again with 12 in a separate
test); training with 4 workers ran at 90–166 images/s across runs, versus ~45–60 on one core. That's Amdahl's law, and
there's a discussion prompt for it. The floating‑point reordering (gradients differ in the 16th digit) explains why
GPU runs with `atomicAdd` aren't bit-for-bit reproducible.

## How it maps to cudagrad / infantTorch / WordHunt‑AI

| here (`babytorch.py`) | cudagrad / infantTorch |
|---|---|
| `Tensor(data, shape, children, op)` + `backward_fn` | `Node<T>` (`value`, `children`, `backward_fn`, `op`) |
| `Tensor.backward()` (topological sort, seed 1, reverse) | `Node::backward()` |
| `conv2d`, `maxpool2d`, `linear`, `relu`, `flatten`, `softmax_cross_entropy` | `conv` + `conv_bias`, `maxpool`, `matmul` + `bias_add`, `relu`, `flatten`, `softmax_cross_entropy` |
| `*_forward` / `*_backward` kernels on flat lists | `launch_*` CUDA kernels |
| `Conv`, `MaxPool`, `Flatten`, `Linear(relu=...)`, `CNN().add(...)` | the same names in `AutoGrad.cuh` / `cnn.hpp` |
| `SGD(..., schedule=cosine_annealing_lr)`, `constant_lr`, `linear_lr` | `SGD<T>` + `schedulers.hpp` (`cosine_annealing_LR`, …) |
| He init `random.gauss(0, sqrt(2/fan_in))` | `Init::He` |
| `addons.SplitKTrainer`, `tree_reduce` | `conv2d_dW_kernel` (`blockIdx.y` chunks, shared‑memory reduction, `atomicAdd`) |

Differences on purpose: the schedule steps every *image* (cudagrad steps per epoch); conv + bias and matmul + bias are
fused into one op each; the core lab has no batch dimension.

**CuTe.** `Layout(shape, stride)`, `size`, `cosize`, `print_layout`, `make_layout`, `zipped_divide` and the
`shape:stride` printing follow CuTe. Two deliberate differences, both explained in Lab 2: default strides are
row‑major (CuTe's default is column‑major `LayoutLeft`), and a single integer index counts row‑major (CuTe counts
colexicographically, so a nested `print_layout` shows `0 1 2 3` where CuTe shows `0 2 1 3`). The real CuTe DSL
(`pip install nvidia-cutlass-dsl`) needs an NVIDIA GPU (Ampere or newer) on Linux, so it's only pointed to.

## Editing the lesson

* Reference code: `babytorch.py` / `addons.py`. Student exercises are the blocks between
  `# >>> exercise: NAME` and `# <<< exercise: NAME`; ready-made cells are `# >>> provided: NAME` blocks.
* Lesson text, demo cells and the hints inside each TODO stub: `tools/lesson_content.py`.
* Rebuild: `python3 tools/build_notebooks.py` (rewrites all six notebooks and `solutions/`). The `_solutions`
  notebooks come out without outputs; run them once (in Jupyter: *Run All*) to fill the outputs back in.

## Troubleshooting

| symptom | fix |
|---|---|
| `ImportError: cannot import name 'sumprod'` | Python < 3.12: the setup cell's fallback handles it; it's just slower |
| `FileNotFoundError: Can't find the data/ folder` | open the notebook from `src/`, or set `MEME10_DATA=/path/to/data` |
| training is too slow | `PER_CLASS = 100` (or the pretrained weights in 6.6). On a Mac, keep the notebook in front while it trains: in testing, a kernel left running in the background once trained ~4× slower |
| `NameError` after restarting the kernel | *Run All Above*, or run the catch-up cell |
| `%load` does nothing (some editors) | `%run -i solutions/NAME.py` |
| Add‑on B hangs or errors on Windows/macOS | run it from `src/` (worker processes import `addons.py` from there) |

## Content notes

Same as `../data/README.md`: the memes include real people, including the "67 kid", a minor. The default
six_seven demos (test images 2, 4, 5, 10 and 12) leave out the "67 kid" edits (test images 3, 6, 9, 11, 14), but
students can open any test image with `show(i)`. Imgflip images are for classroom use only; don't publish the
dataset. The notebooks show test images and their originals from `data/test_full/`; skim them before
projecting.
