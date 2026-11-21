# Build a Neural Network From Scratch

Lab materials for the Splash class *Build a Neural Network From Scratch*. Students write **LeNet‑5 in pure
Python** (no numpy, no PyTorch, no AI libraries), train it on 35,000 synthetic memes, and test it on 150 real
memes it has never seen, including people doing the 6‑7 hands.

The lab follows the class description: the architecture and its layers, then how tensors are stored
(CuTe‑style layouts), layers built from tensors and operators, optimizers, cosine‑annealing learning rates,
data loading, and forward and back propagation. Two bonus notebooks cover the closing discussion,
**batching** and **split‑K reduction**.

## What's in here

| folder | what it is |
|---|---|
| [`src/`](src) | **The lab.** `lab.ipynb` for students (todo cells with step-by-step hints, python docs and automatic checks), `lab_solutions.ipynb` for the teacher, two bonus notebooks, `babytorch.py` (the whole network in one readable file), `%load`-able solutions, pretrained weights, and an interactive web overview in `src/web/`. The teacher guide is [`src/README.md`](src/README.md). |
| [`datagen/`](datagen) | How the dataset was made: polite scrapers, hand curation, and a computer‑vision pipeline that turns 457 real memes into 35,000 synthetic training images. |
| [`data/`](data) | The dataset card, [`data/README.md`](data/README.md). The data files are not in this repo (see below). |

## Get the data

The meme images belong to the people who made them, and some show real people, including a minor, so
the dataset is not published here. Get `meme10-data.zip` from your teacher and unzip it in this folder.
Afterwards `data/` should contain `train.bin`, `val.bin`, `test.bin`, `test.csv`, `classes.txt` and `test_full/`.

## Run the lab

You need Python 3.12 or newer (older versions work, just slower) and Jupyter:

```bash
python3 -m pip install notebook
cd src
jupyter notebook lab.ipynb
```

Run cells with Shift + Enter. Write your code in the cells marked `# todo:` and run the check under each one. Stuck?
Put `%load solutions/NAME.py` in the cell and run it twice.

To see the interactive overview, run `python3 -m http.server 9997 --directory src/web` and open
http://localhost:9997.

## Train the model

With the dataset unzipped into `data/` (get `meme10-data.zip` from your teacher), train from `src/`:

```bash
cd src
python3 train.py --per-class 350 --epochs 4
```

That trains one image at a time with plain SGD and cosine annealing, prints validation accuracy each epoch,
then evaluates on the 150 real test memes and saves the weights to `src/weights/my_lenet.json`.
Use `--per-class 100` for a ~2 minute run, `--per-class 0` for all 35,000 images (~46 min),
`--load weights/lenet_meme10.json --epochs 0` to just evaluate the pretrained model,
or `MEME10_DATA=/path/to/data` if the dataset lives elsewhere.

## Results

Measured with the lab's own code (one image per step, plain SGD, cosine learning rate) on an Apple M5 Pro:

| training run | time | accuracy on the 150 real memes |
|---|---|---|
| random guessing | | 10% |
| 100 images per meme × 5 epochs | about 2 min | 77% |
| 350 images per meme × 4 epochs (the in‑class run) | 4–6 min | 83% |
| all 35,000 images × 4 epochs (`src/weights/lenet_meme10.json`) | 46 min | 88.7% |

## Credits and further reading

- [cudagrad](https://github.com/notYnoj/cudagrad) and [infantTorch](https://github.com/TorchLearners/infantTorch): the CUDA autograd engines whose API the lab copies
- [WordHunt‑AI](https://github.com/notYnoj/WordHunt-AI): a LeNet‑style CNN in pure C++
- [CuTe layouts](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md) (NVIDIA CUTLASS)
- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, *Gradient‑based learning applied to document recognition*, 1998

## For maintainers

Notebook outputs that contain pictures are stripped automatically when you commit (see `.gitattributes`), so
the memes in `lab_solutions.ipynb` never reach GitHub while your local copy keeps them. In a fresh clone, turn
the filter on before committing:

```bash
git config filter.strip-images.clean "python3 src/tools/strip_image_outputs.py"
git config filter.strip-images.smudge cat
git config filter.strip-images.required true
```
