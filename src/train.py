#!/usr/bin/env python3
"""Train LeNet-5 on the Meme-10 dataset in plain Python.

Run from this folder:
    python3 train.py --per-class 350 --epochs 4

The dataset is not in this repo. Put train.bin, val.bin, test.bin, test.csv
and classes.txt in ../data, or set MEME10_DATA=/path/to/data.
"""

import argparse
import os
from pathlib import Path

try:
    import babytorch
except FileNotFoundError as exc:
    raise SystemExit(f"{exc}\nSet MEME10_DATA=/path/to/data or put the data folder next to src/.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="train LeNet-5 on Meme-10 (no numpy, no PyTorch)")
    ap.add_argument("--per-class", type=int, default=350,
                    help="training images per class (0 = all 3,500 per class)")
    ap.add_argument("--epochs", type=int, default=4, help="epochs over the training set")
    ap.add_argument("--lr", type=float, default=0.005, help="lr_max of the cosine schedule")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--save", default="weights/my_lenet.json",
                    help="where to write the trained weights")
    ap.add_argument("--load", help="start from these weights (or just evaluate with --epochs 0)")
    ap.add_argument("--val-limit", type=int, default=500,
                    help="validation images checked after each epoch")
    args = ap.parse_args(argv)

    data = Path(os.environ.get("MEME10_DATA") or babytorch.find_data())
    needed = ["classes.txt", "train.bin", "val.bin", "test.bin"]
    missing = [name for name in needed if not (data / name).exists()]
    if missing:
        ap.error("missing dataset files in " + str(data) + ": " + ", ".join(missing))

    print("dataset:", data)
    print("classes:", ", ".join(babytorch.CLASSES))

    forward = [
        "--per-class", str(args.per_class),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--save", args.save,
        "--val-limit", str(args.val_limit),
    ]
    if args.load:
        forward += ["--load", args.load]
    babytorch.main(forward)


if __name__ == "__main__":
    main()
