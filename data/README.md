# Meme-10 — a LeNet-sized meme dataset

10 meme classes, **32×32 RGB** images, built for training a LeNet-style network from scratch.
Training data is **synthetic** (made by a classical computer-vision pipeline from real meme images);
test data is **real images scraped from the web** that the pipeline never saw.

| split   | images | per class | what it is | files |
|---------|-------:|----------:|------------|-------|
| `train` | 35,000 | 3,500 | synthetic, generated from 457 real "seed" memes | `train/`, `train.bin`, `train.csv` |
| `val`   |  3,500 |   350 | synthetic, same seeds, different random recipes (for tuning / early stopping) | `val/`, `val.bin`, `val.csv` |
| `test`  |    150 |    15 | **real** memes from the web, hand-checked, never used for synthesis | `test/` (32×32), `test_full/` (originals), `test.bin`, `test.csv` |

Sample sheets of the training images are in `preview/` (included with the data download, not on GitHub).

## Classes

Label = line number (0-based) in `classes.txt`.

| # | class | what counts | real seeds | real test |
|--:|-------|-------------|-----------:|----------:|
| 0 | `six_seven` | someone doing the 6-7 hands, the "67 kid", 6-7 edits | 69 | 15 |
| 1 | `tung_tung_tung_sahur` | Italian-brainrot wooden log with a bat | 33 | 15 |
| 2 | `tralalero_tralala` | Italian-brainrot shark in Nike sneakers | 18 | 15 |
| 3 | `skibidi_toilet` | head singing out of a toilet | 24 | 15 |
| 4 | `chill_guy` | "my new character" dog in a grey sweater | 49 | 15 |
| 5 | `doge` | Kabosu the Shiba Inu and direct edits | 63 | 15 |
| 6 | `trollface` | rage-comic Trollface (incl. dark "trollge" edits) | 52 | 15 |
| 7 | `surprised_pikachu` | Pikachu with its mouth open | 61 | 15 |
| 8 | `drake_hotline_bling` | Drakeposting two-panel template | 35 | 15 |
| 9 | `distracted_boyfriend` | the distracted-boyfriend stock photo | 53 | 15 |

The 6-7 test images include frames from two video clips whose frames were kept *entirely* out of the
seed set, so the model has to recognise the gesture on people it has never seen.

## Folder layout

```
data/
├── README.md            this file
├── classes.txt          the 10 class names (line i = label i)
├── train.bin  train.csv 35,000 synthetic images, CIFAR-10 binary format + index
├── val.bin    val.csv    3,500 synthetic images
├── test.bin   test.csv     150 real images (centre-cropped, then resized to 32×32)
├── train/<class>/*.png  the same images as PNG files
├── val/<class>/*.png
├── test/<class>/*.png   32×32 real test images (what the network sees)
├── test_full/<class>/   the real test images at original resolution (for inference demos)
├── raw/
│   ├── seeds/<class>/   the 457 real images the synthesizer was allowed to use
│   ├── backgrounds/     300 random photos used as backdrops (Unsplash via picsum.photos)
│   ├── sources.csv      provenance of every real image: URL, page, licence, author
│   └── backgrounds.csv
└── preview/             sample grids + a picture of the pipeline
```

## The `.bin` format (CIFAR-10 layout)

Each image is one fixed-size record of **3,073 bytes**:

| byte offset | contents |
|-------------|----------|
| 0 | label (0–9) |
| 1 … 1024 | red channel, 32×32, row by row |
| 1025 … 2048 | green channel |
| 2049 … 3072 | blue channel |

So pixel (row `r`, column `c`) of channel `ch` (0 = R, 1 = G, 2 = B) is at byte `1 + ch*1024 + r*32 + c`.
`train.bin` is 35,000 × 3,073 = 107,555,000 bytes. Records are **shuffled**, so the classes are mixed;
row *i* of `train.csv` describes record *i* (PNG path, label, which seed image and layout made it).
`test.bin` is in class order (not shuffled).

Reading it needs nothing but plain Python:

```python
def load_bin(path, size=32):
    """Returns (images, labels). images[i] is a bytes object of 3*size*size values in 0..255."""
    record = 1 + 3 * size * size
    data = open(path, "rb").read()
    images, labels = [], []
    for start in range(0, len(data), record):
        labels.append(data[start])
        images.append(data[start + 1 : start + record])
    return images, labels

images, labels = load_bin("data/train.bin")
r, g, b = (images[0][ch * 1024 + 5 * 32 + 7] / 255 for ch in range(3))   # pixel at row 5, col 7
```

## What accuracy to expect

Sanity check only (not part of the class code): a standard LeNet-5 in PyTorch (conv 5×5→6, max-pool,
conv 5×5→16, max-pool, FC 120 → 84 → 10, ReLU; inputs normalised with the training set's per-channel
mean/std; SGD momentum 0.9, lr 0.02, weight decay 5e-4, batch 64, cosine annealing, no extra augmentation).
Trained **only on synthetic images**, scored on the **real** test images (chance = 10%):

| training data | epochs | synthetic val | **real test** |
|---------------|-------:|--------------:|--------------:|
| all 35,000 | 1 | 73–76% | 83–86% |
| all 35,000 | 15 | 88–89% | **86–89%** (3 runs, mean 87.5%) |
| 10,000 (1,000/class) | 15 | 82–83% | 85–86% |
| 3,500 (350/class) | 15 | 75–76% | 84–85% |
| 1,000 (100/class) | 15 | 58–62% | 75–77% |

Per class (mean of the 3 full runs): `doge`, `surprised_pikachu`, `distracted_boyfriend` 100%,
`drake_hotline_bling` 98%, `skibidi_toilet` 93%, `chill_guy` 82%, `trollface` 82%,
`tung_tung_tung_sahur` 78%, `six_seven` 71%, `tralalero_tralala` 71%.

Things worth pointing out in class:

- **Small test set.** 15 images per class means one image is 6.7% of that class's score. Use `val` for
  tuning and early stopping; touch `test` only to report results.
- **Sim-to-real gap.** The synthetic `val` set is deliberately *harder* than the real test set (heavy
  blur, JPEG damage, occluders), so real-test accuracy can come out about the same as `val`, or even higher.
- **Overfitting.** Training accuracy reaches ~99% while val/test plateau after ~5–10 epochs; this is a good
  setting for learning-rate schedules (cosine annealing) and early stopping.
- **Colour shortcuts.** At 32×32 a LeNet leans on colour: typical misses are a person in a blue shirt
  called `tralalero_tralala`, a yellow-on-green Trollface called `doge`, and plush or toy versions of the
  characters. `six_seven` is the hardest class because it is about a *gesture*, not a fixed picture.
- **Speed.** LeNet-5 needs about 0.65 M multiply-adds per image forward (~2 M with backprop). The pure-Python
  reference implementation in `../src/babytorch.py` (the lab's answer key) trains at about 40–55 images/s on an
  Apple M5 Pro, i.e. 11–15 min per epoch over all 35k images; older laptops are likely 2–3× slower (estimate).
  Start on a subset, then use it to motivate batching and split-K.

Pure-Python results with the lab's own code (one image per step, plain SGD, cosine LR; see `../src/README.md`):
100/class × 5 epochs → 77% on the real test set; 350/class × 4 epochs → 83%; all 35k × 4 epochs (46 min) → 88.7%.

## How the data was made

`preview/pipeline_steps.png` (in the data download) shows how one synthetic image is built, stage by stage.

All the code is in `../datagen/` (numpy + Pillow only; no ML libraries):

1. **Scrape candidates.** `scrape_imgflip.py` collects meme templates and real user-made memes from
   imgflip.com (its robots.txt only blocks `/orig/` and `/browse/`; at most about one request per second).
   `scrape_commons.py` adds freely licensed images from Wikimedia Commons, including still frames of
   people doing the 6-7 gesture taken from CC0 / CC-BY(-SA) video clips (via standard-size thumbnails).
2. **Curate by hand.** `review.py` makes numbered contact sheets. Every image was checked by eye: off-topic
   and cross-class images were dropped, and captions were screened for classroom suitability (slurs,
   sexual references, political jokes, violence and profanity were removed). The decisions are recorded
   in `datagen/picks.json`.
3. **Promote.** `promote.py` copies the picks into `raw/seeds/` and `test_full/`, writes `raw/sources.csv`,
   and refuses any test image that is the same picture as a seed (a repost or re-encode). Different captions
   on the same template are allowed, because that is how memes work.
4. **Backgrounds.** `fetch_backgrounds.py` downloads 300 random photos for backdrops.
5. **Synthesize.** `synthesize.py` builds every training image with its own random recipe:
   *random crop/flip* → one of 6 layouts (*fill*, *letterbox*, *caption bar*, *social-media screenshot /
   photo of a screen*, *two-panel collage*, *auto-masked cutout on a new background*) → class-agnostic meme
   text, emoji, watermarks, video-app UI and occluders → tilt/perspective → colour jitter →
   blur / motion blur / noise / pixelation / JPEG / "deep-fried" → antialiased resize to 32×32.
   All text is random and shared by every class, so a network can't cheat by reading captions.
6. **Test prep.** `prepare_test.py` centre-crops each real test image and resizes it with the exact same
   function used for training.
7. **Preview.** `preview.py` draws the sheets in `preview/`.

## Regenerate or customise

```bash
python3 datagen/synthesize.py                     # rebuild train/ + val/ (about 15 s, reproducible: --seed 67)
python3 datagen/prepare_test.py                   # rebuild test/ + test.bin from test_full/
python3 datagen/preview.py
```

- Different size, e.g. MNIST-like 28×28: add `--size 28` to both scripts (records become 1 + 3·28·28 bytes).
- More or less data: `--train-per-class 1000 --val-per-class 100`.
- Test preprocessing: `prepare_test.py --mode squash` or `--mode pad` instead of the centre crop.
- New classes: edit `datagen/config.py`, re-run the scrapers, curate, then `promote.py` → `synthesize.py`.

Requires Python 3.9+ with `numpy` and `Pillow ≥ 10.1` (`datagen/requirements.txt`). The lab in `../src/` that
trains on this data needs only the Python standard library.

## Sources, licences and content notes

- `raw/sources.csv` lists every real image with its source URL, page, licence and author.
- **imgflip images** (most of the seeds and test images) are user-made memes; their rights belong to their
  creators and the owners of the original pictures. They are included for non-commercial classroom use.
  **Don't republish this folder publicly.**
- **Wikimedia Commons** images are CC0, public domain, CC BY 4.0 or CC BY-SA 4.0. If you show them, credit
  the authors listed in `sources.csv`.
- **Backgrounds** are Unsplash photos served by picsum.photos (Unsplash License); see `raw/backgrounds.csv`.
- Some memes show real people, including the "67 kid", a minor who went viral, and public figures. The
  dataset is for recognising meme *formats* and gestures; don't use it for anything about identity.
  Captions were screened, but skim `test_full/` before showing it on a projector.
