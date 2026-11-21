# answer for the todo 'dataloader' - run this cell to define it
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
