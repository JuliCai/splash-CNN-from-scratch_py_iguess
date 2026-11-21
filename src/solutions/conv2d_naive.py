# answer for the todo 'conv2d_naive' - run this cell to define it
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
