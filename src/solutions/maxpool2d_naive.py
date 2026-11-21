# answer for the todo 'maxpool2d_naive' - run this cell to define it
def maxpool2d_naive(fmaps, p=2):
    """keeps the biggest value in every p x p window (stride p) of every feature map"""
    out = []
    for fmap in fmaps:
        H, W = len(fmap), len(fmap[0])
        out.append([[max(fmap[y * p + i][x * p + j] for i in range(p) for j in range(p))
                     for x in range(W // p)]
                    for y in range(H // p)])
    return out
