# answer for the todo 'size_cosize' - run this cell to define it
def size(layout):
    """how many coordinates the layout has, the product of its shape"""
    return math.prod(flat(layout.shape))


def cosize(layout):
    """how much memory it reaches: 1 + the offset of its last coordinate (cute: A(size(A) - 1) + 1)"""
    return 1 + sum((s - 1) * d for s, d in zip(flat(layout.shape), flat(layout.stride)))
