# answer for the todo 'crd2idx' - run this cell to define it
def crd2idx(coord, stride):
    """the heart of a layout: offset = coord[0]*stride[0] + coord[1]*stride[1] + ... (both are flat tuples)"""
    return sum(c * d for c, d in zip(coord, stride))
