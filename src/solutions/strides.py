# answer for the todo 'strides' - run this cell to define it
def row_major(shape):
    """compact row-major strides (cute calls it LayoutRight). the last mode moves fastest. (3, 32, 32) -> (1024, 32, 1)"""
    strides, step = [], 1
    for s in reversed(shape):
        strides.append(step)
        step *= s
    return tuple(reversed(strides))


def col_major(shape):
    """compact column-major strides (cute calls it LayoutLeft). the first mode moves fastest. (3, 32, 32) -> (1, 3, 96)"""
    strides, step = [], 1
    for s in shape:
        strides.append(step)
        step *= s
    return tuple(strides)
