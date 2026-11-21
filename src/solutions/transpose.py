# answer for the todo 'transpose' - run this cell to define it
def transpose(layout):
    """swaps the two modes of a rank-2 layout. no data moves, only the layout changes"""
    (s0, s1), (d0, d1) = layout.shape, layout.stride
    return Layout((s1, s0), (d1, d0))
