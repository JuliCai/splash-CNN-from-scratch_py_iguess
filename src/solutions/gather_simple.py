# answer for the todo 'gather_simple' - run this cell to define it
def gather(storage, layout):
    """reads memory thru a layout: one value for every coordinate, in row-major order"""
    return [storage[o] for o in offsets(layout)]
