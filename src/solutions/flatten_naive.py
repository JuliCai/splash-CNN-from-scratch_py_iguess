# answer for the todo 'flatten_naive' - run this cell to define it
def flatten_naive(fmaps):
    """C x H x W nested lists -> one flat list (channel by channel, row by row)"""
    return [v for fmap in fmaps for row in fmap for v in row]
