# answer for the todo 'broadcast' - run this cell to define it
def broadcast(n_rows, n_cols):
    """reads a list of n_cols numbers as an n_rows x n_cols matrix where every row is that same list.
    a stride of 0 means stepping that way doesnt move in memory"""
    return Layout((n_rows, n_cols), (0, 1))
