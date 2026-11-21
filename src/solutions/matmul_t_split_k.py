# answer for the todo 'matmul_t_split_k' - run this cell to define it
def matmul_t_split_k(A, B, splits):
    """C = A @ B^T like matmul_t, but the K dimension (the sum) gets cut into `splits` slices.
    every slice gives a partial C (they could all run at the same time), then the partials get added up"""
    K = len(A[0])
    chunk = -(-K // splits)                       # K / splits rounded up
    partials = []
    for s in range(splits):
        lo, hi = s * chunk, min(K, (s + 1) * chunk)
        partials.append(matmul_t([a[lo:hi] for a in A], [b[lo:hi] for b in B]))
    return tree_reduce(partials)
