# answer for the todo 'matmul_t' - run this cell to define it
def matmul_t(A, B):
    """C = A times B-transposed for two lists of rows: C[i][j] = dot(A[i], B[j]).

    every row has the same length (the K dimension, the thing being added up). cutlass would say both
    sides are "K-major". gives back C as one flat row-major list of len(A) * len(B) numbers"""
    return [sumprod(a, b) for a in A for b in B]
