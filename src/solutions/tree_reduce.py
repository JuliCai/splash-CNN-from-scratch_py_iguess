# answer for the todo 'tree_reduce' - run this cell to define it
def tree_reduce(partials):
    """adds up a list of same-length lists in pairs, the way a gpu thread block does it in shared memory:
        [p0, p1, p2, p3]  ->  [p0+p1, p2+p3]  ->  [p0+p1+p2+p3]       (log2(n) rounds)
    every add inside one round is independent, so a gpu does a whole round at the same time"""
    while len(partials) > 1:
        paired = [[a + b for a, b in zip(partials[i], partials[i + 1])] for i in range(0, len(partials) - 1, 2)]
        if len(partials) % 2:
            paired.append(partials[-1])   # an odd one out waits for the next round
        partials = paired
    return partials[0]
