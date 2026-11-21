# answer for the todo 'cosine_annealing_lr' - run this cell to define it
def cosine_annealing_lr(t, T, lr_max, lr_min):
    """cosine annealing: start at lr_max, slide down half a cosine wave, land on lr_min at step T"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t / T, 1.0)))
