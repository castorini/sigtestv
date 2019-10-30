import numpy as np


def compute_pr_x_ge_y(x, y):
    x = [(x_, 0) for x_ in x]
    y = [(y_, 1) for y_ in y]
    z = sorted(x + y, key=lambda e: e[0])
    counter = 0
    results = []
    for idx, (elem, lab) in enumerate(z):
        counter += lab
        results.append(counter * (1 - lab))
    return np.sum(results) / (len(x) * len(y))
