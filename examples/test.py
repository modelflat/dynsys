from collections import defaultdict


def compute_distr(phases):
    d = defaultdict(lambda: 0)
    c = 0
    for e in phases:
        if e == 1 and c != 0:
            d[c] += 1
            c = 0
        elif e == 0:
            c += 1
    return d


d = compute_distr([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1])
print(d)