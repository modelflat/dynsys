import numpy as np


def window(a, step, size):
    print(a.shape[0])
    n = a.shape[0]
    # crude, but works, and not so bad from perf perspective
    a = np.concatenate((a, np.zeros((size,), dtype=np.int32)))
    return np.vstack(a[i: i+size] for i in range(0, n, step))


a = np.random.randint(1, 10, size=30)

print(window(a, 7, 10))
