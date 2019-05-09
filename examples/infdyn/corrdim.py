import numpy
import matplotlib.pyplot as plt
from scipy.stats import linregress

from multiprocessing.pool import ThreadPool
pool = ThreadPool()

def generate_series(N):
    return numpy.random.random(size=N).astype(dtype=numpy.float64)


def dist(v1, v2):
    return abs(v1 - v2)


def count_buckets(series, num_buckets):
    sorted_series = numpy.sort(series)
    min_ = min(series)
    max_ = max(series)

    half_a_span = (max_ - min_) / 2
    min_size = half_a_span / num_buckets
    bucket_sizes = numpy.linspace(min_size, half_a_span, num_buckets)

    def _bucket(bucket_size):
        bkt_ = numpy.arange(min_, max_, bucket_size)
        buckets = {
            (left, right): 0 for left, right in zip(bkt_, bkt_[1:])
        }
        for el in series:
            for b in buckets.keys():
                if b[0] <= el <= b[1]:
                    buckets[b] = 1
                    break
        return sum(buckets.items())

    counts_for_sizes = pool.map(_bucket, bucket_sizes)
    





s = generate_series(100)



def corr_dim(series, r):
    N = len(series)
    cnt = 0
    for i, v1 in enumerate(series):
        for j, v2 in enumerate(series):
            if i == j: continue
            if dist(v1, v2) < r: cnt += 1
    return cnt / (N * (N - 1))


def corr_dim_plot(series, r_values):
    x = r_values
    y = numpy.array([corr_dim(series, x_) for x_ in x])
    D, ipt, _, _, _ = linregress(x, y)
    sl = lambda x: D*x + ipt
    plt.loglog(x, y)
    plt.plot(x, sl(x))
    plt.title("D = {:.4f}".format(D))
    plt.show()


# s = generate_series(100)
# r_values = numpy.linspace(1e-3, 1e-1)
# corr_dim_plot(s, r_values)

