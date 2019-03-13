import numpy
import pyopencl as cl

numpy.array([
    0, 1, 3, 1, 3, 2, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
])


def autocorr(seq, p):
    return