import numpy
from matplotlib import pyplot as pp


def rk4(F, v, t, h):
    k1 = h * F(v, t)
    k2 = h * F(v + k1 / 2, t + h / 2)
    k3 = h * F(v + k2 / 2, t + h / 2)
    k4 = h * F(v + k3, t + h)
    return v + (k1 + 2*k2 + 2*k3 + k4) / 6.0


def LCE(F, x0, eps, tau, segments, T):

    for t in numpy.linspace(0, T, tau):
