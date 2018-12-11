import numpy
from numpy import linalg
from matplotlib import pyplot as pp

from mpl_toolkits.mplot3d import Axes3D
assert Axes3D


def vec(*it, dtype=numpy.float):
    return numpy.array(it, dtype=dtype)


sig = 10
b   = 8.0 / 3.0
r   = 28
# x0 = vec(-3.16, -5.31, 13.31)
x0 = vec(1, 2, 10)


def lorenz(x, t):
    x, y, z = x
    return vec(
        sig * (y - x),
        r*x - y - x*z,
        -b*z + y*x
    )


def rk4(F, v, t, h):
    k1 = h * F(v, t)
    k2 = h * F(v + k1 / 2, t + h / 2)
    k3 = h * F(v + k2 / 2, t + h / 2)
    k4 = h * F(v + k3, t + h)
    return (k1 + 2*k2 + 2*k3 + k4) / 6.0


def stepSystem(F, t: float, xs: numpy.ndarray, h: float):
    xs = xs.copy()
    for i in range(xs.shape[0]):
        xs[i] += rk4(F, xs[i], t, h)
    return xs


def orthonormalize(v, p):
    w = numpy.empty(shape=(p, p),  dtype=numpy.float)
    u = numpy.empty(shape=w.shape, dtype=numpy.float)
    for k in range(p):
        u[k] = v[k] - sum(numpy.dot(v[k], w[j]) * w[j] for j in range(k))
        w[k] = u[k] / linalg.norm(u[k])
    return linalg.norm(u, axis=0), w


def renormalize(point, neighbourPoints, eps):
    pass  # do the right thing here
    return vec(1, 1, 1), neighbourPoints


def LCE(F, x0: numpy.array, eps: float, tau: float, T: float, stepsInTau: int):
    xs = numpy.empty((4, 3), dtype=numpy.float)
    xs[0] = x0
    xs[1] = x0 + vec(eps, 0, 0)
    xs[2] = x0 + vec(0, eps, 0)
    xs[3] = x0 + vec(0, 0, eps)
    S = numpy.zeros((3,), dtype=numpy.float)
    step = tau / stepsInTau
    xsHist = numpy.empty((int(T / tau), 3), dtype=numpy.float)
    xsEpsHist = numpy.empty((int(T / tau), 3), dtype=numpy.float)
    k = 0
    for t in numpy.arange(0, T, tau):
        for t_ in numpy.arange(t, t + tau, step):
            xs = stepSystem(F, t_, xs, tau)
        xsHist[k], xsEpsHist[k] = xs[0], xs[1]
        norms, xs[1:] = renormalize(xs[0], xs[1:], eps)
        S += numpy.log(norms)
        k += 1

    fig = pp.figure()
    ax = fig.gca(projection="3d")
    ax.plot(*xsHist.T, "r-")
    ax.plot(*xsEpsHist.T, "b-")

    return S


print(
    LCE(lorenz, x0=x0, eps=1e-4, tau=2e-3, T=5, stepsInTau=10)
)

pp.show()
