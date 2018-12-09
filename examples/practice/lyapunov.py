
import numpy
from numpy import linalg
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

omega = 10
b = 8/3
r = 28

aHenon = 1.4
bHenon = -.3

def lorenz(x, t):
    return numpy.array((
        omega * (x[1] - x[0]),
        r * x[0] - x[1] - x[0] * x[2],
        -b * x[2] + x[1] * x[0]
    ), dtype=numpy.float)


def lorenzJ(x, w, t):
    return numpy.array((
        omega * (w[1] - w[0]),
        r * w[0] - w[1] - x[0]*w[2] - x[2]*w[0],
        -b * w[2] + x[0]*w[1] + w[0]*x[1]
    ), dtype=numpy.float)


# def henonMap(x, t):
#     return numpy.array((
#         1 - aHenon*x[0]**2 - bHenon*x[1],
#         x[0]
#     ), dtype=numpy.float)
#
#
# def henonMapJ(x, w, t):
#     return numpy.array((
#         -2*aHenon*x[0]*w[0] - bHenon*w[1],
#         w[0]
#     ), dtype=numpy.float)


def timerange(t0, t1, h):
    while t0 < t1:
        yield t0
        t0 += h


def mLCE(H, V, x0, w0, h, T0, Tm, X1m, step):
    x = x0
    w = w0
    t = T0
    N = int( (Tm - T0) / h )
    print(N, "steps")
    X1 =    0   #numpy.empty((N,), dtype=numpy.float)
    alpha = 0 #numpy.empty((N,), dtype=numpy.float)
    for k in range(N):
        if X1 < X1m and k > 0: break
        for t_ in timerange(t, t + h, h / 100):
            x_1 = x + step * H(x, t_)
            w_1 = w + step * (V(x, w, t_) * w)
            x, w = x_1, w_1
        alphaK = linalg.norm(w)
        w /= alphaK
        alpha += numpy.log(alphaK)
        X1 = 1 / t * alpha

        t += h
    return X1


def orthonormalize(v, p):
    # print('before or', v)
    w = numpy.empty(shape=(p, p), dtype=numpy.float)
    u = numpy.empty(shape=w.shape, dtype=numpy.float)
    for k in range(p):
        u[k] = v[k] - sum(numpy.dot(v[k], w[j]) * w[j] for j in range(k))
        w[k] = u[k] / linalg.norm(u[k])
    # print("after or", w)
    return linalg.norm(u, axis=0), w


def evolve(H, V, x, w, tr, step):
    print("before", w)
    for t in tr:
        x += step * H(x, t)
        for i in range(w.shape[0]):
            w[i] += step * (V(x, w[i], t) * w[i])
    print("after", w)
    return x, w


def LCE(H, V, p, x0, w0, T, Tm, iterPerH, useMap=False):
    assert p == w0.shape[0]
    step = T / iterPerH / 100
    # return
    x = x0
    M = int(Tm / T)
    ws = numpy.empty((M, *w0.shape), dtype=numpy.float)
    ws2 = numpy.empty((M, *w0.shape), dtype=numpy.float)
    xs = numpy.empty((M, *x.shape), dtype=numpy.float)
    xs[0] = x0
    ws[0] = w0
    ws2[0] = w0

    print(ws.shape)

    S = numpy.zeros((p,), dtype=numpy.float)
    for k in range(1, M):
        x, w = evolve(H, V, x, ws[k-1].copy(), timerange(T*(k - 1), T*k, T / iterPerH), step)
        ws2[k] = w.copy()
        g, w = orthonormalize(w.copy(), p)
        S += numpy.log(g)
        ws[k] = w.copy()
        xs[k] = x
    print(S)

    fig = pp.figure()
    ax = fig.gca(projection="3d")
    ax.plot(*xs.T, "r.")
    ax.quiver(*xs.T, *ws2.T, color="g")
    ax.quiver(*xs.T, *ws.T, color="y")

    return S / (M * T)


    # if useMap: # system is map
    #     for t_ in timerange(t, t + T, T / iterPerH):
    #         x_1 = H(x, t_)
    #         wv_k = tuple(V(x, wv_k[i], t_) * wv_k[i] for i in range(p))
    #         x = x_1
    # else: # contiguous system


# print(mLCE(f, J, numpy.array((0.0, 0, 0, 0)), numpy.array((0.0, 0, 0, 1)), .1, 0, 10, 1e-5, 1e-4))

# w0 = numpy.array((
#     (1, 0, 0),
#     (0, 1, 0),
#     (0, 0, 1)),
#     dtype=numpy.float)

w0 = numpy.array((
    (1, 3, 2),
    (3, -1, 0),
    (1, 3, -5)
), dtype=numpy.float) / 5

print(
    LCE(lorenz, lorenzJ,
        p=3,
        x0=numpy.array(
            (.5, -5, 28),
            dtype=numpy.float),
        w0=w0,
        T=1,
        Tm=150,
        iterPerH=100
        )
)

print("===================================")

# print(
#     evolve(lorenz, lorenzJ, numpy.array((.5, -5, 28), dtype=numpy.float),
#            w0, timerange(0, 1, 0.01), 0.001)[1]
# )

pp.show()

# print(
#     LCE(henonMap, henonMapJ,
#         p=2,
#         x0=numpy.array(
#             (0.001, 0.001),
#             dtype=numpy.float),
#         w0v=numpy.array((
#             (0.001, 0),
#             (0, 0.001)),
#             dtype=numpy.float),
#         T=1,
#         Tm=600,
#         iterPerH=1,
#         useMap=True
#         )
# )


