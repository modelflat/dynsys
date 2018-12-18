import numpy


def rk4(t0, t1, y0, F, steps=100):
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for i in range(steps):
        # k1
        k = h * F(t, y) / 2.0
        r = k / 3.0

        # k2
        k = h * F(t + h / 2.0, y + k) / 2.0
        r += k * (2.0 / 3.0)

        # k3
        k = h * F(t + h / 2.0, y + k)
        r += k / 3.0

        # k4
        pass

        y += r + h * F(t + h, y + k) / 6.0
        t += h
    return y


def Lorenz(t, X, sig=10, bet=8/3, r=28):
    x, y, z = X[0], X[1], X[2]
    res = numpy.empty(shape=(12,), dtype=numpy.float)

    res[0] = sig*(y - x)

    res[1] = -x*z + r*x - y

    res[2] = x*y - bet*z

    #        m
    # cij = sum aik*bkj
    #       k-1

    # 0 0
    res[3] = -sig * X[3] + sig * X[4]
    # 1 0
    res[4] = (r - z) * X[3] - X[4] - x * X[5]
    # 2 0
    res[5] = y * X[3] + x * X[4] - bet * X[5]
    # 0 1
    res[6] = -sig * X[6] + sig * X[7]
    # 1 1
    res[7] = (r - z) * X[6] - X[7] - x * X[8]
    # 2 1
    res[8] = y * X[6] + x * X[7] - bet * X[8]
    # 0 2
    res[9] = -sig * X[9] + sig * X[10]
    # 1 2
    res[10] = (r - z) * X[9] - X[10] - x * X[11]
    # 2 2
    res[11] = y * X[9] + x * X[10] - bet * X[11]

    return res


def lyapunov(n, fn, tStart, tStep, tEnd, y0):
    totalIter = int((tEnd - tStart) / tStep)

    y = numpy.zeros((n + 1, n), numpy.float)
    S = numpy.zeros((n,), numpy.float)

    gsc = numpy.empty((n,), numpy.float)
    norms = numpy.empty((n,), numpy.float)
    series = numpy.empty((totalIter, n + 1), dtype=numpy.float)

    y[0] = y0
    y[1:] = numpy.eye(n)

    t = tStart

    for i in range(totalIter):
        y = rk4(t, t + tStep, y.flat, fn).reshape((n + 1, n))
        t += tStep

        w = y[1:]
        for j in range(n):
            for k in range(j):
                gsc[k] = numpy.dot(w[j], w[k])

            for k in range(j):
                w[j] -= numpy.dot(gsc[k], w[k])

            norms[j] = numpy.linalg.norm(w[j])
            w[j] /= norms[j]
        y[1:] = w

        S += numpy.log(norms)
        series[i] = (t, *(S / (t - tStart)))

        if (i + 1) % 10 == 0:
            print(*map(lambda x: "%.3f" % x, series[i]))
            print("%.3f" % sum(series[i][1:]))

    return series
