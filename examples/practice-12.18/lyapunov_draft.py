import numpy
from scipy import integrate


def Lorenz(t, X, sig=10, bet=8/3, r=28):
    x, y, z = X[0], X[1], X[2]
    res = numpy.empty(shape=(12,), dtype=numpy.float)

    res[0] = sig*(y - x)
    res[1] = -x*z + r*x - y
    res[2] = x*y - bet*z

    Y = numpy.mat((
        (X[3], X[6], X[9]),
        (X[4], X[7], X[10]),
        (X[5], X[8], X[11])
    ))
    J = numpy.mat((
        (-sig, sig, 0),
        (r - z, -1, -x),
        (y, x, -bet)
    ))

    res[3:] = (J*Y).reshape((9,), order="F")

    return res


def lyapunov_sanity(n, fn, tstart, stept, tend, ystart):
    nit = int((tend - tstart) / stept)
    y = numpy.zeros((n + 1, n), numpy.float)
    y0 = y.copy()
    S = numpy.zeros((n,), numpy.float)
    gsc = S.copy()
    znorm = S.copy()
    lp = S.copy()
    series = numpy.empty((nit, 4), dtype=numpy.float)

    y[0] = ystart

    for i in range(n):
        y[i + 1][i] = 1.0

    t = tstart
    integrator = integrate.ode(fn).set_integrator("dopri5")
    integrator.set_initial_value(y.reshape((numpy.prod(y.shape))), t)

    for ITERLYAP in range(nit):
        Y = integrator.integrate(t + stept)

        t += stept
        y = Y

        for i in range(n):
            for j in range(n):
                y0.reshape((n+1, n))[i + 1][j] = y.reshape((n+1, n))[j + 1][i]

        znorm[0] = 0.0
        for j in range(n):
            znorm[0] += y0.reshape((n + 1, n))[j + 1][0] ** 2
        znorm[0] = numpy.sqrt(znorm[0])

        for j in range(n):
            y0.reshape((n + 1, n))[j + 1][0] /= znorm[0]

        for j in range(1, n):
            for k in range(j):
                gsc[k] = 0.0
                for l in range(n):
                    gsc[k] += y0.reshape((n + 1, n))[l + 1][j] * y0.reshape((n + 1, n))[l + 1][k]

            for k in range(n):
                for l in range(j):
                    y0.reshape((n+1, n))[k + 1][j] -= gsc[l] * y0.reshape((n+1, n))[k+1][l]

            znorm[j] = 0.0
            for k in range(n):
                znorm[j] += y0.reshape((n+1, n))[k+1][j] ** 2

            znorm[j] = numpy.sqrt(znorm[j])
            for k in range(n):
                y0.reshape((n+1, n))[k+1][j] /= znorm[j]

        for k in range(n):
            S[k] += numpy.log(znorm[k])
            lp[k] = S[k] / (t - tstart)

        for i in range(n):
            for j in range(n):
                y.reshape((n+1, n))[j + 1][i] = y0.reshape((n+1, n))[i + 1][j]

        series[ITERLYAP] = (t, *lp)

    return series


def lyapunov_for_porting(n, fn, tStart, tStep, tEnd, y0):
    totalIter = int((tEnd - tStart) / tStep)

    y = numpy.zeros((n + 1, n), numpy.float)
    w = numpy.zeros((n, n), dtype=numpy.float)

    S = numpy.zeros((n,), numpy.float)
    L = S.copy()

    gsc = S.copy()
    norms = S.copy()
    series = numpy.empty((totalIter, 4), dtype=numpy.float)

    y[0] = y0
    for i in range(n):
        y[i + 1][i] = 1.0

    t = tStart
    integrator = integrate.ode(fn).set_integrator("dopri5")
    integrator.set_initial_value(y.flat, t)

    for it in range(totalIter):
        Y = integrator.integrate(t + tStep)

        t += tStep
        y = Y.reshape((n + 1, n))

        for i in range(n):
            for j in range(n):
                w[i][j] = y[j + 1][i]

        norms[0] = 0.0
        for j in range(n):
            norms[0] += w[j][0] ** 2
        norms[0] = numpy.sqrt(norms[0])

        for j in range(n):
            w[j][0] /= norms[0]

        for j in range(1, n):
            for k in range(j):
                gsc[k] = 0.0
                for l in range(n):
                    gsc[k] += w[l][j] * w[l][k]

            for k in range(n):
                for l in range(j):
                    w[k][j] -= gsc[l] * w[k][l]

            norms[j] = 0.0
            for k in range(n):
                norms[j] += w[k][j] ** 2

            norms[j] = numpy.sqrt(norms[j])
            for k in range(n):
                w[k][j] /= norms[j]

        for k in range(n):
            S[k] += numpy.log(norms[k])
            L[k] = S[k] / (t - tStart)

        for i in range(n):
            for j in range(n):
                y[j + 1][i] = w[i][j]

        series[it] = (t, *L)

        if (it + 1) % 10 == 0:
            print(t, *L)

    return series


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
    integrator = integrate.ode(fn).set_integrator("dopri5")
    integrator.set_initial_value(y.flat, t)

    for i in range(totalIter):
        Y = integrator.integrate(t + tStep)

        t += tStep
        y = Y.reshape((n + 1, n))

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

    return series


if __name__ == '__main__':
    tMax = 100

    sanity = lyapunov_sanity(3, Lorenz, 0, 1, tMax, [-3.16, -5.31, 13.31])
    actual = lyapunov(3, Lorenz, 0, 1, tMax, [-3.16, -5.31, 13.31])

    assert numpy.allclose(sanity, actual, atol=1e-14, rtol=1e-14)
