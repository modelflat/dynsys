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
    integrator = integrate.ode(fn)\
        .set_integrator("dopri5")\
        .set_initial_value(y.flat, t)

    for i in range(totalIter):
        y = integrator.integrate(t + tStep).reshape((n + 1, n))
        t += tStep

        w = y[1:]
        for j in range(n):
            for k in range(j):
                gsc[k] = numpy.dot(w[j], w[k])

            for k in range(j):
                w[j] -= gsc[k] * w[k]

            norms[j] = numpy.linalg.norm(w[j])
            w[j] /= norms[j]
        y[1:] = w

        S += numpy.log(norms)
        series[i] = (t, *(S / (t - tStart)))

    return series


if __name__ == '__main__':
    print(*lyapunov(3, Lorenz, 0, 1, 100, [-3.16, -5.31, 13.31])[-1])
