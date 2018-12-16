import numpy
from scipy import integrate


def Lorenz(t, X):
    sig = 10
    r = 28
    bet = 8 / 3
    x, y, z = X[0], X[1], X[2]

    Y = numpy.mat(
        [
            [X[3], X[6], X[9]],
            [X[4], X[7], X[10]],
            [X[5], X[8], X[11]]
        ])
    f = numpy.zeros(shape=(12,), dtype=numpy.float)

    f[0] = sig*(y-x)
    f[1] = -x*z+r*x-y
    f[2] = x*y-bet*z

    Jac = numpy.mat([
        [-sig, sig,     0],
        [r-z,    -1,    -x],
        [y,     x, -bet]
        ])

    f[3:] = (Jac*Y).reshape((9,), order="F")

    return f


def lyapunov(n, fn, tstart, stept, tend, ystart):
    n2 = n*(n+1)
    nit = int((tend - tstart) / stept)
    y = numpy.zeros((n + 1, n), numpy.float, order="F")
    y0 = y.copy()
    S = numpy.zeros((n,), numpy.float, order="F")
    gsc = S.copy()
    znorm = S.copy()
    lp = S.copy()
    series = numpy.empty((nit, 4), dtype=numpy.float)

    y[:n] = ystart
    
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

        if (ITERLYAP + 1) % 10 == 0:
            print(t, *lp)

    return series


if __name__ == '__main__':
    lyapunov(3, Lorenz, 0, 1, 500, [-3.16, -5.31, 13.31])
