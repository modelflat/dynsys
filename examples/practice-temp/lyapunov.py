from dynsys import SimpleApp
from dynsys.LCE import *
import time
from matplotlib import pyplot


Fn_Lorenz = r"""
#define userFn_SYSTEM(x, y, z, sig, bet, r) (vec_t)( \
    (sig)*((y) - (x)), \
    -(x)*(z) + (r)*(x) - (y), \
    (x)*(y) - (bet)*(z)\
)

#define userFn_VARIATION(x, y, z, x_, y_, z_, sig, bet, r) (vec_t)( \
    ((y_) - (x_)) * (sig), \
    ((r) - (z))*(x_) - (y_) - (x)*(z_), \
    (y)*(x_) + (x)*(y_) - (bet)*(z_) \
)

#define NP 3
"""


Fn_KPR = r"""
#define userFn_SYSTEM(x, y, z, h, g, eps) (vec_t)( \
    2*h*x + y - g*z, \
    -x, \
    (x - 8.592*z + 22*z*z - 14.408*z*z*z) / eps \
)

#define userFn_VARIATION(x, y, z, x_, y_, z_, h, g, eps) (vec_t)( \
    2*h*x_ + y_ - g*z_, \
    -x_, \
    (x_ - 8.592*z_ + 44*z_*z - 43.224*z_*z*z) / eps \
)

#define NP 3
"""


class Test(SimpleApp):

    def __init__(self):
        super().__init__("123")

        

        exit()

    def lyapTestLorenz(self):
        lyapTest = Lyapunov(self.ctx, self.queue, Fn_Lorenz)
        print(lyapTest((1, 2, 3), 10, 8/3, 28, 0, 1, iter=2000, stepIter=200))

    def lyapTestSeriesLorenz(self):
        lyap = LyapunovSeries(self.ctx, self.queue, Fn_Lorenz, 3)
        par = numpy.linspace(9, 11, 64)
        t = time.perf_counter()
        res = lyap((1, 1, 1), (10, 8/3, 28), 0, par, 0, 1, iter=2000, stepIter=200)
        print("%.3f s" % (time.perf_counter() - t))
        pyplot.plot(par, res.T[0], "r.")
        pyplot.plot(par, res.T[1], "g.")
        pyplot.plot(par, res.T[2], "b.")
        pyplot.show()


if __name__ == '__main__':
    Test().run()
