from dynsys import SimpleApp, vStack, QLabel, Image2D
from dynsys import LCE
from dynsys import LCE2Map
import numpy
import time
from matplotlib import pyplot

from dynsys.ui.ImageWidgets import toPixmap

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

Fn_Henon = r"""

#define userFn_SYSTEM(x, y, a, b) (vec_t)( \
    1 - a*x*x + b*y, \
    x \
)

#define userFn_VARIATION(x, y, x_, y_, a, b) (vec_t)( \
    -2*a*x_*x + b*y_, \
    x_ \
)

#define NP 2

"""

Fn_Ikeda = r"""

#define userFn_SYSTEM(x, y, A, B) (vec_t)( \
    A + B * (x * cos(x*x + y*y) - y * sin(x*x + y*y)), \
    B * (x * sin(x*x + y*y) + y * cos(x*x + y*y)) \
)

#define userFn_VARIATION(x, y, x_, y_, A, B) (vec_t)( \
    B * (cos(x*x + y*y) - 2*x*x*sin(x*x + y*y) - 2*x*y*cos(x*x + y*y)) * x_ + \
    B *(-2*x*y*sin(x*x + y*y) - sin(x*x + y*y) - 2*y*y*cos(x*x + y*y)) * y_, \
    B * (sin(x*x + y*y) + 2*x*x*cos(x*x + y*y) - 2*x*y*sin(x*x + y*y)) * x_ + \
    B * (2*x*y*cos(x*x + y*y) + cos(x*x + y*y) - 2*y*y*sin(x*x + y*y)) * y_ \
)

#define NP 2

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
        self.lyapMapIkeda()

    def lyapLorenz(self):
        lyapTest = LCE.Lyapunov(self.ctx, self.queue, Fn_Lorenz)
        print(lyapTest((1, 2, 3), 10, 8/3, 28, 0, 1, iter=2000, stepIter=200))

    def lyapHenon(self):
        lyapTest = LCE2Map.Lyapunov(self.ctx, self.queue, Fn_Henon)
        res = lyapTest((-.5, .5), 1.4, .3, iter=2000)
        print(res)
        print("Actual sum:\t", sum(res), "\nTheor sum:\t", numpy.log(abs(-0.3)))

    def lyapIkeda(self):
        lyapTest = LCE2Map.Lyapunov(self.ctx, self.queue, Fn_Ikeda)
        res = lyapTest((-.5, .5), 2.5, .25, iter=2000)
        print(res)

    def lyapMapIkeda(self, lyapId=0):
        lyap = LCE2Map.LyapunovMap(self.ctx, self.queue, Fn_Ikeda, (400, 256))
        t = time.perf_counter()
        res, min_, max_ = lyap((-.8, .8), lyapId, (1.0, 6.0), (0.0, 0.3), iter=100)
        print("%.3f s" % (time.perf_counter() - t))
        self.label = Image2D(targetShape=(False, False))
        self.label2 = QLabel("L%d varies from %.3f (blue) to 0 (red). L > 0 is black. Lmax is %.3f" % (lyapId, min_, max_))
        self.label.setTexture(res)
        self.setLayout(vStack(
            self.label2, self.label
        ))

    def lyapTestSeriesLorenz(self):
        lyap = LCE.LyapunovSeries(self.ctx, self.queue, Fn_Lorenz, 3)
        par = numpy.linspace(9, 11, 64)
        t = time.perf_counter()
        res = lyap((1, 1, 1), (10, 8/3, 28), 0, par, 0, 1, iter=2000, stepIter=200)
        print("%.3f s" % (time.perf_counter() - t))
        pyplot.plot(par, res.T[0], "r.")
        pyplot.plot(par, res.T[1], "g.")
        pyplot.plot(par, res.T[2], "b.")
        pyplot.show()

    def lyapTestMapLorenz(self, lyapId=2):
        lyap = LCE.LyapunovMap(self.ctx, self.queue, Fn_Lorenz, (128, 128))
        t = time.perf_counter()
        res, min_, max_ = lyap((1, 2, 3), lyapId, (8, 12), (2, 4), 28, t0=0, dt=1, iter=100, stepIter=100)
        print("%.3f s" % (time.perf_counter() - t))
        self.label = Image2D(targetShape=(False, False))
        self.label2 = QLabel("L%d varies from %.3f (blue) to %.3f (red)" % (lyapId, min_, max_))
        self.label.setTexture(res)
        self.setLayout(vStack(
            self.label2, self.label
        ))


if __name__ == '__main__':
    Test().run()
