"""

1.4 and 1.5

"""

import time
from typing import Iterable

import numpy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dynsys import LCE
from dynsys import SimpleApp, vStack, QLabel, Image2D, hStack

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

Fn_Rossler = r"""
#define userFn_SYSTEM(x, y, z, a, b, r) (vec_t)( \
    -y - z, \
    x + a*y, \
    b - r*z + x*z \
)

#define userFn_VARIATION(x, y, z, x_, y_, z_, a, b, r) (vec_t)( \
    -y_ - z_, \
    x_ + a*y_, \
    -r*z_ + x_*z + x*z_ \
)

#define NP 3
"""

startPoint = (0.1, 0.1, 0.1)
eps = 0.3


class ContinuousMapLyapunov(SimpleApp):

    def __init__(self):
        super().__init__("Continuous")

        self.figure = Figure()
        self.lyapSeries(
            h=0.1,#numpy.linspace(0.05, 0.15, 500),
            g=numpy.linspace(0.0, 1.0, 500),
            fig=self.figure
        )
        self.canvas = FigureCanvas(self.figure)

        lyapId = 0
        res, min_, max_ = self.lyapMap((0.06, 0.15), (0.75, 0.95), lyapId)

        self.label = Image2D(targetShape=(False, False))
        self.label2 = QLabel("L%d varies from %.3f (blue) to %.3f (red)" % (lyapId, min_, max_))
        self.label.setTexture(res)
        self.setLayout(
            hStack(
                vStack(self.canvas),
                vStack(self.label2, self.label)
            )
        )

    def lyapSeries(self, h, g, fig):
        sub = fig.add_subplot(111)
        lyap = LCE.LyapunovSeries(self.ctx, self.queue, Fn_KPR, 3)
        if isinstance(h, Iterable) and not isinstance(g, Iterable):
            pars = 0, g
            parLin = h
            parId = 0
        elif isinstance(g, Iterable) and not isinstance(h, Iterable):
            pars = h, 0
            parLin = g
            parId = 1
        else:
            raise ValueError("should be only one Iterable")

        t = time.perf_counter()
        res = lyap(startPoint, (*pars, eps), parId, parLin, 0, 1, iter=2000, stepIter=15)
        print("Series %.3f s" % (time.perf_counter() - t))
        sub.plot(parLin, res.T[0], "r-", label="L0")
        sub.plot(parLin, res.T[1], "g-", label="L1")
        sub.plot(parLin, res.T[2], "b-", label="L2")
        sub.legend()

    def lyapMap(self, hBounds, gBounds, lyapId):
        lyap = LCE.LyapunovMap(self.ctx, self.queue, Fn_KPR, (256, 256))
        t = time.perf_counter()
        res, min_, max_ = lyap(startPoint, lyapId, hBounds, gBounds, eps, t0=0, dt=1, iter=1000, stepIter=15)
        print("Map %.3f s" % (time.perf_counter() - t))
        return res, min_, max_


class ContinuousMapRossler(SimpleApp):

    def __init__(self):
        super().__init__("Continuous")

        self.figure = Figure()
        self.lyapSeries(
            a=0.2, b=0.2,
            r=numpy.linspace(2, 6, 500),
            fig=self.figure
        )
        self.canvas = FigureCanvas(self.figure)

        lyapId = 0
        res, min_, max_ = self.lyapMap((0.06, 0.15), (0.75, 0.95), lyapId)

        self.label = Image2D(targetShape=(False, False))
        self.label2 = QLabel("L%d varies from %.3f (blue) to %.3f (red)" % (lyapId, min_, max_))
        self.label.setTexture(res)
        self.setLayout(
            hStack(
                vStack(self.canvas),
                vStack(self.label2, self.label)
            )
        )

    def lyapSeries(self, a, b, r, fig):
        sub = fig.add_subplot(111)
        lyap = LCE.LyapunovSeries(self.ctx, self.queue, Fn_Rossler, 3)
        t = time.perf_counter()
        res = lyap((1, 1, 1), (a, b, 0), 2, r, 0, 1, iter=10000, stepIter=5)
        print("Series %.3f s" % (time.perf_counter() - t))
        sub.plot(r, res.T[0], "r-", label="L0")
        sub.plot(r, res.T[1], "g-", label="L1")
        sub.plot(r, res.T[2], "b-", label="L2")
        sub.legend()

    def lyapMap(self, hBounds, gBounds, lyapId):
        lyap = LCE.LyapunovMap(self.ctx, self.queue, Fn_KPR, (1, 1))
        t = time.perf_counter()
        res, min_, max_ = lyap((1, 1, 1), lyapId, hBounds, gBounds, 0.2, t0=0, dt=1, iter=1, stepIter=1)
        print("Map %.3f s" % (time.perf_counter() - t))
        return res, min_, max_



if __name__ == '__main__':
    ContinuousMapLyapunov().run()
