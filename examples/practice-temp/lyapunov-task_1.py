"""

1.4 and 1.5

"""

import time

import numpy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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


class ContinuousMapLyapunov(SimpleApp):

    def __init__(self):
        super().__init__("Continuous")

        # a figure instance to plot on
        self.figure = Figure()
        self.lyapSeriesIkeda(self.figure)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        lyapId = 0
        res, min_, max_ = self.lyapMapIkeda(lyapId)

        self.label = Image2D(targetShape=(False, False))
        self.label2 = QLabel("L%d varies from %.3f (blue) to %.3f (red)" % (lyapId, min_, max_))
        self.label.setTexture(res)
        self.setLayout(
            hStack(
                vStack(self.canvas, self.toolbar),
                vStack(self.label2, self.label)
            )
        )

    def lyapLorenz(self):
        lyapTest = LCE.Lyapunov(self.ctx, self.queue, Fn_Lorenz)
        print(lyapTest((1, 2, 3), 10, 8/3, 28, 0, 1, iter=2000, stepIter=200))

    def lyapTestSeriesLorenz(self, fig):
        lyap = LCE.LyapunovSeries(self.ctx, self.queue, Fn_Lorenz, 3)
        par = numpy.linspace(9, 11, 64)
        t = time.perf_counter()
        res = lyap((1, 1, 1), (10, 8/3, 28), 0, par, 0, 1, iter=2000, stepIter=200)
        print("%.3f s" % (time.perf_counter() - t))
        fig.plot(par, res.T[0], "r.",
                 par, res.T[1], "g.",
                 par, res.T[2], "b.")

    def lyapTestMapLorenz(self, lyapId=2):
        lyap = LCE.LyapunovMap(self.ctx, self.queue, Fn_Lorenz, (128, 128))
        t = time.perf_counter()
        res, min_, max_ = lyap((1, 2, 3), lyapId, (8, 12), (2, 4), 28, t0=0, dt=1, iter=100, stepIter=100)
        print("%.3f s" % (time.perf_counter() - t))



if __name__ == '__main__':
    ContinuousMapLyapunov().run()
