"""

2.4 and 2.5

"""

import time

import numpy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from dynsys import LCE2Map
from dynsys import SimpleApp, vStack, QLabel, Image2D, hStack


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


class DiscreteMapLyapunov(SimpleApp):

    def __init__(self):
        super().__init__("Discrete")
        self.lyapIkeda()

        self.figure = Figure()
        self.lyapSeriesIkeda(self.figure)

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

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

    def lyapIkeda(self):
        lyapTest = LCE2Map.Lyapunov(self.ctx, self.queue, Fn_Ikeda)
        res = lyapTest((-.5, .5), 2.5, .25, iter=2000)
        print(res)

    def lyapMapIkeda(self, lyapId=0):
        lyap = LCE2Map.LyapunovMap(self.ctx, self.queue, Fn_Ikeda, (400, 256))
        t = time.perf_counter()
        res, min_, max_ = lyap((-.8, .8), lyapId, (1.0, 6.0), (0.0, 0.3), iter=100)
        print("%.3f s" % (time.perf_counter() - t))
        return res, min_, max_

    def lyapSeriesIkeda(self, fig):
        sub = fig.add_subplot(111)
        sub.clear()
        lyap = LCE2Map.LyapunovSeries(self.ctx, self.queue, Fn_Ikeda)
        par = numpy.linspace(1.0, 6.0, 512)
        t = time.perf_counter()
        res = lyap((-.8, .8), (2.0, 0.3), 0, par, iter=600)
        print("%.3f s" % (time.perf_counter() - t))
        sub.plot(par, res.T[0], "r-", par, res.T[1], "g-")


if __name__ == '__main__':
    DiscreteMapLyapunov().run()
