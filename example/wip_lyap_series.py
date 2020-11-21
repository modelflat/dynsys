import numpy

from algorithms.lyapunov import Lyapunov
from app import SimpleApp, stack
from equation import Parameters, EquationSystem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Henon(SimpleApp):

    def __init__(self):
        super(Henon, self).__init__("Henon")

        system = EquationSystem(
            "x = 1 + b*y - a*x*x",
            "y = x",
            parameters=["a", "b"],
            kind="discrete",
        )

        variations = EquationSystem(
            "_x = b*_y - 2*a*x*_x",
            "_y = _x",
            parameters=["a", "b"],
            kind="discrete"
        )

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.henon_lyap = Lyapunov(system=system, variations=variations)

        self.recompute()

        self.setLayout(stack(
            self.canvas,
            kind="h"
        ))

    def recompute(self, *_):
        sub = self.figure.add_subplot(111)
        sub.clear()

        par = numpy.linspace(1.0, 1.43, 1000)
        res = self.henon_lyap.compute(
            self.queue,
            parameters=[
                Parameters(a=a, b=0.3) for a in par
            ],
            n_iter=10000
        )

        sub.plot(par, res.T[0], "r-", linewidth=1)
        sub.grid()


class Ressler(SimpleApp):

    def __init__(self):
        super(Ressler, self).__init__("Ressler")

        system = EquationSystem(
            "x = -y - z",
            "y = x + a*y + w",
            "z = b + x*z",
            "w = c*z + d*w",
            parameters=["a", "b", "c", "d"],
            kind="continuous"
        )

        variations = EquationSystem(
            "_x = -_y - _z",
            "_y = _x + a*_y + _w",
            "_z = _x*z + x*_z",
            "_w = c*_z + d*_w",
            parameters=["a", "b", "c", "d"],
            kind="continuous"
        )

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.ressler_lyap = Lyapunov(system=system, variations=variations)

        self.recompute()

        self.setLayout(stack(
            self.canvas,
            kind="h"
        ))

    def recompute(self, *_):
        sub = self.figure.add_subplot(111)
        sub.clear()

        # par = numpy.linspace(0.0, 0.5, 500)
        # par = numpy.linspace(2.0, 4.0, 500)
        # par = numpy.linspace(-1.0, 0.0, 500)
        par = numpy.linspace(0.0, 0.1, 500)
        res = self.ressler_lyap.compute(
            self.queue,
            parameters=[
                # Parameters(a=a, b=3.0, c=-0.5, d=0.05) for a in par
                # Parameters(a=0.25, b=b, c=-0.5, d=0.05) for b in par
                # Parameters(a=0.25, b=3.0, c=c, d=0.05) for c in par
                Parameters(a=0.25, b=3.0, c=-0.5, d=d) for d in par
            ],
            n_iter=2000,
            t_step=1e-2,
            n_integrator_steps=100,
        )

        for i, color in enumerate(["r", "g", "b", "c"]):
            sub.plot(par, res.T[i], f"{color}-", linewidth=1)
        sub.grid()


if __name__ == '__main__':
    Ressler().run()
