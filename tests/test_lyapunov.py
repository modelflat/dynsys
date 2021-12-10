import unittest

import numpy
from pyopencl import create_some_context, CommandQueue

from dynsys.algorithms.lyapunov import Lyapunov
from dynsys.equation import EquationSystem, Parameters


class TestLyapunov(unittest.TestCase):

    def setUp(self):
        self.ctx = create_some_context(answers=[0, 0])
        self.queue = CommandQueue(self.ctx)

    def test_henon(self):
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

        solver = Lyapunov(system, variations)

        result = solver.compute(
            self.queue,
            parameters=[Parameters(a=1.4, b=0.3)],
            n_iter=1 << 16
        )[0]

        result = list(numpy.round(result, 3))
        print(result)

        assert numpy.allclose(result, [0.606, -2.343])

    def test_ressler_chaos(self):
        system = EquationSystem(
            "x = -(y + z)",
            "y = x + a*y",
            "z = b + z*(x - c)",
            parameters=["a", "b", "c"],
            kind="continuous"
        )

        variations = EquationSystem(
            "_x = -(_y + _z)",
            "_y = _x + a*_y",
            "_z = _z*(x - c) + z*_x",
            parameters=["a", "b", "c"],
            kind="continuous"
        )

        solver = Lyapunov(system, variations)

        result = solver.compute(
            self.queue,
            parameters=[Parameters(a=0.15, b=0.20, c=10.0)],
            n_iter=1 << 18,
            t_step=1e-2
        )[0]

        result = list(numpy.round(result, 2))

        assert numpy.allclose(result, [0.13, 0.0, -14.14])

    def test_lorenz(self):
        system = EquationSystem(
            "x = a*(y - x)",
            "y = -x*z + b*x - y",
            "z = x*y - c*z",
            parameters=["a", "b", "c"],
            kind="continuous"
        )

        variations = EquationSystem(
            "_x = a*(_y - _x)",
            "_y = -_x*z - _z*x + b*_x - _y",
            "_z = _x*y + _y*x - c*_z",
            parameters=["a", "b", "c"],
            kind="continuous"
        )

        solver = Lyapunov(system, variations)

        result = solver.compute(
            self.queue,
            parameters=[Parameters(a=10, b=28, c=8 / 3)],
            init=(-3.16, -5.31, 13.31),
            n_iter=10000,
            t_step=5e-3
        )[0]

        assert numpy.allclose(list(numpy.round(result, 3)), [0.911, -0.02, -14.558])
        assert numpy.allclose(sum(result), -13.66667)
