import unittest

import numpy
from pyopencl import create_some_context, CommandQueue

from algorithms.lyapunov import Lyapunov
from dynsys.equation import EquationSystem, Parameters


class TestLyapunov(unittest.TestCase):

    def setUp(self):
        self.ctx = create_some_context(answers=[0, 0])
        self.queue = CommandQueue(self.ctx)

    def test_lorenz(self):
        system = EquationSystem(
            "x = a*(y - x)",
            "y = -x*z + b*x - y",
            "z = x*y - c*z",
            parameters=["a", "b", "c"]
        )

        variations = EquationSystem(
            "_x = a*(_y - _x)",
            "_y = -_x*z - _z*x + b*_x - _y",
            "_z = _x*y + _y*x - c*_z",
            parameters=["a", "b", "c"],
        )

        solver = Lyapunov(system, variations)

        result = solver.compute_at_point(
            self.queue,
            n_iter=10000,
            t_step=5e-3,
            point=(-3.16, -5.31, 13.31),
            parameters=Parameters(a=10, b=28, c=8/3),
        )

        assert numpy.allclose(list(numpy.round(result, 3)), [0.911, -0.02, -14.558])
        assert numpy.allclose(sum(result), -13.66667)
