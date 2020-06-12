import unittest

import numpy
from pyopencl import create_some_context, CommandQueue

from dynsys.equation import EquationSystem, Parameters
from dynsys.grid import Grid
from algorithms.attractors import Attractors


class TestAttractorFinder(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(42)

        self.ctx = create_some_context(answers=[0, 0])
        self.queue = CommandQueue(self.ctx)

        self.grid = Grid(x=((-1, 1), 32), y=((-1, 1), 32))

    def test_known_period(self):
        for real_type, real_type_name in [(numpy.float32, "float"), (numpy.float64, "double")]:
            henon_map = EquationSystem(
                "x = 1 - a*x*x - b*y",
                "y = x",
                parameters=["a", "b"],
                real_type=real_type, real_type_name=real_type_name
            )

            solver = Attractors(henon_map)

            attractors = solver.find_attractors(self.queue, self.grid, 1000, 25, Parameters(a=1.05, b=-0.2))
            assert len(attractors) == 1

            attractor, = attractors

            assert attractor["count"] == 1024
            assert attractor["period"] == 4

            expected_attractor = numpy.array(
                [[-0.11,  0.981], [1.183, -0.11], [-0.492, 1.183], [0.981, -0.492]],
                dtype=real_type
            )

            assert numpy.allclose(attractor["attractor"], expected_attractor)
