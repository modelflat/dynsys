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

    def test_real_system(self):
        grid = Grid([
            (-1, 1, 32),
            (-1, 1, 32)
        ])

        for real_type, real_type_name in [(numpy.float32, "float"), (numpy.float64, "double")]:
            henon_map = EquationSystem(
                "x = 1 - a*x*x - b*y",
                "y = x",
                parameters=["a", "b"],
                real_type=real_type, real_type_name=real_type_name
            )

            solver = Attractors(henon_map)

            attractors = solver.find_attractors(self.queue, grid, Parameters(a=1.05, b=-0.2), 1000, 16)
            assert len(attractors) == 1

            attractor, = attractors

            assert attractor.count == 1024
            assert attractor.period == 4

            expected_attractor = numpy.array(
                [[-0.11,  0.981], [1.183, -0.11], [-0.492, 1.183], [0.981, -0.492]],
                dtype=real_type
            )

            assert numpy.allclose(attractor.values, expected_attractor)

    def test_dimensions(self):
        for d in range(1, 16):
            grid_granularity = 6
            grid = Grid([
                (-1, 1, grid_granularity),
                *[
                    0.0
                    for _ in range(d - 1)
                ]
            ])

            system = EquationSystem(
                *[
                    f"x{i} = a"
                    for i in range(d)
                ],
                parameters=["a"],
            )
            params = Parameters(a=1)

            solver = Attractors(system)

            attractors = solver.find_attractors(self.queue, grid, params, n_skip=1, n_iter=4, occurrence_threshold=0)

            assert len(attractors) == 1
            assert attractors[0].count == grid_granularity
            assert attractors[0].values.shape == (1, d)
            assert list(attractors[0].values[0]) == [1.0]*d
