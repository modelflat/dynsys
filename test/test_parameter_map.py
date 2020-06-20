import unittest

import numpy
from pyopencl import create_some_context, CommandQueue

from algorithms.parameter_map import ParameterMap
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
            ("b", -0.5, 0.5, 32),
            ("a", 0, 2, 32),
        ])

        for real_type, real_type_name in [(numpy.float32, "float"), (numpy.float64, "double")]:
            henon_map = EquationSystem(
                "x = 1 - a*x*x - b*y",
                "y = x",
                parameters=["a", "b"],
                real_type=real_type, real_type_name=real_type_name
            )

            solver = ParameterMap(henon_map)

            periods = solver.compute(self.queue, grid, (0.1, 0.1), 1000, 16)

            total_area = 32 * 32
            periods, counts = numpy.unique(periods, return_counts=True)
            assert periods[0] == 0
            assert total_area * 0.05 < counts[0] < total_area * 0.25
            assert periods[1] == 1
            assert total_area * 0.25 < counts[1] < total_area * 0.5
            assert periods[2] == 2
            assert total_area * 0.05 < counts[2] < total_area * 0.25

    def test_variable_dimensions(self):
        grid_granularity = 4

        for d in range(1, 16):
            grid = Grid([
                ("a", -1, 1, grid_granularity),
                ("b", -1, 1, grid_granularity),
            ])

            system = EquationSystem(
                *[
                    f"x{i} = a"
                    for i in range(d)
                ],
                parameters=["a", "b"],
            )

            solver = ParameterMap(system)

            periods = solver.compute(self.queue, grid, tuple([0.1] * d), n_skip=16, n_iter=4, return_periods=True)

            assert numpy.allclose(periods, 1)

    def test_parameter_dimensions(self):
        grid_granularity = 4

        for d in range(1, 16):
            grid = Grid([
                ("a", 0, 1.0, grid_granularity),
                ("b", 0, 0.5, grid_granularity),
            ])

            system = EquationSystem(
                "x = a",
                "y = x",
                parameters=["a", "b"] + [f"c{i}" for i in range(d - 2)],
            )

            solver = ParameterMap(system)

            params = Parameters(a=1.0, b=0.0, **{
                f"c{i}": 1.0 for i in range(d - 2)
            })

            periods = solver.compute(self.queue, grid, (0.1, 0.1), n_skip=16, n_iter=4,
                                     parameters=params, return_periods=True)

            assert numpy.allclose(periods, 1)
