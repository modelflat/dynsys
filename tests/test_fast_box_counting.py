import unittest

import numpy
import pyopencl as cl
from pyopencl import create_some_context, CommandQueue

from algorithms.fast_box_counting import FastBoxCounting
from dynsys.core import CLImage


class TestAttractorFinder(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(42)

        self.ctx = create_some_context(answers=[0, 0])
        self.queue = CommandQueue(self.ctx)

        self.image = CLImage(self.ctx, (256, 256))

        self.solver = FastBoxCounting()

    def test_empty_image(self):
        self.image.clear(self.queue, color=(1.0, 1.0, 1.0, 1.0))

        D = self.solver.compute(self.queue, self.image)

        assert numpy.isnan(D)

    def test_full_image(self):
        self.image.clear(self.queue, color=(0.0, 0.0, 0.0, 1.0))

        D = self.solver.compute(self.queue, self.image)

        assert numpy.isclose(D, 2)

    def test_one_filled_pixel_image(self):
        self.image.clear(self.queue, color=(1.0, 1.0, 1.0, 1.0))

        cl.enqueue_fill_image(
            self.queue, self.image.dev,
            color=numpy.array((0.0, 0.0, 0.0, 1.0), dtype=numpy.float32), origin=(128, 128), region=(1, 1)
        )

        D = self.solver.compute(self.queue, self.image)

        assert numpy.isclose(D, 0)
