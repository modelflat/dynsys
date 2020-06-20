import warnings
from typing import Tuple, Callable, List, Optional

import numpy
import pyopencl as cl

from core import CLImage
from cl import WithProgram, assemble, load_template
from equation import EquationSystem, Parameters
from grid import Grid


SOURCE = load_template("parameter_map/kernels.cl")


UINT_SIZE = numpy.uint32(0).nbytes


class ParameterMap(WithProgram):

    def __init__(self, system: EquationSystem = None, ctx: cl.Context = None):
        super().__init__(ctx)
        self._system = system
        self._points_dev = None
        self._periods_dev = None

    @property
    def system(self):
        return self._system

    def src(self, **kwargs):
        return assemble(SOURCE, system=self.system, **kwargs)

    def _allocate_buffers(self, shape: Tuple[int], n_iter: int, dimensions: int):
        real_size = self.system.real_type(0).nbytes
        new_size = dimensions * numpy.prod(shape) * n_iter * real_size

    def compute(
            self,
            queue: cl.CommandQueue,
            grid: Grid,
            n_skip: int,
            n_iter: int,
            init: Tuple[float] = None,
            tolerance: int = 3
    ):


        self.compile(queue.context, template_variables={"varied": varied})
