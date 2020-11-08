import warnings
from typing import Tuple, Callable, List, NamedTuple

import numpy
import pyopencl as cl

from core import CLImage, reallocate
from cl import WithProgram, assemble, load_template
from equation import EquationSystem, Parameters
from grid import Grid


SOURCE = load_template("lyapunov/kernels.cl")


UINT_SIZE = numpy.uint32(0).nbytes


class Lyapunov(WithProgram):

    def __init__(self, system: EquationSystem, variations: EquationSystem = None, ctx: cl.Context = None):
        super().__init__(ctx)
        self._system = system
        self._variations = variations

    def src(self, **kwargs):
        return assemble(
            SOURCE,
            system=self._system,
            **kwargs
        )

    def compute_at_point(
            self,
            queue: cl.CommandQueue,
            point: Tuple,
            parameters: Parameters,
            variations: EquationSystem = None,
            n_iter: int = 1000,
            t_start: float = 0.0,
            t_step: float = 1e-2,
            n_integrator_steps: int = 10,
    ):
        variation_equations = variations or self._variations
        self.compile(queue.context, template_variables={
            'variations': variation_equations
        })

        params = parameters.to_cl_object(self._system)
        init = numpy.array(point, dtype=self._system.real_type)
        variation_values = numpy.eye(self._system.dimensions, dtype=self._system.real_type)
        variation_values_dev = cl.Buffer(
            queue.context,
            cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
            hostbuf=variation_values,
        )
        result = numpy.empty((self._system.dimensions,), dtype=self._system.real_type)
        result_dev = cl.Buffer(
            queue.context,
            cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR,
            size=result.nbytes
        )

        self.program.single_lyapunov_at_point(
            queue, (1,), None,
            numpy.int32(n_iter),
            self._system.real_type(t_start),
            self._system.real_type(t_step),
            numpy.int32(n_integrator_steps),
            init,
            params,
            variation_values_dev,
            result_dev,
        )

        cl.enqueue_copy(queue, result, result_dev)

        return result
