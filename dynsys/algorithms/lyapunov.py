from typing import Tuple, List

import numpy
import pyopencl as cl

from cl import WithProgram, assemble, load_template
from core import send_to_device, allocate_on_device
from equation import EquationSystem, Parameters

SOURCE = load_template("lyapunov/kernels.cl")


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

    def compute(
            self,
            queue: cl.CommandQueue,
            parameters: List[Parameters],
            init: Tuple = None,
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

        if init is None:
            init = tuple([0]*self._system.dimensions)

        init = numpy.array(init, dtype=self._system.real_type)
        init_dev = send_to_device(queue, init)

        params = numpy.array([
            p.to_cl_object(self._system) for p in parameters
        ], dtype=self._system.parameters_dtype())
        params_dev = send_to_device(queue, params)

        variation_values = numpy.eye(self._system.dimensions, dtype=self._system.real_type)
        variation_values_dev = send_to_device(queue, variation_values)

        result = numpy.empty((len(parameters), self._system.dimensions), dtype=self._system.real_type)
        result_dev = allocate_on_device(queue, result)

        if self._system.is_continuous:
            temporal_args = (
                self._system.real_type(t_start),
                self._system.real_type(t_step),
                numpy.int32(n_integrator_steps),
            )
        else:
            temporal_args = tuple()

        self.program.lyapunov_variations(
            queue, (len(parameters),), None,
            numpy.int32(n_iter),
            *temporal_args,
            init_dev,
            params_dev,
            variation_values_dev,
            result_dev,
        )

        cl.enqueue_copy(queue, result, result_dev)

        return result
