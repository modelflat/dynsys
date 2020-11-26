from typing import Tuple, List

import numpy
import pyopencl as cl

from dynsys.cl import WithProgram, assemble, load_template
from dynsys.core import send_to_device, allocate_on_device
from dynsys.equation import EquationSystem, Parameters

SOURCE = load_template("phase/kernels.cl")


class Phase(WithProgram):

    def __init__(self, system: EquationSystem, ctx: cl.Context = None):
        super().__init__(ctx)
        self._system = system

    def src(self, **kwargs):
        return assemble(SOURCE, system=self._system, **kwargs)

    def compute(
            self,
            queue: cl.CommandQueue,
            init: List[Tuple],
            parameters: List[Parameters],
            n_skip: int = 0,
            n_iter: int = 1000,
            t_start: float = 0.0,
            t_step: float = 1e-2,
    ):
        self.compile(queue.context)

        if init is None:
            init = [tuple([0] * self._system.dimensions)]
        else:
            for cond in init:
                assert len(cond) == self._system.dimensions

        init = numpy.array(init, dtype=self._system.real_type)
        init_dev = send_to_device(queue, init)

        params = numpy.array(
            [p.to_cl_object(self._system) for p in parameters],
            dtype=self._system.parameters_dtype()
        )
        params_dev = send_to_device(queue, params)

        result = numpy.empty(
            (len(init), len(parameters), n_iter, self._system.dimensions),
            dtype=self._system.real_type
        )
        result_dev = allocate_on_device(queue, result)

        if self._system.is_continuous:
            temporal_args = (
                self._system.real_type(t_start),
                self._system.real_type(t_step),
            )
        else:
            temporal_args = tuple()

        self.program.capture_phase(
            queue, (len(init), len(parameters)), None,
            numpy.int32(n_skip),
            numpy.int32(n_iter),
            *temporal_args,
            init_dev,
            params_dev,
            result_dev,
        )

        cl.enqueue_copy(queue, result, result_dev)

        return result
