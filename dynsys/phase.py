from typing import Iterable, Union, Tuple

import numpy
import pyopencl as cl

from grid import Grid
from .equation import EquationSystem, Parameters
from .cl import WithProgram, assemble, load_template


SOURCE = load_template("phase/kernels.cl")


class Phase(WithProgram):

    def __init__(self, system: EquationSystem, ctx: cl.Context = None):
        super(Phase, self).__init__(ctx)
        self.system = system

    def src(self, **kwargs):
        return assemble(SOURCE, system=self.system, **kwargs)

    def compute(
            self,
            queue: cl.CommandQueue,
            parameters: Parameters,
            init: Union[numpy.ndarray, Grid] = None
    ):
        self.compile(queue.context)

        parameters_obj = parameters.to_cl_object(self.system)
        parameters_dev = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=parameters_obj
        )

        points_buf = send_to_device(queue, init)

        result = numpy.empty(len(init) * params.iter, dtype=init.dtype)
        result_buf = allocate_on_device(queue, result)

        if init is None:
            self.program.phase_gen(queue, (len(init),), None, params_buf, result_buf)
        else:
            self.program.phase(queue, (len(init),), None, params_buf, points_buf, result_buf)


        return result_buf
