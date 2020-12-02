import numbers
from typing import Tuple, List, Dict, Union, Any, Iterable, Sized

import numpy
import pyopencl as cl

from dynsys.cl import WithProgram, assemble, load_template
from dynsys.core import send_to_device, allocate_on_device, CLImage
from dynsys.equation import EquationSystem, Parameters


SOURCE = load_template("phase/kernels.cl")


def parse_bounds(bounds: Any, system: EquationSystem):
    varying_id = 0
    global_work_size = []
    varying = dict()
    bounds_arr = []
    unset = set()
    unknown_types = dict()
    for variable in system.variables:
        value = bounds.get(variable)
        if value is None:
            unset.add(variable)
            continue
        try:
            min_, max_, n = value
            bounds_arr.append((min_, max_))
            if min_ != max_:
                varying[variable] = varying_id
                global_work_size.append(max(n, 2))
                varying_id += 1
        except TypeError or ValueError:
            if isinstance(value, numbers.Real):
                bounds_arr.append((value, value))
            else:
                unknown_types[variable] = value

    if unset:
        raise RuntimeError(f"The following variables' bounds was not set: {unset}")

    if unknown_types:
        raise RuntimeError(f"The following variables' types was not understood: {unknown_types}")

    if len(varying) > 3:
        raise RuntimeError(f"Can't vary more than 3 variables at once! Varying variables are: {varying}")

    return varying, bounds_arr, global_work_size


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

    def compute_grid(
            self,
            queue: cl.CommandQueue,
            bounds: Dict[str, Union[Tuple, float, int]],
            parameters: Union[Parameters, Dict[str, Union[Tuple, float, int]]],
            n_skip: int = 0,
            n_iter: int = 1000,
            t_start: float = 0.0,
            t_step: float = 1e-2,
            image: CLImage = None,
            variables_to_draw: Union[List, Tuple] = None,
            drawing_bounds: Dict[str, Union[Tuple, float, int]] = None
    ):
        varying, bounds_arr, global_work_size = parse_bounds(bounds, self._system)

        _, drawing_bounds_arr, _ = parse_bounds(drawing_bounds, self._system)

        if image:
            if not variables_to_draw:
                if len(image.shape) == self._system.dimensions:
                    variables_to_draw = self._system.variables
                else:
                    raise ValueError(
                        "Cannot determine which variables to draw (variables_to_draw is not set, and system's "
                        f"dimension {self._system.dimensions} does not correspond to image dimension "
                        f"({len(image.shape)}))"
                    )
            elif len(variables_to_draw) not in {2, 3}:
                raise ValueError(
                    f"Can only draw in 2D or 3D, requested dim: {len(variables_to_draw)} (from {variables_to_draw})"
                )
        else:
            variables_to_draw = None

        self.compile(queue.context, template_variables=dict(varying=varying, variables_to_draw=variables_to_draw))

        if self._system.is_continuous:
            temporal_args = (
                self._system.real_type(t_start),
                self._system.real_type(t_step),
            )
        else:
            temporal_args = tuple()

        bounds_arr = numpy.array(bounds_arr, dtype=self._system.real_type)
        bounds_dev = send_to_device(queue, bounds_arr)

        drawing_bounds_arr = numpy.array(drawing_bounds_arr, dtype=self._system.real_type)
        drawing_bounds_dev = send_to_device(queue, drawing_bounds_arr)

        if not isinstance(parameters, Parameters):
            parameters = Parameters(**parameters)

        params = parameters.to_cl_object(self._system)
        params_dev = send_to_device(queue, params)

        if image is None:
            result = numpy.array((*global_work_size, n_iter, self._system.dimensions), dtype=self._system.real_type)
            result_dev = allocate_on_device(queue, result)

            self.program.compute_grid(
                queue,
                tuple(global_work_size),
                None,
                numpy.int32(n_skip),
                numpy.int32(n_iter),
                *temporal_args,
                bounds_dev,
                params_dev,
                result_dev
            )

            cl.enqueue_copy(queue, result, result_dev)

            return result
        else:
            self.program.draw_phase_plane(
                queue,
                tuple(global_work_size or [1]),
                None,
                numpy.int32(n_skip),
                numpy.int32(n_iter),
                *temporal_args,
                bounds_dev,
                drawing_bounds_dev,
                params_dev,
                image.dev
            )

            return image
