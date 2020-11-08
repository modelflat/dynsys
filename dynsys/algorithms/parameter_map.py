from copy import copy
from typing import Tuple

import numpy
import pyopencl as cl

from cl import WithProgram, assemble, load_template
from core import CLImage, reallocate
from equation import EquationSystem, Parameters
from grid import Grid

SOURCE = load_template("parameter_map/kernels.cl")


UINT_SIZE = numpy.uint32(0).nbytes


class ParameterMap(WithProgram):

    def __init__(self, system: EquationSystem, ctx: cl.Context = None):
        super().__init__(ctx)
        self._system = system
        self._points_dev = None
        self._points = None
        self._periods_dev = None
        self._periods = numpy.empty((1,), dtype=numpy.int32)

    @property
    def system(self):
        return self._system

    def src(self, **kwargs):
        return assemble(SOURCE, system=self.system, **kwargs)

    def _allocate_points_buffer(self, shape: Tuple[int], n_iter: int, dimensions: int):
        new_size = dimensions * numpy.prod(shape) * n_iter * self.system.real_type(0).nbytes
        self._points_dev = reallocate(self.ctx, self._points_dev, cl.mem_flags.READ_WRITE, new_size)
        self._points = numpy.empty((*shape, n_iter, dimensions), dtype=self.system.real_type)

    def _allocate_periods_buffer(self, shape):
        new_size = numpy.prod(shape) * UINT_SIZE
        self._periods_dev = reallocate(self.ctx, self._periods_dev, cl.mem_flags.READ_WRITE, new_size)
        self._periods.resize(shape)

    def compute(
            self,
            queue: cl.CommandQueue,
            grid: Grid,
            init: Tuple,
            n_skip: int,
            n_iter: int,
            parameters: Parameters = None,
            tolerance: int = 3,
            infinity_check: float = 1e4,
            return_periods: bool = True,
            return_points: bool = False,
            image: CLImage = None,
            color_scheme: int = 0
    ):
        """
        Computes parameter map.

        :param queue: OpenCL command queue.
        :param grid: parameter plane topology. See ``Grid`` class for more info.
        :param init: initial point of iterative process.
        :param n_skip: number of iterations to skip. No limit, but consider the running time.
        :param n_iter: number of iterations to consider for period detection. No limit, but consider the running time.
        :param parameters: parameters for the system
        :param tolerance: which tolerance to use when attractors performing attractor detection. This is a floating
            point comparison precision, measured in decimals after floating point.
        :param infinity_check: maximum value allowed for a point coordinate. If any of a point's coordinates exceeds
            this value, point is considered to be "lost in the infinity".
        :param return_periods: whether to return array of detected periods.
        :param return_points: whether to capture and return points of iterative process. Using this with high values of
            ``n_iter`` can lead to high memory consumption.
        :param image: image to draw resulting parameter map onto.
        :param color_scheme: which color scheme to use. Supported values are ``0`` and ``1``.
        :return: nothing, unless ``return_points`` or ``return_periods`` options are specified.
        """
        if len(init) != self.system.dimensions:
            raise ValueError("Initial conditions should be the same size as system's dimensionality!")

        if len(grid.varied) != 2:
            raise ValueError(f"Parameter map can only have 2 varied parameters, got {len(grid.varied)}")

        self.compile(
            queue.context,
            template_variables={"varied": grid.varied},
            defines={"USE_OLD_COLORS": color_scheme}
        )

        self._allocate_periods_buffer(grid.shape)

        tolerance = 1 / 10 ** tolerance

        # TODO less hacky way to update parameters?
        params = copy(parameters) if parameters else Parameters()
        for name in grid.varied:
            params._values[name] = numpy.nan
        params = params.to_cl_object(self.system)
        params_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)

        args = (
            numpy.uint32(n_skip),
            numpy.uint32(n_iter),
            self.system.real_type(tolerance),
            self.system.real_type(infinity_check),
            numpy.array(init, dtype=self.system.real_type),
            *(
                numpy.array(bounds, dtype=self.system.real_type)
                for bounds in grid.bounds
            ),
            params_dev
        )

        if return_points:
            self._allocate_points_buffer(grid.shape, n_iter, self.system.dimensions)
            self.program.iterate_capture_with_periods(
                queue, grid.shape, None,
                *args, self._points_dev, self._periods_dev
            )
        else:
            self.program.iterate_with_periods(
                queue, grid.shape, None,
                *args, self._periods_dev
            )

        if image is not None:
            if grid.shape != image.shape:
                raise ValueError(f"Grid should be the same size as shape; got {grid.shape} vs {image.shape}")
            self.program.draw_map(
                queue, grid.shape, None,
                numpy.uint32(n_iter),
                self._periods_dev,
                image.dev
            )

        ret = []

        if return_periods:
            cl.enqueue_copy(queue, self._periods, self._periods_dev)
            ret.append(numpy.flip(self._periods, axis=0))

        if return_points:
            cl.enqueue_copy(queue, self._points, self._points_dev)
            ret.append(numpy.flip(self._points, axis=0))

        return ret[0] if len(ret) == 1 else (tuple(ret) or None)
