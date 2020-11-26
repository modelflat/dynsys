import warnings
from typing import Tuple, Callable, List, NamedTuple

import numpy
import pyopencl as cl

from dynsys.core import CLImage, reallocate
from dynsys.cl import WithProgram, assemble, load_template
from dynsys.equation import EquationSystem, Parameters
from dynsys.grid import Grid


SOURCE = load_template("attractors/kernels.cl")


UINT_SIZE = numpy.uint32(0).nbytes


class Attractor(NamedTuple):
    count: int
    hash: int
    period: int
    values: numpy.ndarray


class Attractors(WithProgram):

    def __init__(self, system: EquationSystem, ctx: cl.Context = None):
        super().__init__(ctx)
        self._system = system
        self._points_dev = None
        self._periods_dev = None
        self._sequence_hashes_dev = None
        self._table_dev = None
        self._table_data_dev = None

    @property
    def system(self):
        return self._system

    def src(self, **kwargs):
        return assemble(SOURCE, system=self.system, **kwargs)

    def _allocate_buffers(self, shape: Tuple[int], n_iter: int, dimensions: int):
        new_size = dimensions * numpy.prod(shape) * n_iter * self.system.real_type(0).nbytes
        self._points_dev = reallocate(self.ctx, self._points_dev, cl.mem_flags.READ_WRITE, new_size)
        new_size_periods = numpy.prod(shape) * UINT_SIZE
        self._periods_dev = reallocate(self.ctx, self._periods_dev, cl.mem_flags.READ_WRITE, new_size_periods)

    def _allocate_hash_table(self, queue: cl.CommandQueue, shape: Tuple[int], table_size: int):
        self._sequence_hashes_dev = reallocate(
            self.ctx, self._sequence_hashes_dev, cl.mem_flags.READ_WRITE, size=UINT_SIZE * numpy.prod(shape)
        )
        new_size = UINT_SIZE * table_size
        self._table_dev = reallocate(self.ctx, self._table_dev, cl.mem_flags.READ_WRITE, new_size)
        self._table_data_dev = reallocate(self.ctx, self._table_data_dev, cl.mem_flags.READ_WRITE, new_size)

        cl.enqueue_fill_buffer(queue, self._table_dev, numpy.uint32(0), 0, new_size)
        cl.enqueue_fill_buffer(queue, self._table_data_dev, numpy.uint32(0), 0, new_size)

    def _capture_points_with_periods(
            self,
            queue: cl.CommandQueue,
            grid: Grid,
            n_skip: int,
            n_iter: int,
            tolerance: float,
            infinity_check: float,
            parameters: Parameters
    ):
        self._allocate_buffers(grid.shape, n_iter, dimensions=self.system.dimensions)

        params = parameters.to_cl_object(self.system)
        params_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)

        self.program.iterate_capture_with_periods(
            queue, grid.shape, None,
            numpy.int32(n_skip),
            numpy.int32(n_iter),
            self.system.real_type(tolerance),
            self.system.real_type(infinity_check),
            grid.init(self.system.real_type),
            *[
                numpy.array(bounds, dtype=self.system.real_type)
                for bounds in grid.bounds
            ],
            params_dev,
            self._points_dev,
            self._periods_dev
        )

    def _find_attractors(
            self,
            queue: cl.CommandQueue,
            shape: Tuple[int],
            n_iter: int,
            table_size: int,
            check_collisions: bool,
            tolerance: float
    ):
        n_points = numpy.prod(shape)
        table_size = max(table_size or (n_points * 2 - 1), 8191)

        if table_size > 2 ** 32:
            # TODO support
            raise ValueError("Table sizes >= 4 GB are not supported")

        self._allocate_hash_table(queue, shape, table_size)

        period_counts = numpy.zeros((n_iter,), dtype=numpy.uint32)
        period_counts_dev = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=period_counts
        )

        self.program.round_points(
            queue, (n_points * n_iter,), None,
            self.system.real_type(tolerance),
            self._points_dev
        )

        self.program.rotate_sequences(
            queue, (n_points,), None,
            numpy.int32(n_iter),
            self.system.real_type(tolerance),
            self._periods_dev,
            self._points_dev
        )

        self.program.hash_sequences(
            queue, (n_points,), None,
            numpy.uint32(n_iter),
            numpy.uint32(table_size),
            self._periods_dev,
            self._points_dev,
            self._sequence_hashes_dev
        )

        self.program.count_unique_sequences(
            queue, (n_points,), None,
            numpy.uint32(n_iter),
            numpy.uint32(table_size),
            self._periods_dev,
            self._sequence_hashes_dev,
            self._table_dev,
            self._table_data_dev
        )

        if check_collisions:
            collisions = numpy.zeros((1,), dtype=numpy.uint32)
            collisions_dev = cl.Buffer(
                self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=collisions
            )

            self.program.check_collisions(
                queue, (n_points,), None,
                self.system.real_type(tolerance),
                numpy.uint32(n_iter),
                numpy.uint32(table_size),
                self._periods_dev,
                self._sequence_hashes_dev,
                self._points_dev,
                self._table_dev,
                self._table_data_dev,
                collisions_dev
            )

            cl.enqueue_copy(queue, collisions, collisions_dev)
            n_collisions = int(collisions[0])

            if n_collisions != 0:
                warnings.warn(
                    "Hash collisions were detected while computing basins of attraction.\n"
                    "This may result in non-deterministic results and/or incorrect counts "
                    "of sequences of some of the periods.\nIn some cases, setting `table_size` "
                    f"manually might help. Current `table_size` is {table_size}."
                )
        else:
            n_collisions = None

        self.program.count_periods_of_unique_sequences(
            queue, (table_size,), None,
            numpy.uint32(n_iter),
            self._periods_dev,
            self._table_dev,
            self._table_data_dev,
            period_counts_dev
        )

        cl.enqueue_copy(queue, period_counts, period_counts_dev)

        hash_positions = numpy.concatenate(
            (
                # shift array by one position to get exclusive cumsum
                numpy.zeros((1,), dtype=numpy.uint32),
                numpy.cumsum(period_counts, axis=0, dtype=numpy.uint32)[:-1]
            )
        )
        hash_positions_dev = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hash_positions
        )

        volume_per_period = numpy.arange(1, n_iter) * period_counts[:-1]
        sequence_positions = numpy.concatenate(
            (
                numpy.zeros((1,), dtype=numpy.uint32),
                numpy.cumsum(volume_per_period, axis=0, dtype=numpy.uint32)[:-1]
            )
        )
        sequence_positions_dev = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sequence_positions
        )

        current_positions = numpy.zeros((n_iter,), dtype=numpy.uint32)
        current_positions_dev = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_positions
        )

        total_points_count = sum(numpy.arange(1, n_iter) * period_counts[:-1])
        if total_points_count == 0:
            # no attractors were found at all
            return [], n_collisions

        unique_sequences = numpy.empty((total_points_count, self.system.dimensions), dtype=self.system.real_type)
        unique_sequences_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=unique_sequences.nbytes)

        # hashes and counts
        unique_sequences_info = numpy.empty((sum(period_counts[:-1]), 2), dtype=numpy.uint32)
        unique_sequences_info_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=unique_sequences_info.nbytes)

        self.program.gather_unique_sequences(
            queue, (table_size,), None,
            numpy.uint32(n_iter),
            self._periods_dev,
            self._points_dev,
            self._table_dev,
            self._table_data_dev,
            period_counts_dev,
            hash_positions_dev,
            sequence_positions_dev,
            current_positions_dev,
            unique_sequences_dev,
            unique_sequences_info_dev
        )

        cl.enqueue_copy(queue, unique_sequences, unique_sequences_dev)
        cl.enqueue_copy(queue, unique_sequences_info, unique_sequences_info_dev)

        raw_attractors = []
        current_pos = 0
        current_pos_info = 0
        for period, n_sequences in enumerate(period_counts[:-1], start=1):
            if n_sequences == 0:
                continue
            sequences = unique_sequences[current_pos:current_pos + n_sequences * period] \
                .reshape((n_sequences, period, self.system.dimensions))
            current_pos += n_sequences * period
            info = unique_sequences_info[current_pos_info:current_pos_info + n_sequences] \
                .reshape((n_sequences, 2))
            current_pos_info += n_sequences

            raw_attractors.append((int(period), sequences, *info.T))

        return raw_attractors, n_collisions

    def _color_attractors(
            self,
            queue: cl.CommandQueue,
            image: CLImage,
            attractors: List[Attractor],
            color_fn: Callable
    ):
        # TODO should use raw results to avoid doing this
        attractors.sort(key=lambda a: a.hash)

        hashes = numpy.array([attractor.hash for attractor in attractors], dtype=numpy.uint32)
        colors = numpy.array([color_fn(attractor) for attractor in attractors], dtype=numpy.float32)

        # TODO can we avoid searching inside this kernel? probably reuse a hash table
        hashes_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hashes)
        colors_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=colors)

        self.program.color_attractors(
            queue, image.shape, None,
            numpy.int32(len(attractors)),
            hashes_dev,
            colors_dev,
            self._sequence_hashes_dev,
            image.dev
        )

    def find_attractors(
            self,
            queue: cl.CommandQueue,
            grid: Grid,
            parameters: Parameters,
            n_skip: int,
            n_iter: int,
            tolerance: int = 3,
            infinity_check: float = 1e4,
            table_size: int = None,
            check_collisions: bool = False,
            occurrence_threshold: int = 100,
            image: CLImage = None,
            color_fn: Callable = None,
    ):
        """
        Finds attractors in an ``EquationSystem``.

        :param queue: OpenCL command queue.
        :param grid: phase plane topology. See ``Grid`` class for more info.
        :param n_skip: number of iterations to skip. No limit, but consider the running time.
        :param n_iter: number of iterations to consider for period detection. Current implementation limits this to 255.
        :param parameters: parameters for the system
        :param tolerance: which tolerance to use when attractors performing attractor detection. This is a floating
            point comparison precision, measured in decimals after floating point.
        :param infinity_check: maximum value allowed for a point coordinate. If any of a point's coordinates exceeds
            this value, point is considered to be "lost in the infinity", and no further analysis is done on it.
        :param table_size: size of hash table to use. Defaults to 2*N - 1, where N is the number of points in ``grid``.
        :param check_collisions: whether to check for hash collisions after hashing step. Currently algorithm outputs
            a warning message if any collisions were found.
        :param occurrence_threshold: the least number of times a sequence of points should be encountered for it to be
            considered as an attractor. Sequences which failed to met the threshold does not appear in the output.
        :param image: image to draw attractors onto.
        :param color_fn: color function to use for coloring attractors.
        :return: a list of attractors.
        """
        if n_iter >= 256:
            # TODO support. this is due to how sequence rotation currenly works
            raise ValueError("n_iter values >= 256 are not supported")

        tolerance = 1 / 10 ** tolerance

        self.compile(
            queue.context,
            defines={"ROTATION_MAX_ITER": max(n_iter, 16)},
            template_variables={"varied": grid.varied}
        )

        self._capture_points_with_periods(
            queue, grid, n_skip, n_iter, tolerance, infinity_check, parameters
        )

        # TODO add an option to raise exception if n_collisions != 0?
        raw_attractors, _n_collisions = self._find_attractors(
            queue, grid.shape, n_iter, table_size, check_collisions, tolerance
        )

        # TODO the following code is really slow compared to _find_attractors. Is there any way to speed it up?
        attractors = []
        for period, sequences, hashes, counts in raw_attractors:
            mask = counts > occurrence_threshold
            sequences = sequences[mask]
            hashes = hashes[mask]
            counts = counts[mask]
            for sequence, hash, count in zip(sequences, hashes, counts):
                attractors.append(Attractor(
                    hash=hash,
                    count=count,
                    period=period,
                    values=sequence,
                ))

        if image is not None and color_fn is not None:
            if image.shape != grid.shape:
                raise ValueError("Image shape should be the same as grid shape")
            self._color_attractors(queue, image, attractors, color_fn)

        return attractors
