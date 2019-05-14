from utils import *
from PyQt5.QtCore import Qt
from dynsys import FLOAT, ParameterizedImageWidget, ParameterSurface, allocateImage as alloc_image
from multiprocessing import Lock


class IFSFractal:

    def __init__(self, ctx, img_shape, include_dir, options=None):
        self.ctx = ctx
        self.img = alloc_image(self.ctx, img_shape)
        self.img_shape = img_shape
        self.prg = cl.Program(ctx, IFS_SOURCE).build(
            options=["-I", include_dir, *(options if options is not None else [])]
        )
        self.map_points = None
        self.basin_points = numpy.empty((numpy.prod(img_shape), 2), dtype=numpy.float64)
        self.basin_points_dev = alloc_like(self.ctx, self.basin_points)
        self.compute_lock = Lock()

    def draw_phase_portrait(self, queue, skip, iter, h, alpha, c,
                            bounds=(-1, 1, -1, 1), grid_size=1, z0=None, root_seq=None, clear=True):
        if clear:
            clear_image(queue, self.img[1], self.img_shape)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.newton_fractal(
            queue, (grid_size, grid_size) if z0 is None else (1, 1), None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=numpy.float64),

            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.float64(h),
            numpy.float64(alpha),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,

            numpy.int32(1 if z0 is not None else 0),
            numpy.array((0, 0) if z0 is None else (z0.real, z0.imag), dtype=numpy.float64),

            self.img[1]
        )

        return read_image(queue, *self.img, self.img_shape)

    def _compute_map(self, queue, skip, iter, z0, c, tol, param_bounds, root_seq, resolution, lossless, buf=None):

        elem_size = 16 if lossless else 8
        reqd_size = iter * numpy.prod(self.img_shape) // resolution ** 2

        kernel = self.prg.compute_points_lossless if lossless else self.prg.compute_points

        if buf is None and (self.map_points is None or self.map_points.size != reqd_size):
            self.map_points = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=reqd_size * elem_size)
            buf = self.map_points

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        kernel(
            queue, (self.img_shape[0] // resolution, self.img_shape[1] // resolution), (1, 1),
            # z0
            numpy.array((z0.real, z0.imag), dtype=numpy.float64),
            # c
            numpy.array((c.real, c.imag), dtype=numpy.float64),
            # bounds
            numpy.array(param_bounds, dtype=numpy.float64),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            numpy.float64(random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            buf
        )

    def _render_map(self, queue, num_points, resolution, lossless):
        color_scheme = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=4)

        if lossless:
            points_host = numpy.empty((self.img_shape[0] // resolution,
                                       self.img_shape[1] // resolution, num_points, 2),
                                      dtype=numpy.float64)
            cl.enqueue_copy(queue, points_host, self.map_points)

            periods = numpy.empty((self.img_shape[0] // resolution,
                                   self.img_shape[1] // resolution),
                                  dtype=numpy.int32)

            points_host = numpy.round(points_host, decimals=2)

            for i in range(periods.shape[0]):
                for j in range(periods.shape[1]):
                    un, cnt = numpy.unique(points_host[i][j], axis=0, return_counts=True)
                    periods[i][j] = un.shape[0]

            # uniques, un_counts = numpy.unique(points_host, axis=0, return_counts=True)
            # print(un_counts.shape)
            # print(un_counts)
            # TODO period detection here?

            # periods = un_counts.astype(numpy.int32)
            periods_device = copy_dev(self.ctx, periods)
        else:
            periods = numpy.empty((self.img_shape[0] // resolution,
                                   self.img_shape[1] // resolution),
                                  dtype=numpy.int32)
            periods_device = alloc_like(self.ctx, periods)

        kernel = self.prg.draw_periods_lossless if lossless else self.prg.draw_periods

        kernel(
            queue, self.img_shape, None,
            numpy.int32(resolution),
            numpy.int32(num_points),
            color_scheme,
            self.map_points,
            periods_device,
            self.img[1]
        )

        if not lossless:
            cl.enqueue_copy(queue, periods, periods_device)

        return read_image(queue, *self.img, self.img_shape), periods

    def draw_parameter_map(self, queue, skip, iter, z0, c, tol, param_bounds,
                           root_seq=None, resolution=1, lossless=False):
        self._compute_map(queue, skip, iter, z0, c, tol, param_bounds, root_seq, resolution, lossless)
        return self._render_map(queue, iter, resolution, lossless)

    def _compute_basins(self, queue, skip, h, alpha, c, bounds, root_seq, resolution):
        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_basins(
            queue, (self.img_shape[0] // resolution, self.img_shape[1] // resolution), None,

            numpy.int32(skip),
            numpy.array(bounds, dtype=numpy.float64),
            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.float64(h),
            numpy.float64(alpha),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,
            self.basin_points_dev
        )

    def _render_basins(self, queue, bounds, resolution, algo):
        clear_image(queue, self.img[1], self.img_shape)

        if algo == "c":
            cl.enqueue_copy(queue, self.basin_points, self.basin_points_dev)
            unique_points = numpy.unique(self.basin_points, axis=0)
            unique_points_dev = copy_dev(self.ctx, unique_points)

            print("Unique attraction points: {} / {}".format(unique_points.shape[0], self.basin_points.shape[0]))

            self.prg.draw_basins_colored(
                queue, self.img_shape, (1, 1),
                numpy.int32(resolution),
                numpy.int32(unique_points.shape[0]),
                unique_points_dev,
                self.basin_points_dev,
                self.img[1]
            )
        elif algo == "b":
            self.prg.draw_basins(
                queue, self.img_shape, (1, 1),
                numpy.int32(resolution),
                numpy.array(bounds, dtype=numpy.float64),
                self.basin_points_dev,
                self.img[1]
            )
        else:
            raise ValueError("Unknown algo: \"{}\"".format(algo))

        return read_image(queue, *self.img, self.img_shape)

    def draw_basins(self, queue, skip, h, alpha, c, bounds, root_seq=None, resolution=1, algo="c"):
        self._compute_basins(queue, skip, h, alpha, c, bounds, root_seq, resolution)
        return self._render_basins(queue, bounds, resolution, algo)


def make_param_placeholder(ctx, queue, image_shape, h_bounds, alpha_bounds):
    pm = ParameterSurface(ctx, queue, image_shape, (*h_bounds, *alpha_bounds),
                          colorFunctionSource=r"""
float3 userFn(real2 v);
float3 userFn(real2 v) {
    real h = v.x;
    real alpha = v.y;
    if (h > 0 && alpha > 0.5) {
        if (get_global_id(0) % 2 == 0 && get_global_id(1) % 2 == 0) {
            return 0.0f;
        }
    }
    if (fabs(alpha - 1) < 0.002 && h < 0) {
        return (float3)(1, 0, 0);
    }
    if (length(v - (real2)(1.0, 0.0)) < 0.01) {
        return (float3)(1, 0, 0);
    }
    if (length(v - (real2)(-1.0, 0.0)) < 0.01) {
        return (float3)(1, 0, 0);
    }
    return 1.0f;
}
""", typeConfig=FLOAT)
    return pm


def make_phase_wgt(space_shape, image_shape):
    return ParameterizedImageWidget(
        space_shape, ("z_real", "z_imag"), shape=(True, True), textureShape=image_shape,
    )


def make_param_wgt(h_bounds, alpha_bounds, image_shape):
    return ParameterizedImageWidget(
        bounds=(*h_bounds, *alpha_bounds),
        names=("h", "alpha"),
        shape=(True, True),
        textureShape=image_shape,
        targetColor=Qt.gray
    )


