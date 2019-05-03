import os

import numpy
import pyopencl as cl
import sys

from PyQt5.QtCore import Qt

from dynsys import ComputedImage, FLOAT, ParameterizedImageWidget, ParameterSurface

SCRIPT_DIR = os.path.abspath(sys.path[0])


def read_file(path):
    with open(path) as file:
        return file.read()


def random_seed():
    return numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),


def prepare_root_seq(ctx, root_seq):
    if root_seq is None:
        seq = numpy.empty((1,), dtype=numpy.int32)
        seq[0] = -1
    else:
        seq = numpy.array(root_seq, dtype=numpy.int32)

    seq_buf = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seq)
    return seq.size if root_seq is not None else 0, seq_buf


def make_simple_param_surface(ctx, queue, image_shape, h_bounds, alpha_bounds):
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


SOURCE = read_file(os.path.join(SCRIPT_DIR, "kernels.cl"))


class IFSFractal(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[]):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options, "-w",
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)

    def __call__(self, h: float, alpha: float, c: complex, grid_size: int, iterCount: int, skip: int,
                 z0=None, root_seq=None, clear_image=True):

        if clear_image:
            self.clear()

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.program.newton_fractal(
            self.queue, (grid_size, grid_size) if z0 is None else (1, 1), None,
            numpy.int32(skip),
            numpy.int32(iterCount),

            numpy.array(self.spaceShape, dtype=numpy.float64),

            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.float64(h),
            numpy.float64(alpha),

            numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),

            numpy.int32(seq_size),
            seq,

            numpy.int32(z0 is not None),
            numpy.array((0, 0) if z0 is None else (z0.real, z0.imag), dtype=numpy.float64),

            self.deviceImage
        )

        return self.readFromDevice()


def make_phase_plot(ctx, queue, image_shape, space_shape,):

    fr = IFSFractal(
        ctx, queue, image_shape, space_shape,
        fractalSource=SOURCE,
        options=[
            *"-I {}".format(os.path.join(SCRIPT_DIR, "include")).split(),
        ]
    )

    frw = ParameterizedImageWidget(
        space_shape, ("z_real", "z_imag"), shape=(True, True), textureShape=image_shape,
    )

    return fr, frw


class IFSFractalParameterMap(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[]):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options, "-w",
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)
        self.points: cl.Buffer = None

    def compute_points(self, z0: complex, c: complex, skip: int, iter: int, tol: float, root_seq=None,
                       wait=False, resolution=1, lossless=False):
        reqd_size = iter * numpy.prod(self.imageShape) // resolution ** 2

        elem_size = 16 if lossless else 8
        kernel = self.program.compute_points_lossless if lossless else self.program.compute_points

        if self.points is None or self.points.size != reqd_size:
            self.points = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                    size=reqd_size * elem_size)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        kernel(
            self.queue, (self.imageShape[0] // resolution, self.imageShape[1] // resolution), (1, 1),
            # z0
            numpy.array((z0.real, z0.imag), dtype=numpy.float64),
            # c
            numpy.array((c.real, c.imag), dtype=numpy.float64),
            # bounds
            numpy.array(self.spaceShape, dtype=numpy.float64),
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
            self.points
        )

        if wait:
            self.queue.finish()

    def display(self, num_points: int, resolution=1, lossless=False):
        color_scheme = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=4)

        if lossless:
            points_host = numpy.empty((self.imageShape[0] // resolution,
                                       self.imageShape[1] // resolution, num_points, 2),
                                      dtype=numpy.float64)
            cl.enqueue_copy(self.queue, points_host, self.points)

            periods = numpy.empty((self.imageShape[0] // resolution,
                                  self.imageShape[1] // resolution),
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
            periods_device = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=periods)
        else:
            periods = numpy.empty((self.imageShape[0] // resolution, self.imageShape[1] // resolution),
                                  dtype=numpy.int32)
            periods_device = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=periods.nbytes)

        kernel = self.program.draw_periods_lossless if lossless else self.program.draw_periods

        kernel(
            self.queue, self.imageShape, None,
            numpy.int32(resolution),
            numpy.int32(num_points),
            color_scheme,
            self.points,
            periods_device,
            self.deviceImage
        )

        if not lossless:
            cl.enqueue_copy(self.queue, periods, periods_device)

        return self.readFromDevice(), periods


def make_parameter_map(ctx, queue, image_shape, h_bounds, alpha_bounds):
    pm = IFSFractalParameterMap(
        ctx, queue, image_shape, (*h_bounds, *alpha_bounds),
        fractalSource=SOURCE,
        options=[
            *"-I {}".format(os.path.join(SCRIPT_DIR, "include")).split(),
        ]
    )

    pmw = ParameterizedImageWidget(
        bounds=(*h_bounds, *alpha_bounds),
        names=("h", "alpha"),
        shape=(True, True),
        textureShape=image_shape,
        targetColor=Qt.gray
    )

    return pm, pmw


class IFSFractalBasinsOfAttraction(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[]):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options, "-w",
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)
        self.points = numpy.empty((numpy.prod(imageShape), 2), dtype=numpy.float64)
        self.points_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=self.points.nbytes)

    def compute_points(self, h: float, alpha: float, c: complex, skip: int, root_seq=None, resolution=1):
        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.program.compute_basins(
            self.queue, (self.imageShape[0] // resolution, self.imageShape[1] // resolution), None,

            numpy.int32(skip),

            numpy.array(self.spaceShape, dtype=numpy.float64),

            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.float64(h),
            numpy.float64(alpha),

            numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),

            numpy.int32(seq_size),
            seq,

            self.points_dev
        )

    def draw_points(self, resolution=1, algo="c"):
        self.clear()

        if algo == "c":
            cl.enqueue_copy(self.queue, self.points, self.points_dev)
            unique_points = numpy.unique(self.points, axis=0)
            unique_points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=unique_points)

            print("Unique attraction points: {} / {}".format(unique_points.shape[0], self.points.shape[0]))

            self.program.draw_basins_colored(
                self.queue, self.imageShape, (1, 1),
                numpy.int32(resolution),
                numpy.int32(unique_points.shape[0]),
                unique_points_dev,
                self.points_dev,
                self.deviceImage
            )
        elif algo == "b":
            self.program.draw_basins(
                self.queue, self.imageShape, (1, 1),
                numpy.int32(resolution),
                numpy.array(self.spaceShape, dtype=numpy.float64),
                self.points_dev,
                self.deviceImage
            )
        else:
            raise ValueError("Unknown algo: \"{}\"".format(algo))

        return self.readFromDevice()


def make_basins(ctx, queue, image_shape, space_shape):
    fr = IFSFractalBasinsOfAttraction(
        ctx, queue, image_shape, space_shape,
        fractalSource=SOURCE,
        options=[
            *"-I {}".format(os.path.join(SCRIPT_DIR, "include")).split(),
        ]
    )

    frw = ParameterizedImageWidget(
        space_shape, ("z_real", "z_imag"), shape=(True, True), textureShape=image_shape,
        targetColor=Qt.gray
    )

    return fr, frw