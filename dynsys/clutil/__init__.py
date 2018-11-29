import numpy
import pyopencl as cl


def allocate_image(ctx: cl.Context, dim: tuple, flags=cl.mem_flags.WRITE_ONLY):
    #
    fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)
    return numpy.empty((*dim, 4), dtype=numpy.uint8), cl.Image(ctx, flags, fmt, shape=dim)
