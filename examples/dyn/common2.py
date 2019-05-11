import os

import pyopencl as cl
import numpy


os.environ["PYOPENCL_NO_CACHE"] = "1"


def clear_image(queue, img, shape, color=(1.0, 1.0, 1.0, 1.0)):
    cl.enqueue_fill_image(
        queue, img,
        color=numpy.array(color, dtype=numpy.float32), origin=(0,)*len(shape), region=shape
    )


def read(queue, host_img, dev_img, shape):
    cl.enqueue_copy(
        queue, host_img, dev_img, origin=(0,)*len(shape), region=shape
    )
    return host_img


def make_type(ctx, type_name, type_desc, device=None):
    """
    :return: CL code generated for given type and numpy.dtype instance
    """
    import pyopencl.tools
    dtype, cl_decl = cl.tools.match_dtype_to_c_struct(
        ctx.devices[0] if device is None else device, type_name, numpy.dtype(type_desc), context=ctx
    )
    type_def = cl.tools.get_or_register_dtype(type_name, dtype)
    return cl_decl, type_def


def copy_dev(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)


def alloc_like(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=buf.nbytes)

