import os
import sys
import numpy
import pyopencl as cl
from dynsys import vStack, hStack


os.environ["PYOPENCL_NO_CACHE"] = "1"


def stack(*args, kind="v", cm=(0, 0, 0, 0), sp=0):
    if kind == "v":
        l = vStack(*args)
        l.setSpacing(sp)
        l.setContentsMargins(*cm)
    elif kind == "h":
        l = hStack(*args)
        l.setSpacing(sp)
        l.setContentsMargins(*cm)
    else:
        raise ValueError("Unknown kind of stack: \"{}\"".format(kind))
    return l


def clear_image(queue, img, shape, color=(1.0, 1.0, 1.0, 1.0)):
    cl.enqueue_fill_image(
        queue, img,
        color=numpy.array(color, dtype=numpy.float32), origin=(0,)*len(shape), region=shape
    )


def read_image(queue, host_img, dev_img, shape):
    cl.enqueue_copy(
        queue, host_img, dev_img, origin=(0,)*len(shape), region=shape
    )
    return host_img


def copy_dev(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)


def alloc_like(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=buf.nbytes)


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


SCRIPT_DIR = os.path.abspath(sys.path[0])

CL_SOURCE_PATH = os.path.join(SCRIPT_DIR, "cl")
CL_INCLUDE_PATH = os.path.join(CL_SOURCE_PATH, "include")
