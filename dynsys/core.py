import logging
import sys

from typing import Iterable, Optional, Union, Tuple

import numpy
import pyopencl as cl

from PIL import Image


def create_context_and_queue(config: dict = None):
    """
    Creates OpenCL context and command queue.
    :return: ctx, queue
    """
    def alt(d: dict, *alternatives):
        for alt in alternatives:
            val = d.get(alt)
            if val is not None:
                return val
        raise RuntimeError("No alternative key were found in given dict (alt: {})".format(str(alternatives)))

    if config is None or config.get("autodetect"):
        ctx = cl.create_some_context(interactive=False)
        logging.info(f"Using auto-detected device: {ctx.get_info(cl.context_info.DEVICES)}")
    else:
        pl = cl.get_platforms()[alt(config, "pid", "platform", "platformId")]
        dev = pl.get_devices()[alt(config, "did", "device", "deviceId")]
        logging.info(f"Using specified device: {str(dev)}")
        ctx = cl.Context([dev])
    return ctx, cl.CommandQueue(ctx)


def send_to_device(queue, buf):
    return cl.Buffer(queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)


def allocate_on_device(queue, buf):
    return cl.Buffer(queue.context, cl.mem_flags.WRITE_ONLY, size=buf.nbytes)


def fetch_from_device(queue, dest, buf):
    cl.enqueue_copy(queue, dest, buf)
    return dest


def reallocate(
        ctx: cl.Context,
        buf: Optional[cl.Buffer],
        flags: cl.mem_flags,
        size: Union[int, Tuple[int]],
        shrink_threshold: float = 0.5,
):
    if buf is None:
        return cl.Buffer(ctx, flags, size)
    if buf.size < size or size <= shrink_threshold * buf.size:
        del buf
        return cl.Buffer(ctx, flags, size)
    return buf


def get_endianness(ctx: cl.Context):
    de = ((dev, dev.get_info(cl.device_info.ENDIAN_LITTLE)) for dev in ctx.get_info(cl.context_info.DEVICES))
    if all(map(lambda x: x[1], de)):
        return "little"
    if all(map(lambda x: not x[1], de)):
        return "big"
    return "both"


def alloc_image(ctx: cl.Context, dim: tuple, flags=cl.mem_flags.READ_WRITE):
    endianness = get_endianness(ctx)
    if endianness == "both":
        raise RuntimeError("Context has both little and big endian devices, which is not currently supported")
    elif endianness == sys.byteorder:
        order = cl.channel_order.BGRA
    else:
        if endianness == "little":
            order = cl.channel_order.BGRA
        else:
            order = cl.channel_order.ARGB
    fmt = cl.ImageFormat(order, cl.channel_type.UNORM_INT8)
    return numpy.empty((*dim, 4), dtype=numpy.uint8), cl.Image(ctx, flags, fmt, shape=dim)


def clear_image(queue, img, shape, color):
    cl.enqueue_fill_image(
        queue, img,
        color=numpy.array(color, dtype=numpy.float32), origin=(0,)*len(shape), region=shape
    )


def read_image(queue, host_img, dev_img, shape):
    cl.enqueue_copy(
        queue, host_img, dev_img, origin=(0,)*len(shape), region=shape
    )
    return host_img


def bgra2rgba(arr):
    arr.T[numpy.array((0, 1, 2, 3))] = arr.T[numpy.array((2, 1, 0, 3))]
    return arr


def make_img(arr):
    return Image.fromarray(bgra2rgba(arr), mode="RGBA")


class CLImage:

    host = property(lambda self: self.img[0])
    dev = property(lambda self: self.img[1])

    def __init__(self, ctx, shape):
        self.ctx = ctx
        self.img = alloc_image(ctx, shape)
        self.shape = shape

    def read(self, queue):
        read_image(queue, self.img[0], self.img[1], self.shape)
        return self.img[0]

    def clear(self, queue, color=(1.0, 1.0, 1.0, 1.0)):
        clear_image(queue, self.img[1], self.shape, color)

    def as_img(self):
        return make_img(self.host.reshape((*self.shape[::-1], self.host.shape[-1])))

    def save(self, path):
        self.as_img().save(path)
