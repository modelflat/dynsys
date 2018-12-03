import numpy
import pyopencl as cl
import sys

from typing import Union

from .CodeGen import makeSource


def getEndianness(ctx: cl.Context):
    de = ((dev, dev.get_info(cl.device_info.ENDIAN_LITTLE)) for dev in ctx.get_info(cl.context_info.DEVICES))
    if all(map(lambda x: x[1], de)):
        return "little"
    if all(map(lambda x: not x[1], de)):
        return "big"
    return "both"


def allocateImage(ctx: cl.Context, dim: tuple, flags=cl.mem_flags.WRITE_ONLY):
    endianness = getEndianness(ctx)
    if endianness == "both":
        raise RuntimeError("Context has both little and big endian devices, which is not currently supported")
    elif endianness == sys.byteorder:
        chOrder = cl.channel_order.BGRA
    else:
        if endianness == "little":
            chOrder = cl.channel_order.BGRA
        else:
            chOrder = cl.channel_order.ARGB
    fmt = cl.ImageFormat(chOrder, cl.channel_type.UNORM_INT8)
    return numpy.empty((*dim, 4), dtype=numpy.uint8), cl.Image(ctx, flags, fmt, shape=dim)


def getAlternatives(d: dict, *alternatives):
    for alt in alternatives:
        val = d.get(alt)
        if val is not None:
            return val
    raise RuntimeError("No alternative key were found in given dict (alt: {})".format(str(alternatives)))


def createContextAndQueue(jsonConfig: dict = None):
    if jsonConfig is None or jsonConfig.get("autodetect"):
        ctx = cl.create_some_context(interactive=False)
        print("Using auto-detected device:", ctx.get_info(cl.context_info.DEVICES))
    else:
        pl = cl.get_platforms()[getAlternatives(jsonConfig, "pid", "platform", "platformId")]
        dev = pl.get_devices() [getAlternatives(jsonConfig, "did", "device", "deviceId")]
        print("Using specified device:", dev)
        ctx = cl.Context([dev])
    return ctx, cl.CommandQueue(ctx)


class TypeConfig:

    TYPES = {
        numpy.float16: "half",
        numpy.float32: "float",
        numpy.float64: "double"
    }

    def __init__(self, realType, varType=None, paramType=None, boundsType=None):
        self.realType = realType
        self.realTypeName = TypeConfig.TYPES[self.realType]
        self.varType = varType if varType is not None else realType
        self.varTypeName = TypeConfig.TYPES[self.varType]
        self.paramType = paramType if paramType is not None else realType
        self.paramTypeName = TypeConfig.TYPES[self.paramType]
        self.boundsType = boundsType if boundsType is not None else realType
        self.boundsTypeName = TypeConfig.TYPES[self.boundsType]

    def real(self, arg=None):
        return self.realType(arg) if arg is not None else self.realType

    def realSize(self):
        return numpy.dtype(self.realType).itemsize

    def cl(self):
        return """
        // type config
        #define real {}
        #define real2 {}2
        #define real3 {}3
        #define real4 {}4
        #define real6 {}6
        #define convert_real convert_{}
        #define convert_real2 convert_{}2
        #define convert_real3 convert_{}3
        #define convert_real4 convert_{}4
        //
        """.format(*[TypeConfig.TYPES[self.realType], ] * 9)

    def __call__(self):
        return self.realType, self.realSize()


HALF =   TypeConfig(numpy.float16)
FLOAT =  TypeConfig(numpy.float32)
DOUBLE = TypeConfig(numpy.float64)


class ComputedImage:

    def __init__(self, ctx: cl.Context, queue: cl.CommandQueue,
                 imageShape: Union[tuple, int], spaceShape: tuple,
                 *sources: str,
                 typeConfig: TypeConfig):
        self.ctx, self.queue, self.tc = ctx, queue, typeConfig
        self.imageShape = imageShape
        self.spaceShape = spaceShape
        self.hostImage, self.deviceImage = allocateImage(ctx, imageShape)
        src = makeSource(*sources, typeConfig=typeConfig)
        self.program = cl.Program(ctx, src).build(["-DDIM={}".format(len(imageShape))])

    def wrapArgs(self, requiredArgCount, *args, skipIndex=None):
        if requiredArgCount < len(args):
            args = args[:args]
            raise RuntimeWarning(
                "wrapArgs: {} arguments is required, but {} supplied. Taking first {} to process".format(
                    requiredArgCount, len(args), requiredArgCount
                ))
        if requiredArgCount == len(args):
            return list(map(self.tc.realType, args))
        if requiredArgCount - 1 == len(args) and skipIndex is not None:
            return list(map(self.tc.realType, args[:skipIndex])) \
                   + [self.tc.realType(0.0),] \
                   + list(map(self.tc.realType, args[skipIndex+1:]))
        # not enough args
        raise ValueError("Out of %d arguments, only %d were provided." % (requiredArgCount, len(args)))

    def clear(self, readBack=False, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(
            self.queue, self.deviceImage,
            color=numpy.array(color, dtype=numpy.float32),
            origin=(0,)*len(self.imageShape), region=self.imageShape
        )
        if readBack:
            self.readFromDevice()

    def readFromDevice(self):
        cl.enqueue_copy(
            self.queue, self.hostImage, self.deviceImage,
            origin=(0,)*len(self.imageShape), region=self.imageShape
        )
        return self.hostImage


class Bounds:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def clamp_x(self, v):
        return numpy.clip(v, self.x_min, self.x_max)

    def from_integer_x(self, v, v_max):
        return self.x_min + v / v_max * (self.x_max - self.x_min)

    def clamp_y(self, v):
        return numpy.clip(v, self.y_min, self.y_max)

    def from_integer_y(self, v, v_max, invert=True):
        return self.y_min + ((v_max - v) if invert else v)/ v_max * (self.y_max - self.y_min)

    def to_integer(self, x, y, w, h, invert_y=True):
        y_val = int((y - self.y_min) / (self.y_max - self.y_min) * h)
        return (
            int((x - self.x_min) / (self.x_max - self.x_min) * w),
            y_val if not invert_y else h - y_val
        )

    def asTuple(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    @staticmethod
    def x(x_min, x_max):
        return Bounds(x_min, x_max, 0, 0)

    @staticmethod
    def y(y_min, y_max):
        return Bounds(0, 0, y_min, y_max)
