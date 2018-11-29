import numpy
import pyopencl as cl
import sys

from .codegen import makeSource


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


def _getAlternatives(d: dict, *alternatives):
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
        pl = cl.get_platforms()[_getAlternatives(jsonConfig, "pid", "platform", "platformId")]
        dev = pl.get_devices() [_getAlternatives(jsonConfig, "did", "device", "deviceId")]
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
        self.varType = TypeConfig.TYPES[paramType] if varType is not None else TypeConfig.TYPES[realType]
        self.paramType = TypeConfig.TYPES[paramType] if paramType is not None else TypeConfig.TYPES[realType]
        self.boundsType = TypeConfig.TYPES[boundsType] if boundsType is not None else TypeConfig.TYPES[realType]

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


def wrapParameterArgs(total_params, params, type, active_idx=None):
    if total_params < len(params):
        params = params[:total_params] # todo raise warning?
    if total_params == len(params):
        return list(map(type, params))
    if total_params - 1 == len(params) and active_idx is not None:
        return list(map(type, params[:active_idx])) + [type(0.0),] + list(map(type, params[active_idx+1:]))
    raise ValueError("Out of %d arguments, only %d were provided." % (total_params, len(params)))


class ComputedImage:

    def __init__(self, ctx, queue, imageShape, spaceShape, *sources, typeConfig):
        self.ctx, self.queue, self.tc = ctx, queue, typeConfig
        self.imageShape = imageShape
        self.spaceShape = spaceShape
        self.hostImage, self.deviceImage = allocateImage(ctx, imageShape)
        src = makeSource(*sources, typeConfig=typeConfig)
        self.program = cl.Program(ctx, src).build(["-DDIM={}".format(len(imageShape))])

    def clear(self, readBack=False, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(self.queue, self.deviceImage,
                              color=numpy.array(color),
                              origin=(0,)*len(self.imageShape),
                              region=self.imageShape)
        if readBack:
            cl.enqueue_copy(self.queue, self.hostImage, self.deviceImage,
                            origin=(0,)*len(self.imageShape),
                            region=self.imageShape)

    def readFromDevice(self, queue=None):
        cl.enqueue_copy(queue if queue is not None else self.queue,
                        self.hostImage, self.deviceImage,
                        origin=(0,)*len(self.imageShape),
                        region=self.imageShape)
        return self.hostImage
