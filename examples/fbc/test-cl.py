import numpy
import pyopencl as cl


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])


ctx = cl.Context(dev_type=cl.device_type.GPU)


def getKernel():
    knl = r"""
    kernel void f(
        const global int* data,
        local int* dataLocal,
        int value
    ) {
        printf("Things: %d %d %d\n", data[0], dataLocal[0], value);
    }
    
    """
    prg = cl.Program(ctx, knl).build(options=[dummyOption()])
    return prg.f


k = getKernel()
queue = cl.CommandQueue(ctx)


a = numpy.random.randint(0, 42, size=(1,), dtype=numpy.int32)
aD = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b = cl.LocalMemory(size=4)
c = numpy.int32(15)

print(a, type(a))

k(queue, (1,), None, aD, b, c)
