import pyopencl as cl
import numpy

from dynsys.LCE import dummyOption
from ifs_fractal import SCRIPT_DIR, SOURCE

test_kernel = r"""

kernel void test_heap_sort(
    int n,
    global ulong* data
) {
    heap_sort(data, n);
}

"""

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context(devices=[dev])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, SOURCE + test_kernel).build(options=["-I", SCRIPT_DIR + "/include", dummyOption()])

n = 32
arr = numpy.random.randint(
    0, high=32#2**64-1
    , size=n, dtype=numpy.uint64
)
arr_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

prg.test_heap_sort(
    queue, (1,), None, numpy.int32(n), arr_dev
)

arr_res = arr.copy()
cl.enqueue_copy(queue, arr_res, arr_dev)

print(arr)
print(arr_res)
