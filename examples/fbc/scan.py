import numpy
import pyopencl as cl

# pyopencl already has an implementation of scan, but I really want to roll my own, minimal variant
# (also, my impl seems to run faster on HUGE inputs B-) )
from pyopencl.scan import InclusiveScanKernel
from pyopencl.array import to_device


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])


TYPES_SRC = r"""
#ifdef _DATA_TYPE
typedef _DATA_TYPE data_t;
#else 
#error _DATA_TYPE should be specified from host!
#endif

#ifdef _INDEX_TYPE
typedef _INDEX_TYPE idx_t;
#else 
typedef size_t idx_t;
#endif
"""


PREFIX_SUM_SRC = r"""

#ifndef CHUNK_SIZE
#error Specify CHUNK_SIZE as a compile-time constant!
#endif

#define WG_SIZE (CHUNK_SIZE >> 1)

// Parallel prefix sum (algorithm of Blelloch 1990)

inline void up_sweep(local data_t* temp, const int item) {
    size_t ai = (item << 1) + 1;
    size_t bi = (item + 1) << 1;
    
    temp[bi] += temp[ai];

    for (int d = WG_SIZE >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (item < d) {
            ai <<= 1;
            bi <<= 1;
            temp[bi] += temp[ai];
        }
    }
}

inline void down_sweep(local data_t* temp, const int item) {
    data_t swp;
    
    size_t ai = WG_SIZE * ((item << 1) + 1);
    size_t bi = WG_SIZE * ((item + 1) << 1);
    
    for (int d = 1; d < WG_SIZE; d <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (item < d) {
            swp      =  temp[ai];
            temp[ai] =  temp[bi];
            temp[bi] += swp;
        }
        
        ai >>= 1;
        bi >>= 1;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
        
    swp = temp[ai];
    temp[ai] = temp[bi];
    temp[bi] += swp;
}

// !!! ensure that all items in work group encounter calls to this function,
// !!! local barriers inside 
inline void par_prefix_sum(local data_t* temp, global data_t* sum) {
    const int item = get_local_id(0);
    
    // Move global sum pointer to current chunk position
    sum += get_group_id(0);
    
    // All accesses to the temp variable in this function will require this
    temp -= 1;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // up sweep phase
    up_sweep(temp, item);
    
    // store the last element in the global sum vector and clear the last element
    if (item == 0) {
        *sum = temp[CHUNK_SIZE]; temp[CHUNK_SIZE] = 0;
    }
    
    // down sweep phase
    down_sweep(temp, item);
    
    barrier(CLK_LOCAL_MEM_FENCE);
}
"""


SCAN_KERNEL_SRC = r"""
inline void load_or_0(int n, global data_t* in, int i, local data_t* out, int j) {
    if        (i + 1 < n) {
        out[j]     = in[i];
        out[j + 1] = in[i + 1];
    } else if (i < n) {
        out[j]     = in[i];
        out[j + 1] = 0;
    } else {
        out[j]     = 0;
        out[j + 1] = 0;
    }
}

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
kernel void in_scan(
    const  idx_t   input_size,
    global data_t *input,
    global data_t *sum
) {
    local data_t temp[CHUNK_SIZE];

    const idx_t gid2 = get_global_id(0) << 1;
    const int   lid2 = get_local_id(0) << 1;
    
    load_or_0(input_size, input, gid2, temp, lid2);
    par_prefix_sum(temp, sum);
    
    if        (gid2 + 1 < input_size) {
        input[gid2]     += temp[lid2];
        input[gid2 + 1] += temp[lid2 + 1];
    } else if (gid2 < input_size) {
        input[gid2]     += temp[lid2];
    }
}
"""


MERGE_KERNEL_SRC = r"""
kernel void merge(
    const idx_t sum_size,
    const global data_t *sum,
    global data_t *result
) {
    const idx_t res_idx = get_global_id(0);
    const idx_t sum_idx = res_idx / CHUNK_SIZE - 1;
    
    if (sum_idx < sum_size) {
        result[res_idx] += sum[sum_idx];
    }
}
"""


TYPE_MAP = {
    numpy.int16: "short",
    numpy.int32: "int",
    numpy.int64: "long",

    numpy.uint16: "ushort",
    numpy.uint32: "uint",
    numpy.uint64: "ulong"
}


class Scanner:

    def __init__(self, ctx,
                 element_type=numpy.uint32,
                 index_type=numpy.uint32,
                 chunk_size=None,
                 preallocate_iterations=1):
        """
        :param ctx: pyopencl.Context for which to build kernels/allocate buffers
        :param element_type: element type.
        :param index_type: type of index variables inside kernels. If None, size_t is used
        :param chunk_size: size of prefix sum chunk. If None, computed from device capabilities (local memory
        and work group size)
        :param preallocate_iterations: how many buffers to preallocate for scan iterations beginning from the last
        (will allocate buffers of size `chunk_size, chunk_size ** 2, ... , chunk_size ** preallocate_iterations`).
        """
        self.ctx = ctx

        if index_type is not None:
            if index_type not in TYPE_MAP.keys():
                raise NotImplementedError("Only {} types are supported".format(TYPE_MAP.keys()))
            index_type_opt = "-D_INDEX_TYPE={}".format(TYPE_MAP[index_type])
        else:
            index_type_opt = None

        if element_type not in TYPE_MAP.keys():
            raise NotImplementedError("Only {} types are supported".format(TYPE_MAP.keys()))
        else:
            element_type_opt = "-D_DATA_TYPE={}".format(TYPE_MAP[element_type])

        # TODO probably worth implementing different input/sum types, as types can rapidly overflow...
        # TODO find out how big libraries deal with it
        self.element_type = element_type
        self.element_size = element_type().nbytes

        self.max_local_items = min(map(lambda d: d.get_info(cl.device_info.MAX_WORK_GROUP_SIZE), self.ctx.devices))
        self.max_local_mem = min(map(lambda d: d.get_info(cl.device_info.LOCAL_MEM_SIZE), self.ctx.devices))

        if chunk_size is None:
            chunk_size = min(self.max_local_items * 2, self.max_local_mem // self.element_size)
        else:
            assert self.element_size * chunk_size <= self.max_local_mem, \
                "Local memory too small to fit chunk of size {} (max {})".format(chunk_size, self.max_local_mem // self.element_size)
            assert self.max_local_items >= chunk_size // 2, \
                "Max work group size ({}) should be >= chunkSize/2 ({})".format(self.max_local_items, chunk_size // 2)
        self.chunk_size = chunk_size

        self.prg = cl.Program(ctx, "\n".join([
            TYPES_SRC, PREFIX_SUM_SRC, SCAN_KERNEL_SRC, MERGE_KERNEL_SRC
        ])).build(
            options=list(filter(None, (element_type_opt, index_type_opt))) + [
                "-w", "-Werror", "-DCHUNK_SIZE={}".format(self.chunk_size), dummyOption()
            ]
        )

        self._local_buf = cl.LocalMemory(self.element_size * chunk_size)

        self._single_value_buf = cl.Buffer(
            self.ctx, flags=cl.mem_flags.WRITE_ONLY, size=self.element_size
        )

        # Pre-allocate buffers of various sizes to reduce allocations when calling scan
        # TODO guard against OOM?
        self._prealloc = []
        for i in range(preallocate_iterations):
            self._prealloc.append(cl.Buffer(
                self.ctx, flags=cl.mem_flags.READ_WRITE, size=self.element_size * self.chunk_size ** (i + 1)
            ))

    def _n_scans(self, size):
        if size <= 1:
            return 0
        return int(numpy.ceil(numpy.log(size) / numpy.log(self.chunk_size)))

    def _result_size(self, input_size):
        return input_size // self.chunk_size + (0 if input_size % self.chunk_size == 0 else 1)

    def _get_buffer(self, size: int):
        if size <= 0:
            raise RuntimeError("Buffer of invalid size requested")
        if size == 1:
            # size == 1 means that this will be a terminal stage of scan algorithm.
            # for that special case we have this buffer
            return self._single_value_buf
        if size <= self.chunk_size ** len(self._prealloc):
            # buffer of such size already allocated. buffer index in self._prealloc is:
            idx = int(numpy.floor(numpy.log(size - 1) / numpy.log(self.chunk_size)))
            # Note: pointer aliasing cannot happen here, because scan algorithm reduces problem size by
            # self.chunk_size on each step, so index will be different for each call of
            # _get_buffer() inside single scan() call
            return self._prealloc[idx]

        #
        return cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.element_size * size)

    def _merge(self, queue, sum_size, sum_buf, result_size, result_buf):
        self.prg.merge(
            queue, (result_size,), None,
            numpy.uint32(sum_size),
            sum_buf,
            result_buf,
            global_offset=(self.chunk_size,)
        )

    def _scan(self, queue, input_size, input_buf):
        output_size = self._result_size(input_size)
        output_buf = self._get_buffer(output_size)

        assert output_buf.int_ptr != input_buf.int_ptr

        local_size = self.chunk_size // 2

        if (input_size // 2) % local_size == 0:
            global_size = input_size // 2
        else:
            global_size = int(numpy.ceil(input_size / self.chunk_size)) * local_size

        print(self.prg.in_scan.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0]
        ))

        self.prg.in_scan(
            queue, (global_size,), (local_size,),
            numpy.int32(input_size),
            input_buf,
            output_buf
        )
        return output_size, output_buf

    def scan(self, queue: cl.CommandQueue, buffer: cl.Buffer, size: int = None) -> None:
        """
        Performs in-place parallel scan on given buffer.

        :param queue: queue to which to enqueue kernels.
        :param buffer: buffer to perform scan on.
        :param size: size of buffer. If None, computed as `pyopencl.Buffer.size // self.elementSize`
        """
        if size is None:
            assert buffer.size % self.element_size == 0
            size = buffer.size // self.element_size

        if size <= 1:
            return

        if size <= self.chunk_size:
            # buffer fits in local memory, just scan and return
            self._scan(queue, size, buffer)
            return

        if size <= self.chunk_size ** 2:
            # TODO insert some clever codepath here :)
            # for now, lets just hard-code what we have in general case, as the case with scan + merge is pretty common
            # when you have 2 or more levels of preallocated buffers :)
            res_size, res_buf = self._scan(queue, size, buffer)
            _,  total_sum_buf = self._scan(queue, res_size, res_buf)
            self._merge(queue, res_size, res_buf, size, buffer)
            return

        # based on the size determine number of scan calls
        num_scans = self._n_scans(size)

        # as we already handled case with only two scan calls, >= 3 scans should be performed then.
        # perform the first one:
        res_size, res_buf = self._scan(queue, size, buffer)

        # the rest of scans we perform in loop and save results
        results = [(size, buffer), (res_size, res_buf)]
        for _ in range(num_scans - 1):
            res_size, res_buf = self._scan(queue, res_size, res_buf)
            results.append((res_size, res_buf))

        # finally, merge buffers in reverse order
        results.reverse()
        for p1, p2 in zip(results, results[1:]):
            self._merge(queue, *p1, *p2)


def create_random_buffer(size, dtype):
    input = numpy.random.randint(0, 2, size=(size,), dtype=dtype)
    input_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=input)
    return input, input_buf


def test_small_sizes(ctx, queue):
    scanner = Scanner(ctx, chunk_size=32)

    for size in range(1, scanner.chunk_size + 1):
        b, bb = create_random_buffer(size, scanner.element_type)

        scanner.scan(queue, bb)
        queue.finish()

        res = b.copy()
        cl.enqueue_copy(queue, res, bb)

        assert numpy.allclose(numpy.cumsum(b), res), "Failed for size {}".format(size)


def test_arbitrary_sizes(ctx, queue):
    scanner = Scanner(ctx, chunk_size=32)

    for i in range(500):
        size = numpy.random.randint(scanner.chunk_size, scanner.chunk_size ** 4)
        # print("Running validation for size {}...".format(size))
        input = numpy.random.randint(0, 2, size=(size,), dtype=numpy.uint32)
        input_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=input)

        scanner.scan(queue, input_buf)
        queue.finish()

        prefix = input.copy()
        cl.enqueue_copy(queue, prefix, input_buf)

        valid = numpy.allclose(numpy.cumsum(input), prefix)

        assert valid


def benchmark(ctx, queue, size, count, impl, silent=False):
    import time
    scanner = Scanner(ctx, element_type=numpy.uint32, preallocate_iterations=2)
    knl = InclusiveScanKernel(ctx, numpy.int32, "a + b", neutral="0", options=[
        dummyOption()
    ])

    results = []

    for i in range(count):
        input = numpy.random.randint(0, 2, size=(size,), dtype=scanner.element_type)
        input_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=input.nbytes)
        cl.enqueue_copy(queue, input_buf, input)
        dev_data = to_device(queue, input)
        queue.finish()
        tm = time.perf_counter()
        #
        if impl == "my":
            scanner.scan(queue, input_buf)
        else:
            knl(dev_data)
        queue.finish()
        #
        tm = time.perf_counter() - tm
        queue.finish()

        results.append(tm * 1000)
        # time.sleep(0.08)

    av = sum(results) / len(results)
    mn = min(results)
    mx = max(results)

    avsq = sum(map(lambda x: x*x, results)) / len(results)

    stddev = (avsq - av**2) ** .5

    if not silent:
        print("Bench for size {:10d} : ave: {:8.3f}; min: {:8.3f}; max: {:8.3f}, std: +/-{:.3f} | {:10.3f} MiB/s throughput".format(
            size, av, mn, mx, stddev, scanner.element_size * size * count / sum(results) * 1000 / 2 ** 20
        ))

    return size, av, mn, mx, stddev, scanner.element_size * size * count / sum(results) * 1000 / 2 ** 20


def big_bench(impl, output_file=None):
    bench_runs = dict()

    runs = 3
    for run in range(runs):
        print("run", run)
        for i in range(12, 26):
            sz, av, _, _, _, _ = benchmark(ctx, queue, (1 << i), count=20, impl=impl, silent=False)
            bench_runs[sz] = bench_runs.get(sz, 0) + av

    print("Bench '{}' completed!".format(impl))
    if output_file is None:
        output_file = "{}.txt".format(impl)
    with open(output_file, "w") as f:
        for sz, tm in bench_runs.items():
            print("{:10d} {:10.3f}".format(sz, tm / runs), file=f)


if __name__ == '__main__':
    ctx = cl.Context(devices=(cl.get_platforms()[0].get_devices()[0],))
    queue = cl.CommandQueue(ctx)

    # test_small_sizes(ctx, queue)
    # test_arbitrary_sizes(ctx, queue)



    big_bench("pc")
    # big_bench("my")
