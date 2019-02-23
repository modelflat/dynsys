import numpy
import pyopencl as cl

_src_other = r"""
#define SIZE_TYPE size_t
"""

_src_kernelScan = r"""
__attribute__((vec_type_hint(KEY_TYPE)))
kernel void merge(
    const global KEY_TYPE *sum,
    global KEY_TYPE *histogram
) {
    const KEY_TYPE s = sum[ get_group_id(0) ];
    histogram += get_global_id(0) << 1;
    histogram[0] += s;
    histogram[1] += s;
}

__attribute__((vec_type_hint(KEY_TYPE)))
kernel void scan(
    global KEY_TYPE *input,
    global KEY_TYPE *sum,
    local  KEY_TYPE *temp
) {
    const uint gid2  = get_global_id(0) << 1;
    const uint group = get_group_id(0);
    const uint item  = get_local_id(0);
    const uint n     = get_local_size(0) << 1;
    
    temp[ item << 1 ]     = input[gid2];
    temp[(item << 1) + 1] = input[gid2 + 1];
    
    // parallel prefix sum (algorithm of Blelloch 1990)
    int decale = 1;
    // up sweep phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (item < d) {
            const int ai = decale * ((item << 1) + 1) - 1;
            const int bi = decale * ((item << 1) + 2) - 1;
            temp[bi] += temp[ai];
        }
        
        decale <<= 1;
    }
    
    // store the last element in the global sum vector and clear the last element
    if (item == 0) {
        sum[group] = temp[n - 1];
        temp[n - 1] = 0;
    }
    
    // down sweep phase
    for (int d = 1; d < n; d <<= 1) {
        decale >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (item < d) {
            const int ai = decale * ((item << 1) + 1) - 1;
            const int bi = decale * ((item << 1) + 2) - 1;
            const int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    input[gid2]     = temp[ item << 1 ];
    input[gid2 + 1] = temp[(item << 1) + 1];
}
"""

_src_kernelHistogramAndReorder = r"""

#define EXTRACT_DIGIT(key, pass) \
    (((key) >> ((pass) * _BITS)) & (_RADIX - 1))

__attribute__((vec_type_hint(KEY_TYPE)))
kernel void histogram(
    const uint length,
    const uint pass,
    const global KEY_TYPE *keys,
    global KEY_TYPE *histograms,
    local  KEY_TYPE *localHistograms
) {
    const uint item  = get_local_id(0);
    const uint group = get_group_id(0);
    const uint size  = length / (_GROUPS * _ITEMS);
    
    // adjust pointers to const positions
    localHistograms += item * _RADIX;
    keys            += get_global_id(0) * size;
    histograms      += _ITEMS * group + item;
    // histograms      += _RADIX * group * _ITEMS + item * _GROUPS;
    
    for (int i = 0; i < _RADIX; ++i) {
        localHistograms[i] = 0; 
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (SIZE_TYPE i = 0; i < size; ++i) {
        ++localHistograms[EXTRACT_DIGIT(keys[i], pass)];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int i = 0; i < _RADIX; ++i) {
        histograms[i * _GROUPS * _ITEMS] = localHistograms[i];
        // histograms[i] = localHistograms[i];
    }
}

__attribute__((vec_type_hint(KEY_TYPE)))
kernel void reorder(
// in
    const uint length,
    const uint pass,
    const global KEY_TYPE *keysIn,
    const global KEY_TYPE *histograms,
// out
    global KEY_TYPE *keysOut,
// local
    local KEY_TYPE *localHistograms
) {
    const uint group = get_group_id(0);
    const uint item  = get_local_id(0);
    const uint size  = length / (_GROUPS * _ITEMS);
    
    // adjust pointers to const positions
    localHistograms += item * _RADIX;
    keysIn          += get_global_id(0) * size;
    histograms      += _ITEMS * group + item;
    // histograms      += _RADIX * group * _ITEMS + item * _GROUPS;
    
    for (int i = 0; i < _RADIX; ++i) {
        localHistograms[i] = histograms[i * _GROUPS * _ITEMS];
        // localHistograms[i] = histograms[i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (SIZE_TYPE i = 0; i < size; ++i) {
        const KEY_TYPE key = keysIn[i];
        keysOut[localHistograms[EXTRACT_DIGIT(key, pass)]++] = key;
    }
}
"""


def determineSizeT(ctx):
    types = {
        32: numpy.uint32,
        64: numpy.uint64
    }
    bitness = numpy.unique(list(map(lambda d: d.get_info(cl.device_info.ADDRESS_BITS), ctx.devices)))
    if bitness.shape[0] != 1:
        raise RuntimeError("Context contains both 32 and 64 bit devices, currently not supported")
    return types[bitness[0]]


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])

# todo things to check:
# && ( totalBits / 8 <= sizeof(_DataType) )
# && is_power_of_2<groups>::value
# && is_power_of_2<items>::value
# && ((groups * items * radix) % histosplit == 0)


merge = r"""

typedef uint _Type;

int my_binary_search(_Type value, const global _Type* a, int left, int right );
int my_binary_search(_Type value, const global _Type* a, int left, int right ) {
    long low  = left;
    long high = max( left, right + 1 );
    while( low < high ) {
        long mid = ( low + high ) / 2;
        if ( value <= a[ mid ] ) 
            high = mid;
        else
            low  = mid + 1;
    }
    return high;
}

typedef uint data_t;

kernel void merge_Stage2(
    global data_t* in,
    const uint subSize
) {

}

kernel void merge_Stage1(
    global data_t* in,
    local data_t* aux
) {
  int i = get_local_id(0);
  int wg = get_local_size(0);

  int offset = get_group_id(0) * wg;
  in += offset;
  aux[i] = in[i];
  
  barrier(CLK_LOCAL_MEM_FENCE);

  // Now we will merge sub-sequences of length 1,2,...,WG/2
  for (int length = 1; length < wg; length <<= 1) {
    data_t iData = aux[i];
    data_t iKey = iData;
    int ii = i & (length - 1);  // index in our sequence in 0..length-1
    int sibling = (i - ii) ^ length; // beginning of the sibling sequence
    
    int pos = 0;
    for (int inc = length; inc > 0; inc >>= 1) { // increment for dichotomic search 
      int j = sibling + pos + inc - 1;
      data_t jKey = aux[j];
      bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );
      
      pos += smaller ? inc : 0;
      pos = min(pos, length);
    }
    
    int bits = (length << 1) - 1; // mask for destination
    int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
    
    barrier(CLK_LOCAL_MEM_FENCE);
    aux[dest] = iData;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write output
  in[i] = aux[i];
}
"""


class DichtomicMergeSort:

    def __init__(self, ctx, queue):
        self.ctx, self.queue = ctx, queue
        # with open("SortKernels.cl") as f:
        #     src = f.read()
        self.program = cl.Program(ctx, merge).build()
        self.sortKernel = self.program.merge_Stage1

    def sort(self, a: numpy.ndarray):
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        # outBuf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=a.nbytes)
        localTemp = cl.LocalMemory(a.itemsize * 512)

        print(a.size)

        def run():
            self.sortKernel(
                self.queue, (a.size,), (512,),  # use max work group size here, power of two is a must
                buf, localTemp
            )
            self.queue.finish()

        _, t = timed(run)
        print("Sorted in {:.3f} ms".format(t * 1000))

        cl.enqueue_copy(self.queue, a, buf)

        for i in range(a.shape[0] // 512):
            assert_is_sorted(a[i*512:(i+1)*512])

        print("Contains sorted subs")

        return a


TYPEMAP = {
    numpy.uint16: "short",
    numpy.uint32: "uint",
    numpy.uint64: "ulong"
}


class RadixSort:

    def __init__(self, ctx: cl.Context, queue: cl.CommandQueue,
                 dtype=numpy.uint32,
                 radixBits=None,
                 totalBits=None,
                 groups=64,
                 items=32,
                 histosplit=512,
                 ):
        self.ctx, self.queue = ctx, queue

        self.datatype = dtype
        self.keyTypeName = TYPEMAP[self.datatype]
        self.dataTypeSize = self.datatype().nbytes
        self.deviceSizeT = determineSizeT(self.ctx)

        if radixBits is None:
            radixBits = 8

        if totalBits is None:
            totalBits = self.dataTypeSize * radixBits

        if totalBits > self.dataTypeSize * 8:
            raise RuntimeError("totalBits should be less than or equal to data type size in bits")
        elif totalBits % radixBits != 0:
            raise RuntimeError("totalBits should be divisible by number of radix bits")

        self.groups = groups
        self.items = items
        self.histosplit = histosplit
        self.passes = totalBits // radixBits
        self.radix = 1 << radixBits
        self.maxInt = (1 << (totalBits - 1)) - 1

        self.program = cl.Program(self.ctx, "\n\n".join([
            _src_other, _src_kernelHistogramAndReorder, _src_kernelScan
        ])).build(options=list(filter(lambda x: bool(x), [
            "-w",
            "-D_RADIX={}".format(self.radix),
            "-D_BITS={}".format(radixBits),
            "-D_GROUPS={}".format(self.groups),
            "-D_ITEMS={}".format(self.items),
            "-DKEY_TYPE={}".format(self.keyTypeName),
            dummyOption()
        ])))

        self.kernelHistogram: cl.Kernel = self.program.histogram
        self.kernelScan     : cl.Kernel = self.program.scan
        self.kernelMerge    : cl.Kernel = self.program.merge
        self.kernelReorder  : cl.Kernel = self.program.reorder

        self.d_Histograms = \
            cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                      size=self.dataTypeSize * self.radix * self.groups * self.items)
        self.deviceSum = \
            cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                      size=self.dataTypeSize * self.histosplit)
        self.deviceTempSum = \
            cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                      size=self.dataTypeSize * self.histosplit)

        self.histogramLocal = cl.LocalMemory(
            size=self.dataTypeSize *
                 self.radix * self.items
        )
        self.reoderLocal = self.histogramLocal

        print("Local scan memory: {} {}".format(self.histosplit,
                                                self.radix * self.groups * self.items // self.histosplit))
        self.scanLocal = cl.LocalMemory(
            size=self.dataTypeSize *
            max(self.histosplit, self.radix * self.groups * self.items // self.histosplit)
        )

        self.d_DataIn = None
        self.d_DataOut = None

    def wait(self):
        self.queue.flush()
        self.queue.finish()

    def _init(self, a: numpy.ndarray):
        baseSize = a.shape[0]
        rest = baseSize % (self.groups * self.items)

        print("Rest base: ", self.groups * self.items)

        self.size = baseSize if rest == 0 else (baseSize - rest + (self.groups * self.items))
        sizeInBytes = a.itemsize * self.size

        print(baseSize, rest, self.size, sizeInBytes)

        self.d_DataIn = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=sizeInBytes)
        cl.enqueue_copy(self.queue, self.d_DataIn, a)

        if rest != 0:
            # fixme some kind of bug here
            cl.enqueue_fill_buffer(self.queue, self.d_DataIn,
                                   self.datatype(self.maxInt), a.nbytes, a.nbytes + rest * a.itemsize)

        self.d_DataOut = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=sizeInBytes)
        self.wait()

    def _scan(self):
        globalItems = (self.radix * self.groups * self.items) // 2
        localItems = globalItems // self.histosplit

        self.kernelScan(
            self.queue, (globalItems,), (localItems,),
            self.d_Histograms,
            self.deviceSum,
            self.scanLocal
        )

        globalItems = self.histosplit // 2
        localItems = globalItems

        self.kernelScan(
            self.queue, (globalItems,), (localItems,),
            self.deviceSum,
            self.deviceTempSum,
            self.scanLocal
        )

        globalItems = self.radix * self.groups * self.items // 2
        localItems = globalItems // self.histosplit

        self.kernelMerge(
            self.queue, (globalItems,), (localItems,),
            # args
            self.deviceSum,
            self.d_Histograms
        )
        self.wait()

    def _histogram(self, passNo):
        self.kernelHistogram(
            self.queue, (self.groups * self.items,), (self.items,),
            # args
            numpy.uint32(self.size),
            numpy.uint32(passNo),
            self.d_DataIn,
            self.d_Histograms,
            self.histogramLocal
        )
        self.wait()

    def _reorder(self, passNo):
        self.kernelReorder(
            self.queue, (self.groups * self.items,), (self.items,),
            # args
            numpy.uint32(self.size),
            numpy.uint32(passNo),
            self.d_DataIn,
            self.d_Histograms,
            self.d_DataOut,
            self.reoderLocal
        )
        self.d_DataIn, self.d_DataOut = self.d_DataOut, self.d_DataIn
        self.wait()

    def sort(self, arr: numpy.ndarray) -> numpy.ndarray:
        """
        Sorts in-place.

        :param arr: array to sort
        :return: same object as passed
        """
        _, tInit = timed(lambda: self._init(arr))

        print("[sort init]\t{:3f}".format(tInit * 1000))

        l = []

        for passNo in range(self.passes):
            _, tHist = timed(lambda: self._histogram(passNo))

            _, tScan = timed(lambda: self._scan())

            _, tReord = timed(lambda: self._reorder(passNo))

            l.append((tHist, tScan, tReord))

        for passNo in range(self.passes):
            print("[sort pass #{}] hist\t{:3f} scan\t{:3f} reor\t{:3f}".format(
                passNo, l[passNo][0] * 1000, l[passNo][1] * 1000, l[passNo][2] * 1000
            ))

        _, tCopy = timed(lambda: cl.enqueue_copy(self.queue, arr, self.d_DataIn))

        print("[sort copy]\t{:3f}".format(tCopy * 1000))

        return arr


TYPE = numpy.uint32

def timed(fn):
    import time
    t = time.perf_counter()
    res = fn()
    t = time.perf_counter() - t
    return (res, t)


def generate(max_value, size, dtype):
    return numpy.random.randint(0, max_value, size=(size,), dtype=dtype)


def run_comparison(ctx, queue):
    rs = RadixSort(ctx, queue, groups = 128, items = 16, dtype=TYPE)

    l = []

    for i in range(10, 26):

        arr = generate(1 << 15, 1 << i, TYPE)
        arr2 = arr.copy()

        import time

        t = time.perf_counter()

        rs.sort(arr)

        t = time.perf_counter() - t

        print("(Radix) {:.3f} ms".format(t * 1000))

        t2 = time.perf_counter()
        numpy.sort(arr2)
        t2 = time.perf_counter() - t2

        print("(NumPy) {:.3f} ms".format(t2 * 1000))

        time.sleep(1)

        l.append((arr.shape[0], t, t2))

    print(l)


def assert_is_sorted(arr):
    for (e1, e2) in zip(arr, arr[1:]):
        if e1 > e2:
            raise RuntimeError("Array is not sorted!")
    if arr[-1] < arr[-2]:
        raise RuntimeError("Array is not sorted!")


def run_validation(ctx, queue):

    rs = RadixSort(ctx, queue, dtype=TYPE)
    #rs = DichtomicMergeSort(ctx, queue)

    def run(i):
        arr = generate(1 << 10, 1 << 20, TYPE)
        arr, time = timed(lambda: rs.sort(arr))
        assert_is_sorted(arr)
        print("[{}] sorted in {:.3f} ms".format(i, time * 1000))
        return time

    count = 1
    times = [ run(i) * 1000 for i in range(count) ]

    avg = sum(times) / len(times)
    min_ = min(times)
    max_ = max(times)
    std = sum([abs(avg - t) for t in times]) / len(times)

    print("Time: {:.3f} (min {:.3f} max {:.3f} std+/- {:.3f}) ms".format(avg, min_, max_, std))


if __name__ == '__main__':
    ctx = cl.Context(devices=(cl.get_platforms()[0].get_devices()[1],))
    queue = cl.CommandQueue(ctx)

    run_validation(ctx, queue)
    # run_comparison(ctx, queue)

# Time: 47.315 (min 39.679 max 102.517 std+/- 5.851) ms