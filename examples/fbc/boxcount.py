import pyopencl as cl
import numpy
from scan import Scanner
import time

from pyopencl.scan import InclusiveScanKernel
from pyopencl.bitonic_sort import BitonicSort
from pyopencl.array import Array

from scipy.stats import linregress
from matplotlib.pyplot import imread


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])


def make_inclusive_scanner(ctx, dtype,):
    return InclusiveScanKernel(ctx, dtype, "a + b", neutral="0", options=[
        dummyOption()
    ])


INTERCALATE_KERNEL = r"""
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#ifndef COORD_MAX_SIGNIFICANT_BITS
#error Define COORD_MAX_SIGNIFICANT_BITS from host!
#endif

inline bool is_not_empty(uint4 color) {
    return color.r != 255; //color.a != 0;
}

inline uint intercalate(uint2 coord) {
    uint v = 0;
    // todo unroll
    for (uint i = 0, mask = 1; i < COORD_MAX_SIGNIFICANT_BITS; mask <<= 1, ++i) {
        v |= (((coord.x & mask) << (i + 1)) | (coord.y & mask) << i);
    }
    return v;
}

kernel void bit_intercalate_and_count(
    const uint counts_size,
    read_only image2d_t input,
    global uint* counts
) {
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const uint iv = intercalate(as_uint2(coord));
    
    if (iv < counts_size) {
        *(counts + iv) = (uint)(is_not_empty(read_imageui(input, sampler, coord)));
    }
}

"""


MATERIALIZE_KERNEL = r"""
kernel void materialize_bit_strings(
    const int m,
    const int bit_strings_size,
    const global uint* counts,
    global uint* bit_strings
) {
    // point in image in intercalated coords
    const int id = get_global_id(0);
    
    // we need to produce a sorted vector of all unique non-empty points masked by `m` bits
    // (or "materialize" counts)
    // for this, we should first determine if this point is unique by examining its neighbour to the left
    
    // hanlde the first element
    if (id == 0) {
        const int idx = counts[id];
        // first box is unique because it's first
        // if `idx` > 0 at `id` == 0 that means that the first box is occupied, so put it in place
        // otherwise, the first box is empty and we do nothing 
        if (idx > 0) {
            bit_strings[0] = 0;
        }    
    } else {
        // index of this point in sorted vector of all non-empty points masked by `m` bits
        const int idx      = (counts[id] - 1) >> m;
        const int prev_idx = (counts[id - 1] - 1) >> m;
        
        // if idx == prev_idx that point is not unique and we dont need to do anything
        if (idx != prev_idx) {
            // that point is unique, i.e. it is the first in the sequence of same points
            // that mean that `idx` points to its place in sorted vector
            if (idx >= 0 && idx < bit_strings_size) { 
                bit_strings[idx] = id;
            }
        }
    }
}
"""


CHECK_BOXES_KERNEL = r"""
#define PLACEHOLDER (uint)(0xFFFFFFFF)

kernel void check_boxes(
    const int size,
    const uint box_size,
    const global uint* coords, // coordinates of non-empty boxes
    //
    global int *boxes
) { 
    // position of given box in sorted array
    const int id    = get_global_id(0);
    // box id
    const uint left = coords[id];
    
    // in order to decide whether the box is empty we need to just consider its value :)
    if (left != PLACEHOLDER) {
        atomic_inc(boxes);
        // TODO avoid atomics
    }
}
"""


class FastBoxCounting:

    def __init__(self, ctx, significands=20, preallocate_bit_strings=False,
                 use_scanner=None):
        self.ctx = ctx
        self.significands = significands
        self.prg = cl.Program(
            ctx, "\n".join([INTERCALATE_KERNEL, MATERIALIZE_KERNEL, CHECK_BOXES_KERNEL])
        ).build(options=[
            "-DCOORD_MAX_SIGNIFICANT_BITS={}".format(significands),
            dummyOption()
        ])
        self.counts: numpy.ndarray = numpy.zeros((1 << significands,), dtype=numpy.uint32)
        self.countsBuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=self.counts)
        # self.scanner = Scanner(ctx, element_type=numpy.uint32, preallocate_iterations=2)
        self.scanner = make_inclusive_scanner(self.ctx, numpy.uint32)
        self.totalBoxes = 0
        self.nonEmptyPixels = 0
        self.preallocate_bit_strings = preallocate_bit_strings
        self.imageShape = None

        self.dtype=numpy.uint32
        self.element_size = self.dtype().nbytes

        self.countsBuf = Array(self.ctx, self.counts.shape, self.counts.dtype, data=self.countsBuf)

    def intercalate_and_count(self, queue, imageShape: tuple, imageBuf: cl.Buffer):
        # print("-- calling intercalate_and_count")
        self.totalBoxes = numpy.prod(imageShape)
        self.imageShape = imageShape
        self.prg.bit_intercalate_and_count(
            queue, imageShape, None,
            numpy.int32(self.counts.size),
            imageBuf,
            self.countsBuf.data
        )
        # cl.enqueue_copy(queue, self.counts, self.countsBuf)
        # print("-- done intercalating")
        # self.bitStrings = numpy.empty((self.nonEmptyBoxes,), dtype=numpy.uint32)
        # self.bitStringsBuf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
        #                                size=self.bitStrings.nbytes)
        # print("Intercalate")
        return self.counts

    def scan_counts(self, queue):
        # self.scanner.scan(queue, self.countsBuf, size=self.counts.size)
        self.scanner(self.countsBuf, queue=queue)
        # cl.enqueue_copy(queue, self.counts, self.countsBuf)
        self.nonEmptyPixels = self.counts[-1]
        if not self.preallocate_bit_strings:
            self.bitStrings = numpy.empty((numpy.prod(self.imageShape),), dtype=numpy.uint32)
            self.bitStringsBuf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.bitStrings.nbytes)
        # print("Scan")
        return self.counts

    def materialize_bit_strings(self, queue, mask):
        cl.enqueue_fill_buffer(queue, self.bitStringsBuf, numpy.uint32(0xFFFFFFFF),
                               offset=0,
                               size=self.bitStringsBuf.size)
        self.prg.materialize_bit_strings(
            queue, (self.counts.size,), None,
            numpy.uint32(mask << 1),
            numpy.uint32(self.bitStrings.size),
            self.countsBuf.data,
            self.bitStringsBuf
        )
        queue.finish()
        # print("Materialize")
        # cl.enqueue_copy(queue, self.bitStrings, self.bitStringsBuf)
        return self.bitStrings

    def check_boxes(self, queue, mask):
        boxSize = 1 << mask

        globalWorkSize = (numpy.prod(self.imageShape))

        self.blacksBuf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.element_size)
        # self.graysBuf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.scanner.element_size)

        cl.enqueue_fill_buffer(queue, self.blacksBuf, numpy.uint32(0), offset=0, size=self.blacksBuf.size)
        # cl.enqueue_fill_buffer(queue, self.graysBuf, numpy.uint32(0), offset=0, size=self.graysBuf.size)

        self.prg.check_boxes(
            queue, (globalWorkSize,), None,
            numpy.uint32(numpy.prod(self.imageShape)),
            numpy.uint32(boxSize),
            self.bitStringsBuf,
            self.blacksBuf
        )

        # todo split here for easier debug

        # self.scanner.scan(queue, self.blacksBuf, colorBufSize)
        # self.scanner.scan(queue, self.graysBuf, colorBufSize)

        black = numpy.empty(shape=(1,), dtype=self.bitStrings.dtype)
        cl.enqueue_copy(queue, black, self.blacksBuf)

        # cl.enqueue_copy(queue, gray, self.graysBuf.get_sub_region(
        #     origin=gray.itemsize*(colorBufSize - 1), size=gray.itemsize
        # ))

        return numpy.log(boxSize), numpy.log(black[0])

    def all_counts(self, queue, imsize):
        queue.finish()
        mask_sizes = int(numpy.ceil(numpy.log2(max(imsize))))
        box_counts = []
        for mask in range(1, mask_sizes):
            # print("mask", mask)
            self.materialize_bit_strings(queue, mask)
            box_counts.append(self.check_boxes(queue, mask))
        return box_counts


def ceil_to(v, div):
    if v % div == 0:
        return v
    return (v // div + 1) * div


def is_pow_2(v):
    return v == 0 or (v & (v - 1)) == 0


def ceil_to_next_pow_2(v):
    if v == 0: return 1
    if is_pow_2(v): return v
    return ceil_to(v, 1 << (int(numpy.log2(v)) + 1))


def assert_intercalated(x1, x2, v):

    def nicebin(v, fill_to=16):
        x = str(bin(v))[2:]
        return ((fill_to - len(x)) * "0") + x
    x1 = nicebin(x1, fill_to=16)
    x2 = nicebin(x2, fill_to=16)
    v = nicebin(v, fill_to=32)
    print(" ".join(x1), " " + " ".join(x2), v, sep="\n")
    x1 = x1[::-1]
    x2 = x2[::-1]
    v = v[::-1]
    i = 0
    for b1, b2 in zip(x1, x2):
        val = v[2*i:2*(i+1)]
        assert (b1 + b2) == val[::-1], "{} : {}{} != {}".format(2*i, b1, b2, val)
        i += 1


def intercalation_test(ctx, queue):
    prg = cl.Program(ctx, BIT_INTERCALATE).build()

    a: numpy.ndarray = numpy.random.randint(0, 512, size=(2,1), dtype=numpy.uint32)
    aBuf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=a)
    b = numpy.empty((a.shape[1],), dtype=numpy.uint32)
    bBuf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=b.nbytes)

    prg.test_intercalate(
        queue, (1,), None,
        aBuf, bBuf
    )

    cl.enqueue_copy(queue, b, bBuf)

    assert_intercalated(a[0][0], a[1][0], b[0])


def assert_is_sorted(arr):
    for i, e in enumerate(zip(arr, arr[1:])):
        e1, e2 = e
        if e1 > e2:
            raise RuntimeError("Array is not sorted! (at i={}: {} is not < than {}) ({})".format(i, e1, e2, arr[i - 10: i + 10]))
    if arr[-1] < arr[-2]:
        raise RuntimeError("Array is not sorted!")


def count_for_image(ctx, queue, counter, imageShape, imageBuf):
    assert numpy.prod(imageShape) <= (1 << counter.significands)

    # imageBuf = cl.Image(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
    #                     format=cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
    #                     hostbuf=image)
    counts = counter.intercalate(queue, imageShape, imageBuf)

    # print("After intercalation: ", counts[:64])

    counts = counter.scan_counts(queue)

    # print("After scan: ", counts[:16], "...", counts[-16:])

    # print("Non-empty count:", counter.nonEmptyPixels)

    results = counter.all_counts(queue, imageShape)

    x, y = numpy.array(results).T
    # print(x, y)

    s = linregress(x, y)
    return -s[0]


def test_counting(ctx, queue):

    import os
    filename = "~/Downloads/Ikeda_map_a=1_b=0.9_k=0.4_p=6.jpg"
    filename = os.path.expanduser(filename)

    w, h = (128, 128)

    counter = FastBoxCounting(ctx)

    image: numpy.ndarray = numpy.random.randint(0, 2, size=(w, h, 4), dtype=numpy.uint8)
    # image: numpy.ndarray = numpy.ones((w, h, 4), dtype=numpy.uint8)
    # for i in range(-w // 2, w // 2):
    #     for j in range(-h // 2, h // 2):
    #         if int((i * i + j * j)**.5) == int(w / 4):
    #             image[i + w//2][j + h//2] = 1

    imageBuf = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        format=cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
                        hostbuf=image)


    count_for_image(ctx, queue, counter, imageShape=(w, h), imageBuf=imageBuf)

    # for line in image.T[3]:
    #     print("".join(map(str, line)))




if __name__ == '__main__':
    ctx = cl.Context(devices=(cl.get_platforms()[0].get_devices()[0],))
    queue = cl.CommandQueue(ctx)

    test_counting(ctx, queue)
