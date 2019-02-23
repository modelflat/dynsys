import pyopencl as cl
import numpy

from pyopencl.scan import InclusiveScanKernel
from pyopencl.bitonic_sort import BitonicSort
from pyopencl.array import Array

from scipy.stats import linregress


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])


def make_inclusive_scanner(ctx, dtype,):
    return InclusiveScanKernel(ctx, dtype, "a + b", neutral="0", options=[
        dummyOption()
    ])


def ceil_to(v, div):
    if v % div == 0:
        return v
    return (v // div + 1) * div


INTERCALATE_KERNEL = r"""
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#ifndef COORD_MAX_SIGNIFICANT_BITS
#error Define COORD_MAX_SIGNIFICANT_BITS from host!
#endif

#ifndef INTERCALATED_INFINITY
#error Define INTERCALATED_INFINITY from host!
#endif

inline bool is_not_empty(uint4 color) {
    return color.r != 255; //color.a != 0;
}

inline uint bit_intercalate(uint2 coord) {
    uint v = 0;
    // todo unroll
    for (uint i = 0, mask = 1; i < COORD_MAX_SIGNIFICANT_BITS; mask <<= 1, ++i) {
        v |= (((coord.x & mask) << (i + 1)) | (coord.y & mask) << i);
    }
    return v;
}

kernel void intercalate(
    const uint counts_size,
    read_only image2d_t input,
    global uint* intercalated_coords
) {
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    
    if (is_not_empty(read_imageui(input, sampler, coord))) {
        intercalated_coords[coord.s0 + coord.s1 * get_global_size(0)] = bit_intercalate((uint2)coord);
    } else {
        intercalated_coords[coord.s0 + coord.s1 * get_global_size(0)] = INTERCALATED_INFINITY;
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

    def __init__(self, ctx, significands=20):
        self.ctx = ctx
        self.significands = significands
        self.prg = cl.Program(
            ctx, "\n".join([INTERCALATE_KERNEL, MATERIALIZE_KERNEL, CHECK_BOXES_KERNEL])
        ).build(options=[
            "-DCOORD_MAX_SIGNIFICANT_BITS={}".format(significands),
            "-DINTERCALATED_INFINITY={}".format(0xFFFFFFFF),
            dummyOption()
        ])

        # self.scanner = make_inclusive_scanner(self.ctx, numpy.uint32)

        self.sorter = BitonicSort(self.ctx)

        self.totalBoxes = 0
        self.nonEmptyPixels = 0
        self.imageShape = None

        self.dtype=numpy.uint32
        self.element_size = self.dtype().nbytes

    def set_image_params(self, image_shape, image_format):
        self.image_shape = image_shape
        self.image_format = image_format
        self.counts = numpy.zeros((numpy.prod(image_shape),), dtype=numpy.uint32)
        # self.counts_arr = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=self.counts.nbytes)
        self.counts_arr = Array(self.ctx, self.counts.shape, self.counts.dtype)

    def intercalate(self, queue, image_buf: cl.Buffer):
        self.prg.intercalate(
            queue, self.image_shape, None,
            #
            numpy.int32(numpy.prod(self.image_shape)),
            image_buf,
            self.counts_arr.data
        )

    def sort_intercalated(self, queue):
        self.sorter(self.counts_arr, queue=queue)

    def scan_counts(self, queue):
        # self.scanner.scan(queue, self.countsBuf, size=self.counts.size)
        self.scanner(self.counts_arr, queue=queue)
        # cl.enqueue_copy(queue, self.counts, self.countsBuf)
        self.nonEmptyPixels = self.counts[-1]
        self.bitStrings = numpy.empty((numpy.prod(self.imageShape),), dtype=numpy.uint32)
        self.bitStringsBuf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.bitStrings.nbytes)
        # print("Scan")
        return self.counts

    def materialize_bit_strings(self, queue, mask):
        cl.enqueue_fill_buffer(queue, self.bitStringsBuf, numpy.uint32(0xFFFFFFFF), offset=0,
                               size=self.bitStringsBuf.size)
        self.prg.materialize_bit_strings(
            queue, (self.counts.size,), None,
            numpy.uint32(mask << 1),
            numpy.uint32(self.bitStrings.size),
            self.counts_arr.data,
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

    def count_for_image(self, queue, imageShape, imageBuf):
        assert numpy.prod(imageShape) <= (1 << self.significands)

        self.intercalate(queue, imageShape, imageBuf)

        self.sorter(self.counts, queue=queue)

        # self.scan_counts(queue)
        #
        # results = self.all_counts(queue, imageShape)
        #
        # x, y = numpy.array(results).T
        #
        # s = linregress(x, y)
        return 0 #-s[0]


def test_box_count(ctx, queue):
    counter = FastBoxCounting(ctx)

    w, h = 256, 256

    counter.set_image_params((w, h), None)

    image: numpy.ndarray = numpy.ones((w, h, 4), dtype=numpy.uint8)

    imageBuf = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        format=cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
                        hostbuf=image)

    counter.intercalate(queue, imageBuf)

    counter.sort_intercalated(queue)

    queue.finish()




if __name__ == '__main__':
    ctx = cl.Context(devices=(cl.get_platforms()[0].get_devices()[0],))
    queue = cl.CommandQueue(ctx)

    test_box_count(ctx, queue)
