import numpy
import pyopencl as cl

from .cl import ComputedImage, generateParameterCode


basins_of_attraction_source = """

#ifndef FALLBACK_COLOR 
#define FALLBACK_COLOR (float4)(0, 0, 0, 1)
#endif

#ifndef DETECTION_PRECISION 
#define DETECTION_PRECISION 1e-4
#endif

#ifndef DETECTION_PRECISION_EXPONENT 
#define DETECTION_PRECISION_EXPONENT 4
#endif

long2 round_and_compress(real2 point, real to_sign) {
    return convert_long2_rtz(point / to_sign);
}

real2 round_point(real2 point, int to_sign) {
    return convert_real2(convert_long2_rtz(point * pow(10.0f, (float)to_sign))) / pow(10.0f, (float)(to_sign)); 
}

kernel void create_attraction_map(
    const real x_min, const real x_max, const real y_min, const real y_max,
    PARAM_SIGNATURES,
    const int iter_count,
    global real2* result
) {
    const int2 id = ID_2D;
    const int2 size = SIZE_2D;
    
    real2 point = TRANSLATE_2D_INV_Y(id, size, x_min, x_max, y_min, y_max);
    
    for (int i = 0; i < iter_count; ++i) {
        point = system(point, PARAM_VALUES);
    }
    
    result[id.x * size.y + id.y] = round_point(point, DETECTION_PRECISION_EXPONENT);
}

int pair_eq(const real2 p1, const real2 p2) {
    return NEAR(p1.x, p2.x, DETECTION_PRECISION) && NEAR(p1.y, p2.y, DETECTION_PRECISION);
}

// C++ STL pair operator< implementation
int pair_lt(const real2 p1, const real2 p2) {
    if (p1.x < p2.x) {
        return 1;
    } else if (p2.x < p1.x) {
        return 0;
    } else if (p1.y < p2.y) {
        return 1;
    } else {
        return 0;
    }
}

int pair_gt(const real2 p1, const real2 p2) {
    return !(pair_lt(p1, p2) || pair_eq(p1, p2));
}

int binary_search(const int size, const global real2* arr, const real2 value) {
    int l = 0, r = size;
    
    while (l < r) {
        const int mid = (r + l) / 2;
        if (pair_eq(arr[mid], value)) {
            return mid;
        }
        if (r == l + 1) {
            return -1;
        }
        if (pair_lt(arr[mid], value)) {
            l = mid;
        } else {
            r = mid;
        }
    }
    
    return (r + l) / 2;
}

kernel void draw_attraction_map(
    const int attraction_points_count,
    const global real2* attraction_points,
    const global real2* result,
    
    write_only image2d_t map
) {
    const int2 id = ID_2D;
    const real2 val = result[id.x * get_global_size(1) + id.y];
    const int color_idx = binary_search(attraction_points_count, attraction_points, val);
    
    if (color_idx == -1 || length(val) < DETECTION_PRECISION) { // in case of trivial solution (0, 0), we want to fallback too
        write_imagef(map, id, FALLBACK_COLOR);
        return;
    }
    
    const float ratio = (float)(color_idx) / (float)(attraction_points_count);
    write_imagef(map, id, (float4)( hsv2rgb((float3)(240.0 * ratio, 1.0, 1.0)), 1 ));
}

kernel void find_attraction(
    const real x, const real y,
    PARAM_SIGNATURES,
    const int iter_count,
    global real2* result
) {
    real2 p = (real2)(x, y);
    
    for (int i = 0; i < iter_count; ++i) {
        p = system(p, PARAM_VALUES);
    }
    
    *result = p;
}

"""

class BasinsOfAttraction(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, systemFunction, paramCount, typeConfig):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         # sources
                         systemFunction,
                         generateParameterCode(typeConfig, paramCount),
                         basins_of_attraction_source,
                         #
                         typeConfig=typeConfig)
        self.param_count = paramCount
        self._find_attr_kernel = self.program.find_attraction
        self.num_basins = 0

    def find_attraction(self, x, y, iter_count, *params):
        real, real_size = self.tc()

        result_device = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, real_size*2)

        pl = self.wrapArgs(self.param_count, *params)

        self._find_attr_kernel.set_args(
            real(x), real(y), *pl,
            numpy.int32(iter_count),
            result_device
        )
        cl.enqueue_task(self.queue, self._find_attr_kernel)

        result = numpy.empty((2,), dtype=real)

        cl.enqueue_copy(self.queue, result, result_device)

        return result

    def __call__(self, iter_count, *params):

        bounds = self.bounds
        real, real_size = self.tc()

        pl = self.wrapArgs(self.param_count, params)

        result_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.width*self.height*2*real_size)

        self.program.create_attraction_map(
            self.queue, (self.width, self.height), None,
            real(bounds.x_min), real(bounds.x_max), real(bounds.y_min), real(bounds.y_max),
            *pl,
            numpy.int32(iter_count), result_device
        )

        result = numpy.empty(shape=(self.width * self.height, 2), order="C", dtype=real)

        cl.enqueue_copy(self.queue, result, result_device)

        result_unique = numpy.unique(result, axis=0)

        self.num_basins = len(result_unique) - 1 # except trivial solution

        result_unique_device = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=result_unique.itemsize*result_unique.size)

        cl.enqueue_copy(self.queue, result_unique_device, result_unique)

        self.program.draw_attraction_map(
            self.queue, (self.width, self.height), None,
            numpy.int32(len(result_unique)),
            result_unique_device, result_device, self.deviceImage
        )

        return self.readFromDevice(), self.num_basins

