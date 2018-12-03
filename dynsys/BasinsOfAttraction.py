import numpy
import pyopencl as cl

from .cl import ComputedImage, generateParameterCode, generateBoundsCode


utility = r"""

#ifndef DETECTION_PRECISION 
#define DETECTION_PRECISION 1e-4
#endif

long2 roundCompress(real2, real);
long2 roundCompress(real2 point, real to_sign) {
    return convert_long2_rtz(point / to_sign);
}

real2 roundPoint(real2, int);
real2 roundPoint(real2 point, int to_sign) {
    return convert_real2(convert_long2_rtz(point * pow(10.0f, (float)to_sign))) / pow(10.0f, (float)(to_sign)); 
}


// C++ STL pair operator< implementation
int pair_lt(const real2, const real2);
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

int pair_eq(real2, real2);
int pair_eq(real2 p1, real2 p2) {
    return NEAR_1D(p1.x, p2.x, DETECTION_PRECISION) && NEAR_1D(p1.y, p2.y, DETECTION_PRECISION);
}

int pair_gt(const real2, const real2);
int pair_gt(const real2 p1, const real2 p2) {
    return !(pair_lt(p1, p2) || pair_eq(p1, p2));
}

int binary_search(int, const global real2*, real2);
int binary_search(int size, const global real2* arr, real2 value) {
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

"""


basins_of_attraction_source = utility + r"""

#ifndef FALLBACK_COLOR 
#define FALLBACK_COLOR (float4)(0, 0, 0, 1)
#endif

#ifndef DETECTION_PRECISION_EXPONENT 
#define DETECTION_PRECISION_EXPONENT 4
#endif

#define user_SYSTEM system_fn

kernel void createAttractionMap(
    const BOUNDS bounds,
    const PARAMETERS_SIGNATURE,
    const int iterations,
    global real2* result
) {
    const int2 id = ID_2D;
    const int2 size = SIZE_2D;
    
    real2 point = TRANSLATE_INV_Y_2D(real2, id, size, bounds);
    
    for (int i = 0; i < iterations; ++i) {
        point = user_SYSTEM(point, PARAMETERS);
    }
    
    #define FLAT_ID id.x * size.y + id.y
    result[FLAT_ID] = roundPoint(point, DETECTION_PRECISION_EXPONENT);
}

kernel void drawAttractionMap(
    const int attraction_points_count,
    const global real2* attraction_points,
    const global real2* result,
    write_only image2d_t map
) {
    const int2 id = ID_2D;
    const real2 val = result[id.x * get_global_size(1) + id.y];
    const int color_idx = binary_search(attraction_points_count, attraction_points, val);
    
    if (color_idx == -1 || length(val) < DETECTION_PRECISION) { 
        // in case of trivial solution (0, 0), we want to fallback too
        write_imagef(map, id, FALLBACK_COLOR);
        return;
    }
    
    const float ratio = (float)(color_idx) / (float)(attraction_points_count);
    write_imagef(map, id, (float4)( hsv2rgb((float3)(240.0 * ratio, 1.0, 1.0)), 1 ));
}

kernel void findAttraction(
    const real x, const real y,
    const PARAMETERS_SIGNATURE,
    const int iterations,
    global real2* result
) {
    real2 p = (real2)(x, y);
    for (int i = 0; i < iterations; ++i) {
        p = user_SYSTEM(p, PARAMETERS);
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
                         generateBoundsCode(typeConfig, len(imageShape)),
                         basins_of_attraction_source,
                         #
                         typeConfig=typeConfig)
        self.paramCount = paramCount

    def findAttraction(self, targetPoint: tuple, parameters: tuple, iterations: int):
        real, realSize = self.tc()

        resultDevice = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, realSize*2)

        self.program.findAttraction(
            self.queue, (1,), None,
            *numpy.array(targetPoint, real),
            *self.wrapArgs(self.paramCount, *parameters),
            numpy.int32(iterations),
            resultDevice
        )

        result = numpy.empty((2,), dtype=real)
        cl.enqueue_copy(self.queue, result, resultDevice)
        return result

    def __call__(self, parameters, iterations):
        real, real_size = self.tc()

        pl = self.wrapArgs(self.paramCount, *parameters)

        resultDevice = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 size=numpy.prod(self.imageShape) * real_size * 2)

        self.program.createAttractionMap(
            self.queue, self.imageShape, None,
            numpy.array(self.spaceShape, dtype=real),
            *pl,
            numpy.int32(iterations),
            resultDevice
        )

        result = numpy.empty(shape=(*self.imageShape, 2), order="C", dtype=real)

        cl.enqueue_copy(self.queue, result, resultDevice)

        resultUnique = numpy.unique(result, axis=0)

        resultUniqueDevice = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY,
                                       size=resultUnique.itemsize * resultUnique.size)

        cl.enqueue_copy(self.queue, resultUniqueDevice, resultUnique)

        self.program.drawAttractionMap(
            self.queue, self.imageShape, None,
            numpy.int32(len(resultUnique)),
            resultUniqueDevice,
            resultDevice,
            self.deviceImage
        )

        return self.readFromDevice(), len(resultUnique) - 1

