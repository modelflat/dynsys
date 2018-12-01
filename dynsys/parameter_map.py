from .cl.core import *
from .cl.codegen import *

import numpy as np

parameter_map_source = """

#ifndef DIVERGENCE_THRESHOLD
#define DIVERGENCE_THRESHOLD 1e100
#endif

#ifndef VALUE_DETECTION_PRECISION 
#define VALUE_DETECTION_PRECISION 1e-3
#endif

#ifndef DIVERGENCE_COLOR
#define DIVERGENCE_COLOR (float4)(1.0)
#endif

void makeHeap(global real*, int, int);

void makeHeap(global real* data, int n, int i) {
    while (true) {
        int smallest = i;
        int l = (i << 1) + 1;
        int r = (i << 1) + 2;

        if (l < n && data[l] < data[smallest]) {
            smallest = l;
        }
        if (r < n && data[r] < data[smallest]) {
            smallest = r;
        }
        if (smallest == i) {
            return; // already smallest
        }
        real t = *(data + i); *(data + i) = *(data + smallest); *(data + smallest) = t;
        i = smallest;
    }
}

void heapSort(global real*, int);

void heapSort(global real* data, int n)
{
    for (int i = n / 2 - 1; i >= 0; --i) {
        makeHeap(data, n, i);
    }
    
    for (int i = n - 1; i >= 0; --i) {
        real t = *(data); *(data) = *(data + i); *(data + i) = t;
        makeHeap(data, i, 0);
    }
}

float3 color_for_count(int, int);

float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.0;
    }
#ifdef GENERATE_COLORS
    const float ratio = (float)count / (float)total;
    return hsv2rgb((float3)(ratio * 240.0, 1.0, 1.0));
#else
    const float d = count < 8? 1.0 : .5;
    switch(count % 8) {
        case 1:
            return (float3)(1.0, 0.0, 0.0)*d;
        case 2:
            return (float3)(0.0, 1.0, 0.0)*d;
        case 3:
            return (float3)(0.0, 0.0, 1.0)*d;
        case 4:
            return (float3)(1.0, 0.0, 1.0)*d;
        case 5:
            return (float3)(1.0, 1.0, 0.0)*d;
        case 6:
            return (float3)(0.0, 1.0, 1.0)*d;
        case 7:
            return (float3)(0.5, 0.0, 0.0)*d;
        default:
            return count == 8 ? .5 : 0;
    }
#endif
}

kernel void compute_map(
    const BOUNDS,
    VARIABLE_SIGNATURES,
    const int samples_count,
    const int skip,
    global VARIABLE_TYPE* samples,
    write_only image2d_t map
) {
    const int2 id = ID_2D;
    samples += (id.x * get_global_size(1) + id.y)*samples_count;
    const real2 v = TRANSLATE_2D_INV_Y(id, SIZE_2D, x_min, x_max, y_min, y_max);
    
    VARIABLE_TYPE x = VARIABLES;

    for (int i = 0; i < skip; ++i) {
        x = map_function(x, v.x, v.y);
    }

    int uniques = 0;
    for (int i = 0; i < samples_count; ++i) {
        x = map_function(x, v.x, v.y);
        samples[i] = x;
        
        if (VARIABLE_VECTOR_ANY_ISNAN(x) || VARIABLE_VECTOR_ANY_ABS_GREATER(x, DIVERGENCE_THRESHOLD)) {
            write_imagef(map, id, DIVERGENCE_COLOR);
            return;
        }
        
        int found = 0;
        for (int j = 0; j < i; ++j) {
            const VARIABLE_ACCEPTOR_TYPE t = samples[j];
            if (VARIABLE_VECTOR_NEAR(x, t, VALUE_DETECTION_PRECISION)) {
                found = 1;
                break;
            }
        }
        if (!found) ++uniques;
    }
    
    // TODO: make heapSort work with vector values ???
    
//    if (samples_count > 16) {
//        heapSort(samples, samples_count);
//        real prev = samples[0]; 
//        for (int i = 1; i < samples_count; ++i) {
//            if (fabs(samples[i] - prev) > VALUE_DETECTION_PRECISION) {
//                prev = samples[i];
//                ++uniques;
//            }
//        }
//    }

    write_imagef(map, id, (float4)(color_for_count(uniques, samples_count), 1.0));
}

"""


class ParameterMap(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, mapFunctionSource, varCount, typeConfig):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         # source
                         mapFunctionSource, generateVariableCode(typeConfig, varCount),
                         parameter_map_source,
                         #
                         typeConfig=typeConfig)
        self.varCount = varCount

    def __call__(self, *variables, iterations, skip):
        real, realSize = self.tc()

        samplesDevice = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE,
            size=numpy.prod(self.imageShape) * iterations * realSize * self.varCount
        )

        self.program.compute_map(
            self.queue, self.imageShape, None,
            numpy.array(self.spaceShape, dtype=self.tc.boundsType)
            *wrapParameterArgs(self.varCount, variables, real),
            np.int32(iterations), np.int32(skip),
            samplesDevice,
            self.deviceImage
        )

        return self.readFromDevice()
