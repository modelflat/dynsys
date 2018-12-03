import numpy
import pyopencl as cl

from .cl import ComputedImage, \
    generateBoundsCode, generateImageBoundsCode, generateParameterCode

SOURCE = """

#ifndef carrying_function
#define carrying_function map_function
#endif

// compute samples for diagram, single-threaded 
// (usually iterations count is small enough, and we avoid copying data)
kernel void computeSamples(
    const real start, 
    const PARAMETERS_SIGNATURE,
    const int skip,
    const int iterations,
    global real* samples
) {
    real x = start;

    for (int i = 0; i < skip; ++i) {
        x = map_function(x, PARAMETERS);
    }
    
    samples[0] = x;
    samples[1] = x;
    for (int i = skip + 2; i < iterations; ++i) {
        x = map_function(x, PARAMETERS);
        samples[i - skip] = x;   
    }
}

#define ABS_ERROR 2e-3

#define CROSSING_COLOR (float4)(1, 0, 0, 1)
#define CARRY_COLOR    (float4)(.5, 1, 0, 1)
#define FILL_COLOR     (float4)(1.0)

// draw background (secant line and carrying function) for this cobweb diagram
kernel void drawBackground(
    const PARAMETERS_SIGNATURE, 
    const BOUNDS bounds,
    write_only image2d_t result
) {
    const int2 id = ID_2D;
    const real2 v = TRANSLATE_INV_Y_2D(real2, id, SIZE_2D, bounds);
    
    if (NEAR_1D(v.y, v.x, ABS_ERROR)) {
        write_imagef(result, id, CROSSING_COLOR);
    } else if (NEAR_1D(v.y, carrying_function(v.x, PARAMETERS), ABS_ERROR * 5)) {
        write_imagef(result, id, CARRY_COLOR);
    } else {
        write_imagef(result, id, FILL_COLOR);
    }
}

#define ITER_COLOR (float4)(0, 0, 0, 1)

kernel void drawCobwebDiagram(
    const global real* samples,
    const BOUNDS bounds,
    const IMAGE_BOUNDS image_bounds,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    
    if (id + 2u >= get_global_size(0)) {
        return;
    }
    
    const real3 x = (real3)(samples[id], samples[id+1], samples[id+2]);

    if (isnan(x.s0) || isnan(x.s1) || isnan(x.s2)) {
        return;   
    }

    const int2 p1 = CONVERT_SPACE_TO_COORD(TRANSLATE_BACK_INV_Y_2D(real2, x.s01, bounds, image_bounds));
    const int2 p2 = CONVERT_SPACE_TO_COORD(TRANSLATE_BACK_INV_Y_2D(real2, x.s12, bounds, image_bounds));
    
    int2 line = (int2)(min(p1.x, p2.x), max(p1.x, p2.x));
    if (p1.y < image_bounds.y && p1.y >= 0) {
        for (int i = clamp(line.s0, 0, image_bounds.x); i <= line.s1; ++i) {
            if (i < image_bounds.y && i >= 0) {
                write_imagef(result, (int2)(i, p1.y), ITER_COLOR);
            }
        }
    }
    
    line = (int2)(min(p1.y, p2.y), max(p1.y, p2.y));
    if (p2.x < image_bounds.x && p2.x >= 0) {
        for (int i = clamp(line.s0, 0, image_bounds.y); i <= line.s1; ++i) {
            if (i < image_bounds.y && i >= 0) {
                write_imagef(result, (int2)(p2.x, i), ITER_COLOR);
            }
        }
    }
}

"""


class CobwebDiagram(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, functionSource, paramCount, typeConfig):
        ComputedImage.__init__(self, ctx, queue, imageShape, spaceShape,
                               # sources
                               functionSource,
                               generateParameterCode(typeConfig, paramCount),
                               generateBoundsCode(typeConfig, 2),
                               generateImageBoundsCode(2),
                               SOURCE,
                               #
                               typeConfig=typeConfig)
        self.paramCount = paramCount

    def __call__(self, startPoint, parameters, iterations, skip=0):
        real, realSize = self.tc()

        samplesDevice = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, realSize * iterations)

        paramList = self.wrapArgs(self.paramCount, *parameters)

        self.program.computeSamples(
            self.queue, (1,), None,
            real(startPoint), *paramList,
            numpy.int32(skip),
            numpy.int32(iterations), samplesDevice
        )

        self.program.drawBackground(
            self.queue, self.imageShape, None,
            *paramList,
            numpy.array(self.spaceShape, dtype=self.tc.boundsType),
            self.deviceImage
        )

        self.program.drawCobwebDiagram(
            self.queue, (iterations,), None,
            samplesDevice,
            numpy.array(self.spaceShape, dtype=self.tc.boundsType),
            numpy.array(self.imageShape, dtype=numpy.int32),
            self.deviceImage
        )

        return self.readFromDevice()
