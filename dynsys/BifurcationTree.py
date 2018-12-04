import numpy
import pyopencl as cl

from .cl import ComputedImage, generateCode


SOURCE = """

kernel void computeBifurcationTree(
    const real x0,
    PARAMETERS_SIGNATURE, 
    const int activeParamIdx,
    const real2 bounds,
    const real maxAllowedValue,
    const int skip, 
    const int iterations,
    global real* result, global real2* resultMinMax
) {
    const int id = ID_1D;
    const real param = TRANSLATE_1D(id, SIZE_1D, bounds);

    result += id * iterations;
    
    SET_PARAMETER(activeParamIdx, param);

    real x    = x0;
    real min_ = x0;
    real max_ = x0;
    for (int i = 0; i < skip; ++i) {
        x = map_function(x, PARAMETERS);
    }

    for (int i = 0; i < iterations; ++i) {
        x = map_function(x, PARAMETERS);
        if (x < min_ && x > -maxAllowedValue) min_ = x;
        if (x > max_ && x <  maxAllowedValue) max_ = x;
        result[i] = clamp(x, -maxAllowedValue, maxAllowedValue);
    }

    resultMinMax[id] = (real2)(min_, max_); // save min/max
}

#define TREE_COLOR (float4)(0, 0, 0, 1.0)

kernel void drawBifurcationTree(
    const global real* samples,
    const int iterations,
    const real2 bounds, const real height,
    write_only image2d_t result
) {
    const int id = ID_1D;
    
    samples += id * iterations;
    
    for (int i = 0; i < iterations; ++i) {
        int2 coord = (int2)(id, TRANSLATE_BACK_INV_1D(samples[i], bounds, height));
        write_imagef(result, coord, TREE_COLOR);
    }
}

"""


class BifurcationTree(ComputedImage):

    def __init__(self, ctx, queue, imageShape, mapFunctionSource, paramCount, typeConfig):
        ComputedImage.__init__(self, ctx, queue, imageShape, (),
                               # sources
                               generateCode(typeConfig,
                                            parameterCount=paramCount),
                               mapFunctionSource,
                               SOURCE,
                               #
                               typeConfig=typeConfig)
        self.paramCount = paramCount

    def __call__(self, startPoint, paramIndex, paramRange, otherParams, iterations,
                 skip: int = 0, maxAllowedValue: float = 1000.0):
        real, realSize = self.tc()
        width = self.imageShape[0]

        resultDevice = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE, size=iterations * width * realSize
        )
        resultMinMaxDevice = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, size=width * 2 * realSize
        )

        self.program.computeBifurcationTree(
            self.queue, (self.imageShape[0],), None,
            real(startPoint),
            *self.wrapArgs(self.paramCount, otherParams, skipIndex=paramIndex),
            numpy.int32(paramIndex),
            numpy.array(paramRange, dtype=self.tc.real()),
            real(maxAllowedValue),
            numpy.int32(skip), numpy.int32(iterations),
            resultDevice, resultMinMaxDevice
        )

        resultMinMaxHost = numpy.empty((self.imageShape[0]*2,), dtype=real)

        cl.enqueue_copy(self.queue, resultMinMaxHost, resultMinMaxDevice)

        minMax = min(resultMinMaxHost), max(resultMinMaxHost)

        self.clear()

        self.program.drawBifurcationTree(
            self.queue, (self.imageShape[0],), None,
            resultDevice,
            numpy.int32(iterations),
            numpy.array(minMax, dtype=self.tc.real()),
            real(self.imageShape[1]),
            self.deviceImage
        )

        return self.readFromDevice()
