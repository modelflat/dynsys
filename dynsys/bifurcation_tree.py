from .cl.core import *
from .cl.codegen import *
import numpy as np


bifurcation_tree_source = """

kernel void compute_bifurcation_tree(
    const real x0,
    PARAM_SIGNATURES, 
    const int active_param,
    const real2 bounds,
    const real max_allowed_value,
    const int skip, 
    const int samples_count,
    global real* result, global real2* result_minmax
) {
    const int id = get_global_id(0);
    const real param = TRANSLATE_1D(id, get_global_size(0), bounds);

    result += id * samples_count;
    
    SET_PARAM_VALUE(active_param, param);

    real x    = x0;
    real min_ = x0;
    real max_ = x0;
    for (int i = 0; i < skip; ++i) {
        x = map_function(x, PARAM_VALUES);
    }

    for (int i = 0; i < samples_count; ++i) {
        x = map_function(x, PARAM_VALUES);
        if (x < min_ && x > -max_allowed_value) min_ = x;
        if (x > max_ && x < max_allowed_value) max_ = x;
        result[i] = clamp(x, -max_allowed_value, max_allowed_value);
    }

    result_minmax[id] = (real2)(min_, max_); // save minmax
}

#define TREE_COLOR (float4)(0, 0, 0, 1.0)

kernel void draw_bifurcation_tree(
    const global real* samples,
    const int samples_count,
    const real2 bounds,
    const real height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    samples += id * samples_count;
    for (int i = 0; i < samples_count; ++i) {
        int2 coord = (int2)(id, TRANSLATE_BACK_INV_1D(samples[i], bounds, height));
        write_imagef(result, coord, TREE_COLOR);
    }
}

"""


class BifurcationTree(ComputedImage):

    def __init__(self, ctx, queue, imageShape, mapFunctionSource, paramCount, typeConfig):
        ComputedImage.__init__(self, ctx, queue, imageShape, (),
                               # sources
                               mapFunctionSource,
                               generateParameterCode(typeConfig, paramCount),
                               bifurcation_tree_source,
                               #
                               typeConfig=typeConfig)
        self.paramCount = paramCount

    def __call__(self, startPoint, iterations, paramRange, paramIndex,
                 *otherParams,
                 skip=0, maxAllowedValue=1000):
        real, realSize = self.tc()
        width = self.imageShape[0]

        resultDevice = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE, size=iterations * width * realSize
        )
        resultMinMaxDevice = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, size=width * 2 * realSize
        )

        self.program.compute_bifurcation_tree(
            self.queue, (self.imageShape[0],), None,
            real(startPoint),
            *wrapParameterArgs(self.paramCount, otherParams, real, active_idx=paramIndex),
            np.int32(paramIndex),
            numpy.array(paramRange, dtype=self.tc.real()),
            real(maxAllowedValue),
            np.int32(skip), np.int32(iterations),
            resultDevice, resultMinMaxDevice
        )

        resultMinMaxHost = np.empty((self.imageShape[0]*2,), dtype=real)

        cl.enqueue_copy(self.queue, resultMinMaxHost, resultMinMaxDevice)

        minMax = min(resultMinMaxHost), max(resultMinMaxHost)

        self.clear()

        self.program.draw_bifurcation_tree(
            self.queue, (self.imageShape[0],), None,
            resultDevice,
            np.int32(iterations),
            numpy.array(minMax, dtype=self.tc.real()),
            real(self.imageShape[1]),
            self.deviceImage
        )

        return self.readFromDevice()
