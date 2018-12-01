import numpy

from .cl.core import *
from .cl.codegen import *

PHASE_PLOT_SOURCE = """
#define MAIN_COLOR (float4)(0, 0, 0, 1)

kernel void draw_phase_portrait(
    const PARAM_SIGNATURES,
    const BOUNDS, const IMAGE_BOUNDS,
    const int skip, const int iterations,
    write_only IMAGE_TYPE result
) {
    const COORD_TYPE id = ID;
    VAR_TYPE point = TRANSLATE_INV_Y(VAR_TYPE, id, SIZE, _DS_bs);
    
    for (int i = 0; i < skip; ++i) {
        point = system(point, PARAM_VALUES);
    }
    
    for (int i = skip; i < iterations; ++i) {
        point = system(point, PARAM_VALUES);
        const COORD_TYPE coord = TRANSLATE_BACK_INV_Y(VAR_TYPE, point, _DS_bs, _DS_ibs);
        
        if (VALID_POINT_2D(_DS_ibs, coord)) {
#ifdef DYNAMIC_COLOR
            const float ratio = (float)(i - skip) / (float)(iterations - skip);
            write_imagef(result, coord, (float4)(hsv2rgb((float3)( 240.0 * (1.0 - ratio), 1.0, 1.0)), 1.0));
#else
            write_imagef(result, coord, MAIN_COLOR);
#endif
        }
    }
}
"""


class PhasePortrait(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, systemSource, paramCount, typeConfig):
        super().__init__(ctx, queue, imageShape, spaceShape.asTuple(),
                         # sources
                         systemSource,
                         generateImageBoundsCode(len(imageShape)),
                         generateBoundsCode(typeConfig, len(imageShape)),
                         generateVariableCode(typeConfig, len(imageShape)),
                         generateParameterCode(typeConfig, paramCount),
                         PHASE_PLOT_SOURCE,
                         #
                         typeConfig=typeConfig)
        self.paramCount = paramCount

    def __call__(self, *params, sparse=8, iterations=256, skip=0):
        self.clear()

        self.program.draw_phase_portrait(
            self.queue, tuple(map(lambda x: x // sparse, self.imageShape)), None,
            *wrapParameterArgs(self.paramCount, params, self.tc.real()),
            numpy.array(self.spaceShape, dtype=self.tc.boundsType),
            numpy.array(self.imageShape, dtype=numpy.int32),
            numpy.int32(skip), numpy.int32(iterations),
            self.deviceImage
        )

        return self.readFromDevice()
