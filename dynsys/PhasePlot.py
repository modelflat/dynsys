import numpy
import pyopencl as cl

from .cl import ComputedImage, TypeConfig,\
    generateParameterCode, generateImageBoundsCode, generateBoundsCode, generateVariableCode

PHASE_PLOT_SOURCE = """
#define user_SYSTEM system_fn

kernel void draw_phase_plot(
    const PARAMETERS_SIGNATURE,
    const BOUNDS bounds, 
    const IMAGE_BOUNDS image_bounds,
    const int skip, const int iterations,
    write_only IMAGE_TYPE result
) {
    const COORD_TYPE id = ID;
    VARIABLE_TYPE point = TRANSLATE_INV_Y(VARIABLE_TYPE, id, SIZE, bounds);
    
    for (int i = 0; i < skip; ++i) {
        point = user_SYSTEM(point, PARAMETERS);
    }
    
    for (int i = skip; i < iterations; ++i) {
        point = user_SYSTEM(point, PARAMETERS);
        const COORD_TYPE_EXPORT coord = CONVERT_SPACE_TO_COORD(TRANSLATE_BACK_INV_Y(VARIABLE_TYPE, point, bounds, image_bounds));
        
        if (VALID_POINT(image_bounds, coord)) {
#ifdef DYNAMIC_COLOR
            const float ratio = (float)(i - skip) / (float)(iterations - skip);
            write_imagef(result, coord, (float4)(hsv2rgb((float3)( 240.0 * (1.0 - ratio), 1.0, 1.0)), 1.0));
#else
            write_imagef(result, coord, DEFAULT_ENTITY_COLOR);
#endif
        }
    }
}
"""


def ceilToPow2(x):
    return 1 << int(numpy.ceil(numpy.log2(x)))


class PhasePlot(ComputedImage):

    def __init__(self,
                 ctx: cl.Context, queue: cl.CommandQueue,
                 imageShape: tuple, spaceShape: tuple,
                 systemSource: str, paramCount: int,
                 backColor: tuple,
                 typeConfig: TypeConfig):
        super().__init__(ctx, queue, imageShape, spaceShape,
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
        self.backColor = backColor

    def __call__(self, parameters, iterations, skip=0, gridSparseness=8):
        self.clear(color=self.backColor)

        space = tuple(self.spaceShape[i] if i < len(self.spaceShape) else numpy.nan
                      for i in range(ceilToPow2(len(self.spaceShape))))

        image = tuple(self.imageShape[i] if i < len(self.imageShape) else 0
                      for i in range(ceilToPow2(len(self.imageShape))))

        self.program.draw_phase_plot(
            self.queue, tuple(map(lambda x: x // gridSparseness + 1, self.imageShape)), None,
            *self.wrapArgs(self.paramCount, *parameters),
            numpy.array(space, dtype=self.tc.boundsType),
            numpy.array(image, dtype=numpy.int32),
            numpy.int32(skip), numpy.int32(iterations),
            self.deviceImage
        )

        return self.readFromDevice()
