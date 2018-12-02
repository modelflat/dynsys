import numpy

from .cl.core import *
from .cl.codegen import *


parameterSurfaceSource = """
kernel void fill_parameter_surface(
    const BOUNDS_2D bounds, write_only image2d_t result
) {    
    const int2 id = ID_2D;
    const real2 v = TRANSLATE_2D(real2, id, SIZE_2D, bounds);
    write_imagef(result, (int2)(id.x, get_global_size(1) - id.y), (float4)(color_for_point(v), 1.0));
}
"""


class ParameterSurface(ComputedImage):

    def __init__(self, ctx, queue, width, height, bounds, colorFunctionSource, typeConfig):
        super().__init__(ctx, queue, (width, height), bounds.asTuple(),
                         colorFunctionSource,
                         generateBoundsCode(typeConfig, 2),
                         parameterSurfaceSource,
                         typeConfig=typeConfig)

    def __call__(self):
        self.program.fill_parameter_surface(
            self.queue, self.imageShape, None,
            numpy.array(self.spaceShape, dtype=numpy.float32),
            self.deviceImage
        )
        return self.readFromDevice()
