import numpy

from .cl import ComputedImage, generateCode

SOURCE = """
kernel void fillParameterSurface(
    const float4 bounds, write_only image2d_t result
) {    
    const int2 id = ID_2D;
    const real2 v = TRANSLATE_2D(real2, id, SIZE_2D, bounds);
    write_imagef(result, (int2)(id.x, get_global_size(1) - id.y - 1), (float4)(userFn(v), 1.0));
}
"""


class ParameterSurface(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, colorFunctionSource, typeConfig):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         generateCode(typeConfig, boundsDims=2),
                         colorFunctionSource,
                         SOURCE,
                         typeConfig=typeConfig,
                         options=[
                             "-cl-std=CL1.0",
                             "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])]
                         )

    def __call__(self):
        self.program.fillParameterSurface(
            self.queue, self.imageShape, None,
            numpy.array(self.spaceShape, dtype=numpy.float32),
            self.deviceImage
        )
        return self.readFromDevice()
