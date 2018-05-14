from .common import *

parameter_sufrace_source = """

kernel void fill_parameter_surface(
    const real x_min, const real x_max, const real y_min, const real y_max,
    write_only image2d_t result
) {    
    const int2 id = ID_2D;
    const int2 size = SIZE_2D;    
    const real2 v = TRANSLATE_2D(id, size, x_min, x_max, y_min, y_max);
    
    float4 color = (float4)(color_for_point(v), 1.0);
    
    write_imagef(result, (int2)(id.x, get_global_size(1) - id.y), color);
}

"""


class ParameterSurface(ComputedImage):

    def __init__(self, ctx, queue, width, height, bounds, color_for_point_function_source, type_config=float_config):
        super().__init__(ctx, queue, width, height, bounds, color_for_point_function_source, parameter_sufrace_source, type_config=type_config)

    def __call__(self, bounds=None):
        if bounds is None:
            bounds = self.bounds

        real, _ = self.tc()

        self.program.fill_parameter_surface(
            self.queue, (self.width, self.height,), None,
            real(bounds.x_min), real(bounds.x_max), real(bounds.y_min), real(bounds.y_max),
            self.image_device
        )

        return self.read_from_device()
