from dynsys.common import *

phase_portrait_source = """

#define MAIN_COLOR (uint4)(0, 0, 0, 255)

kernel void draw_phase_portrait(
    const real a, const real b,
    const real x_min, const real x_max, const real y_min, const real y_max,
    const int step_count, 
    const int draw_last_points,
    const int w, const int h,
    write_only image2d_t result
) {
    const int2  id = ID_2D;
    const int2 size = SIZE_2D;
    const real2 grid_step = (real2)( (x_max - x_min)/get_global_size(0), (y_max - y_min)/get_global_size(1));
    real2 point = (real2)(x_min + id.x*grid_step.x, y_min + (get_global_size(1) - id.y)*grid_step.y);
    
    //real2 point = TRANSLATE_2D_INV_Y(id, size, x_min, x_max, y_min, y_max);
    
    for (int i = 0; i < step_count; ++i) {
        point = system(point, a, b);
        if (step_count - i <= draw_last_points) {
            int2 coord = TRANSLATE_BACK_2D( point, x_min, x_max, y_min, y_max, (int2)(w, h));
#ifdef DYNAMIC_COLOR
            write_imagef (result, coord, (float4)(hsv2rgb( (float3)( (float)(i) / step_count * 360.0, 1.0, 1.0 )), 1.0) );
#else
            write_imageui(result, coord, MAIN_COLOR);
#endif
        }
    }
}

"""


class PhasePortrait(ComputedImage):

    def __init__(self, ctx, queue, width, height, bounds, system_function_source, param_count=2, type_config=float_config):
        super().__init__(ctx, queue, width, height, bounds,
                         system_function_source, generate_param_code(param_count),
                         phase_portrait_source, type_config=type_config)
        self.param_count = param_count

    def __call__(self, step_count, *params, grid_sparseness = 5, draw_last_points = 1):

        bounds = self.bounds

        real, real_size = self.tc()

        self.clear()

        param_list = make_param_list(self.param_count, params, real)

        self.program.draw_phase_portrait(
            self.queue, (self.width // grid_sparseness, self.height // grid_sparseness), None,
            *param_list, real(bounds.x_min), real(bounds.x_max), real(bounds.y_min), real(bounds.y_max),
            np.int32(step_count), np.int32(draw_last_points), np.int32(self.width), np.int32(self.height),
            self.image_device
        )

        return self.read_from_device()
