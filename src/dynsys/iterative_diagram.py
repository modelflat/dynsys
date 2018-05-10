from .common import Bounds, allocate_image, float_config
import numpy as np
import pyopencl as cl

iterative_diagram_kernel = """

#define ID_2D (int2)(get_global_id(0), get_global_id(1))
#define ID_2D_Y_INV (int2)(get_global_id(0), get_global_size(1) - get_global_id(1))

#define SIZE_2D (int2)(get_global_size(0), get_global_size(1))

#define TRANSLATE(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id.x)*((x_max) - (x_min))/(size.x), (y_min) + (id.y)*((y_max) - (y_min))/(size.y))

#define TRANSLATE_INV_Y(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id.x)*((x_max) - (x_min))/(size.x), (y_min) + (size.y - id.y)*((y_max) - (y_min))/(size.y))

#define NEAR(a, b, abs_error) (fabs((a) - (b)) < (abs_error))


/**
 * Iterative diagram kernel source 
 * Kernel name: draw_background, draw_iterative_diagram
 * 
 * Requires defines:
 * - real, real2, real4, as_real4 : for types
 * - carrying_function
 * - map_function  
 */

#define ABS_ERROR 2e-3

#define CROSSING_COLOR (uint4)(255, 0, 0, 255)
#define CARRY_COLOR    (uint4)(128, 255, 0, 255)
#define FILL_COLOR     (uint4)(255)

kernel void draw_background(
    const real lambda,
    const real x_min, const real x_max, const real y_min, const real y_max,
    write_only image2d_t result
) {
    const int2 id = ID_2D;
    const real2 v = TRANSLATE_INV_Y(id, SIZE_2D, x_min, x_max, y_min, y_max);
    
    if (NEAR(v.y, v.x, ABS_ERROR)) {
        write_imageui(result, id, CROSSING_COLOR);
#ifdef CARRYING_FUNCTION_NO_EXTRA_PARAM
    } else if (NEAR(v.y, carrying_function(v.x), ABS_ERROR)) {
#else
    } else if (NEAR(v.y, carrying_function(v.x, lambda), ABS_ERROR)) {
#endif
        write_imageui(result, id, CARRY_COLOR);
    } else {
        write_imageui(result, id, FILL_COLOR);
    }
}

kernel void draw_iterative_diagram(
    const global real* samples,
    const int iter_count,
    const real param,
    
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    //const real4 sample = (real4)(samples[4*id], samples[4*id + 1], samples[4*id + 2], samples[4*id + 3]);
    
    
    
}

"""


class IterativeDiagram:

    def __init__(self, ctx, queue, width, height, carrying_function_source: str, type_config = float_config):
        self.ctx, self.queue, self.tc = ctx, queue, type_config
        self.width, self.height = width, height
        self.image, self.image_device = allocate_image(ctx, width, height)
        self._bounds = None
        self.program = cl.Program(ctx,
                                  self.tc.cl() + "\n" +
                                  carrying_function_source + "\n" +
                                  iterative_diagram_kernel
                                  ).build()

    def bounds(self, bounds: Bounds):
        self._bounds = bounds

    def __call__(self, lam, iter_count, bounds: Bounds = None, queue = None):
        if bounds is None:
            bounds = self._bounds

        if queue is None:
            queue = self.queue

        real, real_size = self.tc()

        self.program.draw_background(queue, (self.width, self.height), None,
                                     real(lam),
                                     real(bounds.x_min), real(bounds.x_max),
                                     real(bounds.y_min), real(bounds.y_max),
                                     self.image_device
                                     )

        cl.enqueue_copy(queue, self.image, self.image_device, origin=(0, 0), region=(self.width, self.height))

        return self.image





















