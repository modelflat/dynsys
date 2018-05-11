import pyopencl as cl
import numpy as np
from .common import Bounds, allocate_image, make_cl_source, float_config

iteration_diagram_source = """

// compute samples for diagram, single-threaded (usually iterations count are small enough, and we avoid copying data)
kernel void compute_samples(
    const real start, const real lambda,
    const int samples_count,
    global real* samples
) {
    real x = start;
    samples[0] = x;
    samples[1] = x;
    for (int i = 2; i < samples_count; ++i) {
        x = map_function(x, lambda);
        samples[i] = x; 
    }
}

#define ABS_ERROR 2e-3

#define CROSSING_COLOR (uint4)(255, 0, 0, 255)
#define CARRY_COLOR    (uint4)(128, 255, 0, 255)
#define FILL_COLOR     (uint4)(255)

// draw background (secant line and carrying function) for this iteration diagram
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
    } else if (NEAR(v.y, carrying_function(v.x, lambda), ABS_ERROR * 5)) {
#endif
        write_imageui(result, id, CARRY_COLOR);
    } else {
        write_imageui(result, id, FILL_COLOR);
    }
}

#define ITER_COLOR (uint4)(0, 0, 0, 255)

kernel void draw_iteration_diagram(
    const global real* samples,
    const real x_min, const real x_max, const real y_min, const real y_max,
    const int width, const int height,    
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    
    if (id + 2 >= get_global_size(0)) {
        return;
    }
    
    const real3 x = (real3)(samples[id], samples[id+1], samples[id+2]);
    const int2 size = (int2)(width, height);
    const int2 p1 = TRANSLATE_BACK_INV_Y( x.s01, size, x_min, x_max, y_min, y_max );
    const int2 p2 = TRANSLATE_BACK_INV_Y( x.s12, size, x_min, x_max, y_min, y_max );
    
    int2 line = (int2)(min(p1.x, p2.x), max(p1.x, p2.x));
    for (int i = clamp(line.s0, 0, width); i <= line.s1; ++i) {
        if (i < width && i >= 0) {
            write_imageui(result, (int2)(i, p1.y), ITER_COLOR);
        }
    }
    
    line = (int2)(min(p1.y, p2.y), max(p1.y, p2.y));
    for (int i = clamp(line.s0, 0, height); i <= line.s1; ++i) {
        if (i < height && i >= 0) {
            write_imageui(result, (int2)(p2.x, i), ITER_COLOR);
        }
    }
        
}

"""


class IterationDiagram:

    def __init__(self, ctx, queue, width, height, carrying_function_source: str, type_config=float_config):
        self.ctx, self.queue, self.tc = ctx, queue, type_config
        self.width, self.height = width, height
        self.image, self.image_device = allocate_image(ctx, width, height)
        self._bounds = None
        self.program = cl.Program(ctx, make_cl_source(
            carrying_function_source, iteration_diagram_source, type_config=type_config
        )).build()
        self.compute_samples = self.program.compute_samples

    def bounds(self, bounds: Bounds):
        self._bounds = bounds

    def __call__(self, x0, lam, iter_count, bounds: Bounds = None, queue=None):
        if bounds is None:
            bounds = self._bounds

        if queue is None:
            queue = self.queue

        real, real_size = self.tc()

        samples_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, real_size*iter_count)

        self.compute_samples.set_args(
            real(x0), real(lam), np.int32(iter_count),
            samples_device
        )
        cl.enqueue_task(queue, self.compute_samples)
        self.program.draw_background(queue, (self.width, self.height), None,
                                     real(lam),
                                     real(bounds.x_min), real(bounds.x_max),
                                     real(bounds.y_min), real(bounds.y_max),
                                     self.image_device
                                     )

        self.program.draw_iteration_diagram(queue, (iter_count,), None,
                                            samples_device,
                                            real(bounds.x_min), real(bounds.x_max),
                                            real(bounds.y_min), real(bounds.y_max),
                                            np.int32(self.width), np.int32(self.height),
                                            self.image_device
                                            )

        cl.enqueue_copy(queue, self.image, self.image_device, origin=(0, 0), region=(self.width, self.height))

        return self.image
