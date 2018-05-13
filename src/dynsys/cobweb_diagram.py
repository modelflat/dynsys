from .common import *

cobweb_diagram_source = """

#ifndef carrying_function
#define carrying_function map_function
#endif

// compute samples for diagram, single-threaded (usually iterations count is small enough, and we avoid copying data)
kernel void compute_samples(
    const real start, PARAM_SIGNATURES,
    const int skip_first,
    const int samples_count,
    global real* samples
) {
    real x = start;

    for (int i = 0; i < skip_first; ++i) {
        x = map_function(x, PARAM_VALUES);
    }
    
    samples[0] = x;
    samples[1] = x;
    for (int i = 2; i < samples_count; ++i) {
        x = map_function(x, PARAM_VALUES);
        samples[i] = x;   
    }
}

#define ABS_ERROR 2e-3

#define CROSSING_COLOR (uint4)(255, 0, 0, 255)
#define CARRY_COLOR    (uint4)(128, 255, 0, 255)
#define FILL_COLOR     (uint4)(255)

// draw background (secant line and carrying function) for this cobweb diagram
kernel void draw_background(
    PARAM_SIGNATURES,
    const real x_min, const real x_max, const real y_min, const real y_max,
    write_only image2d_t result
) {
    const int2 id = ID_2D;
    const real2 v = TRANSLATE_2D_INV_Y(id, SIZE_2D, x_min, x_max, y_min, y_max);
    
    if (NEAR(v.y, v.x, ABS_ERROR)) {
        write_imageui(result, id, CROSSING_COLOR);
    } else if (NEAR(v.y, carrying_function(v.x, PARAM_VALUES), ABS_ERROR * 5)) {
        write_imageui(result, id, CARRY_COLOR);
    } else {
        write_imageui(result, id, FILL_COLOR);
    }
}

#define ITER_COLOR (uint4)(0, 0, 0, 255)

kernel void draw_cobweb_diagram(
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

    if (isnan(x.s0) || isnan(x.s1) || isnan(x.s2)) {
        return;   
    }

    const int2 p1 = TRANSLATE_BACK_2D_INV_Y( x.s01, x_min, x_max, y_min, y_max, size );
    const int2 p2 = TRANSLATE_BACK_2D_INV_Y( x.s12, x_min, x_max, y_min, y_max, size );
    
    int2 line = (int2)(min(p1.x, p2.x), max(p1.x, p2.x));
    if (p1.y < height && p1.y >= 0) {
        for (int i = clamp(line.s0, 0, width); i <= line.s1; ++i) {
            if (i < width && i >= 0) {
                write_imageui(result, (int2)(i, p1.y), ITER_COLOR);
            }
        }
    }
    
    line = (int2)(min(p1.y, p2.y), max(p1.y, p2.y));
    if (p2.x < width && p2.x >= 0) {
        for (int i = clamp(line.s0, 0, height); i <= line.s1; ++i) {
            if (i < height && i >= 0) {
                write_imageui(result, (int2)(p2.x, i), ITER_COLOR);
            }
        }
    }
}

"""


class CobwebDiagram(ComputedImage):

    def __init__(self, ctx, queue, width, height, bounds, carrying_function_source, param_count=1, type_config=float_config):
        ComputedImage.__init__(self, ctx, queue, width, height, bounds,
                               carrying_function_source, generate_param_code(param_count), cobweb_diagram_source,
                               type_config=type_config)
        self.param_count = param_count
        self.compute_samples = self.program.compute_samples

    def __call__(self, x0, iter_count, *params, skip_first=0, bounds=None, queue=None):
        if bounds is None:
            bounds = self.bounds
        if queue is None:
            queue = self.queue

        iter_count = max( iter_count - skip_first, 1 )

        real, real_size = self.tc()

        samples_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, real_size * iter_count)

        param_list = make_param_list(self.param_count, params, real)

        self.compute_samples.set_args(real(x0), *param_list,
                                      np.int32(skip_first),
                                      np.int32(iter_count), samples_device)

        cl.enqueue_task(queue, self.compute_samples)

        self.program.draw_background(queue, (self.width, self.height), None,
                                     *param_list,
                                     real(bounds.x_min), real(bounds.x_max),
                                     real(bounds.y_min), real(bounds.y_max),
                                     self.image_device
                                     )

        self.program.draw_cobweb_diagram(queue, (iter_count,), None,
                                         samples_device,
                                         real(bounds.x_min), real(bounds.x_max),
                                         real(bounds.y_min), real(bounds.y_max),
                                         np.int32(self.width), np.int32(self.height),
                                         self.image_device
                                         )

        return self.read_from_device(queue)
