from dynsys.common import ComputedImage, float_config, generate_param_code, make_param_list
import pyopencl as cl
import numpy as np

bifurcation_tree_source = """

kernel void compute_bifurcation_tree(
    const real x0,
    PARAM_SIGNATURES, const int active_param,
    const real start, const real stop,
    const real max_allowed_value,
    const int skip, const int samples_count,
    
    global real* result, global real2* result_minmax
) {
    const int id = get_global_id(0);
    const real param = TRANSLATE(id, get_global_size(0), start, stop);

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

#define TREE_COLOR (uint4)((uint3)(0), 255)

kernel void draw_bifurcation_tree(
    const global real* samples,
    const int samples_count,
    const real min_, const real max_,
    const real height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    samples += id * samples_count;
    for (int i = 0; i < samples_count; ++i) {
        int2 coord = (int2)(id, TRANSLATE_BACK_INV(samples[i], min_, max_, height));
        write_imageui(result, coord, TREE_COLOR);
    }
}

"""


class BifurcationTree(ComputedImage):

    def __init__(self, ctx, queue, width, height, map_function_source, param_count=1, type_config=float_config):
        ComputedImage.__init__(self, ctx, queue, width, height, None,
                               map_function_source, generate_param_code(param_count), bifurcation_tree_source,
                               type_config=type_config)
        self.param_count = param_count

    def __call__(self, x0, samples_count, param_start, param_stop, *params, active_idx=0, skip=0, max_allowed_value=1000):
        real, real_size = self.tc()

        result_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=samples_count*self.width * real_size)
        result_minmax_device = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=self.width * 2 * real_size)

        param_list = make_param_list(self.param_count, params, real, active_idx=active_idx)

        self.program.compute_bifurcation_tree(
            self.queue, (self.width, ), None,
            real(x0),
            *param_list, np.int32(active_idx),
            real(param_start), real(param_stop),
            real(max_allowed_value),
            np.int32(skip), np.int32(samples_count),
            result_device, result_minmax_device
        )

        result_minmax = np.empty((self.width*2,), dtype=real)

        cl.enqueue_copy(self.queue, result_minmax, result_minmax_device)

        min_, max_ = min(result_minmax), max(result_minmax)

        self.clear()

        self.program.draw_bifurcation_tree(
            self.queue, (self.width, ), None,
            result_device,
            np.int32(samples_count),
            real(min_), real(max_),
            real(self.height),
            self.image_device
        )

        cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0, 0), region=(self.width, self.height))

        return self.image
