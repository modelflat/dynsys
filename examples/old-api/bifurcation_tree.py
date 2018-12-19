from dynsys import ComputedImage, FLOAT, generateCode, SimpleApp, ParameterizedImageWidget, vStack
import pyopencl as cl
import numpy as np

from dynsys.LCE import dummyOption


bifurcation_tree_source = """

#undef TRANSLATE
#define TRANSLATE(id, size, min_, max_) \
    ((min_) + (id)*((max_) - (min_))/(size))

#undef TRANSLATE_BACK
#define TRANSLATE_BACK(v, min_, max_, size) \
    (((v) - (min_)) / ((max_) - (min_)) * (size))

#undef TRANSLATE_BACK_INV
#define TRANSLATE_BACK_INV(v, min_, max_, size) \
    ((size) - TRANSLATE_BACK((v), (min_), (max_), (size)))

kernel void compute_bifurcation_tree(
    real3 x,
    real p1, real p2, real p3,
    real sliceValue,
    const real start,
    const real stop,
    const int skip,
    const int samples_count,
    global real* result,
    global int* resultCount,
    global real* result_minmax
) {
    const int id = get_global_id(0);
    result += id * samples_count;
    p3 = TRANSLATE(id, get_global_size(0), start, stop);

    for (int i = 0; i < skip; ++i) {
        x = userFn(x, p1, p2, p3);
    }

    real3 prevX = x;
    real min_ = x.x;
    real max_ = x.x;
    int count = 0;
    for (int i = 0; i < samples_count; ++i) {
        x = userFn(x, p1, p2, p3);
        
        if (x.x < sliceValue && prevX.x >= sliceValue) {
            result[count++] = x.x;
            if (x.x < min_) {
                min_ = x.x;
            }
            if (x.x > max_) {
                max_ = x.x;
            }
        }
        
        prevX = x;
    }
    
    resultCount[get_global_id(0)] = count;
}

#define TREE_COLOR (float4)(0, 0, 0, 1.0)

kernel void draw_bifurcation_tree(
    const global real* samples,
    const global int* counts,
    const int samples_count,
    const real min_, const real max_,
    const real height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    samples += id * samples_count;
    
    for (int i = 0; i < counts[id]; ++i) {
        int2 coord = (int2)(id, TRANSLATE_BACK_INV(samples[i], min_, max_, height));
        write_imagef(result, coord, TREE_COLOR);
    }
}

"""


user_fn = r"""

#define real float
#define real3 float3

real3 userFn(real3 v, real a, real b, real r);
real3 userFn(real3 v, real a, real b, real r) {
    
}

"""


class BifurcationTree(ComputedImage):

    def __init__(self, ctx, queue, width, height, map_function_source, param_count=1):
        ComputedImage.__init__(self, ctx, queue, (width, height), (-1, 1, -1, 1),
                               map_function_source,
                               generateCode(FLOAT, parameterCount=param_count),
                               bifurcation_tree_source,
                               typeConfig=FLOAT, options=[dummyOption()])
        self.param_count = param_count

    def __call__(self, x0, samples_count, param_start, param_stop, p1, p2,
                 active_idx=0, skip=0, max_allowed_value=1000):
        real, real_size = self.typeConf()

        result_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=samples_count*self.imageShape[0] * real_size)
        result_minmax_device = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=self.imageShape[0] * 2 * real_size)

        t = np.float32

        sliceValue = .5

        resultCountHost = np.empty((self.imageShape[0],), dtype=np.int32)
        resultCountDevice = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=resultCountHost.nbytes)

        self.program.compute_bifurcation_tree(
            self.queue, (self.imageShape[0], ), None,
            np.array(x0, dtype=t),
            real(p1), real(p2), real(0),
            np.float32(sliceValue),
            real(param_start),
            real(param_stop),
            np.int32(skip),
            np.int32(samples_count),
            result_device,
            resultCountDevice,
            result_minmax_device
        )

        result_minmax = np.empty((self.imageShape[0]*2,), dtype=real)

        cl.enqueue_copy(self.queue, result_minmax, result_minmax_device)

        min_, max_ = min(result_minmax), max(result_minmax)

        self.clear()

        self.program.draw_bifurcation_tree(
            self.queue, (self.imageShape[0], ), None,
            result_device,
            resultCountDevice,
            np.int32(samples_count),
            real(min_), real(max_),
            real(self.imageShape[1]),
            self.deviceImage
        )

        return self.readFromDevice()


class Test(SimpleApp):

    def __init__(self):
        super().__init__("123")
        self.btree = BifurcationTree(self.ctx, self.queue, 512, 512, user_fn)
        self.btreeUi = ParameterizedImageWidget(bounds=(0, 1, 0, 1),
                                                names=("x", "y"),
                                                shape=(512, 512))

        self.setLayout(vStack(
            self.btreeUi
        ))

        self.draw()

    def draw(self):
        r = self.btree(.5, 512, 0, 3, 0.5, 0.5, active_idx=0, skip=512, max_allowed_value=10)
        self.btreeUi.setImage(r)


if __name__ == '__main__':
    Test().run()
