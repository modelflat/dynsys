from common2 import *

from dynsys import SimpleApp, allocateImage, vStack, Image2D


PARAM_MAP_SOURCE = r"""
#define real double
#define TWOPI (2 * M_PI)

inline real map(real x, real om, real k) { 
    return x + om + fmod(k * sin(x), TWOPI);
}

kernel void compute_parameter_map(
    const real om_min, const real om_max,
    const real k_min, const real k_max,
    const int skip, const int iter,
    real x, const int decimals,
    global int* res
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    
    const real om = om_min + id.x * (om_max - om_min) / get_global_size(0);
    const real k = k_min + id.y * (k_max - k_min) / get_global_size(1);
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, om, k);
    }
    
    res += (id.x + id.y * get_global_size(0)) * iter;
    
    for (int i = 0; i < iter; ++i) {
        res[i] = convert_int_rtz(x * pow(10.0, decimals));
        x = map(x, om, k);
    }
}

void make_heap(global int*, int, int);
void make_heap(global int* data, int n, int i) {
    while (true) {
        int smallest = i;
        int l = (i << 1) + 1;
        int r = (i << 1) + 2;

        if (l < n && data[l] > data[smallest]) {
            smallest = l;
        }
        if (r < n && data[r] > data[smallest]) {
            smallest = r;
        }
        if (smallest == i) {
            return; // already smallest
        }

        int t = *(data + i); *(data + i) = *(data + smallest); *(data + smallest) = t;

        i = smallest;
    }
}

void heap_sort(global int*, int);
void heap_sort(global int* data, int n) {
    for (int i = n / 2 - 1; i >= 0; --i) {
        make_heap(data, n, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        int t = *(data); *(data) = *(data + i); *(data + i) = t;
        make_heap(data, i, 0);
    }
}

inline int count_unique(global int* data, int n) {
    heap_sort(data, n);
    int prev = data[0];
    int uniques = 1;
    
    for (int i = 1; i < n; ++i) {
        int next = data[i];
        if (prev != next) {
            prev = next;
            ++uniques;
        }
    }
    
    return uniques;
}


inline float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.0;
    }
    const float d = count < 8? 1.0 : .5;
    switch(count % 8) {
        case 1:
            return (float3)(1.0, 0.0, 0.0)*d;
        case 2:
            return (float3)(0.0, 1.0, 0.0)*d;
        case 3:
            return (float3)(0.0, 0.0, 1.0)*d;
        case 4:
            return (float3)(1.0, 0.0, 1.0)*d;
        case 5:
            return (float3)(1.0, 1.0, 0.0)*d;
        case 6:
            return (float3)(0.0, 1.0, 1.0)*d;
        case 7:
            return (float3)(0.5, 0.0, 0.0)*d;
        default:
            return count == 8 ? .5 : 0;
    }
}

kernel void draw_parameter_map(
    const int iter,
    const global int* res,
    write_only image2d_t out
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    res += (id.x + id.y * get_global_size(0)) * iter;
    
    int uniques = count_unique(res, iter);
    
    float3 color = color_for_count(uniques, iter);
    
    write_imagef(out, (int2)(id.x, get_image_height(out) - 1 - id.y), (float4)(color, 1.0));
}

"""


LYAP_MAP = r"""

kernel void compute_lyap_map(
    const real om_min, const real om_max,
    const real k_min, const real k_max,
    const int skip, const int iter,
    real x,
    global real* res
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    
    const real om = om_min + id.x * (om_max - om_min) * get_global_size(0);
    const real k = k_min + id.y * (k_max - k_max) * get_global_size(1);
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, om, k);
    }
    
    res += id.x + id.y * get_global_size(0);
    
    // for (int i = 0; i < iter; ++i) {
    //     res[i] = convert_int_rtz(x * pow(10.0, decimals));
    //     x = map(x, om, k);
    // }
}

"""


class CircularMap:

    def __init__(self, ctx, shape):
        self.ctx = ctx
        self.prg = cl.Program(ctx, "\n".join([
            PARAM_MAP_SOURCE
        ])).build()
        self.shape = shape
        self.img = allocateImage(ctx, shape)

    def draw_param_map(self, queue, skip, iter, om_min, om_max, k_min, k_max, x0=0, decimals=3):
        """
        kernel void compute_parameter_map(
            const real om_min, const real om_max,
            const real k_min, const real k_max,
            const int skip, const int iter,
            const real x, const int decimals,
            global int* res
        )
        """
        clear_image(queue, self.img[1], self.shape)

        res_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * iter * numpy.prod(self.shape))

        self.prg.compute_parameter_map(
            queue, self.shape, None,
            numpy.float64(om_min),
            numpy.float64(om_max),
            numpy.float64(k_min),
            numpy.float64(k_max),
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0), numpy.int32(decimals),
            res_dev
        )

        self.prg.draw_parameter_map(
            queue, self.shape, None,
            numpy.int32(iter),
            res_dev,
            self.img[1]
        )

        return read(queue, *self.img, self.shape)


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("3")

        self.cm = CircularMap(self.ctx, (640, 480))

        self.param_map = Image2D()

        layout = vStack(
            self.param_map
        )

        self.setLayout(layout)

        self.draw_param_map()

    def draw_param_map(self, *_):
        texture = self.cm.draw_param_map(
            self.queue,
            skip=1 << 10, iter=1 << 6,
            om_min=0, om_max=2 * numpy.pi,
            k_min=0, k_max=4,
            decimals=4
        )
        self.param_map.setTexture(texture)


if __name__ == '__main__':
    App().run()
