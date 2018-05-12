import numpy as np
import pyopencl as cl

from dynsys.common import ComputedImage, float_config

parameter_map_source = """

void makeHeap(global real* data, int n, int i) {
    while (true) {
        int smallest = i;
        int l = (i << 1) + 1;
        int r = (i << 1) + 2;

        if (l < n && data[l] < data[smallest]) {
            smallest = l;
        }
        if (r < n && data[r] < data[smallest]) {
            smallest = r;
        }
        if (smallest == i) {
            return; // already smallest
        }
        real t = *(data + i); *(data + i) = *(data + smallest); *(data + smallest) = t;
        i = smallest;
    }
}

void heapSort(global real* data, int n)
{
    for (int i = n / 2 - 1; i >= 0; --i) {
        makeHeap(data, n, i);
    }
    
    for (int i = n - 1; i >= 0; --i) {
        real t = *(data); *(data) = *(data + i); *(data + i) = t;
        makeHeap(data, i, 0);
    }
}

float3 hsv2rgb(float3 hsv) {
    const float c = hsv.y * hsv.z;
    const float x = c * (1 - fabs(fmod( hsv.x / 60, 2 ) - 1));
    float3 rgb;
    if      (0 <= hsv.x && hsv.x < 60) {
        rgb = (float3)(c, x, 0);
    } else if (60 <= hsv.x && hsv.x < 120) {
        rgb = (float3)(x, c, 0);
    } else if (120 <= hsv.x && hsv.x < 180) {
        rgb = (float3)(0, c, x);
    } else if (180 <= hsv.x && hsv.x < 240) {
        rgb = (float3)(0, x, c);
    } else if (240 <= hsv.x && hsv.x < 300) {
        rgb = (float3)(x, 0, c);
    } else {
        rgb = (float3)(c, 0, x);
    }
    return (rgb + (hsv.z - c));
}

#define VALUE_DETECTION_PRECISION 1e-3

//#define GENERATE_COLORS

float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.0;
    }
#ifdef GENERATE_COLORS
    const float d = (float)total / (float)count;
    return hsv2rgb((float3)(d * 360.0, 1.0, 1.0));
#else
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
            return (float3)(0.5, 0.5, 1.0)*d;
        default:
            return count == 8 ? 1 : 0;
    }
#endif
}

kernel void compute_map(
    const real x_min, const real x_max,
    const real y_min, const real y_max,
    const real x0,
    const int samples_count,
    const int skip,
    global real* samples,
    write_only image2d_t map
) {
    const int2 id = ID_2D;
    samples += (id.x * get_global_size(1) + id.y)*samples_count;
    const real2 v = TRANSLATE_2D_INV_Y(id, SIZE_2D, x_min, x_max, y_min, y_max);
    
    real x = x0;

    for (int i = 0; i < skip; ++i) {
        x = map_function(x, v.x, v.y);
    }

    int uniques = 0;
    for (int i = 0; i < samples_count; ++i) {
        x = map_function(x, v.x, v.y);
        samples[i] = x;
        
        if (fabs(x) > 1e2) {
            write_imageui(map, id, (uint4)(255));
            return;
        }
        
        if (samples_count <= 16) {
            int found = 0;
            for (int j = 0; j < i; ++j) {
                if (fabs(x - samples[j]) < VALUE_DETECTION_PRECISION) {
                    found = 1;
                    break;
                }
            }
            if (!found) ++uniques;
        }
    }
    
    if (samples_count > 16) {
        heapSort(samples, samples_count);
        real prev = samples[0]; 
        for (int i = 1; i < samples_count; ++i) {
            if (fabs(samples[i] - prev) > VALUE_DETECTION_PRECISION) {
                prev = samples[i];
                ++uniques;
            }
        }
    }

    write_imageui(map, id, convert_uint4_rtz(255*(float4)(color_for_count(uniques, samples_count), 1.0)).zyxw);
}

"""


class ParameterMap(ComputedImage):

    def __init__(self, ctx, queue, width, height, bounds, map_function_source, type_config=float_config):
        super().__init__(ctx, queue, width, height, bounds, map_function_source, parameter_map_source, type_config=type_config)

    def __call__(self, x0, samples_count, skip, bounds=None):
        if bounds is None:
            bounds = self.bounds

        real, real_size = self.tc()

        samples_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                   size=self.width * self.height * samples_count * real_size)

        self.program.compute_map(self.queue, (self.width, self.height), None,
                                 real(bounds.x_min), real(bounds.x_max),
                                 real(bounds.y_min), real(bounds.y_max),
                                 real(x0),
                                 np.int32(samples_count), np.int32(skip),
                                 samples_device, self.image_device)

        return self.read_from_device()
