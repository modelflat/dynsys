BIFURCATION_TREE_KERNEL_SOURCE = """

#define real double
#define real2 double2

real2 map_function(real2 v, real param_A, real param_B) {
    real xp = 1 - param_A*v.x*v.x - param_B*v.y; 
    real yp = v.x;
    return (real2)(xp, yp);
}

kernel void clear(write_only image2d_t img) {
    write_imageui(img, (int2)(get_global_id(0), get_global_id(1)), (uint4)(0));
}


kernel void prepare_bifurcation_tree(
    const real param_A, const real param_B,
    const int use_a,
    const real start, const real stop,
    const real x_start, const real y_start,
    const real x_max,
    const int skip, const int samples_count,

    global real* result,
    global real2* result_minmax
) {
    const int id = get_global_id(0);
    const real value = start + ( (stop - start)*id ) / get_global_size(0);

    real2 v = (real2)(x_start, y_start);
    real min_ = x_start;
    real max_ = x_start;
    for (int i = 0; i < skip; ++i) {
        if (use_a) {
            v = map_function(v, param_A, value);
        } else {
            v = map_function(v, value, param_B);
        }
        //if (x < min_ && x > -x_max) min_ = x;
        //if (x > max_ && x < x_max) max_ = x;
    }


    for (int i = 0; i < samples_count; ++i) {
        if (use_a) {
            v = map_function(v, param_A, value);
        } else {
            v = map_function(v, value, param_B);
        }
        if (v.y < min_ && v.y > -x_max) min_ = v.y;
        if (v.y > max_ && v.y < x_max) max_ = v.y;
        result[id * samples_count + i] = v.y;
    }

    result_minmax[id] = (real2)(min_, max_); // save minmax
}

kernel void draw_bifurcation_tree(
    const global double* samples,
    const int samples_count,
    const real min_, const real max_,
    const real height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    const real part = (max_ - min_) / height;
    samples += id * samples_count;
    for (int i = 0; i < samples_count; ++i) {
        int y_coord = (samples[i] - min_) / part;
        write_imageui(result, (int2)(id, height - y_coord ), (uint4)((uint3)(0), 255));
    }
}

"""

DYNAMIC_MAP_KERNEL_SOURCE = """

#define real double
#define real2 double2

real2 map_function(real2 v, real param_A, real param_B) {
    real xp = 1 - param_B*v.x*v.x - param_A*v.y; 
    real yp = v.x;
    return (real2)(xp, yp);
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
        return 0.3;
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
            return (float3)(1.0, 0.5, 1.0)*d;
        default:
            return count == 8 ? 1 : .7;
    }
#endif
}

kernel void compute_map(
    const real x_min, const real x_max,
    const real y_min, const real y_max,
    const real x_start, const real y_start,
    const int samples_count,
    const int skip,
    global real2* samples,
    write_only image2d_t map
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    samples += (id.x * get_global_size(1) + id.y)*samples_count;
    const real xVal = x_min + id.x * ((x_max - x_min) / get_global_size(0));
    const real yVal = y_min + (get_global_size(1) - id.y) * ((y_max - y_min) / get_global_size(1));

    #define APPLY_MAP(x) map_function((x), xVal, yVal)

    real2 v = (real2)(xVal, yVal);

    for (int i = 0; i < skip; ++i) {
        v = APPLY_MAP(v);
    }

    int uniques = 0;
    for (int i = 0; i < samples_count; ++i) {
        v = APPLY_MAP(v);
        samples[i] = v;
        //if (samples_count <= 16) {
            int found = 0;
            for (int j = 0; j < i; ++j) {
                if (fabs(samples[j].x - v.x) < VALUE_DETECTION_PRECISION &&
                    fabs(samples[j].y - v.y) < VALUE_DETECTION_PRECISION) {
                    found = 1;
                    break;
                }
            }
            if (!found) ++uniques;
        //}
    }
    
    write_imageui(map, id, convert_uint4_rtz(255*(float4)(color_for_count(uniques, samples_count), 1.0)).zyxw);
}

"""


ATTRACTOR_KS = """

#define real double
#define real2 double2

real2 system(real2 v, real param_A, real param_B) {
    real xp = 1 - param_B*v.x*v.x - param_A*v.y; 
    real yp = v.x;
    return (real2)(xp, yp);
}

kernel void clear(write_only image2d_t img) {
    write_imageui(img, (int2)(get_global_id(0), get_global_id(1)), (uint4)(1.0));
}

kernel void draw_phase_portrait(
    const real a, const real b,
    const real x_min, const real x_max, const real y_min, const real y_max,
    const int step_count, 
    const int w, const int h,
    write_only image2d_t result
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    const real2 grid_step = (real2)((x_max - x_min)/get_global_size(0), (y_max - y_min)/get_global_size(1));
    
    real2 point = (real2)(x_min + id.x*grid_step.x, y_min + id.y*grid_step.y);
    
    for (int i = 0; i < step_count; ++i) {
        point = system(point, a, b);
    }
    int2 coord = (int2)( (point.x - x_min)/(x_max - x_min)*w, (point.y - y_min)/(y_max - y_min)*h);
    write_imageui(result, coord, (uint4)((uint3)(0), 255));
    
}

"""