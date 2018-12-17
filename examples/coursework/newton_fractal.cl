#include "complex.clh"
#include "random.clh"

#ifdef cl_khr_fp64
    #define PRECISION 1e-9
#else
    #define PRECISION 1e-4
#endif

#define DYNAMIC_COLOR 0

float3 hsv2rgb(float3);
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
    return (rgb + (hsv.z - c)); //* 255;
}

// Draws newton fractal
kernel void newton_fractal(
    // plane bounds
    real min_x, real max_x, real min_y, real max_y,
    // fractal parameters
    const real2 C,
    int backward,
    const real alpha,
    const real h,
    // how many times solve equation for certain initial point
    int points_count,
    // how many initial steps will be skipped
    int iter_skip,
    // seed - seed value for pRNG; see "random.clh"
    ulong seed,
    // color. this color will only be used as static!
    global float4* color_in,
    // image buffer for output
    write_only image2d_t out_image)
{
    // color
    #if (DYNAMIC_COLOR)
        float4 color = {fabs(sin(PI * h / 3.)), fabs(cos(PI * h / 3.)), 0.0, 0.0};
        float3 color_hsv = { 0.0, 1.0, 1.0 };
    #else
        float4 color = {0.0, 0.0, 0.0, 1.0};
    #endif
    // initialize pRNG
    uint2 rng_state;
    init_state(seed, &rng_state);
    // spans, scales, sizes
    real span_x = max_x - min_x;
    real span_y = max_y - min_y;
    int image_width = get_image_width(out_image);
    int image_height = get_image_height(out_image);
    real scale_x = span_x / image_width;
    real scale_y = span_y / image_height;
    int2 coord;
    // for each run
    real2 roots[3];
    //real2 a;
    const real2 c = (alpha + 1.0f / ( 1 - h )) * alpha * C;
    const real2 b_modifier = { -(1.0f / (h - h*h)), 0.0 };
    real total_distance = 0.0;

    // choose starting point
    real2 point = {
        ((random(&rng_state)) * span_x + min_x) / 2.0,
        ((random(&rng_state)) * span_y + min_y) / 2.0
    };
    uint is = iter_skip;
    int frozen = 0;

    // iterate through solutions of cubic equation
    for (int i = 0; i < points_count; ++i) {

        // compute next point:
        uint root_number = (as_uint(random(&rng_state)) >> 7) % 3;
        if (backward) {
            const real2 zn = point;
            const real2 zero = (real2)(0.0, 0.0);
            //
            real2 z_a = alpha * zn;
            real2 z_a2 = { z_a.x*z_a.x - z_a.y*z_a.y, 2.0*z_a.x*z_a.y };
            z_a2 += 1.0f / h;
            const real2 z_na2 = { zn.x*z_a2.x - zn.y*z_a2.y, zn.y*z_a2.x + zn.x*z_a2.y };
            const real2 c_eq = z_na2 + c;
            //
            solve_cubic(zero, b_modifier, c_eq, 1e-8, root_number, roots);

            real distance_from_prev = length(point - roots[root_number]);
            total_distance += distance_from_prev;

            point = roots[root_number];
        } else {
            const real2 zn = point;
            const real2 zero = {0.0, 0.0};
            //
            real2 z_a = alpha * zn;
            real2 z_a2 = { z_a.x*z_a.x - z_a.y*z_a.y, 2.0*z_a.x*z_a.y };
            z_a2 += 1.0f / h;
            const real2 z_na2 = { zn.x*z_a2.x - zn.y*z_a2.y, zn.y*z_a2.x + zn.x*z_a2.y };
            const real2 c_eq = z_na2 + c;
            //
            solve_cubic(zero, b_modifier, c_eq, 1e-8, root_number, roots);

            real distance_from_prev = length(point - roots[root_number]);
            total_distance += distance_from_prev;

            point = roots[root_number];
        }

        // the first iter_skip points will  be skipped
        if (is == 0) {
            // transform coords:
            coord.x = (point.x - min_x) / scale_x;
            coord.y = image_height - 1 - (int)((point.y - min_y) / scale_y);

            // draw next point:
            #if (DYNAMIC_COLOR)
                //(1 - distance_from_prev / max_distance_from_prev)
                if (backward) {
                    color_hsv.x = convert_float(360.0 * (root_number / 3.0));
                } else {
                    color_hsv.x = convert_float(360.0 * sin(total_distance));
                }
                color_hsv.y = convert_float(1.0 * ((float)(i) / (points_count - iter_skip)));
                color_hsv.z = convert_float(1.0 * (total_distance / (max_distance_from_prev * (points_count - iter_skip))));
            #endif
            if (coord.x < image_width && coord.y < image_height && coord.x >= 0 && coord.y >= 0) {
                #if DYNAMIC_COLOR
                    write_imagef(out_image, coord, (float4)(hsv2rgb( color_hsv ), 1.0));
                #else
                    write_imagef(out_image, coord, color);
                #endif
                frozen = 0;
            } else {
                if (++frozen > 15) {
                    // this generally means that solution is going to approach infinity
                    //printf("[OCL] error at slave %d: frozen!\n", get_global_id(0));
                    break;
                }
            }
        } else {
            --is;
        }
    }
}