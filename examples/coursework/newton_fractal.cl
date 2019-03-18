#include "newton.clh"
#include "heapsort.clh"

#define POINT_RADIUS 1
#define POINT_COLOR (float4)(0.0, 0.0, 0.0, 1.0)
#include "util.clh"

// Draws Newton fractal (phase plot)
kernel void newton_fractal(
    // algo parameters
    const int skip,
    const int iter,

    // plane bounds
    const real4 bounds,

    // fractal parameters
    const real2 c,
    const real h,
    const real alpha,

    // root selection
    // seed
    const ulong seed,

    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    // starting point
    const int use_single_point,
    const real2 z0,

    // output
    write_only image2d_t out
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    real2 point = use_single_point ? z0 : point_from_id(bounds) / 3;

    const real A =  h * alpha / 3;
    const real B = -h * (1 - alpha) / 3;

    int seq_pos = 0;

    for (int i = 0; i < skip; ++i) {
        point = next_point(point, c, A, B, &rng_state, &seq_pos, seq_size, seq);
    }

    const int2 image_size = (int2)(get_image_width(out), get_image_height(out));

    for (int i = 0, frozen = 0; i < iter; ++i) {
        const int2 coord = to_size(point, bounds, image_size);

        if (in_image(coord, image_size)) {
            put_point(out, coord, image_size);
            frozen = 0;
        } else {
            if (++frozen > 32) {
                // this likely means that solution is going to approach infinity
                // printf("[OCL] error at slave %d: frozen!\n", get_global_id(0));
                break;
            }
        }

        point = next_point(point, c, A, B, &rng_state, &seq_pos, seq_size, seq);
    }
}

// Computes samples for parameter map
kernel void compute_points(
    const real2 z0,
    const real2 c,

    const real4 bounds,

    const int skip,
    const int iter,
    const float tol,

    // root selection
    // seed
    const ulong seed,

    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    global ulong* result
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    const int2 coord = { get_global_id(0), get_global_id(1) };
    const real2 uv = convert_real2(coord) / convert_real2((int2)(get_global_size(0), get_global_size(1)));
    const real2 param = bounds.s02 + uv * (bounds.s13 - bounds.s02);

    const real A = param.x * param.y / 3.0;
    const real B = -param.x * (1 - param.y) / 3.0;

    int seq_pos = 0;
    int mul = convert_int_rtz(1.0 / tol);

    real2 point = z0;
    result += (coord.y * get_global_size(0) + coord.x) * iter;

    for (int i = 0; i < skip; ++i) {
        point = next_point(point, c, A, B, &rng_state, &seq_pos, seq_size, seq);
    }

    for (int i = 0; i < iter; ++i) {
        point = next_point(point, c, A, B, &rng_state, &seq_pos, seq_size, seq);
        result[i] = as_ulong(convert_int2_rtz(point / tol));
    }
}

// Draws parameter map using computed samples
kernel void draw_periods(
    const int num_points,
    const global float* color_scheme,
    global ulong* points,
    global int* periods,
    write_only image2d_t out
) {
    const int2 coord = { get_global_id(0), get_global_id(1) };
    points  += (coord.y * get_global_size(0) + coord.x) * num_points;

    int unique = count_unique(points, num_points);
    float h = (unique % 23) / (float)(22);
    float v = 1.0 - (unique / (float)(num_points));

    float3 hsvcolor = (float3)(240.0 * h, 1.0,v);

    float3 color = hsv2rgb(hsvcolor);

    periods[(get_global_size(1) - coord.y - 1) * get_global_size(0) + coord.x] = unique;
    write_imagef(out, (int2)(coord.x, get_global_size(1) - coord.y - 1), (float4)(color, 1.0));
}
