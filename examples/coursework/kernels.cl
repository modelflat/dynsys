#include "newton.clh"
#include "heapsort.clh"

#define POINT_RADIUS 1
#define POINT_COLOR (float4)(0.0, 0.0, 0.0, 1.0)
#define DETECTION_PRECISION 1e-4
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

    INIT_VARIABLES(use_single_point ? z0 : point_from_id(bounds) / 1.5, c, h, alpha);

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    const int2 image_size = get_image_dim(out);

    for (int i = 0, frozen = 0; i < iter; ++i) {
        const int2 coord = to_size(Z_VAR, bounds, image_size);

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

        NEXT_POINT(c, seq_size, &rng_state);
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

    // NOTE flipped y
    const int2 coord = COORD_2D_INV_Y;
    result += (coord.y * get_global_size(0) + coord.x) * iter;

    const real2 param = point_from_id_dense(bounds);

    INIT_VARIABLES(z0, c, param.x, param.y);
//    INIT_VARIABLES(z0, c, param.x, 0);

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    for (int i = 0; i < iter; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
        result[i] = as_ulong(convert_int2_rtz(Z_VAR / tol));
    }
}

// Draws parameter map using computed samples
kernel void draw_periods(
    const int scale_factor,
    const int num_points,
    const global float* color_scheme,
    global ulong* points,
    global int* periods,
    write_only image2d_t out
) {
    // NOTE flipped y to correspond compute_periods
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_size(0) / scale_factor;

    points += (coord.y * size_x + coord.x) * num_points;

    int unique = count_unique(points, num_points);
    float h = (unique % 23) / (float)(22);
    float v = 1.0 - (unique / (float)(num_points));

    float3 hsvcolor = (float3)(240.0 * h, 1.0,v);

    float3 color = hsv2rgb(hsvcolor);

    periods[coord.y * size_x + coord.x] = unique;
    // NOTE flipped y to correspond to image coordinates (top left (0,0))
    write_imagef(out, COORD_2D_INV_Y, (float4)(color, 1.0));
}

// Compute where points would be after N iterations
kernel void compute_basins(
    const int skip,

    const real4 bounds,

    const real2 c,
    const real h,
    const real alpha,

    const ulong seed,
    const int seq_size,
    const global int* seq,

    global real* endpoints
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    // NOTE y flipped
    const int2 coord = COORD_2D_INV_Y;

    INIT_VARIABLES(point_from_id_dense(bounds), c, h, alpha);

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    Z_VAR = round_point(Z_VAR, 4);

    vstore2(Z_VAR, coord.y * get_global_size(0) + coord.x, endpoints);
}

//
kernel void draw_basins(
    const int scale_factor,
    const real4 bounds,
    const global real* endpoints,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_size(0) / scale_factor;
    const real2 origin = point_from_id_dense(bounds);
    const real2 end = vload2(coord.y * size_x + coord.x, endpoints);

    real x_gran = (real)(1) / (get_global_size(0) - 1);
    real y_gran = (real)(1) / (get_global_size(1) - 1);
    real av_len = 0.0;
    float edge = 1.0;

    if (coord.x > 0) {
        const real2 west_end = vload2(coord.y * size_x + coord.x - 1, endpoints);
        const real dst = length(west_end - end);
        av_len += dst;
        if (dst > x_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.x < get_global_size(1) - 1) {
        const real2 east_end = vload2(coord.y * size_x + coord.x + 1, endpoints);
        const real dst = length(east_end - end);
        av_len += dst;
        if (dst > x_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.y > 0) {
        const real2 north_end = vload2((coord.y - 1) * size_x + coord.x, endpoints);
        const real dst = length(north_end - end);
        av_len += dst;
        if (dst > y_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.y < get_global_size(1) - 1) {
        const real2 south_end = vload2((coord.y + 1) * size_x + coord.x, endpoints);
        const real dst = length(south_end - end);
        av_len += dst;
        if (dst > y_gran) {
            edge -= 0.25f;
        }
    }

    av_len /= 4;

    float mod = 0.005 / length(end - origin);
//    float mod = length(end - origin);

    float col = 240;

    if (mod > 1) {
//        printf("STABLE: %.6f, %.6f\n", origin.x, origin.y);

    }

    mod = 0.5;

    if (end.x >= 0 && end.y >= 0) {
        col = 0;
        if (!(origin.x >= 0 && origin.y >= 0)) {
            mod = 0.5;
        }
    }
    else if (end.x >= 0 && end.y < 0) {
        col = 60;
        if (!(origin.x >= 0 && origin.y < 0)) {
            mod = 0.5;
        }
    }
    else if (end.x < 0 && end.y >= 0) {
        col = 120;
        if (!(origin.x < 0 && origin.y >= 0)) {
            mod = 0.5;
        }
    }
    else if (end.x < 0 && end.y < 0) {
        col = 180;
        if (!(origin.x < 0 && origin.y < 0)) {
            mod = 0.5;
        }
    }

    float3 color = hsv2rgb((float3)(
        col,
        mod,
        edge
    ));

    write_imagef(image, COORD_2D_INV_Y, (float4)(color, 1.0));
//    write_imagef(image, (int2)(coord.x, get_global_size(1) - coord.y - 1), (float4)(color, 1.0));
//    write_imagef(image, (int2)(get_global_size(1) - coord.y - 1, coord.x), (float4)(color, 1.0));
}

kernel void draw_basins_colored(
    const int scale_factor,
    const int attraction_points_count,
    const global real* attraction_points, // TODO make real and use vload in binary_search
    const global real* result,
    write_only image2d_t map
) {
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_id(0) / scale_factor;

    const real2 val = vload2(coord.y * size_x + coord.x, result);

    const int color_idx = binary_search(attraction_points_count, attraction_points, val);

    const int v = 1 - (int)(color_idx == -1 || length(val) < DETECTION_PRECISION);
    const float ratio = (float)(color_idx) / (float)(attraction_points_count);

    float3 color = hsv2rgb((float3)(240.0 * ratio, 1.0, v));

    write_imagef(map, COORD_2D_INV_Y, (float4)(color, 1.0));
}
