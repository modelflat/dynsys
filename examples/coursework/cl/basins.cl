#include "newton.clh"
#include "util.clh"

int binary_search(int, const global real*, real2);
int binary_search(int size, const global real* arr, real2 value) {
    int l = 0, r = size;

    while (l < r) {
        const int mid = (r + l) / 2;
        const real2 mid_value = vload2(mid, arr);

        if (pair_eq(mid_value, value)) {
            return mid;
        }
        if (r == l + 1) {
            return -1;
        }
        if (pair_lt(mid_value, value)) {
            l = mid;
        } else {
            r = mid;
        }
    }

    return (r + l) / 2;
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

// Draw basins' bounds and color them approximately
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

    float value = 0.8;

    real arg = atan2(end.y, end.x) + PI;

    if        (0 <= arg && arg < 2 * PI / 3) {
        col = 0;
    } else if (2 * PI / 3 <= arg && arg < 4 * PI / 3) {
        col = 60;
    } else if (4 * PI / 3 <= arg && arg < 2 * PI) {
        col = 120;
    }

//    if (end.x >= 0 && end.y >= 0) {
//        col = 0;
//        if (!(origin.x >= 0 && origin.y >= 0)) {
//            mod = 0.5;
//        }
//    }
//    else if (end.x >= 0 && end.y < 0) {
//        col = 60;
//        if (!(origin.x >= 0 && origin.y < 0)) {
//            mod = 0.5;
//        }
//    }
//    else if (end.x < 0 && end.y >= 0) {
//        col = 120;
//        if (!(origin.x < 0 && origin.y >= 0)) {
//            mod = 0.5;
//        }
//    }
//    else if (end.x < 0 && end.y < 0) {
//        col = 180;
//        if (!(origin.x < 0 && origin.y < 0)) {
//            mod = 0.5;
//        }
//    }

    float3 color = hsv2rgb((float3)(
        col,
        value,
        edge
    ));

    write_imagef(image, COORD_2D_INV_Y, (float4)(color, 1.0));
//    write_imagef(image, (int2)(coord.x, get_global_size(1) - coord.y - 1), (float4)(color, 1.0));
//    write_imagef(image, (int2)(get_global_size(1) - coord.y - 1, coord.x), (float4)(color, 1.0));
}

// Draw basins in precise colors
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