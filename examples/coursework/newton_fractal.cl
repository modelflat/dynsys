#include "complex.clh"
#include "random.clh"

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

#define RANDOM_ROOT(rngState) ((as_uint(random(rngState)) >> 7) % 3)

inline real2 cCoef(real2 z, real2 c, real B) {
    return B / (1 + B) * c;
}

inline real2 aCoef(real2 z, real2 c, real A, real B) {
    return -(z + A*(z + cdiv(c, csq(z)))) / (1 + B);
}

inline real2 next_point(real2 z, real2 c, real A, real B, uint2* rngState) {
    const uint rootNumber = RANDOM_ROOT(rngState);
    real2 roots[3];
    solve_cubic_newton_fractal_optimized(
        aCoef(z, c, A, B), cCoef(z, c, B), 1e-8, rootNumber, roots
    );
    return roots[rootNumber];
}

inline real2 next_point_with_seq(real2 z, real2 c, real A, real B, int seq_size, int* seq_pos, const global int* seq) {
    if (*(seq_pos) >= seq_size) {
        *(seq_pos) = 0;
    }
    const int rootNumber = seq[(*seq_pos)++];
    real2 roots[3];
    solve_cubic_newton_fractal_optimized(
        aCoef(z, c, A, B), cCoef(z, c, B), 1e-8, rootNumber, roots
    );
    return roots[rootNumber];
}

#define COLOR (float4)(0.0, 0.0, 0.0, 1.0)

// Draws newton fractal
kernel void newton_fractal(
    // plane bounds
    const real4 bounds,
    // fractal parameters
    const real2 c, const real alpha, const real h,
    // iteration config
    const int iterations, const int skip,
    // seed - seed value for pRNG; see "random.clh"
    const int root_seq_size,
    const global int* root_seq,
    const ulong seed,
    // image buffer for output
    write_only image2d_t out_image)
{
    uint2 rng_state;
    init_state(seed, &rng_state);

    const real span_x = bounds.s1 - bounds.s0;
    const real span_y = bounds.s3 - bounds.s2;

    // choose starting point
    real2 point = {
        ((random(&rng_state)) * span_x + bounds.s0) / 2.0,
        ((random(&rng_state)) * span_y + bounds.s2) / 2.0
    };

    const real A = h * alpha / 3.0;
    const real B = -h * (1 - alpha) / 3.0;

    int seq_pos = 0;

    for (int i = 0; i < skip; ++i) {
        if (root_seq_size > 0) {
            point = next_point_with_seq(point, c, A, B, root_seq_size, &seq_pos, root_seq);
        } else {
            point = next_point(point, c, A, B, &rng_state);
        }
    }

    const int imageW = get_image_width (out_image);
    const int imageH = get_image_height(out_image);

    for (int i = skip, frozen = 0; i < iterations; ++i) {
        if (root_seq_size > 0) {
            point = next_point_with_seq(point, c, A, B, root_seq_size, &seq_pos, root_seq);
        } else {
            point = next_point(point, c, A, B, &rng_state);
        }

        const int2 coord = (int2)(
            (point.x - bounds.s0) / span_x * imageW,
            imageH - 1 - (int)((point.y - bounds.s2) / span_y * imageH)
        );

        if (coord.x < imageW && coord.y < imageH && coord.x >= 0 && coord.y >= 0) {
            write_imagef(out_image, coord, COLOR);
            frozen = 0;
        } else {
            if (++frozen > 15) {
                // this generally means that solution is going to approach infinity
                // printf("[OCL] error at slave %d: frozen!\n", get_global_id(0));
                break;
            }
        }
    }
}