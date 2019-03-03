#include "complex.clh"
#include "random.clh"

#define RANDOM_ROOT(rngState) ((as_uint(random(rngState))) % 3)

#define POINT_RADIUS 1

#define COLOR (float4)(0.0, 0.0, 0.0, 1.0)

inline real2 c_coef(real2 z, real2 c, real B) {
    return B / (1 + B) * c;
}

inline real2 a_coef(real2 z, real2 c, real A, real B) {
    return -(z + A*(z + cdiv(c, csq(z)))) / (1 + B);
}

inline real2 next_point_(real2 z, real2 c, real A, real B, int root_number) {
    real2 roots[3];
    solve_cubic_newton_fractal_optimized(
        a_coef(z, c, A, B), c_coef(z, c, B), 1e-8, root_number, roots
    );
    return roots[root_number];
}

inline real2 next_point(
    real2 z, real2 c, real A, real B,
    uint2* rng_state, int* seq_pos, const int seq_size, const global int* seq
) {
    int root;

    if (seq_size > 0) {
        if ((*seq_pos) >= seq_size) {
            (*seq_pos) = 0;
        }
        root = seq[(*seq_pos)++];
    } else {
        root = RANDOM_ROOT(&rng_state);
    }

    return next_point_(z, c, A, B, root);
}

inline int2 to_size(real2 point, const real4 bounds, const int2 size) {
    point = (point - bounds.s02) / (bounds.s13 - bounds.s02);
    return (int2)(
        (int)(point.x * size.x),
        size.y - (int)(point.y * size.y)
    );
}

inline bool in_image(const int2 coord, const int2 size) {
    return coord.x < size.x && coord.y < size.y && coord.x >= 0 && coord.y >= 0;

}

inline real2 point_from_id(const real4 bounds) {
    const real2 uv = {
        (real)(get_global_id(0) + 0.5) / (real)get_global_size(0),
        1.0 - ((real)(get_global_id(1) + 0.5) / (real)get_global_size(1))
    };
    return bounds.s02 + uv * (bounds.s13 - bounds.s02);
}

inline void put_point(write_only image2d_t image, const int2 coord, const int2 image_size) {
    // brute-force non-zero radius for point
    if (POINT_RADIUS > 0) {
        for (int x = -POINT_RADIUS; x <= POINT_RADIUS; ++x) {
            for (int y = -POINT_RADIUS; y <= POINT_RADIUS; ++y) {
                const int2 coord_new = (int2)(coord.x + x, coord.y + y);
                if (in_image(coord_new, image_size) && (x*x + y*y <= POINT_RADIUS*POINT_RADIUS)) {
                    write_imagef(image, coord_new, COLOR);
                }
            }
        }
    } else {
        write_imagef(image, coord, COLOR);
    }
}

// Draws newton fractal
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

    // output
    write_only image2d_t out
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    real2 point = point_from_id(bounds) / 3.0;

    const real A = h * alpha / 3.0;
    const real B = -h * (1 - alpha) / 3.0;

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