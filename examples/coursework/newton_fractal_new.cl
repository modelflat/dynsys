#include "complex.clh"
#include "random.clh"

#define RANDOM_ROOT(rng_state) ((as_uint(random(rng_state)) >> 7) % 3)

#define COLOR (float4)(0.0, 0.0, 0.0, 1.0)

#define POINT_RADIUS 1

inline real2 a_coef(real2 z, real2 c, real A, real B) {
    return -(z + A*(z + cdiv(c, csq(z)))) / (1 + B);
}

inline real2 c_coef(real2 z, real2 c, real B) {
    return B / (1 + B) * c;
}

inline real2 next_point(
    real2 z, real2 c, real2 AB,
    ulong* rng_state,
    int* seq_pos, int seq_size, const global int* seq
) {
    uint root_number;

    if (seq_size == 0) {
        // use pRNG to obtain new root index
        root_number = RANDOM_ROOT(rng_state);
    } else {
        // use sequence to obtain new root index
        if (*(seq_pos) >= seq_size) {
            *(seq_pos) = 0;
        }
        root_number = seq[(*seq_pos)++];
    }

    real2 root;
    solve_cubic_newton_fractal_optimized(
        a_coef(z, c, AB.x, AB.y),
        c_coef(z, c, AB.y),
        1e-8, root_number, &root
    );

    return root;
}

inline real2 point_from_id(const real4 bounds) {
    const real2 uv = {
        (real)get_global_id(0) / (real)get_global_size(0),
        1.0 - ((real)get_global_id(1) / (real)get_global_size(1))
    };
    return bounds.s02 + uv * (bounds.s13 - bounds.s02);
}

inline real2 prepare_AB(const real h, const real alpha) {
    return (real2)(
        h * alpha / 3.0,
        -h * (1 - alpha) / 3.0
    );
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

inline long make_state(ulong seed) {
    ulong state;
    init_state(&state, seed);
    return state;
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
    const real alpha,
    const real h,

    // root selection
    // seed
    const ulong seed,
    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    // output
    write_only image2d_t out
) {
    ulong rng_state = make_state(seed);
    int seq_pos = 0;

    // real2 point = point_from_id(bounds / 4) + 0.1 * random(&rng_state);
    real2 point = {
            ((random(&rng_state)) * (bounds.s1 - bounds.s0) + bounds.s0) / 2.0,
            ((random(&rng_state)) * (bounds.s3 - bounds.s2) + bounds.s2) / 2.0
        };

    const real2 AB = prepare_AB(h, alpha);

    for (int i = 0; i < skip; ++i) {
        point = next_point(point, c, AB, &rng_state, &seq_pos, seq_size, seq);
    }

    const int2 image_size = (int2)(get_image_width(out), get_image_height(out));

    for (int i = 0, frozen = 0; i < iter; ++i) {
        point = next_point(point, c, AB, &rng_state, &seq_pos, seq_size, seq);

        const int2 coord = to_size(point, bounds, image_size);

        if (in_image(coord, image_size)) {
            // brute-force non-zero radius for point
            if (POINT_RADIUS > 0) {
                for (int x = -POINT_RADIUS; x <= POINT_RADIUS; ++x) {
                    for (int y = -POINT_RADIUS; y <= POINT_RADIUS; ++y) {
                        const int2 coord_new = (int2)(coord.x + x, coord.y + y);
                        if (in_image(coord_new, image_size)) {
                            if (x*x + y*y <= POINT_RADIUS*POINT_RADIUS) {
                                write_imagef(out, coord_new, COLOR);
                            }
                        }
                    }
                }
            } else {
                write_imagef(out, coord, COLOR);
            }
            frozen = 0;
        } else {
            if (++frozen > 15) {
                // this likely means that solution is going to approach infinity
                // printf("[OCL] error at slave %d: frozen!\n", get_global_id(0));
                break;
            }
        }
    }
}

// Draws parameter map for out system
kernel void parameter_map(
    // algo parameters
    const int skip,
    const int periods,
    const real tolerance,

    // starting point
    real2 point,
    // c
    const real2 c,

    // surface parameters
    const real4 bounds,

    // root selection
    // seed
    const ulong seed,
    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    // color scheme for periods
    const global float* period_colors,

    // temporary storage for values
    // should be 2 * periods * group_size in size
    global long* temp,

    // output
    write_only image2d_t out
) {
    ulong rng_state = make_state(seed);
    int seq_pos = 0;

    const int2 id = (int2)(get_global_id(0), get_global_size(1) - get_global_id(1) - 1);

    const real2 h_alpha = point_from_id(bounds);

    const real2 AB = prepare_AB(h_alpha.x, h_alpha.y);

    temp += (id.y * get_global_size(0) + id.x) * periods;

    for (int i = 0; i < skip; ++i) {
        point = next_point(point, c, AB, &rng_state, &seq_pos, seq_size, seq);
    }

    for (int i = 0; i < periods; ++i) {
        point = next_point(point, c, AB, &rng_state, &seq_pos, seq_size, seq);
        temp[i] = as_long(
            convert_int2_rtz(point / tolerance)
        );
    }

    // brute-force period search
    int period = 0;
    for (int p = 1; p < periods; ++p) {

        bool this_is_it = true;
        for (int j = 0; j + p < periods; ++j) {
            int next = (j + p);// % periods;

            if (temp[j] != temp[next]) {
                this_is_it = false;
                break;
            }
        }

        if (this_is_it) {
            period = p;
            break;
        }
    }

    const float4 color = vload4(period, period_colors);
    write_imagef(out, id, color);
}
