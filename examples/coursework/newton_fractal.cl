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
        root = RANDOM_ROOT(rng_state);
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
    if (POINT_RADIUS > 1) {
        for (int x = -POINT_RADIUS; x <= POINT_RADIUS; ++x) {
            for (int y = -POINT_RADIUS; y <= POINT_RADIUS; ++y) {
                const int2 coord_new = (int2)(coord.x + x, coord.y + y);
                if (in_image(coord_new, image_size) && (x*x + y*y <= POINT_RADIUS*POINT_RADIUS)) {
                    write_imagef(image, coord_new, COLOR);
                }
            }
        }
    } else if (POINT_RADIUS == 1) {
        int2 coord_new = (int2)(coord.x, coord.y);
        write_imagef(image, coord_new, COLOR);

        coord_new.x = coord.x - 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, COLOR);
        }
        coord_new.x = coord.x + 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, COLOR);
        }
        coord_new.x = coord.x;
        coord_new.y = coord.y - 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, COLOR);
        }
        coord_new.y = coord.y + 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, COLOR);
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

inline ulong intercalate(uint2 coord) {
    ulong v = 0;
    // todo unroll
    for (uint i = 0, mask = 1; i < 16; mask <<= 1, ++i) {
        v |= (((coord.x & mask) << (i + 1)) | (coord.y & mask) << i);
    }
    return v;
}

inline void make_heap(global ulong* data, int n, int i) {
    while (true) {
        int smallest = i;
        int l = (i << 1) + 1;
        int r = (i << 1) + 2;

        ulong sm = data[smallest];

        if (l < n && data[l] < sm) {
            smallest = l;
        }
        if (r < n && data[r] < sm) {
            smallest = r;
        }
        if (smallest == i) {
            return; // already smallest
        }

        ulong t = *(data + i); *(data + i) = *(data + smallest); *(data + smallest) = t;

        i = smallest;
    }
}

void heap_sort(global ulong*, int);
void heap_sort(global ulong* data, int n) {
    for (int i = n / 2 - 1; i >= 0; --i) {
        make_heap(data, n, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        ulong t = *(data); *(data) = *(data + i); *(data + i) = t;
        make_heap(data, i, 0);
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

    const int2 coord = { get_global_id(0), get_global_size(1) - get_global_id(1) - 1 };
    result += (coord.y * get_global_size(0) + coord.x) * iter;

    const real2 param = point_from_id(bounds);
    const real A = param.x * param.y / 3.0;
    const real B = -param.x * (1 - param.y) / 3.0;

    int seq_pos = 0;

    real2 point = z0;

    for (int i = 0; i < skip; ++i) {
        point = next_point(point, c, A, B, &rng_state, &seq_pos, seq_size, seq);
    }

    for (int i = 0; i < iter; ++i) {
        point = next_point(point, c, A, B, &rng_state, &seq_pos, seq_size, seq);
        result[i] = intercalate(as_uint2(convert_int2_rtz(point / tol)));
    }
}

int count_unique(global ulong*, int);
int count_unique(global ulong* data, int n) {
    heap_sort(data, n);
    ulong prev = data[0];
    int uniques = 1;
    for (int i = 1; i < n; ++i) {
        ulong next = data[i];
        if (prev != next) {
            prev = next;
            ++uniques;
        }
    }
    return uniques;
}

float3 color_for_count(int, int);
float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.0;
    }
    const float d = count < 8 ? 1.0 : .5;
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

//
kernel void draw_periods(
    const int num_points,
    const global float* color_scheme,
    global ulong* points,
    write_only image2d_t out
) {
    const int2 coord = { get_global_id(0), get_global_size(1) - get_global_id(1) - 1 };
    points += (coord.y * get_global_size(0) + coord.x) * num_points;

    int unique = count_unique(points, num_points);

    float3 color = color_for_count(unique, num_points);

    //float val = unique / (float)num_points;

    write_imagef(out, coord, (float4)(color, 1.0));
}