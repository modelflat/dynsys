<%include file="/common/bsearch.clh" />
<%include file="/common/colors.clh" />
<%include file="/common/hash.clh" />
<%include file="/common/types.clh" />
<%include file="/attractors/rotations.clh" />

// TODO include on demand
<%include file="/common/complex.clh" />

<%include file="/common/iterate.clh" />
<%namespace name="iterate" file="/common/iterate.clh" />

<%iterate:iterate_capture_with_periods>
    <%def name="args()">
        const point_t init,
        % for i in range(len(varied)):
        const real2_t varied_${i},
        % endfor
    </%def>
    <%def name="init()">
        point_t point = init;
        // bounds.s02 + uv * (bounds.s13 - bounds.s02)
        % if system.dimensions == 1 and varied:
        point = (
            varied_0.s0 + (real_t)(get_global_id(0)) / (real_t)(get_global_size(0) - 1) * (varied_0.s1 - varied_0.s0)
        );
        % else:
        % for i, idx in enumerate(varied):
        point.s${idx} = (
            varied_${i}.s0 +
            (real_t)(get_global_id(${i})) / (real_t)(get_global_size(${i}) - 1) * (varied_${i}.s1 - varied_${i}.s0)
        );
        % endfor
        % endif
    </%def>
</%iterate:iterate_capture_with_periods>

// Round points to a certain precision
kernel void round_points(
    const real_t tol,
    global real_t* points
) {
    const size_t id = get_global_id(0);
    pt_store(
        pt_round(pt_load(id, points), tol),
        id, points
    );
}

// Rotate all sequences with known periods to a "minimal rotation"
kernel void rotate_sequences(
    const int iter,
    const real_t tol,
    const global int* periods,
    global real_t* points
) {
    const int id = get_global_id(0);
    const int period = periods[id];

    if (period <= 1 || period >= iter) {
        return;
    }

    int start = find_minimal_rotation(period, id * iter, points, tol);

    if (start <= 0) {
        return;
    }

    rotate_sequence(period, id * iter, points, start);
}

// Compute sequence hashes using FNV-1a hash
kernel void hash_sequences(
    const uint iter,
    const uint table_size,
    const global uint* periods,
    const global real_t* points,
    global uint* sequence_hashes
) {
    const uint id = get_global_id(0);
    const uint period = periods[id];

    if (period < 1 || period >= iter) {
        return;
    }

    const size_t shift = id * iter;
    sequence_hashes[id] = fnv_hash(table_size - 1, period, shift, points);
}

// Count sequences using hash table approach
kernel void count_unique_sequences(
    const uint iter,
    const uint table_size,
    const global uint* periods,
    const global uint* sequence_hashes,
    global uint* table,
    global uint* table_data
) {
    const uint id = get_global_id(0);
    const uint period = periods[id];

    if (period < 1 || period >= iter) {
        return;
    }

    const uint hash = sequence_hashes[id];

    if (atomic_inc(table + hash) != 0) {
        table_data[hash] = id;
    }
}

// TODO current implementation does not really calculate the exact number of hash collisions,
// but rather it calculates the number of sequences with colliding hashes. Therefore it cannot
// be relied upon for the purposes other than verifying whether there were *any* collisions or not.
kernel void check_collisions(
    const real_t tol,
    const uint iter,
    const uint table_size,
    const global uint* periods,
    const global uint* sequence_hashes,
    const global real_t* points,
    const global uint* table,
    const global uint* table_data,
    global uint* collisions
) {
    const uint id = get_global_id(0);
    const uint period = periods[id];

    if (period < 1 || period >= iter) {
        return;
    }

    const size_t shift = id * iter;
    const uint hash = sequence_hashes[id];

    size_t shift_by_hash = table_data[hash] * iter;
    for (int i = 0; i < (int)period; ++i) {
        const POINT point1 = pt_load(shift + i, points);
        const POINT point2 = pt_load(shift_by_hash + i, points);
        if (!pt_similar(point1, point2, tol)) {
            atomic_inc(collisions);
            break;
        }
    }
}

// Find out how many sequences of each period there are
kernel void count_periods_of_unique_sequences(
    const uint iter,
    const global int* periods,
    const global uint* table,
    const global uint* table_data,
    global uint* period_counts
) {
    // TODO this kernel has very high parallelism but very little actual work to do
    const uint hash = get_global_id(0);
    if (table[hash] == 0) {
        return;
    }

    const uint period = periods[table_data[hash]];
    if (period < 1 || period >= iter) {
        return;
    }

    atomic_inc(period_counts + (uint)period - 1);
}

// Extract and align unique sequences into chunks sorted by period
kernel void gather_unique_sequences(
    const uint iter,
    const global int* periods,
    const global real_t* points,
    const global uint* table,
    const global uint* table_data,
    const global uint* period_counts,
    const global uint* hash_positions,
    const global uint* sequence_positions,
    global uint* current_positions,
    global real_t* unique_sequences,
    global uint* unique_sequences_info
) {
    const uint hash = get_global_id(0);
    const uint count = table[hash];

    if (count == 0) {
        return;
    }

    const uint id = table_data[hash];
    const int period = periods[id];

    if (period < 1 || (uint)period >= iter) {
        return;
    }

    const uint shift = id * iter;
    const uint base = (period == 1) ? 0 : sequence_positions[period - 1];
    const uint position = atomic_inc(current_positions + period - 1);

    for (int i = 0; i < period; ++i) {
        pt_store(pt_load(shift + i, points), base + position * period + i, unique_sequences);
    }

    const uint hash_base = (period == 1) ? 0 : hash_positions[period - 1];
    vstore2(
        (uint2)(hash, count),
        hash_base + position, unique_sequences_info
    );
}

// Color attractors relying on their hashes
kernel void color_attractors(
    const int n,
    const global uint* hashes,
    const global float* colors,
    const global uint* hashed_points,
    write_only image2d_t image
) {
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const int id = (get_global_size(1) - coord.y - 1) * get_global_size(0) + coord.x;

    const uint hash = hashed_points[id];

    // TODO maybe reuse table from _find_attractors here?
    const int color_no = binary_search(n, hashes, hash);

    if (color_no != -1) {
        const float4 color = hsv2rgb(vload3(color_no, colors));
        write_imagef(image, coord, color);
    } else {
        write_imagef(image, coord, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
    }
}
