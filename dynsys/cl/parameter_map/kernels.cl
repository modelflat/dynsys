<%include file="/common/types.clh" />
<%include file="/common/colors.clh" />

// TODO include on demand
<%include file="/common/complex.clh" />

<%include file="/common/iterate.clh" />
<%namespace name="iterate" file="/common/iterate.clh" />

<%iterate:iterate_with_periods>
    <%def name="args()">
        const point_t init,
        % for i in range(len(varied)):
        const real2_t varied_${i},
        % endfor
    </%def>
    <%def name="init()">
        point_t point = init;
        % for i, name in enumerate(varied):
        params.${name} = (
            varied_${i}.s0 +
            (real_t)(get_global_id(${i})) / (real_t)(get_global_size(${i}) - 1) * (varied_${i}.s1 - varied_${i}.s0)
        );
        % endfor
    </%def>
</%iterate:iterate_with_periods>

<%iterate:iterate_capture_with_periods>
    <%def name="args()">
        const point_t init,
        % for i in range(len(varied)):
        const real2_t varied_${i},
        % endfor
    </%def>
    <%def name="init()">
        point_t point = init;
        % for i, name in enumerate(varied):
        params.${name} = (
            varied_${i}.s0 +
            (real_t)(get_global_id(${i})) / (real_t)(get_global_size(${i}) - 1) * (varied_${i}.s1 - varied_${i}.s0)
        );
        % endfor
    </%def>
</%iterate:iterate_capture_with_periods>


#if (defined(USE_OLD_COLORS) && USE_OLD_COLORS)
#define color_for_period color_for_count_v1
#else
#define color_for_period color_for_count_v2
#endif


kernel void draw_map(
    const int iter,
    const global int* periods,
    write_only image2d_t out
) {
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float3 color = color_for_period(periods[coord.y * get_global_size(0) + coord.x], iter);
    coord.y = get_global_size(1) - 1 - coord.y;
    write_imagef(out, coord, (float4)(color, 1.0));
}

kernel void get_color(const int total, global int* count, global float* res) {
    const int id = get_global_id(0);
    res += id * 3;

    float3 color = color_for_period(count[id], total);

    res[0] = color.x;
    res[1] = color.y;
    res[2] = color.z;
}
